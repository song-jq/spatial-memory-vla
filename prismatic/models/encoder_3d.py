"""
3D point-cloud encoder module adapted from 3dcavla.

The encoder expects point clouds in ``[batch, 3, num_points]`` format and
returns a global feature aligned to the VLA hidden size by default.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_batch_norm1d(bn: nn.BatchNorm1d, x: torch.Tensor) -> torch.Tensor:
    """Use running stats when a training batch has a single sample."""
    if bn.training and x.dim() == 2 and x.shape[0] == 1:
        return F.batch_norm(
            x,
            bn.running_mean,
            bn.running_var,
            bn.weight,
            bn.bias,
            training=False,
            momentum=bn.momentum,
            eps=bn.eps,
        )
    return bn(x)


class STN3d(nn.Module):
    """Predict a 3x3 spatial transform for xyz point clouds."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0].view(-1, 1024)

        x = F.relu(_safe_batch_norm1d(self.bn4, self.fc1(x)))
        x = F.relu(_safe_batch_norm1d(self.bn5, self.fc2(x)))
        x = self.fc3(x)

        identity = torch.eye(3, device=x.device, dtype=x.dtype).flatten().view(1, 9)
        x = x + identity.repeat(batch_size, 1)
        return x.view(-1, 3, 3)


class STNkd(nn.Module):
    """Predict a k x k feature-space transform."""

    def __init__(self, k: int = 64) -> None:
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0].view(-1, 1024)

        x = F.relu(_safe_batch_norm1d(self.bn4, self.fc1(x)))
        x = F.relu(_safe_batch_norm1d(self.bn5, self.fc2(x)))
        x = self.fc3(x)

        identity = torch.eye(self.k, device=x.device, dtype=x.dtype).flatten().view(1, self.k * self.k)
        x = x + identity.repeat(batch_size, 1)
        return x.view(-1, self.k, self.k)


class PointNetProjector(nn.Module):
    """Project PointNet features into the target model hidden dimension."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 2 * input_dim, bias=True),
            nn.GELU(),
            nn.Linear(2 * input_dim, output_dim, bias=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features)


class PointNet3DEncoder(nn.Module):
    """
    PointNet encoder copied from 3dcavla's depth projector.

    Args:
        output_dim: Global feature dimension. Use the VLA/LLM hidden size when
            appending the output as an additional token.
        global_feat: If True returns ``[B, output_dim]``. If False returns
            dense point features ``[B, 1088, N]`` as in PointNet.
        feature_transform: Enables the optional feature transform network.
        use_projector: Projects the 1024-d global PointNet feature to
            ``output_dim``.
    """

    def __init__(
        self,
        output_dim: int = 4096,
        global_feat: bool = True,
        feature_transform: bool = False,
        use_projector: bool = True,
    ) -> None:
        super().__init__()
        self.stn = STN3d()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self.projector = nn.Linear(1024, output_dim) if use_projector else nn.Identity()
        self.fstn = STNkd(k=64) if feature_transform else None

    def _encode(
        self, points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
        if points.dim() != 3 or points.size(1) != 3:
            raise ValueError(f"PointNet3DEncoder expects [B, 3, N] points, got {tuple(points.shape)}")

        n_points = points.size(2)
        spatial_transform = self.stn(points)
        x = points.transpose(2, 1)
        x = torch.bmm(x, spatial_transform)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.fstn is not None:
            feature_transform = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, feature_transform)
            x = x.transpose(2, 1)
        else:
            feature_transform = None

        point_features = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        last_hidden_states = x.transpose(2, 1).contiguous()
        global_features = torch.max(x, 2, keepdim=True)[0].view(-1, 1024)
        return global_features, point_features, last_hidden_states, spatial_transform, feature_transform, n_points

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Encode xyz points.

        Args:
            points: Tensor shaped ``[B, 3, N]``.

        Returns:
            ``(features, spatial_transform, feature_transform)``.
        """
        x, point_features, _, spatial_transform, feature_transform, n_points = self._encode(points)

        if self.global_feat:
            return self.projector(x), spatial_transform, feature_transform

        x = x.view(-1, 1024, 1).repeat(1, 1, n_points)
        return torch.cat([x, point_features], 1), spatial_transform, feature_transform

    def forward_with_hidden_states(
        self, points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return global features plus final point-wise latent tokens shaped [B, N, 1024]."""
        x, point_features, last_hidden_states, spatial_transform, feature_transform, n_points = self._encode(points)

        if self.global_feat:
            return self.projector(x), last_hidden_states, spatial_transform, feature_transform

        x = x.view(-1, 1024, 1).repeat(1, 1, n_points)
        dense_features = torch.cat([x, point_features], 1)
        return dense_features, last_hidden_states, spatial_transform, feature_transform


class PointNetfeat(PointNet3DEncoder):
    """3dcavla-compatible class name and constructor."""

    def __init__(
        self,
        global_feat: bool = True,
        feature_transform: bool = False,
        use_MLP: bool = True,
        output_dim: int = 4096,
    ) -> None:
        super().__init__(
            output_dim=output_dim,
            global_feat=global_feat,
            feature_transform=feature_transform,
            use_projector=use_MLP,
        )
