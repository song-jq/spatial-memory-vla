from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

DEFAULT_3D_ENCODER_CHECKPOINT = (
    "/home/data/users/sjq/ckpts/3dcavla/"
    "31f090d05236101ebfc381b61c674dd4746d4ce0+libero_spatial_cotdep+b8+lr-5e-05+lora-r32+dropout-0.0"
    "--image_aug--libero-spatial-cotdep-3dcavla--80000_chkpt"
)


def _safe_batch_norm1d(bn: nn.BatchNorm1d, x: torch.Tensor) -> torch.Tensor:
    """Fallback to running stats when the local batch has a single sample."""
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
    """Spatial transformer for canonicalizing 3D point clouds."""

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
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(_safe_batch_norm1d(self.bn4, self.fc1(x)))
        x = F.relu(_safe_batch_norm1d(self.bn5, self.fc2(x)))
        x = self.fc3(x)

        identity = Variable(
            torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))
        ).view(1, 9).repeat(batch_size, 1)
        if x.is_cuda:
            identity = identity.cuda()
        x = x + identity
        return x.view(-1, 3, 3)


class STNkd(nn.Module):
    """Spatial transformer for arbitrary feature dimension k."""

    def __init__(self, k: int = 64) -> None:
        super().__init__()
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
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(_safe_batch_norm1d(self.bn4, self.fc1(x)))
        x = F.relu(_safe_batch_norm1d(self.bn5, self.fc2(x)))
        x = self.fc3(x)

        identity = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(
            1, self.k * self.k
        ).repeat(batch_size, 1)
        if x.is_cuda:
            identity = identity.cuda()
        x = x + identity
        return x.view(-1, self.k, self.k)


class PointNetfeat(nn.Module):
    """PointNet feature encoder with an MLP projection into the OpenVLA hidden size."""

    def __init__(self, global_feat: bool = True, feature_transform: bool = False, use_mlp: bool = True) -> None:
        super().__init__()
        self.stn = STN3d()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.use_mlp = use_mlp
        self.proj = nn.Linear(1024, 4096)
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def _encode(self, x: torch.Tensor):
        num_points = x.size(2)
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        point_feat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        last_hidden_states = x.transpose(2, 1).contiguous()
        global_x = torch.max(x, 2, keepdim=True)[0]
        global_x = global_x.view(-1, 1024)
        return global_x, point_feat, last_hidden_states, trans, trans_feat, num_points

    def forward(self, x: torch.Tensor):
        global_x, point_feat, _, trans, trans_feat, num_points = self._encode(x)
        if self.global_feat:
            if self.use_mlp:
                global_x = self.proj(global_x)
            return global_x, trans, trans_feat

        expanded_global_x = global_x.view(-1, 1024, 1).repeat(1, 1, num_points)
        return torch.cat([expanded_global_x, point_feat], 1), trans, trans_feat

    def forward_with_hidden_states(self, x: torch.Tensor):
        """
        Return the standard PointNet output together with the final point-wise hidden states.

        Hidden states have shape `[B, N, 1024]` and are taken from the last Conv1d block
        before global max pooling, which makes them suitable for projecting into perception tokens.
        """
        global_x, point_feat, last_hidden_states, trans, trans_feat, num_points = self._encode(x)
        if self.global_feat:
            if self.use_mlp:
                global_x = self.proj(global_x)
            return global_x, last_hidden_states, trans, trans_feat

        expanded_global_x = global_x.view(-1, 1024, 1).repeat(1, 1, num_points)
        dense_features = torch.cat([expanded_global_x, point_feat], 1)
        return dense_features, last_hidden_states, trans, trans_feat


class DepthPerceptionProjector(nn.Module):
    """
    Project PointNet's final point-wise hidden states into the perception-token width.

    The default `perception_token_dim=256` matches MemoryVLA's `per_token_size`.
    """

    def __init__(self, point_hidden_dim: int = 1024, perception_token_dim: int = 256) -> None:
        super().__init__()
        self.point_hidden_dim = point_hidden_dim
        self.perception_token_dim = perception_token_dim
        self.fc1 = nn.Linear(self.point_hidden_dim, self.perception_token_dim, bias=True)
        self.fc2 = nn.Linear(self.perception_token_dim, self.perception_token_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, point_hidden_states: torch.Tensor) -> torch.Tensor:
        if point_hidden_states.ndim != 3:
            raise ValueError(
                f"Expected point_hidden_states with shape [B, N, {self.point_hidden_dim}], "
                f"got {tuple(point_hidden_states.shape)}"
            )
        projected_features = self.fc1(point_hidden_states)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features


def depth_to_point_cloud(
    depth_imgs: torch.Tensor,
    fx: float = 309.02,
    fy: float = 309.02,
    cx: float = 128.0,
    cy: float = 128.0,
) -> torch.Tensor:
    """Project depth maps of shape [B, H, W] into point clouds [B, H, W, 3]."""
    batch_size, height, width = depth_imgs.shape
    u = torch.arange(width, device=depth_imgs.device, dtype=depth_imgs.dtype)
    v = torch.arange(height, device=depth_imgs.device, dtype=depth_imgs.dtype)
    v, u = torch.meshgrid(v, u, indexing="ij")
    u = u.unsqueeze(0).expand(batch_size, -1, -1)
    v = v.unsqueeze(0).expand(batch_size, -1, -1)

    x = (u - cx) * depth_imgs / fx
    y = (v - cy) * depth_imgs / fy
    z = depth_imgs
    return torch.stack((x, y, z), dim=-1)


def coerce_depth_tensor(
    depth_map,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(depth_map, torch.Tensor):
        depth_map = torch.as_tensor(depth_map)
    depth_map = depth_map.to(device=device, dtype=dtype)

    if depth_map.ndim == 4 and depth_map.shape[-1] == 1:
        depth_map = depth_map.squeeze(-1)
    elif depth_map.ndim == 4 and depth_map.shape[1] == 1:
        depth_map = depth_map.squeeze(1)
    elif depth_map.ndim == 2:
        depth_map = depth_map.unsqueeze(0)

    if depth_map.ndim != 3:
        raise ValueError(f"Expected depth map with shape [B, H, W], got {tuple(depth_map.shape)}")

    return depth_map


def process_depth_features(
    projected_patch_embeddings: torch.Tensor,
    depth_map: Optional[torch.Tensor],
    depth_projector: Optional[nn.Module],
) -> torch.Tensor:
    if depth_map is None or depth_projector is None:
        return projected_patch_embeddings

    depth_map = coerce_depth_tensor(
        depth_map,
        device=projected_patch_embeddings.device,
        dtype=projected_patch_embeddings.dtype,
    )
    point_cloud = depth_to_point_cloud(depth_map)
    batch_size, height, width, _ = point_cloud.shape
    points = point_cloud.reshape(batch_size, height * width, 3).permute(0, 2, 1)
    projector_dtype = next(depth_projector.parameters()).dtype
    points = points.to(dtype=projector_dtype)

    autocast_enabled = points.is_cuda
    with torch.autocast(device_type="cuda", dtype=projector_dtype, enabled=autocast_enabled):
        with torch.no_grad():
            point_features, _, _ = depth_projector(points)

    point_features = point_features.to(projected_patch_embeddings.dtype).unsqueeze(1)
    return torch.cat((projected_patch_embeddings, point_features), dim=1)


def process_depth_perception_tokens(
    depth_map: Optional[torch.Tensor],
    depth_projector: Optional[PointNetfeat],
    depth_perception_projector: Optional[nn.Module],
    output_device: Optional[torch.device] = None,
    output_dtype: Optional[torch.dtype] = None,
) -> Optional[torch.Tensor]:
    """
    Convert a depth map into projected perception tokens using PointNet's final hidden states.

    Returns tokens of shape `[B, N, perception_token_dim]`, where `N` is the number of depth points.
    """
    if depth_map is None or depth_projector is None or depth_perception_projector is None:
        return None

    projector_param = next(depth_projector.parameters())
    depth_map = coerce_depth_tensor(
        depth_map,
        device=projector_param.device,
        dtype=projector_param.dtype,
    )
    point_cloud = depth_to_point_cloud(depth_map)
    batch_size, height, width, _ = point_cloud.shape
    points = point_cloud.reshape(batch_size, height * width, 3).permute(0, 2, 1)
    points = points.to(dtype=projector_param.dtype)

    autocast_enabled = points.is_cuda
    with torch.autocast(device_type="cuda", dtype=projector_param.dtype, enabled=autocast_enabled):
        with torch.no_grad():
            _, point_hidden_states, _, _ = depth_projector.forward_with_hidden_states(points)

    perception_param = next(depth_perception_projector.parameters())
    point_hidden_states = point_hidden_states.to(
        device=perception_param.device,
        dtype=perception_param.dtype,
    )
    with torch.autocast(device_type="cuda", dtype=perception_param.dtype, enabled=point_hidden_states.is_cuda):
        perception_tokens = depth_perception_projector(point_hidden_states)

    if output_device is not None or output_dtype is not None:
        perception_tokens = perception_tokens.to(
            device=output_device if output_device is not None else perception_tokens.device,
            dtype=output_dtype if output_dtype is not None else perception_tokens.dtype,
        )
    return perception_tokens


def append_depth_tokens(
    projected_patch_embeddings: torch.Tensor,
    depth_maps=None,
    depth_projectors_list=None,
) -> torch.Tensor:
    if depth_maps is None or depth_projectors_list is None:
        return projected_patch_embeddings

    for depth_map, depth_projector in zip(depth_maps, depth_projectors_list):
        projected_patch_embeddings = process_depth_features(
            projected_patch_embeddings,
            depth_map,
            depth_projector,
        )
    return projected_patch_embeddings


def _model_is_local_checkpoint(model_path: str) -> bool:
    return os.path.isdir(model_path)


def _find_checkpoint_file(pretrained_checkpoint: str, file_pattern: str) -> str:
    if not os.path.isdir(pretrained_checkpoint):
        raise FileNotFoundError(f"Checkpoint path must be a directory: {pretrained_checkpoint}")

    checkpoint_files = [
        os.path.join(pretrained_checkpoint, filename)
        for filename in os.listdir(pretrained_checkpoint)
        if file_pattern in filename and "checkpoint" in filename
    ]
    if len(checkpoint_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 {file_pattern} checkpoint but found {len(checkpoint_files)} in {pretrained_checkpoint}"
        )
    return checkpoint_files[0]


def _load_component_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def get_depth_projectors_for_checkpoint(
    pretrained_checkpoint: str,
    device: torch.device,
) -> Tuple[PointNetfeat, PointNetfeat]:
    """Load the pair of PointNet depth projectors saved alongside a local 3D checkpoint."""
    depth_projector1 = PointNetfeat()
    depth_projector2 = PointNetfeat()

    if not _model_is_local_checkpoint(pretrained_checkpoint):
        raise ValueError("HF Hub checkpoints are not supported for additive 3D depth projectors.")

    checkpoint_path1 = _find_checkpoint_file(pretrained_checkpoint, "depth_projector1")
    checkpoint_path2 = _find_checkpoint_file(pretrained_checkpoint, "depth_projector2")
    depth_projector1.load_state_dict(_load_component_state_dict(checkpoint_path1))
    depth_projector2.load_state_dict(_load_component_state_dict(checkpoint_path2))
    depth_projector1 = depth_projector1.to(device=device, dtype=torch.bfloat16)
    depth_projector2 = depth_projector2.to(device=device, dtype=torch.bfloat16)
    depth_projector1.eval()
    depth_projector2.eval()
    return depth_projector1, depth_projector2


__all__ = [
    "DEFAULT_3D_ENCODER_CHECKPOINT",
    "STN3d",
    "STNkd",
    "PointNetfeat",
    "DepthPerceptionProjector",
    "depth_to_point_cloud",
    "coerce_depth_tensor",
    "process_depth_features",
    "process_depth_perception_tokens",
    "append_depth_tokens",
    "get_depth_projectors_for_checkpoint",
]
