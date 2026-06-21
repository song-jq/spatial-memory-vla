from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prismatic.models.encoder_3d import PointNetfeat
from prismatic.models.memory_bank import CogMemBank, GateFusion


DEFAULT_3D_ENCODER_CHECKPOINT = (
    "/home/data/users/sjq/ckpts/3dcavla/"
    "31f090d05236101ebfc381b61c674dd4746d4ce0+libero_spatial_cotdep+b8+lr-5e-05+lora-r32+dropout-0.0"
    "--image_aug--libero-spatial-cotdep-3dcavla--80000_chkpt"
)


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}


def _find_checkpoint_file(checkpoint: str, file_pattern: str) -> str:
    if os.path.isfile(checkpoint):
        return checkpoint
    if not os.path.isdir(checkpoint):
        raise FileNotFoundError(f"3D encoder checkpoint does not exist: {checkpoint}")

    matches = [
        os.path.join(checkpoint, filename)
        for filename in os.listdir(checkpoint)
        if file_pattern in filename and "checkpoint" in filename
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {file_pattern} checkpoint in {checkpoint}, found {len(matches)}")
    return matches[0]


def depth_to_point_cloud(
    depth_imgs: torch.Tensor,
    fx: float = 309.02,
    fy: float = 309.02,
    cx: float = 128.0,
    cy: float = 128.0,
) -> torch.Tensor:
    """Project depth maps shaped [B, H, W] into point clouds shaped [B, H, W, 3]."""
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


def coerce_depth_tensor(depth_maps, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not isinstance(depth_maps, torch.Tensor):
        depth_maps = torch.as_tensor(depth_maps)
    depth_maps = depth_maps.to(device=device, dtype=dtype)

    if depth_maps.ndim == 4 and depth_maps.shape[-1] == 1:
        depth_maps = depth_maps.squeeze(-1)
    elif depth_maps.ndim == 4 and depth_maps.shape[1] == 1:
        depth_maps = depth_maps.squeeze(1)
    elif depth_maps.ndim == 3 and depth_maps.shape[-1] == 1:
        depth_maps = depth_maps.squeeze(-1).unsqueeze(0)
    elif depth_maps.ndim == 2:
        depth_maps = depth_maps.unsqueeze(0)

    if depth_maps.ndim != 3:
        raise ValueError(f"Expected depth maps shaped [B, H, W], got {tuple(depth_maps.shape)}")
    return depth_maps


class DepthLatentProjector(nn.Module):
    """Project PointNet final point-wise latent tokens into the VLM latent width."""

    def __init__(self, point_hidden_dim: int = 1024, llm_dim: int = 4096) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(point_hidden_dim, llm_dim, bias=True),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim, bias=True),
        )

    def forward(self, point_tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(point_tokens)


class DepthMemoryFusion(nn.Module):
    """
    Trainable 3dcavla depth encoder + projector + perception/depth fusion + MemoryVLA bank.

    Input perception tokens should be VLM last-layer perception tokens shaped [B, N, D].
    Output memory-conditioned tokens keep the same shape and can be used as DiT per-attention tokens.
    """

    def __init__(
        self,
        llm_dim: int,
        num_depth_tokens: int,
        depth_encoder_checkpoint: Optional[str] = DEFAULT_3D_ENCODER_CHECKPOINT,
        depth_checkpoint_pattern: str = "depth_projector1",
        dataloader_type: str = "stream",
        group_size: int = 16,
        mem_length: int = 16,
        retrieval_layers: int = 2,
        use_timestep_pe: bool = True,
        consolidate_type: str = "tome",
        update_fused: bool = False,
        depth_perception_fusion_type: str = "gate",
    ) -> None:
        super().__init__()
        if depth_perception_fusion_type not in ("gate", "add"):
            raise ValueError(f"Unsupported depth_perception_fusion_type: {depth_perception_fusion_type}")
        self.llm_dim = llm_dim
        self.num_depth_tokens = num_depth_tokens
        self.depth_perception_fusion_type = depth_perception_fusion_type

        self.depth_encoder = PointNetfeat(global_feat=True, feature_transform=False, use_MLP=True, output_dim=llm_dim)
        if depth_encoder_checkpoint is not None:
            self.load_depth_encoder(depth_encoder_checkpoint, depth_checkpoint_pattern)
        self.enable_depth_encoder_training()

        self.depth_projector = DepthLatentProjector(point_hidden_dim=1024, llm_dim=llm_dim)
        self.depth_perception_gate = GateFusion(llm_dim) if self.depth_perception_fusion_type == "gate" else None
        self.memory_bank = CogMemBank(
            dataloader_type=dataloader_type,
            group_size=group_size,
            token_size=llm_dim,
            mem_length=mem_length,
            retrieval_layers=retrieval_layers,
            use_timestep_pe=use_timestep_pe,
            fusion_type="gate",
            consolidate_type=consolidate_type,
            update_fused=update_fused,
        )

    def load_depth_encoder(self, checkpoint: str, file_pattern: str) -> None:
        checkpoint_file = _find_checkpoint_file(checkpoint, file_pattern)
        state_dict = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
        self.depth_encoder.load_state_dict(_strip_module_prefix(state_dict), strict=False)

    def enable_depth_encoder_training(self) -> None:
        self.depth_encoder.train()
        for param in self.depth_encoder.parameters():
            param.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        return self

    def _depth_to_latent_tokens(self, depth_maps: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
        encoder_param = next(self.depth_encoder.parameters())
        depth_maps = coerce_depth_tensor(depth_maps, device=encoder_param.device, dtype=encoder_param.dtype)
        point_cloud = depth_to_point_cloud(depth_maps)
        batch_size, height, width, _ = point_cloud.shape
        points = point_cloud.reshape(batch_size, height * width, 3).permute(0, 2, 1)
        points = F.adaptive_avg_pool1d(points, self.num_depth_tokens)

        _, point_tokens, _, _ = self.depth_encoder.forward_with_hidden_states(points)

        projector_param = next(self.depth_projector.parameters())
        point_tokens = point_tokens.to(device=projector_param.device, dtype=projector_param.dtype)
        depth_tokens = self.depth_projector(point_tokens)
        return depth_tokens.to(dtype=target_dtype)

    def forward(
        self,
        perception_tokens: torch.Tensor,
        depth_maps: torch.Tensor,
        episode_ids: Optional[np.ndarray],
        timesteps: Optional[np.ndarray],
    ) -> torch.Tensor:
        if perception_tokens.ndim != 3:
            raise ValueError(f"Expected perception tokens shaped [B, N, D], got {tuple(perception_tokens.shape)}")

        depth_tokens = self._depth_to_latent_tokens(depth_maps, perception_tokens.dtype)
        if depth_tokens.shape[1] != perception_tokens.shape[1]:
            depth_tokens = depth_tokens.transpose(1, 2)
            depth_tokens = F.adaptive_avg_pool1d(depth_tokens, perception_tokens.shape[1]).transpose(1, 2)

        if self.depth_perception_fusion_type == "add":
            fused_tokens = 0.5 * (perception_tokens + depth_tokens)
        else:
            fused_tokens = self.depth_perception_gate(perception_tokens, depth_tokens)
        return self.memory_bank.process_batch(fused_tokens, episode_ids=episode_ids, timesteps=timesteps)


class DepthPerceptionTokenizer(nn.Module):
    """
    Trainable 3dcavla depth encoder + projector for DiT per-attention tokens.

    This module intentionally has no memory bank. It converts depth maps into 3D encoder
    latent tokens and projects them into the VLM latent width expected by ActionModel
    `per_token`.
    """

    def __init__(
        self,
        llm_dim: int,
        num_depth_tokens: int,
        depth_encoder_checkpoint: Optional[str] = DEFAULT_3D_ENCODER_CHECKPOINT,
        depth_checkpoint_pattern: str = "depth_projector1",
    ) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.num_depth_tokens = num_depth_tokens

        self.depth_encoder = PointNetfeat(global_feat=True, feature_transform=False, use_MLP=True, output_dim=llm_dim)
        if depth_encoder_checkpoint is not None:
            self.load_depth_encoder(depth_encoder_checkpoint, depth_checkpoint_pattern)
        self.enable_depth_encoder_training()

        self.depth_projector = DepthLatentProjector(point_hidden_dim=1024, llm_dim=llm_dim)

    def load_depth_encoder(self, checkpoint: str, file_pattern: str) -> None:
        checkpoint_file = _find_checkpoint_file(checkpoint, file_pattern)
        state_dict = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
        self.depth_encoder.load_state_dict(_strip_module_prefix(state_dict), strict=False)

    def enable_depth_encoder_training(self) -> None:
        self.depth_encoder.train()
        for param in self.depth_encoder.parameters():
            param.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        return self

    def forward(self, depth_maps: torch.Tensor, target_dtype: torch.dtype, target_num_tokens: Optional[int] = None):
        encoder_param = next(self.depth_encoder.parameters())
        depth_maps = coerce_depth_tensor(depth_maps, device=encoder_param.device, dtype=encoder_param.dtype)
        point_cloud = depth_to_point_cloud(depth_maps)
        batch_size, height, width, _ = point_cloud.shape
        points = point_cloud.reshape(batch_size, height * width, 3).permute(0, 2, 1)
        points = F.adaptive_avg_pool1d(points, self.num_depth_tokens)

        _, point_tokens, _, _ = self.depth_encoder.forward_with_hidden_states(points)

        projector_param = next(self.depth_projector.parameters())
        point_tokens = point_tokens.to(device=projector_param.device, dtype=projector_param.dtype)
        depth_tokens = self.depth_projector(point_tokens)

        if target_num_tokens is not None and depth_tokens.shape[1] != target_num_tokens:
            depth_tokens = depth_tokens.transpose(1, 2)
            depth_tokens = F.adaptive_avg_pool1d(depth_tokens, target_num_tokens).transpose(1, 2)

        return depth_tokens.to(dtype=target_dtype)
