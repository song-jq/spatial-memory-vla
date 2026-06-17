from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from prismatic.models.memory_bank import CogMemBank


class PerceptionMemoryFusion(nn.Module):
    """
    MemoryVLA-style memory bank over VLM last-layer perception tokens only.

    Input perception tokens should be shaped [B, N, D]. No depth encoder or
    depth projector is constructed or imported by this module.
    """

    def __init__(
        self,
        llm_dim: int,
        dataloader_type: str = "stream",
        group_size: int = 16,
        mem_length: int = 16,
        retrieval_layers: int = 2,
        use_timestep_pe: bool = True,
        consolidate_type: str = "tome",
        update_fused: bool = False,
    ) -> None:
        super().__init__()
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

    def forward(
        self,
        perception_tokens: torch.Tensor,
        episode_ids: Optional[np.ndarray],
        timesteps: Optional[np.ndarray],
    ) -> torch.Tensor:
        if perception_tokens.ndim != 3:
            raise ValueError(f"Expected perception tokens shaped [B, N, D], got {tuple(perception_tokens.shape)}")
        return self.memory_bank.process_batch(perception_tokens, episode_ids=episode_ids, timesteps=timesteps)
