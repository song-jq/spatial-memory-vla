"""
Memory bank modules adapted from MemoryVLA.

The bank stores per-episode token histories and retrieves historical tokens with
cross-attention before fusing them back into the current tokens.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedder(nn.Module):
    """Embed scalar timesteps into token-size vectors."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(next(self.mlp.parameters()).device)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(next(self.mlp.parameters()).dtype)
        return self.mlp(t_freq)


class CrossTransformerBlock(nn.Module):
    """Cross-attention retrieval block used by MemoryVLA memory banks."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.attn_norm = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim),
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        x = self.attn_norm(query + attn_out)
        return self.ffn_norm(x + self.ffn(x))


class BottleneckSE(nn.Module):
    """Spatial bottleneck-and-squeeze module for perceptual memory tokens."""

    def __init__(self, C_in: int, C_mid: int, C_out: int) -> None:
        super().__init__()
        self.channels_out = C_out
        self.reduce = nn.Conv2d(C_in, C_mid, 1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.excite = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C_mid, max(C_mid // 16, 1), 1),
            nn.ReLU(),
            nn.Conv2d(max(C_mid // 16, 1), C_mid, 1),
            nn.Sigmoid(),
        )
        self.expand = nn.Conv2d(C_mid, C_out, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, channels_in = x.shape
        height = width = int(math.sqrt(num_tokens))
        if height * width != num_tokens:
            raise ValueError(f"BottleneckSE expects square spatial tokens, got {num_tokens}")

        x = x.reshape(batch_size, height, width, channels_in).permute(0, 3, 1, 2)
        z = self.act(self.reduce(x))
        final = self.expand(z * self.excite(z))
        return final.reshape(batch_size, self.channels_out, num_tokens).permute(0, 2, 1)


class GateFusion(nn.Module):
    """Adaptive gate fusion between current tokens and retrieved memory tokens."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.proj.bias, mean=0.0, std=1e-3)

    def forward(self, current: torch.Tensor, retrieved: torch.Tensor) -> torch.Tensor:
        scale = torch.sigmoid(self.proj(torch.cat([current, retrieved], dim=-1)))
        return scale * current + (1 - scale) * retrieved


class CogMemBank(nn.Module):
    """
    Episode-indexed memory bank copied from MemoryVLA.

    ``process_batch`` accepts current tokens shaped ``[B, N, D]`` plus matching
    episode IDs and timesteps. It returns fused tokens with the same shape.
    """

    def __init__(
        self,
        dataloader_type: str,
        group_size: int,
        token_size: int,
        mem_length: int = 16,
        retrieval_layers: int = 2,
        use_timestep_pe: bool = True,
        fusion_type: str = "gate",
        consolidate_type: str = "tome",
        update_fused: bool = False,
    ) -> None:
        super().__init__()
        if dataloader_type not in ("stream", "group"):
            raise ValueError(f"Unsupported dataloader_type: {dataloader_type}")
        if fusion_type not in ("gate", "add"):
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")
        if consolidate_type not in ("fifo", "tome"):
            raise ValueError(f"Unsupported consolidate_type: {consolidate_type}")

        self.dataloader_type = dataloader_type
        self.group_size = group_size
        self.token_size = token_size
        self.mem_length = mem_length
        self.retrieval_layers = retrieval_layers
        self.use_timestep_pe = use_timestep_pe
        self.fusion_type = fusion_type
        self.consolidate_type = consolidate_type
        self.update_fused = update_fused

        self.retrieval_blocks = nn.ModuleList(
            [CrossTransformerBlock(self.token_size) for _ in range(self.retrieval_layers)]
        )
        self.gate_fusion_blocks = GateFusion(self.token_size) if self.fusion_type == "gate" else None
        self.timestep_encoder = (
            TimestepEmbedder(self.token_size, frequency_embedding_size=self.token_size // 4)
            if self.use_timestep_pe
            else None
        )
        self.reset()

    def reset(self) -> None:
        self.bank = {}
        self.eid_stream = None

    def clear_episode(self, episode_id) -> None:
        self.bank.pop(episode_id, None)

    @torch.no_grad()
    def _consolidate_with_token_merge(self, episode_id) -> None:
        bank = self.bank.get(episode_id, [])
        if len(bank) < 2:
            return

        feats = [feat for _, feat in bank]
        sims = []
        for i in range(len(bank) - 1):
            f1 = feats[i].flatten(1) if feats[i].dim() > 1 else feats[i].unsqueeze(0)
            f2 = feats[i + 1].flatten(1) if feats[i + 1].dim() > 1 else feats[i + 1].unsqueeze(0)
            sims.append(F.cosine_similarity(f1, f2, dim=1).mean().item())

        idx_max = int(torch.tensor(sims).argmax().item())
        timestep_i, feat_i = bank[idx_max]
        _, feat_j = bank[idx_max + 1]
        bank[idx_max] = (timestep_i, (0.5 * (feat_i + feat_j)).detach().clone())
        bank.pop(idx_max + 1)

    @torch.no_grad()
    def _memory_consolidate(self, episode_id, feat: torch.Tensor, timestep: Optional[torch.Tensor]) -> None:
        if episode_id not in self.bank:
            self.bank[episode_id] = []

        self.bank[episode_id].append((timestep, feat.detach().clone()))

        while len(self.bank[episode_id]) > self.mem_length:
            if self.consolidate_type == "fifo":
                self.bank[episode_id] = self.bank[episode_id][-self.mem_length :]
            elif self.consolidate_type == "tome":
                self._consolidate_with_token_merge(episode_id)
            else:
                raise NotImplementedError(self.consolidate_type)

    def process_batch(
        self,
        tokens: torch.Tensor,
        episode_ids: np.ndarray,
        timesteps: Optional[np.ndarray],
    ) -> torch.Tensor:
        if episode_ids is None:
            raise ValueError("episode_ids must be provided")
        if self.use_timestep_pe and timesteps is None:
            raise ValueError("timesteps must be provided when use_timestep_pe=True")

        batch_size, num_tokens, dim = tokens.shape
        if dim != self.token_size:
            raise ValueError(f"Expected token dim {self.token_size}, got {dim}")

        outputs = []

        if self.training:
            if self.dataloader_type == "group":
                self.bank.clear()
                self.eid_stream = None
            elif self.dataloader_type == "stream":
                first_eid = episode_ids[0]
                if self.eid_stream is not None and self.eid_stream != first_eid:
                    self.clear_episode(self.eid_stream)
                self.eid_stream = first_eid

        for i in range(batch_size):
            episode_id = episode_ids[i]
            if self.training:
                if self.dataloader_type == "group" and i > 0 and i % self.group_size == 0:
                    self.clear_episode(episode_ids[i - self.group_size])
                if self.dataloader_type == "stream" and i > 0 and episode_ids[i] != episode_ids[i - 1]:
                    self.clear_episode(episode_ids[i - 1])
                    self.eid_stream = episode_ids[i]

            working_mem = tokens[i].unsqueeze(0)
            hist = self.bank.get(episode_id, [])

            if hist:
                hist_feats = [feat for _, feat in hist]
                episode_mem = torch.stack(hist_feats, dim=0).reshape(-1, dim).unsqueeze(0)

                if self.timestep_encoder is not None:
                    hist_timesteps = torch.tensor([t for t, _ in hist], device=working_mem.device)
                    pos_embed = self.timestep_encoder(hist_timesteps).unsqueeze(0)
                    pos_embed = pos_embed.repeat_interleave(num_tokens, dim=1)
                else:
                    pos_embed = torch.zeros_like(episode_mem)

                retrieved = working_mem
                for block in self.retrieval_blocks:
                    retrieved = block(retrieved, episode_mem + pos_embed, episode_mem)
            else:
                retrieved = working_mem

            if self.fusion_type == "add":
                fused = (working_mem + retrieved) * 0.5
            else:
                fused = self.gate_fusion_blocks(working_mem, retrieved)

            outputs.append(fused)

            timestep_i = timesteps[i] if self.use_timestep_pe else None
            self._memory_consolidate(episode_id, fused.squeeze(0) if self.update_fused else tokens[i], timestep_i)

        return torch.cat(outputs, dim=0)


class PerMemBank(CogMemBank):
    """Perceptual memory bank alias matching MemoryVLA naming."""

    pass
