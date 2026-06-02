from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


EpisodeId = Union[int, str, np.generic, torch.Tensor]
TimestepLike = Union[int, np.generic, torch.Tensor]


class TimestepEmbedder(nn.Module):
    """Embed scalar timesteps into the token feature space."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.to(next(self.mlp.parameters()).device)
        timestep_features = self.timestep_embedding(timesteps, self.frequency_embedding_size)
        timestep_features = timestep_features.to(next(self.mlp.parameters()).dtype)
        return self.mlp(timestep_features)


class CrossTransformerBlock(nn.Module):
    """Cross-attend current tokens against flattened historical memory tokens."""

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


class GateFusion(nn.Module):
    """Learned convex fusion between working tokens and retrieved memory tokens."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.proj.bias, mean=0.0, std=1e-3)

    def forward(self, current_tokens: torch.Tensor, retrieved_tokens: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.proj(torch.cat([current_tokens, retrieved_tokens], dim=-1)))
        return gate * current_tokens + (1.0 - gate) * retrieved_tokens


@dataclass
class MemoryEntry:
    timestep: Optional[int]
    feat: torch.Tensor


class SpatialMemBank(nn.Module):
    """
    Standalone spatial memory bank extracted from MemoryVLA's CogMemBank.

    Expected token shape is `[B, N, D]`, where `N` is the number of spatial tokens and `D`
    is the shared token embedding dimension.
    """

    def __init__(
        self,
        token_size: int,
        mem_length: int = 16,
        retrieval_layers: int = 2,
        dataloader_type: str = "group",
        group_size: int = 1,
        use_timestep_pe: bool = True,
        fusion_type: str = "gate",
        consolidate_type: str = "tome",
        update_fused: bool = False,
    ) -> None:
        super().__init__()
        if dataloader_type not in ("stream", "group", "normal"):
            raise ValueError(f"Unsupported dataloader_type: {dataloader_type}")
        if fusion_type not in ("gate", "add"):
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")
        if consolidate_type not in ("fifo", "tome"):
            raise ValueError(f"Unsupported consolidate_type: {consolidate_type}")
        if mem_length < 1:
            raise ValueError("mem_length must be at least 1")
        if group_size < 1:
            raise ValueError("group_size must be at least 1")

        self.token_size = token_size
        self.mem_length = mem_length
        self.retrieval_layers = retrieval_layers
        self.dataloader_type = dataloader_type
        self.group_size = group_size
        self.use_timestep_pe = use_timestep_pe
        self.fusion_type = fusion_type
        self.consolidate_type = consolidate_type
        self.update_fused = update_fused

        self.retrieval_blocks = nn.ModuleList(
            [CrossTransformerBlock(self.token_size) for _ in range(self.retrieval_layers)]
        )
        self.gate_fusion = GateFusion(self.token_size) if self.fusion_type == "gate" else None
        self.gate_fusion_blocks = self.gate_fusion
        self.timestep_encoder = (
            TimestepEmbedder(self.token_size, frequency_embedding_size=max(1, self.token_size // 4))
            if self.use_timestep_pe
            else None
        )

        self.reset()

    def reset(self) -> None:
        self.bank: Dict[Hashable, List[MemoryEntry]] = {}
        self.eid_stream: Optional[Hashable] = None

    def clear_episode(self, episode_id: EpisodeId) -> None:
        self.bank.pop(self._normalize_episode_id(episode_id), None)

    def get_episode_length(self, episode_id: EpisodeId) -> int:
        return len(self.bank.get(self._normalize_episode_id(episode_id), []))

    def process_batch(
        self,
        tokens: torch.Tensor,
        episode_ids: Sequence[EpisodeId],
        timesteps: Optional[Sequence[TimestepLike]] = None,
        depth_perception_tokens: Optional[torch.Tensor] = None,
        memory_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens with shape [B, N, D], got {tuple(tokens.shape)}")
        if len(episode_ids) != tokens.shape[0]:
            raise ValueError("episode_ids length must match batch size")
        if self.use_timestep_pe and timesteps is None:
            raise ValueError("timesteps must be provided when use_timestep_pe=True")
        if timesteps is not None and len(timesteps) != tokens.shape[0]:
            raise ValueError("timesteps length must match batch size")

        normalized_episode_ids = [self._normalize_episode_id(eid) for eid in episode_ids]
        normalized_timesteps = self._normalize_timesteps(timesteps)
        if memory_tokens is not None:
            storage_batch = self.compose_memory_tokens(
                perception_tokens=tokens,
                memory_tokens=memory_tokens,
            )
        else:
            storage_batch = None
        outputs: List[torch.Tensor] = []

        if self.training:
            self._prepare_training_window(normalized_episode_ids)

        for idx in range(tokens.shape[0]):
            eid = normalized_episode_ids[idx]
            if self.training:
                self._maybe_clear_completed_episode(idx, normalized_episode_ids)

            timestep = normalized_timesteps[idx] if normalized_timesteps is not None else None
            fused_tokens = self.process_step(
                tokens[idx],
                eid,
                timestep=timestep,
                update=True,
                depth_perception_tokens=None if depth_perception_tokens is None else depth_perception_tokens[idx],
                memory_tokens=None if storage_batch is None else storage_batch[idx],
            )
            outputs.append(fused_tokens.unsqueeze(0))

        return torch.cat(outputs, dim=0)

    def get_action_expert_input_batch(
        self,
        perception_tokens: torch.Tensor,
        episode_ids: Sequence[EpisodeId],
        timesteps: Optional[Sequence[TimestepLike]] = None,
        depth_perception_tokens: Optional[torch.Tensor] = None,
        memory_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Return the fused tokens that should be fed into an action expert.

        This matches MemoryVLA's behavior, where the perception branch passed to the action model
        is the output of the memory-bank query/fusion path rather than the raw VLM perception tokens.
        """
        return self.process_batch(
            tokens=perception_tokens,
            episode_ids=episode_ids,
            timesteps=timesteps,
            depth_perception_tokens=depth_perception_tokens,
            memory_tokens=memory_tokens,
        )

    def process_step(
        self,
        tokens: torch.Tensor,
        episode_id: EpisodeId,
        timestep: Optional[TimestepLike] = None,
        update: bool = True,
        depth_perception_tokens: Optional[torch.Tensor] = None,
        memory_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError(f"Expected tokens with shape [N, D], got {tuple(tokens.shape)}")
        if tokens.shape[-1] != self.token_size:
            raise ValueError(f"Expected token_size={self.token_size}, got {tokens.shape[-1]}")
        if self.use_timestep_pe and timestep is None:
            raise ValueError("timestep must be provided when use_timestep_pe=True")

        eid = self._normalize_episode_id(episode_id)
        normalized_timestep = self._normalize_timestep(timestep) if timestep is not None else None
        fused_feats = self.process_perception_step(tokens, eid)

        if update:
            if self.update_fused and memory_tokens is None:
                update_tokens = self.compose_memory_tokens(
                    perception_tokens=fused_feats,
                    depth_perception_tokens=depth_perception_tokens,
                )
            else:
                update_tokens = self.compose_memory_tokens(
                    perception_tokens=tokens,
                    depth_perception_tokens=depth_perception_tokens,
                    memory_tokens=memory_tokens,
                )
            self._memory_consolidate(eid, update_tokens, normalized_timestep)

        return fused_feats

    def get_action_expert_input_step(
        self,
        perception_tokens: torch.Tensor,
        episode_id: EpisodeId,
        timestep: Optional[TimestepLike] = None,
        update: bool = True,
        depth_perception_tokens: Optional[torch.Tensor] = None,
        memory_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Return the single-frame fused tokens that should be fed into an action expert.
        """
        return self.process_step(
            tokens=perception_tokens,
            episode_id=episode_id,
            timestep=timestep,
            update=update,
            depth_perception_tokens=depth_perception_tokens,
            memory_tokens=memory_tokens,
        )

    def process_perception_step(self, perception_tokens: torch.Tensor, episode_id: EpisodeId) -> torch.Tensor:
        """
        Query the memory bank with a single frame's VLM perception tokens and fuse the result.

        This mirrors MemoryVLA's `PerMemBank` flow:
        1. use current perception tokens as the query
        2. retrieve episode memory with cross-attention blocks
        3. fuse current and retrieved features with add/gate fusion
        """
        if perception_tokens.ndim != 2:
            raise ValueError(
                f"Expected perception_tokens with shape [N, D], got {tuple(perception_tokens.shape)}"
            )

        working_mem = perception_tokens.unsqueeze(0)
        retrieved_mem = self.query_perception_tokens(perception_tokens, episode_id)
        return self.fuse_with_retrieved_memory(working_mem, retrieved_mem).squeeze(0)

    def process_perception_batch(
        self,
        perception_tokens: torch.Tensor,
        episode_ids: Sequence[EpisodeId],
    ) -> torch.Tensor:
        """Apply MemoryVLA-style perception querying and fusion for each frame without updating memory."""
        if perception_tokens.ndim != 3:
            raise ValueError(
                f"Expected perception_tokens with shape [B, N, D], got {tuple(perception_tokens.shape)}"
            )
        if len(episode_ids) != perception_tokens.shape[0]:
            raise ValueError("episode_ids length must match batch size")

        normalized_episode_ids = [self._normalize_episode_id(eid) for eid in episode_ids]
        outputs = []
        for idx in range(perception_tokens.shape[0]):
            outputs.append(
                self.process_perception_step(perception_tokens[idx], normalized_episode_ids[idx]).unsqueeze(0)
            )
        return torch.cat(outputs, dim=0)

    def query_perception_tokens(self, perception_tokens: torch.Tensor, episode_id: EpisodeId) -> torch.Tensor:
        """
        Query the memory bank with VLM perception tokens using the same retrieval path as MemoryVLA.

        The returned tensor has shape `[1, N, D]` and corresponds to `retrieved_episode_mem`
        in MemoryVLA's `CogMemBank.process_batch`.
        """
        return self.retrieve(perception_tokens, episode_id)

    def fuse_with_retrieved_memory(
        self,
        current_tokens: torch.Tensor,
        retrieved_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse current perception tokens with retrieved memory using MemoryVLA's gate/add rule.
        """
        return self._fuse_tokens(current_tokens, retrieved_tokens)

    def retrieve(self, tokens: torch.Tensor, episode_id: EpisodeId) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError(f"Expected tokens with shape [N, D], got {tuple(tokens.shape)}")

        eid = self._normalize_episode_id(episode_id)
        working_mem = tokens.unsqueeze(0)
        history = self.bank.get(eid, [])
        if not history:
            return working_mem

        episode_mem = self._build_episode_memory(history, working_mem.device, working_mem.dtype)
        timestep_features = self._build_timestep_features(
            history,
            working_mem.shape[1],
            working_mem.device,
            working_mem.dtype,
        )

        query = working_mem
        for block in self.retrieval_blocks:
            query = block(query, episode_mem + timestep_features, episode_mem)
        return query

    def update(
        self,
        tokens: torch.Tensor,
        episode_id: EpisodeId,
        timestep: Optional[TimestepLike] = None,
        depth_perception_tokens: Optional[torch.Tensor] = None,
        memory_tokens: Optional[torch.Tensor] = None,
    ) -> None:
        if tokens.ndim != 2:
            raise ValueError(f"Expected tokens with shape [N, D], got {tuple(tokens.shape)}")
        normalized_timestep = self._normalize_timestep(timestep) if timestep is not None else None
        storage_tokens = self.compose_memory_tokens(
            perception_tokens=tokens,
            depth_perception_tokens=depth_perception_tokens,
            memory_tokens=memory_tokens,
        )
        self._memory_consolidate(self._normalize_episode_id(episode_id), storage_tokens, normalized_timestep)

    def compose_memory_tokens(
        self,
        perception_tokens: torch.Tensor,
        depth_perception_tokens: Optional[torch.Tensor] = None,
        memory_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build the tensor written into memory.

        By default, the storage object is the elementwise sum of the VLM perception tokens and the
        projected 3D perception tokens. If `memory_tokens` is provided explicitly, it is used as-is.
        """
        if memory_tokens is not None and depth_perception_tokens is not None:
            raise ValueError("Provide either `memory_tokens` or `depth_perception_tokens`, not both")
        if memory_tokens is not None:
            self._validate_token_shape(memory_tokens, expected_ndim=perception_tokens.ndim, name="memory_tokens")
            return memory_tokens
        if depth_perception_tokens is None:
            return perception_tokens

        self._validate_token_shape(
            depth_perception_tokens,
            expected_ndim=perception_tokens.ndim,
            name="depth_perception_tokens",
        )
        if depth_perception_tokens.shape != perception_tokens.shape:
            raise ValueError(
                "depth_perception_tokens must match perception_tokens shape: "
                f"{tuple(depth_perception_tokens.shape)} vs {tuple(perception_tokens.shape)}"
            )
        depth_perception_tokens = depth_perception_tokens.to(
            device=perception_tokens.device,
            dtype=perception_tokens.dtype,
        )
        return perception_tokens + depth_perception_tokens

    def _prepare_training_window(self, episode_ids: Sequence[Hashable]) -> None:
        if self.dataloader_type in ("group", "normal"):
            self.bank.clear()
            self.eid_stream = None
            return

        first_eid = episode_ids[0]
        if self.eid_stream is not None and self.eid_stream != first_eid:
            self.clear_episode(self.eid_stream)
        self.eid_stream = first_eid

    def _maybe_clear_completed_episode(self, index: int, episode_ids: Sequence[Hashable]) -> None:
        if self.dataloader_type == "group":
            if index > 0 and index % self.group_size == 0:
                prev_group_eid = episode_ids[index - self.group_size]
                self.clear_episode(prev_group_eid)
            return
        if self.dataloader_type == "normal":
            if index > 0:
                self.clear_episode(episode_ids[index - 1])
            return

        if index > 0 and episode_ids[index] != episode_ids[index - 1]:
            self.clear_episode(episode_ids[index - 1])
            self.eid_stream = episode_ids[index]

    def _build_episode_memory(
        self,
        history: Sequence[MemoryEntry],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        hist_feats = [entry.feat.to(device=device, dtype=dtype) for entry in history]
        token_dim = hist_feats[0].shape[-1]
        return torch.stack(hist_feats, dim=0).reshape(-1, token_dim).unsqueeze(0)

    def _build_timestep_features(
        self,
        history: Sequence[MemoryEntry],
        num_tokens_per_step: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        episode_mem = len(history) * num_tokens_per_step
        if not self.use_timestep_pe:
            return torch.zeros((1, episode_mem, self.token_size), device=device, dtype=dtype)

        hist_timesteps = [0 if entry.timestep is None else entry.timestep for entry in history]
        timesteps_tensor = torch.tensor(hist_timesteps, device=device)
        timestep_features = self.timestep_encoder(timesteps_tensor).to(dtype).unsqueeze(0)
        return timestep_features.repeat_interleave(num_tokens_per_step, dim=1)

    def _fuse_tokens(self, working_mem: torch.Tensor, retrieved_mem: torch.Tensor) -> torch.Tensor:
        if self.fusion_type == "add":
            return 0.5 * (working_mem + retrieved_mem)
        return self.gate_fusion(working_mem, retrieved_mem)

    @torch.no_grad()
    def _memory_consolidate(
        self,
        episode_id: Hashable,
        feat: torch.Tensor,
        timestep: Optional[int],
    ) -> None:
        if episode_id not in self.bank:
            self.bank[episode_id] = []

        self.bank[episode_id].append(MemoryEntry(timestep=timestep, feat=feat.detach().clone()))
        while len(self.bank[episode_id]) > self.mem_length:
            if self.consolidate_type == "fifo":
                self.bank[episode_id] = self.bank[episode_id][-self.mem_length :]
            elif self.consolidate_type == "tome":
                self._consolidate_with_token_merge(episode_id)
            else:
                raise NotImplementedError(f"Unsupported consolidate_type: {self.consolidate_type}")

    @torch.no_grad()
    def _consolidate_with_token_merge(self, episode_id: Hashable) -> None:
        bank = self.bank.get(episode_id, [])
        if len(bank) < 2:
            return

        similarities = []
        for idx in range(len(bank) - 1):
            feat_i = bank[idx].feat
            feat_j = bank[idx + 1].feat
            flattened_i = feat_i.flatten(1) if feat_i.ndim > 1 else feat_i.unsqueeze(0)
            flattened_j = feat_j.flatten(1) if feat_j.ndim > 1 else feat_j.unsqueeze(0)
            similarity = F.cosine_similarity(flattened_i, flattened_j, dim=1).mean().item()
            similarities.append(similarity)

        merge_index = int(torch.tensor(similarities).argmax().item())
        entry_i = bank[merge_index]
        entry_j = bank[merge_index + 1]
        fused_feat = 0.5 * (entry_i.feat + entry_j.feat)

        bank[merge_index] = MemoryEntry(timestep=entry_i.timestep, feat=fused_feat.detach().clone())
        bank.pop(merge_index + 1)

    @staticmethod
    def _normalize_episode_id(episode_id: EpisodeId) -> Hashable:
        if isinstance(episode_id, torch.Tensor):
            if episode_id.numel() != 1:
                raise ValueError("episode_id tensor must be scalar")
            return episode_id.item()
        if isinstance(episode_id, np.generic):
            return episode_id.item()
        return episode_id

    @staticmethod
    def _normalize_timestep(timestep: TimestepLike) -> int:
        if isinstance(timestep, torch.Tensor):
            if timestep.numel() != 1:
                raise ValueError("timestep tensor must be scalar")
            return int(timestep.item())
        if isinstance(timestep, np.generic):
            return int(timestep.item())
        return int(timestep)

    def _normalize_timesteps(self, timesteps: Optional[Sequence[TimestepLike]]) -> Optional[List[int]]:
        if timesteps is None:
            return None
        return [self._normalize_timestep(timestep) for timestep in timesteps]

    def _validate_token_shape(self, tokens: torch.Tensor, expected_ndim: int, name: str) -> None:
        if tokens.ndim != expected_ndim:
            raise ValueError(f"Expected {name} with ndim={expected_ndim}, got shape {tuple(tokens.shape)}")
        if tokens.shape[-1] != self.token_size:
            raise ValueError(f"Expected {name} last dimension {self.token_size}, got {tokens.shape[-1]}")


__all__ = [
    "CrossTransformerBlock",
    "GateFusion",
    "MemoryEntry",
    "SpatialMemBank",
    "TimestepEmbedder",
]
