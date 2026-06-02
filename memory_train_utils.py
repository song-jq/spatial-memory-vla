from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from memory_vla import MemoryActionPolicyOutput, SpatialMemoryVLA


def build_memory_metadata(
    batch_size: int,
    episode_id: int = 0,
    start_timestep: int = 0,
) -> Tuple[Sequence[int], Sequence[int]]:
    episode_ids = [episode_id for _ in range(batch_size)]
    timesteps = [start_timestep + idx for idx in range(batch_size)]
    return episode_ids, timesteps


def run_memory_training_step(
    memory_vla: SpatialMemoryVLA,
    batch: Dict[str, Any],
    device: torch.device,
    proprio_projector: Optional[torch.nn.Module] = None,
    use_film: bool = False,
) -> MemoryActionPolicyOutput:
    actions = batch["actions"].to(device=device, dtype=torch.bfloat16)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    pixel_values = batch["pixel_values"].to(device=device, dtype=torch.bfloat16)
    labels = batch.get("labels")
    if labels is not None:
        labels = labels.to(device)

    episode_ids = batch.get("episode_ids")
    timesteps = batch.get("timesteps")
    if episode_ids is None or timesteps is None:
        episode_ids, timesteps = build_memory_metadata(batch_size=input_ids.shape[0])

    proprio = batch.get("proprio")
    if proprio is not None:
        proprio = proprio.to(device=device, dtype=torch.bfloat16)

    depth_maps = None
    if "depth_maps" in batch:
        depth_maps = [batch["depth_maps"].to(device=device, dtype=torch.bfloat16)]
        if "depth_maps_wrist" in batch:
            depth_maps.append(batch["depth_maps_wrist"].to(device=device, dtype=torch.bfloat16))

    return memory_vla(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        labels=labels,
        actions=actions,
        episode_ids=episode_ids,
        timesteps=timesteps,
        depth_maps=depth_maps,
        proprio=proprio,
        proprio_projector=proprio_projector,
        use_film=use_film,
    )


__all__ = [
    "build_memory_metadata",
    "run_memory_training_step",
]
