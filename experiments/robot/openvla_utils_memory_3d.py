"""Parallel memory+3D OpenVLA helpers that leave the original inference path untouched."""

from typing import Any, List, Optional

import numpy as np
import torch
from transformers import AutoProcessor

from experiments.robot.openvla_utils import DEVICE, normalize_proprio, prepare_images_for_vla
from experiments.robot.openvla_utils_3d import get_depth_projectors, get_processor, get_vla
from memory_vla import SpatialMemoryVLA
from prismatic.models.depth_projectors import DEFAULT_3D_ENCODER_CHECKPOINT


def get_memory_vla(cfg: Any) -> SpatialMemoryVLA:
    base_vla = get_vla(cfg)
    depth_projectors_list = get_depth_projectors(cfg) if getattr(cfg, "use_depth", False) else None
    memory_vla = SpatialMemoryVLA(
        vla=base_vla,
        per_token_size=getattr(cfg, "per_token_size", 256),
        mem_length=getattr(cfg, "mem_length", 16),
        retrieval_layers=getattr(cfg, "retrieval_layers", 2),
        dataloader_type=getattr(cfg, "dataloader_type", "group"),
        group_size=getattr(cfg, "group_size", 16),
        use_timestep_pe=getattr(cfg, "use_timestep_pe", True),
        fusion_type=getattr(cfg, "fusion_type", "gate"),
        consolidate_type=getattr(cfg, "consolidate_type", "tome"),
        update_fused=getattr(cfg, "update_fused", False),
        depth_projectors_list=depth_projectors_list,
        depth_encoder_checkpoint=getattr(cfg, "depth_encoder_checkpoint", DEFAULT_3D_ENCODER_CHECKPOINT),
    )
    memory_vla.load_memory_components_from_checkpoint(cfg.pretrained_checkpoint)
    memory_vla.eval()
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        memory_vla = memory_vla.to(DEVICE)
    return memory_vla


def get_memory_vla_action(
    cfg: Any,
    memory_vla: SpatialMemoryVLA,
    processor: AutoProcessor,
    obs: dict,
    task_label: str,
    proprio_projector: Optional[torch.nn.Module] = None,
    use_film: bool = False,
    episode_first_frame: bool = False,
) -> List[np.ndarray]:
    with torch.inference_mode():
        all_images = [obs["full_image"]]
        if cfg.num_images_in_input > 1:
            all_images.extend([obs[k] for k in obs.keys() if "wrist_image" in k])

        all_images = prepare_images_for_vla(all_images, cfg)
        primary_image = all_images.pop(0)
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
        inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)

        if all_images:
            all_wrist_inputs = [processor(prompt, image).to(DEVICE, dtype=torch.bfloat16) for image in all_images]
            inputs["pixel_values"] = torch.cat(
                [inputs["pixel_values"]] + [wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs],
                dim=1,
            )

        proprio = None
        if getattr(cfg, "use_proprio", False):
            proprio = obs["state"]
            proprio_norm_stats = memory_vla.vla.norm_stats[cfg.unnorm_key]["proprio"]
            obs["state"] = normalize_proprio(proprio, proprio_norm_stats)
            proprio = torch.as_tensor(obs["state"], device=DEVICE, dtype=torch.bfloat16).reshape(1, -1)

        depth_maps = None
        if getattr(cfg, "use_depth", False):
            depth_maps = (obs.get("depth_image"), obs.get("wrist_depth_image"))

        actions, _, _ = memory_vla.predict_action(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            episode_first_frame=episode_first_frame,
            unnorm_key=cfg.unnorm_key,
            depth_maps=depth_maps,
            proprio=proprio,
            proprio_projector=proprio_projector,
            use_film=use_film,
        )

    return [actions[i] for i in range(min(len(actions), cfg.num_open_loop_steps))]


__all__ = [
    "get_memory_vla",
    "get_memory_vla_action",
    "get_processor",
]
