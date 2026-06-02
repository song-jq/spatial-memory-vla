"""Additive 3D OpenVLA helpers that leave the original loader path untouched."""

from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from experiments.robot.openvla_utils import (
    DEVICE,
    _apply_film_to_vla,
    _load_dataset_stats,
    normalize_proprio,
    prepare_images_for_vla,
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic_3d import OpenVLAForActionPrediction3D
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.depth_projectors import DEFAULT_3D_ENCODER_CHECKPOINT, get_depth_projectors_for_checkpoint


def _safe_register(register_fn, *args) -> None:
    try:
        register_fn(*args)
    except ValueError:
        pass


def _register_openvla_3d() -> None:
    _safe_register(AutoConfig.register, "openvla", OpenVLAConfig)
    _safe_register(AutoImageProcessor.register, OpenVLAConfig, PrismaticImageProcessor)
    _safe_register(AutoProcessor.register, OpenVLAConfig, PrismaticProcessor)
    _safe_register(AutoModelForVision2Seq.register, OpenVLAConfig, OpenVLAForActionPrediction3D)


def get_vla(cfg: Any) -> torch.nn.Module:
    """Load OpenVLA with the additive 3D-capable model class without syncing checkpoint code."""
    _register_openvla_3d()
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )

    if cfg.use_film:
        vla = _apply_film_to_vla(vla, cfg)

    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    vla.eval()
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)
    _load_dataset_stats(vla, cfg.pretrained_checkpoint)
    return vla


def get_processor(cfg: Any) -> AutoProcessor:
    _register_openvla_3d()
    return AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=False)


def get_depth_projectors(cfg: Any) -> Tuple[torch.nn.Module, torch.nn.Module]:
    depth_encoder_checkpoint = getattr(cfg, "depth_encoder_checkpoint", DEFAULT_3D_ENCODER_CHECKPOINT)
    return get_depth_projectors_for_checkpoint(depth_encoder_checkpoint, DEVICE)


def get_vla_action(
    cfg: Any,
    vla: torch.nn.Module,
    processor: Any,
    obs: dict,
    task_label: str,
    action_head: Optional[torch.nn.Module] = None,
    proprio_projector: Optional[torch.nn.Module] = None,
    depth_projectors_list: Optional[Tuple[torch.nn.Module, torch.nn.Module]] = None,
    noisy_action_projector: Optional[torch.nn.Module] = None,
    use_film: bool = False,
) -> List[np.ndarray]:
    """Mirror the original helper, with optional depth maps routed into the 3D model class."""
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
        if cfg.use_proprio:
            proprio = obs["state"]
            proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
            obs["state"] = normalize_proprio(proprio, proprio_norm_stats)
            proprio = obs["state"]

        depth_maps = None
        if getattr(cfg, "use_depth", False):
            depth_maps = (obs.get("depth_image"), obs.get("wrist_depth_image"))

        if action_head is None:
            action, _ = vla.predict_action(**inputs, unnorm_key=cfg.unnorm_key, do_sample=False)
        else:
            action, _ = vla.predict_action(
                **inputs,
                unnorm_key=cfg.unnorm_key,
                do_sample=False,
                proprio=proprio,
                proprio_projector=proprio_projector,
                depth_maps=depth_maps,
                depth_projectors_list=depth_projectors_list,
                noisy_action_projector=noisy_action_projector,
                action_head=action_head,
                use_film=use_film,
            )

    return [action[i] for i in range(min(len(action), cfg.num_open_loop_steps))]
