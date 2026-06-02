"""Robot evaluation helpers that route action queries through the memory+3D policy path."""

import os
import random
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from experiments.robot.openvla_utils_memory_3d import get_memory_vla, get_memory_vla_action

ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

MODEL_IMAGE_SIZES = {
    "openvla": 224,
}


def set_seed_everywhere(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_model(cfg: Any, wrap_diffusion_policy_for_droid: bool = False) -> torch.nn.Module:
    if cfg.model_family == "openvla":
        model = get_memory_vla(cfg)
    else:
        raise ValueError(f"Unsupported model family: {cfg.model_family}")
    print(f"Loaded model: {type(model)}")
    return model


def get_image_resize_size(cfg: Any) -> Union[int, tuple]:
    if cfg.model_family not in MODEL_IMAGE_SIZES:
        raise ValueError(f"Unsupported model family: {cfg.model_family}")
    return MODEL_IMAGE_SIZES[cfg.model_family]


def get_action(
    cfg: Any,
    model: torch.nn.Module,
    obs: Dict[str, Any],
    task_label: str,
    processor: Optional[Any] = None,
    proprio_projector: Optional[torch.nn.Module] = None,
    use_film: bool = False,
    episode_first_frame: bool = False,
) -> Union[List[np.ndarray], np.ndarray]:
    with torch.no_grad():
        if cfg.model_family == "openvla":
            return get_memory_vla_action(
                cfg=cfg,
                memory_vla=model,
                processor=processor,
                obs=obs,
                task_label=task_label,
                proprio_projector=proprio_projector,
                use_film=use_film,
                episode_first_frame=episode_first_frame,
            )
        raise ValueError(f"Unsupported model family: {cfg.model_family}")


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    normalized_action = action.copy()
    normalized_action[..., -1] = 2 * normalized_action[..., -1] - 1
    if binarize:
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])
    return normalized_action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    inverted_action = action.copy()
    inverted_action[..., -1] *= -1.0
    return inverted_action
