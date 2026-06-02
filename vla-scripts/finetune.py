"""
finetune.py

Fine-tunes OpenVLA via LoRA.
"""

import os
import sys
import time
import types
import filecmp
import shutil
from collections import deque
from dataclasses import asdict, dataclass
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple, Type

import draccus
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from accelerate import PartialState
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Some `diffusers` builds assume Intel XPU support exists on `torch`.
# Our CUDA-only training environment does not expose `torch.xpu`, so add a
# no-op compatibility shim before importing modules that transitively import
# `diffusers`.
if not hasattr(torch, "xpu"):
    torch.xpu = types.SimpleNamespace(empty_cache=lambda: None)

from memory_train_utils import run_memory_training_step
from memory_vla import SpatialMemoryVLA

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.depth_projectors import DEFAULT_3D_ENCODER_CHECKPOINT
from prismatic.models.projectors import (
    NoisyActionProjector,
    ProprioProjector,
)
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from prismatic.vla.materialize_memory import get_memory_vla_dataset_and_collator

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def model_is_on_hf_hub(model_path: str) -> bool:
    try:
        HfApi().model_info(model_path)
        return True
    except Exception:
        return False


def update_auto_map(pretrained_checkpoint: str) -> None:
    if not os.path.isdir(pretrained_checkpoint):
        return

    config_path = os.path.join(pretrained_checkpoint, "config.json")
    if not os.path.exists(config_path):
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config["auto_map"] = {
        "AutoConfig": "configuration_prismatic.OpenVLAConfig",
        "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def _check_identical_files(path1: str | Path, path2: str | Path) -> bool:
    path1, path2 = Path(path1), Path(path2)
    if path1.stat().st_size != path2.stat().st_size:
        return False
    return filecmp.cmp(path1, path2, shallow=False)


def _handle_file_sync(curr_filepath: str, checkpoint_filepath: str) -> None:
    if os.path.exists(checkpoint_filepath):
        if _check_identical_files(curr_filepath, checkpoint_filepath):
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(checkpoint_filepath, f"{checkpoint_filepath}.back.{timestamp}")
    shutil.copy2(curr_filepath, checkpoint_filepath)


def check_model_logic_mismatch(pretrained_checkpoint: str) -> None:
    if not os.path.isdir(pretrained_checkpoint):
        return

    curr_files = {"modeling_prismatic.py": None, "configuration_prismatic.py": None}
    for root, _, files in os.walk("./prismatic/"):
        for filename in curr_files.keys():
            if filename in files and curr_files[filename] is None:
                curr_files[filename] = os.path.join(root, filename)

    for filename, curr_filepath in curr_files.items():
        if curr_filepath is None:
            continue
        checkpoint_filepath = os.path.join(pretrained_checkpoint, filename)
        _handle_file_sync(curr_filepath, checkpoint_filepath)


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)

    # Dataset
    data_root_dir: Path = Path("datasets/rlds")      # Directory containing RLDS datasets
    dataset_name: str = "aloha_scoop_x_into_bowl"    # Name of fine-tuning dataset (e.g., `aloha_scoop_x_into_bowl`)
    run_root_dir: Path = Path("runs")                # Path to directory to store logs & checkpoints
    shuffle_buffer_size: int = 32               # Dataloader shuffle buffer size (can reduce if OOM errors occur)

    # Algorithm and architecture
    use_l1_regression: bool = True                   # If True, trains continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, trains continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = False                        # If True, includes robot proprioceptive state in input
    use_spatial_memory: bool = False                # If True, trains the spatial-memory action policy instead of the baseline head
    depth_encoder_checkpoint: str = DEFAULT_3D_ENCODER_CHECKPOINT  # 3D encoder checkpoint used by the memory branch
    dataloader_type: str = "stream"                 # Memory dataloader mode: normal|group|stream
    group_size: int = 16                            # Temporal group size for memory dataloaders
    num_workers: int = 0                            # PyTorch DataLoader workers for the memory branch
    mem_length: int = 16                            # Spatial memory bank capacity
    retrieval_layers: int = 2                       # Cross-attention retrieval layers in the spatial memory bank
    per_token_size: int = 256                       # Width of perception tokens stored in memory
    action_l1_log_freq: int = 10                    # Frequency for sampled action L1 logging
    resume_train: bool = False                      # If True, resumes the spatial-memory branch from `resume_dir`
    resume_dir: Optional[Path] = None               # Checkpoint directory for spatial-memory resume

    # Training configuration
    batch_size: int = 8                              # Batch size per device (total batch size = batch_size * num GPUs)
    learning_rate: float = 5e-4                      # Learning rate
    lr_warmup_steps: int = 0                         # Number of steps to warm up learning rate (from 10% to 100%)
    num_steps_before_decay: int = 100_000            # Number of steps before LR decays by 10x
    grad_accumulation_steps: int = 1                 # Number of gradient accumulation steps
    max_steps: int = 200_000                         # Max number of training steps
    use_val_set: bool = False                        # If True, uses validation set and log validation metrics
    val_freq: int = 10_000                           # (When `use_val_set==True`) Validation set logging frequency in steps
    val_time_limit: int = 180                        # (When `use_val_set==True`) Time limit for computing validation metrics
    save_freq: int = 10_000                          # Checkpoint saving frequency in steps
    save_latest_checkpoint_only: bool = False        # If True, saves only 1 checkpoint, overwriting latest checkpoint
                                                     #   (If False, saves all checkpoints)
    resume: bool = False                             # If True, resumes from checkpoint
    resume_step: Optional[int] = None                # (When `resume==True`) Step number that we are resuming from
    image_aug: bool = True                           # If True, trains with image augmentations (HIGHLY RECOMMENDED)
    diffusion_sample_freq: int = 50                  # (When `use_diffusion==True`) Frequency for sampling in steps

    # LoRA
    use_lora: bool = True                            # If True, uses LoRA fine-tuning
    lora_rank: int = 32                              # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                        # Dropout applied to LoRA weights
    merge_lora_during_training: bool = True          # If True, merges LoRA weights and saves result during training
                                                     #   Note: Merging can be very slow on some machines. If so, set to
                                                     #         False and merge final checkpoint offline!

    # Logging
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    run_id_override: Optional[str] = None            # Optional string to override the run ID with
    wandb_log_freq: int = 10                         # WandB logging frequency in steps

    # fmt: on


def remove_ddp_in_checkpoint(state_dict) -> dict:
    """
    Removes the 'module.' prefix from parameter names in a PyTorch model state dictionary that was saved using
    DistributedDataParallel (DDP).

    When a model is trained using PyTorch's DistributedDataParallel, the saved state dictionary contains parameters
    prefixed with 'module.'. This function removes these prefixes to make the state dictionary compatible when
    loading into models that are not yet wrapped in DDP.

    Args:
        state_dict (dict): PyTorch model state dictionary.

    Returns:
        dict: A new state dictionary with the same contents but with 'module.' prefixes removed from parameter names.
              Parameters without the 'module.' prefix remain unchanged.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def get_run_id(cfg) -> str:
    """
    Generates or retrieves an identifier string for an experiment run.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        str: Experiment run ID.
    """
    if cfg.run_id_override is not None:
        # Override the run ID with the user-provided ID
        run_id = cfg.run_id_override
    elif cfg.resume:
        # Override run ID with the previous resumed run's ID
        run_id = cfg.vla_path.split("/")[-1]
        # Remove the "--XXX_chkpt" suffix from the run ID if it exists
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
    else:
        run_id = (
            f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_lora:
            run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
        if cfg.image_aug:
            run_id += "--image_aug"
        if cfg.run_id_note is not None:
            run_id += f"--{cfg.run_id_note}"
    return run_id


def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    """
    Loads a checkpoint for a given module.

    Args:
        module_name (str): Name of model component to load checkpoint for.
        path (str): Path to checkpoint directory.
        step (int): Gradient step number of saved checkpoint.
        device (str): String specifying how to remap storage locations (default = "cpu").

    Returns:
        dict: PyTorch model state dictionary.
    """
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)


def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    """
    Wrap a module with DistributedDataParallel.

    Args:
        module (nn.Module): PyTorch module.
        device_id (str): Device ID.
        find_unused (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)


def count_parameters(module: nn.Module, name: str) -> None:
    """
    Counts and prints the number of trainable parameters in a module.

    Args:
        module (nn.Module): PyTorch module.
        module_name (str): Name of model component.

    Returns:
        None.
    """
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")


class EpisodeShardDataset(torch.utils.data.IterableDataset):
    """Assign full episodes to a single rank based on episode id for memory training."""

    def __init__(self, base_dataset: torch.utils.data.IterableDataset, rank: int, world_size: int) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.rank = rank
        self.world_size = world_size
        self.dataset_statistics = getattr(base_dataset, "dataset_statistics", None)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for sample in self.base_dataset:
            episode_ids = sample.get("episode_ids")
            if episode_ids is None:
                yield sample
                continue
            eid = int(np.asarray(episode_ids).reshape(-1)[0])
            if eid % self.world_size == self.rank:
                yield sample


def _safe_register(register_fn: Any, *args: Any) -> None:
    try:
        register_fn(*args)
    except ValueError:
        pass


def _register_openvla_auto() -> None:
    _safe_register(AutoConfig.register, "openvla", OpenVLAConfig)
    _safe_register(AutoImageProcessor.register, OpenVLAConfig, PrismaticImageProcessor)
    _safe_register(AutoProcessor.register, OpenVLAConfig, PrismaticProcessor)
    _safe_register(AutoModelForVision2Seq.register, OpenVLAConfig, OpenVLAForActionPrediction)


def _infer_resolution(processor: AutoProcessor) -> tuple[int, int, int]:
    size = getattr(processor.image_processor, "size", None)
    if isinstance(size, dict):
        height = size.get("height") or size.get("shortest_edge") or size.get("width") or 224
        width = size.get("width") or size.get("shortest_edge") or size.get("height") or 224
    elif isinstance(size, int):
        height = width = size
    else:
        height = width = 224
    return (3, int(height), int(width))


def _enable_vla_gradient_checkpointing(vla: torch.nn.Module) -> None:
    if hasattr(vla, "gradient_checkpointing_enable"):
        vla.gradient_checkpointing_enable()
    language_model = getattr(vla, "language_model", None)
    if language_model is not None:
        if hasattr(language_model, "gradient_checkpointing_enable"):
            language_model.gradient_checkpointing_enable()
        if hasattr(language_model, "enable_input_require_grads"):
            language_model.enable_input_require_grads()
        config = getattr(language_model, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False


def _strip_checkpoint_suffix(name: str) -> str:
    if "chkpt" in name.split("--")[-1]:
        return "--".join(name.split("--")[:-1])
    return name


def _get_spatial_memory_run_id(cfg: FinetuneConfig, world_size: int) -> str:
    if cfg.run_id_override is not None:
        return cfg.run_id_override
    if cfg.resume_train and cfg.resume_dir is not None:
        return _strip_checkpoint_suffix(cfg.resume_dir.name)

    run_id = f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
    run_id += f"+b{cfg.batch_size}x{world_size}+lr-{cfg.learning_rate}"
    if cfg.use_lora:
        run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    run_id += f"+{cfg.dataloader_type}+memory3d"
    if cfg.image_aug:
        run_id += "--image_aug"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    return run_id


def _unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if isinstance(module, DDP) else module


def _find_single_checkpoint_file(checkpoint_dir: Path, prefix: str) -> Path:
    matches = sorted(checkpoint_dir.glob(f"{prefix}--*_checkpoint.pt"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 checkpoint matching '{prefix}--*_checkpoint.pt' in {checkpoint_dir}, found {len(matches)}"
        )
    return matches[0]


def _load_training_state(checkpoint_dir: Path) -> dict[str, Any]:
    training_state_path = _find_single_checkpoint_file(checkpoint_dir, "training_state")
    return torch.load(training_state_path, map_location="cpu", weights_only=False)


def _restore_memory_vla_base(
    cfg: FinetuneConfig,
    device: torch.device,
    rank: int,
) -> torch.nn.Module:
    resume_dir = cfg.resume_dir if cfg.resume_train else None
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    if resume_dir is not None:
        adapter_dir = resume_dir / "lora_adapter"
        if cfg.use_lora:
            if not adapter_dir.is_dir():
                raise FileNotFoundError(f"Missing LoRA adapter directory in resume checkpoint: {adapter_dir}")
            vla = PeftModel.from_pretrained(vla, adapter_dir, is_trainable=True)
            if rank == 0:
                vla.print_trainable_parameters()
        else:
            raise ValueError("Spatial memory resume without LoRA is not supported by the current checkpoint format.")
    elif cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        if rank == 0:
            vla.print_trainable_parameters()
    return vla


def _save_memory_training_checkpoint(
    cfg: FinetuneConfig,
    run_dir: Path,
    step: int,
    processor: AutoProcessor,
    train_dataset: torch.utils.data.IterableDataset,
    memory_vla_ddp: DDP,
    optimizer: AdamW,
    rank: int,
) -> None:
    checkpoint_dir = Path(f"{run_dir}--{step}_chkpt")
    adapter_dir = checkpoint_dir / "lora_adapter"
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        if getattr(train_dataset, "dataset_statistics", None) is not None:
            save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        processor.save_pretrained(checkpoint_dir)

        memory_vla = _unwrap_module(memory_vla_ddp)
        vla = memory_vla.vla
        if isinstance(vla, PeftModel):
            vla.save_pretrained(adapter_dir)

        for prefix, state_dict in memory_vla.get_memory_component_state_dicts().items():
            torch.save(state_dict, checkpoint_dir / f"{prefix}--{step}_checkpoint.pt")

        training_state = {
            "global_step": step,
            "optimizer": optimizer.state_dict(),
            "config": asdict(cfg),
        }
        torch.save(training_state, checkpoint_dir / f"training_state--{step}_checkpoint.pt")
    dist.barrier()


def _reduce_mean_tensor(value: torch.Tensor, world_size: int) -> torch.Tensor:
    reduced = value.detach().to(dtype=torch.float32).clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= world_size
    return reduced


def _compute_memory_autoregressive_metrics(
    output: Any,
    batch_labels: torch.Tensor,
    action_tokenizer: ActionTokenizer,
    world_size: int,
) -> dict[str, float]:
    num_condition_tokens = output.base_output.projector_features.shape[1]
    ground_truth_token_ids = batch_labels[:, 1:].to(output.base_output.logits.device)
    predicted_token_ids = output.base_output.logits[:, num_condition_tokens:-1].argmax(dim=2)

    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

    curr_action_accuracy = compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask=current_action_mask)
    next_actions_accuracy = compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask)
    curr_action_l1_loss = compute_actions_l1_loss(
        action_tokenizer,
        predicted_token_ids,
        ground_truth_token_ids,
        mask=current_action_mask,
    ).to(device=predicted_token_ids.device, dtype=torch.float32)
    next_actions_l1_loss = compute_actions_l1_loss(
        action_tokenizer,
        predicted_token_ids,
        ground_truth_token_ids,
        mask=next_actions_mask,
    ).to(device=predicted_token_ids.device, dtype=torch.float32)

    return {
        "train/curr_action_accuracy": float(_reduce_mean_tensor(curr_action_accuracy, world_size).cpu().item()),
        "train/next_actions_accuracy": float(_reduce_mean_tensor(next_actions_accuracy, world_size).cpu().item()),
        "train/curr_action_l1_loss": float(_reduce_mean_tensor(curr_action_l1_loss, world_size).cpu().item()),
        "train/next_actions_l1_loss": float(_reduce_mean_tensor(next_actions_l1_loss, world_size).cpu().item()),
    }


def _build_memory_wandb_config(
    cfg: FinetuneConfig,
    run_dir: Path,
    world_size: int,
    start_step: int,
    checkpoint_resume_step: Optional[int],
) -> dict[str, Any]:
    return {
        "run_name": run_dir.name,
        "output_dir": str(run_dir),
        "resume_train": cfg.resume_train,
        "resume_dir": str(cfg.resume_dir) if cfg.resume_dir is not None else None,
        "resume_step": cfg.resume_step if cfg.resume_step is not None else checkpoint_resume_step,
        "resume_step_arg": cfg.resume_step,
        "checkpoint_resume_step": checkpoint_resume_step,
        "start_step": start_step,
        "max_steps": cfg.max_steps,
        "remaining_steps": max(0, cfg.max_steps - start_step + 1),
        "save_freq": cfg.save_freq,
        "wandb_log_freq": cfg.wandb_log_freq,
        "action_l1_log_freq": cfg.action_l1_log_freq,
        "checkpoint_pattern": f"{run_dir.name}--<step>_chkpt",
        "data_root_dir": str(cfg.data_root_dir),
        "dataset_name": cfg.dataset_name,
        "vla_path": cfg.vla_path,
        "depth_encoder_checkpoint": cfg.depth_encoder_checkpoint,
        "dataloader_type": cfg.dataloader_type,
        "group_size": cfg.group_size,
        "batch_size_per_gpu": cfg.batch_size,
        "world_size": world_size,
        "global_batch_size": cfg.batch_size * world_size,
        "num_workers": cfg.num_workers,
        "shuffle_buffer_size": cfg.shuffle_buffer_size,
        "learning_rate": cfg.learning_rate,
        "use_lora": cfg.use_lora,
        "lora_rank": cfg.lora_rank,
        "lora_dropout": cfg.lora_dropout,
        "image_aug": cfg.image_aug,
        "mem_length": cfg.mem_length,
        "retrieval_layers": cfg.retrieval_layers,
        "per_token_size": cfg.per_token_size,
        "wandb_entity": cfg.wandb_entity,
        "wandb_project": cfg.wandb_project,
    }


def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: FinetuneConfig,
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
    find_unused_params: bool = False,
) -> DDP:
    """
    Initializes a module, optionally loads checkpoint, moves to device, and wraps with DDP.

    Args:
        module_class (Type[nn.Module]): Class of PyTorch module to initialize.
        module_name (str): Name of model component to load checkpoint for.
        cfg (FinetuneConfig): Training configuration.
        device_id (str): Device ID.
        module_args (dict): Args for initializing the module.
        to_bf16 (bool): Whether to convert to torch.bfloat16 data type.
        find_unused_params (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)

    return wrap_ddp(module, device_id, find_unused_params)


def run_forward_pass(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    batch,
    action_tokenizer,
    device_id,
    use_l1_regression,
    use_diffusion,
    use_proprio,
    use_film,
    num_patches,
    compute_diffusion_l1=False,
    num_diffusion_steps_train=None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute model forward pass and metrics for both training and validation.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        batch (dict): Input batch.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        use_l1_regression (bool): Whether to use L1 regression.
        use_diffusion (bool): Whether to use diffusion.
        use_proprio (bool): Whether to use proprioceptive state as input.
        use_film (bool): Whether to use FiLM for better language following.
        num_patches (int): Number of vision patches.
        compute_diffusion_l1 (bool): Whether to sample actions and compute L1 loss for diffusion (do this once every
                                    diffusion_sample_freq steps during training; do it every batch for validation)
        num_diffusion_steps_train (int): Number of diffusion steps for training (only used for diffusion).

    Returns:
        tuple: (loss, metrics_dict)
            loss: The loss tensor with gradient for backpropagation.
            metrics_dict: Dictionary of computed metrics (detached values for logging).
    """
    metrics = {}

    # Get ground-truth action labels
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)

    # [Only for diffusion] Sample noisy actions used as input for noise predictor network
    if use_diffusion:
        noisy_dict = action_head.module.sample_noisy_actions(ground_truth_actions)
        noise, noisy_actions, diffusion_timestep_embeddings = (
            noisy_dict["noise"],
            noisy_dict["noisy_actions"],
            noisy_dict["diffusion_timestep_embeddings"],
        )
    else:
        noise, noisy_actions, diffusion_timestep_embeddings = None, None, None

    # VLA forward pass
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"],
            output_hidden_states=True,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            noisy_actions=noisy_actions if use_diffusion else None,
            noisy_action_projector=noisy_action_projector if use_diffusion else None,
            diffusion_timestep_embeddings=diffusion_timestep_embeddings if use_diffusion else None,
            use_film=use_film,
        )

    # Get action masks needed for logging
    ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

    # Compute metrics for discrete action representation (next-token prediction)
    if not (use_l1_regression or use_diffusion):
        loss = output.loss
        predicted_token_ids = output.logits[:, num_patches:-1].argmax(dim=2)
        curr_action_accuracy = compute_token_accuracy(
            predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
        )
        curr_action_l1_loss = compute_actions_l1_loss(
            action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
        )
        next_actions_accuracy = compute_token_accuracy(
            predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask
        )
        next_actions_l1_loss = compute_actions_l1_loss(
            action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask
        )
        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
                "curr_action_accuracy": curr_action_accuracy.item(),
                "curr_action_l1_loss": curr_action_l1_loss.item(),
                "next_actions_accuracy": next_actions_accuracy.item(),
                "next_actions_l1_loss": next_actions_l1_loss.item(),
            }
        )
    # Compute metrics for continuous action representations (L1 regression | diffusion)
    else:
        # Get last layer hidden states
        last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
        # Get hidden states for text portion of prompt+response (after the vision patches)
        text_hidden_states = last_hidden_states[:, num_patches:-1]
        # Get hidden states for action portion of response
        batch_size = batch["input_ids"].shape[0]
        actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )  # (B, act_chunk_len, D)

        if use_l1_regression:
            # Predict action
            predicted_actions = action_head.module.predict_action(actions_hidden_states)
            # Get full L1 loss
            loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)

        if use_diffusion:
            # Predict noise
            noise_pred = action_head.module.predict_noise(actions_hidden_states)
            # Get diffusion noise prediction MSE loss
            noise_pred = noise_pred.reshape(noise.shape)
            loss = nn.functional.mse_loss(noise_pred, noise, reduction="mean")

            # Only sample actions and compute L1 losses if specified
            if compute_diffusion_l1:
                with torch.no_grad():
                    predicted_actions = run_diffusion_sampling(
                        vla=vla,
                        action_head=action_head,
                        noisy_action_projector=noisy_action_projector,
                        proprio_projector=proprio_projector,
                        batch=batch,
                        batch_size=batch_size,
                        num_patches=num_patches,
                        actions_shape=ground_truth_actions.shape,
                        device_id=device_id,
                        current_action_mask=current_action_mask,
                        next_actions_mask=next_actions_mask,
                        use_proprio=use_proprio,
                        use_film=use_film,
                    )

        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
            }
        )

        # Get detailed L1 losses for logging
        should_log_l1_loss = not use_diffusion or (use_diffusion and compute_diffusion_l1)
        if should_log_l1_loss:
            ground_truth_curr_action = ground_truth_actions[:, 0]
            predicted_curr_action = predicted_actions[:, 0]
            ground_truth_next_actions = ground_truth_actions[:, 1:]
            predicted_next_actions = predicted_actions[:, 1:]
            curr_action_l1_loss = torch.nn.L1Loss()(ground_truth_curr_action, predicted_curr_action)
            next_actions_l1_loss = torch.nn.L1Loss()(ground_truth_next_actions, predicted_next_actions)
            metrics.update(
                {
                    "curr_action_l1_loss": curr_action_l1_loss.item(),
                    "next_actions_l1_loss": next_actions_l1_loss.item(),
                }
            )

    # Return both the loss tensor (with gradients) and the metrics dictionary (with detached values)
    return loss, metrics


def run_diffusion_sampling(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    batch,
    batch_size,
    num_patches,
    actions_shape,
    device_id,
    current_action_mask,
    next_actions_mask,
    use_proprio,
    use_film,
) -> torch.Tensor:
    """
    Run diffusion sampling (reverse diffusion) to generate actions.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        batch (dict): Input batch.
        batch_size (int): Batch size.
        num_patches (int): Number of vision patches.
        actions_shape (tuple): Shape of ground-truth actions.
        device_id (str): Device ID.
        current_action_mask (torch.Tensor): Mask for current action.
        next_actions_mask (torch.Tensor): Mask for next actions.
        use_proprio (bool): Whether to use proprioceptive state as input.
        use_film (bool): Whether to use FiLM for better language following.

    Returns:
        torch.Tensor: Predicted actions.
    """
    # Sample random noisy action, used as the starting point for reverse diffusion
    noise = torch.randn(
        size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM),
        device=device_id,
        dtype=torch.bfloat16,
    )  # (B, chunk_len, action_dim)

    # Set diffusion timestep values
    action_head.module.noise_scheduler.set_timesteps(action_head.module.num_diffusion_steps_train)

    # Reverse diffusion: Iteratively denoise to generate action, conditioned on observation
    curr_noisy_actions = noise
    for t in action_head.module.noise_scheduler.timesteps:
        # Get diffusion model's noise prediction (conditioned on VLA latent embedding, current noisy action embedding,
        # and diffusion timestep embedding)
        timesteps = torch.Tensor([t]).repeat(batch_size).to(device_id)
        diffusion_timestep_embeddings = (
            action_head.module.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
        )  # (B, llm_dim)
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = vla(
                input_ids=batch["input_ids"].to(device_id),
                attention_mask=batch["attention_mask"].to(device_id),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                labels=batch["labels"],
                output_hidden_states=True,
                proprio=batch["proprio"] if use_proprio else None,
                proprio_projector=proprio_projector if use_proprio else None,
                noisy_actions=curr_noisy_actions,
                noisy_action_projector=noisy_action_projector,
                diffusion_timestep_embeddings=diffusion_timestep_embeddings,
                use_film=use_film,
            )
            # Get last layer hidden states
            last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
            # Get hidden states for text portion of prompt+response (after the vision patches)
            text_hidden_states = last_hidden_states[:, num_patches:-1]
            # Get hidden states for action portion of response
            actions_hidden_states = text_hidden_states[current_action_mask | next_actions_mask].reshape(
                batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1
            )  # (B, act_chunk_len, D)
            actions_hidden_states = actions_hidden_states.to(torch.bfloat16)
            # Predict noise
            noise_pred = action_head.module.predict_noise(actions_hidden_states)

        # Compute the action at the previous diffusion timestep: x_t -> x_{t-1}
        curr_noisy_actions = action_head.module.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

    return curr_noisy_actions.reshape(actions_shape)


def compute_smoothened_metrics(metrics_deques) -> dict:
    """
    Compute smoothened metrics from recent deques.

    Args:
        metrics_deques (dict): Dictionary of deques containing recent metrics.

    Returns:
        dict: Dictionary of smoothened metrics.
    """
    smoothened_metrics = {}
    for name, deque in metrics_deques.items():
        if deque and len(deque) > 0:
            smoothened_metrics[name] = sum(deque) / len(deque)
    return smoothened_metrics


def log_metrics_to_wandb(metrics, prefix, step, wandb_entity) -> None:
    """
    Log metrics to Weights & Biases.

    Args:
        metrics (dict): Dictionary of metrics to log
        prefix (str): Prefix for metric names
        step (int): Training step
        wandb_entity (str): W&B entity instance

    Returns:
        None.
    """
    log_dict = {}
    for name, value in metrics.items():
        # Map loss_value to Loss for better readability in W&B
        if name == "loss_value":
            log_dict[f"{prefix}/Loss"] = value
        # Keep other metrics as is
        else:
            log_dict[f"{prefix}/{name.replace('_', ' ').title()}"] = value
    wandb_entity.log(log_dict, step=step)


def save_training_checkpoint(
    cfg,
    run_dir,
    log_step,
    vla,
    processor,
    proprio_projector,
    noisy_action_projector,
    action_head,
    train_dataset,
    distributed_state,
) -> None:
    """
    Save all training checkpoints including model components, LoRA adapter, and dataset statistics.

    Args:
        cfg (FinetuneConfig): Training configuration.
        run_dir (Path): Experiment run directory path.
        log_step (int): Current logging step.
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        processor (PrismaticProcessor): OpenVLA inputs processor.
        proprio_projector (nn.Module): Proprioceptive state projector module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        action_head (nn.Module): Action head module.
        train_dataset (RLDSDataset): Training dataset.
        distributed_state (PartialState): Distributed training state.

    Returns:
        None.
    """
    # Determine checkpoint paths and naming
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        checkpoint_name_suffix = f"{log_step}_checkpoint.pt"

    adapter_dir = checkpoint_dir / "lora_adapter"

    # Create directories and save dataset statistics (main process only)
    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        print(f"Saving Model Checkpoint for Step {log_step}")

    # Wait for directories to be created
    dist.barrier()

    # Save model components (main process only)
    if distributed_state.is_main_process:
        # Save processor and LoRA adapter
        processor.save_pretrained(checkpoint_dir)
        vla.module.save_pretrained(adapter_dir)

        # Save other components
        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}")

        if cfg.use_diffusion and noisy_action_projector is not None:
            torch.save(
                noisy_action_projector.state_dict(), checkpoint_dir / f"noisy_action_projector--{checkpoint_name_suffix}"
            )

        if (cfg.use_l1_regression or cfg.use_diffusion) and action_head is not None:
            torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")

        if cfg.use_film:
            # To be safe, just save the entire vision backbone (not just FiLM components)
            torch.save(
                vla.module.vision_backbone.state_dict(), checkpoint_dir / f"vision_backbone--{checkpoint_name_suffix}"
            )

    # Wait for model components to be saved
    dist.barrier()

    # Merge LoRA weights into base model and save resulting model checkpoint
    # Note: Can be very slow on some devices; if so, we recommend merging offline
    if cfg.use_lora and cfg.merge_lora_during_training:
        base_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        )
        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()

        if distributed_state.is_main_process:
            merged_vla.save_pretrained(checkpoint_dir)
            print(f"Saved merged model for Step {log_step} at: {checkpoint_dir}")

        # Wait for merged model to be saved
        dist.barrier()


def run_validation(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    val_dataloader,
    action_tokenizer,
    device_id,
    cfg,
    num_patches,
    log_step,
    distributed_state,
    val_time_limit,
) -> None:
    """
    Compute validation set metrics for logging.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        val_dataloader (DataLoader): Validation data loader.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        cfg (FinetuneConfig): Training configuration.
        num_patches (int): Number of vision patches.
        log_step (int): Current logging step.
        distributed_state (PartialState): Distributed training state.
        val_time_limit (int): Time limit for computing validation metrics.

    Returns:
        None.
    """
    val_start_time = time.time()
    vla.eval()
    val_batches_count = 0

    # List to store validation metrics
    all_val_metrics = []

    with torch.no_grad():
        for batch in val_dataloader:
            # Always compute L1 loss for validation, even for diffusion
            _, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector,
                proprio_projector=proprio_projector,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_diffusion=cfg.use_diffusion,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=num_patches,
                compute_diffusion_l1=True,
                num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
            )

            # Add the loss value to the metrics
            metrics["loss"] = metrics["loss_value"]
            all_val_metrics.append(metrics)
            val_batches_count += 1

            # Cut testing on validation set short if it exceeds time limit
            if time.time() - val_start_time > val_time_limit:
                break

    # Compute average validation metrics
    avg_val_metrics = {}
    for metric_name in all_val_metrics[0].keys():
        values = [metrics[metric_name] for metrics in all_val_metrics if metric_name in metrics]
        if values:
            avg_val_metrics[metric_name] = sum(values) / len(values)

    # Add batch count to metrics
    avg_val_metrics["val_batches_count"] = val_batches_count

    # Log validation metrics to W&B
    if distributed_state.is_main_process:
        log_metrics_to_wandb(avg_val_metrics, "VLA Val", log_step, wandb)


def finetune_spatial_memory(cfg: FinetuneConfig) -> None:
    assert cfg.use_lora, "Spatial memory training currently expects LoRA to be enabled."
    assert cfg.use_spatial_memory, "Spatial memory branch should only be entered when `use_spatial_memory=True`."

    def _rank_log(message: str) -> None:
        rank_env = os.environ.get("RANK", "?")
        local_rank_env = os.environ.get("LOCAL_RANK", "?")
        print(f"[spatial-memory][rank={rank_env}][local_rank={local_rank_env}] {message}", flush=True)

    cfg.vla_path = cfg.vla_path.rstrip("/")
    if cfg.resume_train and cfg.resume_dir is None:
        raise ValueError("`resume_train=True` requires `resume_dir` to be set.")

    if model_is_on_hf_hub(cfg.vla_path):
        cfg.vla_path = snapshot_download(repo_id=cfg.vla_path)
    else:
        _register_openvla_auto()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=12))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    run_id = _get_spatial_memory_run_id(cfg, world_size)
    run_dir = cfg.run_root_dir / run_id
    if rank == 0:
        _rank_log(f"preparing run directory and checkpoint metadata at {run_dir}")
        os.makedirs(run_dir, exist_ok=True)
        (run_dir / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")
    _rank_log("waiting at post-run-dir barrier")
    dist.barrier()
    _rank_log("passed post-run-dir barrier")

    processor_path = cfg.resume_dir if cfg.resume_train else Path(cfg.vla_path)
    _rank_log(f"loading processor from {processor_path}")
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    _rank_log("processor loaded")
    _rank_log(f"loading base VLA from {cfg.vla_path}")
    vla = _restore_memory_vla_base(cfg, device, rank)
    _rank_log("base VLA loaded")
    _enable_vla_gradient_checkpointing(vla)
    _rank_log("enabled gradient checkpointing on base VLA")

    memory_vla = SpatialMemoryVLA(
        vla=vla,
        per_token_size=cfg.per_token_size,
        mem_length=cfg.mem_length,
        retrieval_layers=cfg.retrieval_layers,
        dataloader_type=cfg.dataloader_type,
        group_size=cfg.group_size,
        use_timestep_pe=True,
        fusion_type="gate",
        consolidate_type="tome",
        update_fused=False,
        depth_projectors_list=None,
        depth_encoder_checkpoint=cfg.depth_encoder_checkpoint,
    ).to(device)
    _rank_log("constructed SpatialMemoryVLA wrapper")
    if cfg.resume_train and cfg.resume_dir is not None:
        _rank_log(f"loading memory components from {cfg.resume_dir}")
        memory_vla.load_memory_components_from_checkpoint(str(cfg.resume_dir))
        _rank_log("memory components loaded")
    memory_vla.train()
    _rank_log("SpatialMemoryVLA set to train mode")

    memory_vla_ddp = DDP(
        memory_vla,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=True,
    )
    memory_vla_ddp._set_static_graph()
    _rank_log("wrapped SpatialMemoryVLA with DDP")

    _rank_log(
        f"building memory dataset: data_root={cfg.data_root_dir}, mix={cfg.dataset_name}, "
        f"dataloader_type={cfg.dataloader_type}, group_size={cfg.group_size}, "
        f"shuffle_buffer_size={cfg.shuffle_buffer_size}"
    )
    dataset, _, collator = get_memory_vla_dataset_and_collator(
        data_root_dir=cfg.data_root_dir,
        data_mix=cfg.dataset_name,
        image_transform=processor.image_processor.apply_transform,
        tokenizer=processor.tokenizer,
        prompt_builder_fn=PurePromptBuilder,
        default_image_resolution=_infer_resolution(processor),
        train=True,
        image_aug=cfg.image_aug,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        dataloader_type=cfg.dataloader_type,
        group_size=cfg.group_size,
        use_wrist_image=cfg.num_images_in_input > 1,
        use_proprio=cfg.use_proprio,
        use_depth=True,
    )
    _rank_log("memory dataset and collator constructed")
    sharded_dataset = EpisodeShardDataset(dataset, rank=rank, world_size=world_size)
    dataloader = DataLoader(
        sharded_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collator,
        num_workers=cfg.num_workers,
    )
    _rank_log("PyTorch DataLoader constructed")

    trainable_params = [param for param in memory_vla_ddp.parameters() if param.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    start_step = 1
    checkpoint_resume_step = None
    if cfg.resume_train and cfg.resume_dir is not None:
        training_state = _load_training_state(cfg.resume_dir)
        optimizer.load_state_dict(training_state["optimizer"])
        checkpoint_resume_step = int(training_state["global_step"])
        if cfg.resume_step is not None and cfg.resume_step != checkpoint_resume_step:
            raise ValueError(
                f"resume_step={cfg.resume_step} does not match checkpoint global_step={checkpoint_resume_step} "
                f"from {cfg.resume_dir}"
            )
        start_step = checkpoint_resume_step + 1

    wandb_run = None
    if rank == 0:
        wandb_run = wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=f"ft+{run_id}",
            config=_build_memory_wandb_config(cfg, run_dir, world_size, start_step, checkpoint_resume_step),
        )
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "world_size": world_size,
                    "checkpoint_resume_step": checkpoint_resume_step,
                    "start_step": start_step,
                    **asdict(cfg),
                },
                indent=2,
                default=str,
            )
        )

    for step, batch in enumerate(dataloader, start=1):
        if step < start_step:
            continue
        optimizer.zero_grad(set_to_none=True)
        output = run_memory_training_step(
            memory_vla=memory_vla_ddp,
            batch=batch,
            device=device,
            proprio_projector=None,
            use_film=False,
        )
        loss = output.loss
        if loss is None:
            raise RuntimeError("Training forward returned no loss.")

        loss.backward()
        optimizer.step()

        reduced_loss = _reduce_mean_tensor(loss, world_size)
        should_log_action_l1 = (
            cfg.action_l1_log_freq > 0
            and (step == 1 or step % cfg.action_l1_log_freq == 0 or step % cfg.save_freq == 0)
        )
        should_log = step == 1 or step % cfg.wandb_log_freq == 0 or step % cfg.save_freq == 0 or should_log_action_l1

        log_metrics = None
        if should_log:
            log_metrics = {
                "train/loss": float(reduced_loss.cpu().item()),
                "train/autoregressive_loss": float(reduced_loss.cpu().item()),
                "train/step": step,
                "train/global_batch_size": cfg.batch_size * world_size,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
            }
            if should_log_action_l1:
                log_metrics.update(
                    _compute_memory_autoregressive_metrics(
                        output=output,
                        batch_labels=batch["labels"].to(device),
                        action_tokenizer=action_tokenizer,
                        world_size=world_size,
                    )
                )

        if rank == 0 and log_metrics is not None:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": log_metrics["train/loss"],
                        "autoregressive_loss": log_metrics["train/autoregressive_loss"],
                        "curr_action_accuracy": log_metrics.get("train/curr_action_accuracy"),
                        "next_actions_accuracy": log_metrics.get("train/next_actions_accuracy"),
                        "curr_action_l1_loss": log_metrics.get("train/curr_action_l1_loss"),
                        "next_actions_l1_loss": log_metrics.get("train/next_actions_l1_loss"),
                    }
                ),
                flush=True,
            )
            wandb_run.log(log_metrics, step=step)

        if step % cfg.save_freq == 0:
            _save_memory_training_checkpoint(
                cfg=cfg,
                run_dir=run_dir,
                step=step,
                processor=processor,
                train_dataset=sharded_dataset,
                memory_vla_ddp=memory_vla_ddp,
                optimizer=optimizer,
                rank=rank,
            )

        if step >= cfg.max_steps:
            if rank == 0:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
            break

    if rank == 0 and wandb_run is not None:
        wandb_run.finish()


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    """
    Fine-tunes base VLA on demonstration dataset via LoRA.

    Allows toggling different action representations (discrete vs. continuous), different learning objectives
    (next-token prediction vs. L1 regression vs. diffusion), FiLM. Also allows for additional model inputs,
    such as additional camera images and robot proprioceptive state. Assumes parallel action generation with
    action chunking.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        None.
    """
    if cfg.use_spatial_memory:
        finetune_spatial_memory(cfg)
        return

    assert cfg.use_lora, "Only LoRA fine-tuning is supported. Please set --use_lora=True!"
    assert not (cfg.use_l1_regression and cfg.use_diffusion), (
        "Cannot do both L1 regression and diffusion. Please pick one of them!"
    )

    # Trim trailing forward slash ('/') in VLA path if it exists
    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # Get experiment run ID
    run_id = get_run_id(cfg)

    # Create experiment run directory
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    # GPU setup
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    # Initialize wandb logging
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{run_id}")

    # Print detected constants
    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPROPRIO_DIM: {PROPRIO_DIM}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )

    # Two options:
    # (1) Base model is on Hugging Face Hub
    #   - Then download it and record the path to the download directory
    # (2) Base model is stored locally
    #   - Then register model config in HF Auto Classes
    # In both cases, we want to check whether any changes have been made to
    # the `modeling_prismatic.py` file in this codebase; if so, we will copy
    # the file to the downloaded or locally stored checkpoint directory so
    # that the user's changes to the VLA class logic go into effect
    if model_is_on_hf_hub(cfg.vla_path):
        # Download model directly from Hugging Face Hub
        vla_download_path = snapshot_download(repo_id=cfg.vla_path)
        # Overwrite VLA path
        cfg.vla_path = vla_download_path
    else:
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Update config.json and sync model files
    if distributed_state.is_main_process:
        update_auto_map(cfg.vla_path)
        check_model_logic_mismatch(cfg.vla_path)

    # Wait for model files to be synced
    dist.barrier()

    # Load processor and VLA
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device_id)

    # Set number of images in VLA input
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    # LoRA setup
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # FiLM setup
    if cfg.use_film:
        count_parameters(vla.vision_backbone, "vla.vision_backbone (original)")
        # Wrap vision backbone with FiLM wrapper
        # Important: For this, must specify `vla.model.vision_backbone` instead of just `vla.vision_backbone`, since the
        # latter would cause the new wrapped backbone to be saved as a new attribute of `vla` instead of overwriting the
        # original one (due to the LoRA wrapper)
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone,
            llm_dim=vla.llm_dim,
        )
        count_parameters(vla.vision_backbone, "vla.vision_backbone (post-wrap)")
        if cfg.resume:
            state_dict = load_checkpoint("vision_backbone", cfg.vla_path, cfg.resume_step)
            vla.model.vision_backbone.load_state_dict(state_dict)
        vla.model.vision_backbone = vla.model.vision_backbone.to(device_id)

    # Wrap VLA with DDP
    vla = wrap_ddp(vla, device_id, find_unused=True)

    # If applicable, instantiate proprio projector
    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector,
            "proprio_projector",
            cfg,
            device_id,
            {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM},
        )

    # If applicable, instantiate continuous action head for L1 regression
    if cfg.use_l1_regression:
        from prismatic.models.action_heads import L1RegressionActionHead

        action_head = init_module(
            L1RegressionActionHead,
            "action_head",
            cfg,
            device_id,
            {"input_dim": vla.module.llm_dim, "hidden_dim": vla.module.llm_dim, "action_dim": ACTION_DIM},
            to_bf16=True,
        )

    # If applicable, instantiate diffusion action head and noisy action projector
    if cfg.use_diffusion:
        from prismatic.models.action_heads import DiffusionActionHead

        action_head = init_module(
            DiffusionActionHead,
            "action_head",
            cfg,
            device_id,
            {
                "input_dim": vla.module.llm_dim,
                "hidden_dim": vla.module.llm_dim,
                "action_dim": ACTION_DIM,
                "num_diffusion_steps_train": cfg.num_diffusion_steps_train,
            },
            to_bf16=True,
        )
        noisy_action_projector = init_module(
            NoisyActionProjector, "noisy_action_projector", cfg, device_id, {"llm_dim": vla.module.llm_dim}
        )

    # Get number of vision patches
    NUM_PATCHES = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
    # If we have proprio inputs, a single proprio embedding is appended to the end of the vision patch embeddings
    if cfg.use_proprio:
        NUM_PATCHES += 1
    # For diffusion, a single diffusion timestep embedding is appended to the end of the vision patch embeddings
    if cfg.use_diffusion:
        NUM_PATCHES += 1

    # Instantiate optimizer
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    if cfg.use_l1_regression or cfg.use_diffusion:
        trainable_params += [param for param in action_head.parameters() if param.requires_grad]
    if cfg.use_diffusion:
        trainable_params += [param for param in noisy_action_projector.parameters() if param.requires_grad]
    if cfg.use_proprio:
        trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Record original learning rate
    original_lr = optimizer.param_groups[0]["lr"]

    # Create learning rate scheduler
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],  # Number of steps after which LR will change
        gamma=0.1,  # Multiplicative factor of learning rate decay
    )

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # train_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder,
    # )
    # ---

    # We assume that the model takes as input one third-person camera image and 1 or 2 optional wrist camera image(s)
    use_wrist_image = cfg.num_images_in_input > 1

    # Create training and optional validation datasets
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
    )
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    if cfg.use_val_set:
        val_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size // 10,
            image_aug=cfg.image_aug,
            train=False,
        )

    # [Important] Save dataset statistics so that we can unnormalize actions during inference
    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    # Create collator and dataloader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
    )
    if cfg.use_val_set:
        val_batch_size = cfg.batch_size
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
        )

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
    }

    # Start training
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            # Compute training metrics and loss
            compute_diffusion_l1 = cfg.use_diffusion and batch_idx % cfg.diffusion_sample_freq == 0
            loss, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_diffusion=cfg.use_diffusion,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=NUM_PATCHES,
                compute_diffusion_l1=compute_diffusion_l1,
                num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
            )

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Store recent train metrics
            for metric_name, value in metrics.items():
                if metric_name in recent_metrics:
                    recent_metrics[metric_name].append(value)

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            smoothened_metrics = compute_smoothened_metrics(recent_metrics)

            # Push Metrics to W&B (every wandb_log_freq gradient steps)
            log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx
            if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                log_metrics_to_wandb(smoothened_metrics, "VLA Train", log_step, wandb)

            # [If applicable] Linearly warm up learning rate from 10% to 100% of original
            if cfg.lr_warmup_steps > 0:
                lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)  # Cap at 1.0
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            if distributed_state.is_main_process and gradient_step_idx % cfg.wandb_log_freq == 0:
                # Log the learning rate
                # Make sure to do this AFTER any learning rate modifications (e.g., warmup/decay)
                wandb.log(
                    {
                        "VLA Train/Learning Rate": scheduler.get_last_lr()[0],
                    },
                    step=log_step,
                )

            # Optimizer and LR scheduler step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()

            # Save model checkpoint: either keep latest checkpoint only or all checkpoints
            if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=log_step,
                    vla=vla,
                    processor=processor,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                    action_head=action_head if (cfg.use_l1_regression or cfg.use_diffusion) else None,
                    train_dataset=train_dataset,
                    distributed_state=distributed_state,
                )

            # Test model on validation set
            if cfg.use_val_set and log_step > 0 and log_step % cfg.val_freq == 0:
                run_validation(
                    vla=vla,
                    action_head=action_head,
                    noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    val_dataloader=val_dataloader,
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    cfg=cfg,
                    num_patches=NUM_PATCHES,
                    log_step=log_step,
                    distributed_state=distributed_state,
                    val_time_limit=cfg.val_time_limit,
                )
                # Set model back to training mode after validation
                vla.train()

            # Stop training when max_steps is reached
            if log_step == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()
