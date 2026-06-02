from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from memory_train_utils import run_memory_training_step
from memory_vla import SpatialMemoryVLA
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.depth_projectors import DEFAULT_3D_ENCODER_CHECKPOINT
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from prismatic.vla.materialize_memory import get_memory_vla_dataset_and_collator

try:
    import wandb
except ImportError:
    wandb = None


@dataclass
class TrainConfig:
    data_root: str
    dataset_name: str
    vla_path: str
    depth_encoder_checkpoint: str
    output_dir: str
    resume_train: bool
    resume_dir: str | None
    resume_step: int | None
    batch_size_per_gpu: int
    dataloader_type: str
    group_size: int
    num_workers: int
    learning_rate: float
    weight_decay: float
    repeated_diffusion_steps: int
    shuffle_buffer_size: int
    max_steps: int
    save_every: int
    log_every: int
    action_l1_log_every: int
    action_l1_num_ddim_steps: int
    use_wandb: bool
    wandb_entity: str | None
    wandb_project: str | None
    wandb_name: str | None
    use_lora: bool
    lora_rank: int
    lora_dropout: float
    image_aug: bool
    mem_length: int
    retrieval_layers: int
    per_token_size: int


class EpisodeShardDataset(IterableDataset):
    """Assign full episodes to a single DDP rank based on episode id."""

    def __init__(self, base_dataset: IterableDataset, rank: int, world_size: int) -> None:
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


def _register_openvla() -> None:
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


def _unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if isinstance(module, DDP) else module


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


def _build_config_from_env() -> TrainConfig:
    return TrainConfig(
        data_root=os.environ["DATA_ROOT"],
        dataset_name=os.environ["DATASET_NAME"],
        vla_path=os.environ["VLA_PATH"].rstrip("/"),
        depth_encoder_checkpoint=os.environ.get("DEPTH_ENCODER_CHECKPOINT", DEFAULT_3D_ENCODER_CHECKPOINT),
        output_dir=os.environ["OUTPUT_DIR"],
        resume_train=os.environ.get("RESUME_TRAIN", "false").lower() == "true",
        resume_dir=(os.environ.get("RESUME_DIR") or None)
        if os.environ.get("RESUME_TRAIN", "false").lower() == "true"
        else None,
        resume_step=(
            int(os.environ["RESUME_STEP"])
            if os.environ.get("RESUME_TRAIN", "false").lower() == "true" and os.environ.get("RESUME_STEP")
            else None
        ),
        batch_size_per_gpu=int(os.environ.get("BATCH_SIZE_PER_GPU", "1")),
        dataloader_type=os.environ.get("DATALOADER_TYPE", "stream"),
        group_size=int(os.environ.get("GROUP_SIZE", "16")),
        num_workers=int(os.environ.get("NUM_WORKERS", "0")),
        learning_rate=float(os.environ.get("LEARNING_RATE", "5e-5")),
        weight_decay=float(os.environ.get("WEIGHT_DECAY", "0.0")),
        repeated_diffusion_steps=int(os.environ.get("REPEATED_DIFFUSION_STEPS", "1")),
        shuffle_buffer_size=int(os.environ.get("SHUFFLE_BUFFER_SIZE", "32")),
        max_steps=int(os.environ.get("MAX_STEPS", "150000")),
        save_every=int(os.environ.get("SAVE_EVERY", "10000")),
        log_every=int(os.environ.get("LOG_EVERY", "10")),
        action_l1_log_every=int(os.environ.get("ACTION_L1_LOG_EVERY", os.environ.get("LOG_EVERY", "10"))),
        action_l1_num_ddim_steps=int(os.environ.get("ACTION_L1_NUM_DDIM_STEPS", "10")),
        use_wandb=os.environ.get("USE_WANDB", "false").lower() == "true",
        wandb_entity=os.environ.get("WANDB_ENTITY") or None,
        wandb_project=os.environ.get("WANDB_PROJECT") or None,
        wandb_name=os.environ.get("WANDB_NAME") or None,
        use_lora=os.environ.get("USE_LORA", "true").lower() == "true",
        lora_rank=int(os.environ.get("LORA_RANK", "32")),
        lora_dropout=float(os.environ.get("LORA_DROPOUT", "0.0")),
        image_aug=os.environ.get("IMAGE_AUG", "true").lower() == "true",
        mem_length=int(os.environ.get("MEM_LENGTH", "16")),
        retrieval_layers=int(os.environ.get("RETRIEVAL_LAYERS", "2")),
        per_token_size=int(os.environ.get("PER_TOKEN_SIZE", "256")),
    )


def _infinite(loader: DataLoader) -> Iterator[dict[str, Any]]:
    while True:
        for batch in loader:
            yield batch


def _save_checkpoint(
    cfg: TrainConfig,
    step: int,
    processor: AutoProcessor,
    train_dataset: IterableDataset,
    memory_vla_ddp: DDP,
    optimizer: AdamW,
    rank: int,
) -> None:
    checkpoint_dir = Path(f"{cfg.output_dir}--{step}_chkpt")
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


def _build_wandb_config(
    cfg: TrainConfig,
    world_size: int,
    start_step: int,
    checkpoint_resume_step: int | None,
) -> dict[str, Any]:
    derived_resume_step = checkpoint_resume_step if cfg.resume_dir is not None else None
    effective_resume_step = cfg.resume_step if cfg.resume_step is not None else derived_resume_step
    output_dir = Path(cfg.output_dir)
    return {
        "run_name": output_dir.name,
        "output_dir": str(output_dir),
        "resume_train": cfg.resume_train,
        "resume_dir": cfg.resume_dir,
        "is_resume": cfg.resume_train and cfg.resume_dir is not None,
        "resume_step": effective_resume_step,
        "resume_step_arg": cfg.resume_step,
        "checkpoint_resume_step": derived_resume_step,
        "start_step": start_step,
        "max_steps": cfg.max_steps,
        "remaining_steps": max(0, cfg.max_steps - start_step + 1),
        "save_every": cfg.save_every,
        "log_every": cfg.log_every,
        "action_l1_log_every": cfg.action_l1_log_every,
        "action_l1_num_ddim_steps": cfg.action_l1_num_ddim_steps,
        "checkpoint_pattern": f"{output_dir.name}--<step>_chkpt",
        "data_root": cfg.data_root,
        "dataset_name": cfg.dataset_name,
        "vla_path": cfg.vla_path,
        "depth_encoder_checkpoint": cfg.depth_encoder_checkpoint,
        "dataloader_type": cfg.dataloader_type,
        "group_size": cfg.group_size,
        "batch_size_per_gpu": cfg.batch_size_per_gpu,
        "world_size": world_size,
        "global_batch_size": cfg.batch_size_per_gpu * world_size,
        "num_workers": cfg.num_workers,
        "shuffle_buffer_size": cfg.shuffle_buffer_size,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "repeated_diffusion_steps": cfg.repeated_diffusion_steps,
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


def _init_wandb(
    cfg: TrainConfig,
    world_size: int,
    start_step: int,
    checkpoint_resume_step: int | None,
):
    if not cfg.use_wandb:
        return None
    if wandb is None:
        raise ImportError("wandb is not installed but USE_WANDB=true was requested.")
    if cfg.wandb_project is None:
        raise ValueError("WANDB_PROJECT must be set when USE_WANDB=true.")

    wandb_name = cfg.wandb_name or Path(cfg.output_dir).name
    return wandb.init(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        name=wandb_name,
        config=_build_wandb_config(cfg, world_size, start_step, checkpoint_resume_step),
    )


def _reduce_mean_tensor(value: torch.Tensor, world_size: int) -> torch.Tensor:
    reduced = value.detach().to(dtype=torch.float32).clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= world_size
    return reduced


def _compute_action_l1_metrics(
    memory_vla_ddp: DDP,
    output,
    world_size: int,
    num_ddim_steps: int,
) -> dict[str, float]:
    if output.action_targets is None:
        return {}

    memory_vla = _unwrap_module(memory_vla_ddp)
    predicted_actions = memory_vla.sample_actions_from_tokens(
        cognition_tokens=output.cognition_tokens.detach(),
        fused_tokens=output.fused_tokens.detach(),
        cfg_scale=1.0,
        use_ddim=True,
        num_ddim_steps=num_ddim_steps,
    )
    action_targets = output.action_targets.to(device=predicted_actions.device, dtype=predicted_actions.dtype)

    full_action_l1 = F.l1_loss(predicted_actions, action_targets)
    curr_action_l1 = F.l1_loss(predicted_actions[:, 0], action_targets[:, 0])
    if predicted_actions.shape[1] > 1:
        next_action_l1 = F.l1_loss(predicted_actions[:, 1:], action_targets[:, 1:])
    else:
        next_action_l1 = torch.zeros((), device=predicted_actions.device, dtype=predicted_actions.dtype)

    return {
        "train/action_l1_full": float(_reduce_mean_tensor(full_action_l1, world_size).cpu().item()),
        "train/curr_action_l1_loss": float(_reduce_mean_tensor(curr_action_l1, world_size).cpu().item()),
        "train/next_actions_l1_loss": float(_reduce_mean_tensor(next_action_l1, world_size).cpu().item()),
    }


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


def _restore_vla_from_checkpoint(
    cfg: TrainConfig,
    device: torch.device,
    rank: int,
) -> torch.nn.Module:
    resume_dir = Path(cfg.resume_dir) if cfg.resume_dir is not None else None
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).to(device)
    vla.vision_backbone.set_num_images_in_input(2)

    if resume_dir is not None:
        adapter_dir = resume_dir / "lora_adapter"
        if cfg.use_lora:
            if not adapter_dir.is_dir():
                raise FileNotFoundError(f"Missing LoRA adapter directory in resume checkpoint: {adapter_dir}")
            vla = PeftModel.from_pretrained(vla, adapter_dir, is_trainable=True)
            if rank == 0:
                vla.print_trainable_parameters()
        else:
            raise ValueError("Resume without LoRA is not supported by the current checkpoint format.")
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


def main() -> None:
    _register_openvla()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=12))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    cfg = _build_config_from_env()
    if cfg.resume_train and cfg.resume_dir is None:
        raise ValueError("RESUME_TRAIN=true requires RESUME_DIR to be set.")
    output_dir = Path(cfg.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    resume_dir = Path(cfg.resume_dir) if cfg.resume_dir is not None else None
    processor_path = resume_dir if resume_dir is not None else Path(cfg.vla_path)
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=False)
    vla = _restore_vla_from_checkpoint(cfg, device, rank)
    _enable_vla_gradient_checkpointing(vla)

    memory_vla = SpatialMemoryVLA(
        vla=vla,
        per_token_size=cfg.per_token_size,
        action_model_type="DiT-L",
        action_dim=ACTION_DIM,
        future_action_window_size=NUM_ACTIONS_CHUNK - 1,
        mem_length=cfg.mem_length,
        retrieval_layers=cfg.retrieval_layers,
        dataloader_type=cfg.dataloader_type,
        group_size=cfg.group_size,
        use_timestep_pe=True,
        fusion_type="gate",
        consolidate_type="tome",
        update_fused=False,
        repeated_diffusion_steps=cfg.repeated_diffusion_steps,
        depth_projectors_list=None,
        depth_encoder_checkpoint=cfg.depth_encoder_checkpoint,
    ).to(device)
    if resume_dir is not None:
        memory_vla.load_memory_components_from_checkpoint(str(resume_dir))
    memory_vla.train()

    memory_vla_ddp = DDP(
        memory_vla,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=True,
    )
    memory_vla_ddp._set_static_graph()

    dataset, _, collator = get_memory_vla_dataset_and_collator(
        data_root_dir=Path(cfg.data_root),
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
        use_wrist_image=True,
        use_proprio=True,
        use_depth=True,
    )
    sharded_dataset = EpisodeShardDataset(dataset, rank=rank, world_size=world_size)
    dataloader = DataLoader(
        sharded_dataset,
        batch_size=cfg.batch_size_per_gpu,
        collate_fn=collator,
        num_workers=cfg.num_workers,
    )
    batch_iter = _infinite(dataloader)

    trainable_params = [param for param in memory_vla_ddp.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    start_step = 1
    checkpoint_resume_step = None
    if resume_dir is not None:
        training_state = _load_training_state(resume_dir)
        optimizer.load_state_dict(training_state["optimizer"])
        checkpoint_resume_step = int(training_state["global_step"])
        if cfg.resume_step is not None and cfg.resume_step != checkpoint_resume_step:
            raise ValueError(
                f"RESUME_STEP={cfg.resume_step} does not match checkpoint global_step={checkpoint_resume_step} "
                f"from {resume_dir}"
            )
        start_step = checkpoint_resume_step + 1

    if rank == 0:
        print(
            json.dumps(
                {
                    "output_dir": cfg.output_dir,
                    "world_size": world_size,
                    "checkpoint_resume_step": checkpoint_resume_step,
                    "start_step": start_step,
                    **asdict(cfg),
                },
                indent=2,
            )
        )
    wandb_run = _init_wandb(cfg, world_size, start_step, checkpoint_resume_step) if rank == 0 else None

    for step in range(start_step, cfg.max_steps + 1):
        batch = next(batch_iter)
        optimizer.zero_grad(set_to_none=True)

        output = run_memory_training_step(
            memory_vla=memory_vla_ddp,
            batch=batch,
            device=device,
            proprio_projector=None,
            use_film=False,
            repeated_diffusion_steps=cfg.repeated_diffusion_steps,
        )
        loss = output.loss
        if loss is None:
            raise RuntimeError("Training forward returned no loss.")

        loss.backward()
        optimizer.step()

        reduced_loss = _reduce_mean_tensor(loss, world_size)

        should_log_action_l1 = (
            cfg.action_l1_log_every > 0
            and (step == 1 or step % cfg.action_l1_log_every == 0 or step % cfg.save_every == 0)
        )
        should_log = step == 1 or step % cfg.log_every == 0 or step % cfg.save_every == 0 or should_log_action_l1
        log_metrics = None
        if should_log:
            log_metrics = {
                "train/loss": float(reduced_loss.cpu().item()),
                "train/action_diffusion_loss": float(reduced_loss.cpu().item()),
                "train/step": step,
                "train/global_batch_size": cfg.batch_size_per_gpu * world_size,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
            }
            base_vla_loss = getattr(output.base_output, "loss", None)
            if base_vla_loss is not None:
                log_metrics["train/base_vla_loss"] = float(_reduce_mean_tensor(base_vla_loss, world_size).cpu().item())
            if should_log_action_l1:
                log_metrics.update(
                    _compute_action_l1_metrics(
                        memory_vla_ddp=memory_vla_ddp,
                        output=output,
                        world_size=world_size,
                        num_ddim_steps=cfg.action_l1_num_ddim_steps,
                    )
                )

        if rank == 0 and log_metrics is not None:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": log_metrics["train/loss"],
                        "action_diffusion_loss": log_metrics["train/action_diffusion_loss"],
                        "base_vla_loss": log_metrics.get("train/base_vla_loss"),
                        "curr_action_l1_loss": log_metrics.get("train/curr_action_l1_loss"),
                        "next_actions_l1_loss": log_metrics.get("train/next_actions_l1_loss"),
                    }
                ),
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(log_metrics, step=step)

        if step % cfg.save_every == 0:
            _save_checkpoint(
                cfg=cfg,
                step=step,
                processor=processor,
                train_dataset=sharded_dataset,
                memory_vla_ddp=memory_vla_ddp,
                optimizer=optimizer,
                rank=rank,
            )

    if rank == 0 and wandb_run is not None:
        wandb_run.finish()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
