from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from memory import SpatialMemBank
from prismatic.extern.hf.modeling_prismatic import PrismaticCausalLMOutputWithPast
from prismatic.models.depth_projectors import (
    DEFAULT_3D_ENCODER_CHECKPOINT,
    DepthPerceptionProjector,
    get_depth_projectors_for_checkpoint,
    process_depth_perception_tokens,
)
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK


class VLMPerceptionProjector(nn.Module):
    """Project VLA projector features into the perception-token width used by the memory bank."""

    def __init__(self, input_dim: int, perception_token_dim: int = 256) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.perception_token_dim = perception_token_dim
        self.fc1 = nn.Linear(self.input_dim, self.perception_token_dim, bias=True)
        self.fc2 = nn.Linear(self.perception_token_dim, self.perception_token_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, projector_features: torch.Tensor) -> torch.Tensor:
        if projector_features.ndim != 3:
            raise ValueError(f"Expected projector_features with shape [B, N, D], got {tuple(projector_features.shape)}")
        projector_features = projector_features.to(
            device=self.fc1.weight.device,
            dtype=self.fc1.weight.dtype,
        )
        projected_features = self.fc1(projector_features)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features


class MemoryConditionProjector(nn.Module):
    """Project memory-bank fused perception tokens back into the VLA language-model width."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim, bias=True)
        self.fc2 = nn.Linear(output_dim, output_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, fused_tokens: torch.Tensor) -> torch.Tensor:
        fused_tokens = fused_tokens.to(device=self.fc1.weight.device, dtype=self.fc1.weight.dtype)
        projected_tokens = self.fc1(fused_tokens)
        projected_tokens = self.act_fn1(projected_tokens)
        projected_tokens = self.fc2(projected_tokens)
        return projected_tokens


@dataclass
class MemoryActionPolicyOutput:
    loss: Optional[torch.Tensor]
    base_output: Any
    cognition_tokens: torch.Tensor
    perception_tokens: torch.Tensor
    depth_perception_tokens: Optional[torch.Tensor]
    fused_tokens: torch.Tensor
    action_targets: Optional[torch.Tensor] = None
    action_loss: Optional[torch.Tensor] = None


class SpatialMemoryVLA(nn.Module):
    """
    Parallel memory-enabled execution flow for the spatial-memory-vla project.

    This wrapper keeps the base OpenVLA model intact and adds:
    - a VLM perception-token projector
    - a 3D hidden-state projector for depth maps
    - a single SpatialMemBank
    - an autoregressive OpenVLA action head conditioned on memory-queried fused tokens
    """

    def __init__(
        self,
        vla: nn.Module,
        per_token_size: int = 256,
        mem_length: int = 16,
        retrieval_layers: int = 2,
        dataloader_type: str = "group",
        group_size: int = 16,
        use_timestep_pe: bool = True,
        fusion_type: str = "gate",
        consolidate_type: str = "tome",
        update_fused: bool = False,
        depth_projectors_list: Optional[Sequence[nn.Module]] = None,
        depth_encoder_checkpoint: str = DEFAULT_3D_ENCODER_CHECKPOINT,
    ) -> None:
        super().__init__()
        self.vla = vla
        self.per_token_size = per_token_size
        self.depth_encoder_checkpoint = depth_encoder_checkpoint
        self.dataloader_type = dataloader_type
        self.group_size = group_size

        self.llm_dim = getattr(self.vla, "llm_dim", None)
        if self.llm_dim is None:
            self.llm_dim = self.vla.language_model.config.hidden_size

        self.perception_projector = VLMPerceptionProjector(self.llm_dim, self.per_token_size)
        self.depth_perception_projector = DepthPerceptionProjector(perception_token_dim=self.per_token_size)
        self.memory_condition_projector = MemoryConditionProjector(self.per_token_size, self.llm_dim)
        self.spatial_mem_bank = SpatialMemBank(
            token_size=self.per_token_size,
            mem_length=mem_length,
            retrieval_layers=retrieval_layers,
            dataloader_type=self.dataloader_type,
            group_size=self.group_size,
            use_timestep_pe=use_timestep_pe,
            fusion_type=fusion_type,
            consolidate_type=consolidate_type,
            update_fused=update_fused,
        )
        self.cur_timestep = 0

        self.depth_projectors_list = nn.ModuleList(depth_projectors_list or [])
        self._align_auxiliary_modules_to_vla()
        self._freeze_3d_encoder_modules()

    def _get_vla_reference_param(self) -> Optional[torch.nn.Parameter]:
        try:
            return next(self.vla.parameters())
        except StopIteration:
            return None

    def _align_auxiliary_modules_to_vla(self) -> None:
        reference_param = self._get_vla_reference_param()
        if reference_param is None:
            return
        module_kwargs = {"device": reference_param.device, "dtype": reference_param.dtype}
        self.perception_projector = self.perception_projector.to(**module_kwargs)
        self.depth_perception_projector = self.depth_perception_projector.to(**module_kwargs)
        self.memory_condition_projector = self.memory_condition_projector.to(**module_kwargs)
        self.spatial_mem_bank = self.spatial_mem_bank.to(**module_kwargs)

    def _freeze_3d_encoder_modules(self) -> None:
        for projector in self.depth_projectors_list:
            projector.requires_grad_(False)
            projector.eval()

    def train(self, mode: bool = True) -> "SpatialMemoryVLA":
        super().train(mode)
        self._freeze_3d_encoder_modules()
        return self

    def reset_memory(self) -> None:
        self.spatial_mem_bank.reset()
        self.cur_timestep = 0

    def set_depth_projectors(self, depth_projectors_list: Sequence[nn.Module]) -> None:
        self.depth_projectors_list = nn.ModuleList(depth_projectors_list)
        self._freeze_3d_encoder_modules()

    def _get_depth_encoder_device(self) -> torch.device:
        try:
            return next(self.depth_perception_projector.parameters()).device
        except StopIteration:
            pass
        try:
            return next(self.vla.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def load_depth_projectors_from_checkpoint(self, checkpoint_path: Optional[str] = None) -> None:
        checkpoint_path = checkpoint_path or self.depth_encoder_checkpoint
        depth_projectors = get_depth_projectors_for_checkpoint(
            checkpoint_path,
            self._get_depth_encoder_device(),
        )
        self.set_depth_projectors(depth_projectors)

    def load_memory_components_from_checkpoint(self, pretrained_checkpoint: str) -> None:
        from experiments.robot.openvla_utils import (
            find_checkpoint_file,
            load_component_state_dict,
            model_is_on_hf_hub,
        )

        if model_is_on_hf_hub(pretrained_checkpoint):
            return

        component_map = {
            "vlm_perception_projector": self.perception_projector,
            "depth_perception_projector": self.depth_perception_projector,
            "memory_condition_projector": self.memory_condition_projector,
            "spatial_mem_bank": self.spatial_mem_bank,
        }
        for prefix, module in component_map.items():
            try:
                checkpoint_path = find_checkpoint_file(pretrained_checkpoint, prefix)
            except FileNotFoundError:
                continue
            module.load_state_dict(load_component_state_dict(checkpoint_path), strict=True)

    def get_memory_component_state_dicts(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "vlm_perception_projector": self.perception_projector.state_dict(),
            "depth_perception_projector": self.depth_perception_projector.state_dict(),
            "memory_condition_projector": self.memory_condition_projector.state_dict(),
            "spatial_mem_bank": self.spatial_mem_bank.state_dict(),
        }

    def _forward_base_vla(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        proprio=None,
        proprio_projector=None,
        use_film: bool = False,
    ):
        common_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            output_hidden_states=True,
            output_projector_features=True,
            proprio=proprio,
            proprio_projector=proprio_projector,
            use_film=use_film,
        )
        depth_kwargs = dict(depth_maps=None, depth_projectors_list=None)
        try:
            return self.vla(**common_kwargs, **depth_kwargs)
        except TypeError as exc:
            if "depth_maps" not in str(exc) and "depth_projectors_list" not in str(exc):
                raise
            return self.vla(**common_kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        actions: Optional[torch.FloatTensor],
        episode_ids: Sequence[Any],
        timesteps: Sequence[Any],
        labels: Optional[torch.LongTensor] = None,
        depth_maps: Optional[Sequence[torch.Tensor]] = None,
        proprio=None,
        proprio_projector=None,
        use_film: bool = False,
    ) -> MemoryActionPolicyOutput:
        memory_inputs = self.encode_memory_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            episode_ids=episode_ids,
            timesteps=timesteps,
            depth_maps=depth_maps,
            proprio=proprio,
            proprio_projector=proprio_projector,
            use_film=use_film,
        )
        conditioned_patch_embeddings = self._build_conditioned_patch_embeddings(
            fused_tokens=memory_inputs.fused_tokens,
            original_projected_patch_embeddings=memory_inputs.base_output.projector_features,
        )
        conditioned_output = self._forward_autoregressive_with_memory(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            conditioned_patch_embeddings=conditioned_patch_embeddings,
        )
        memory_inputs.loss = conditioned_output.loss
        memory_inputs.base_output = conditioned_output
        return memory_inputs

    @torch.inference_mode()
    def predict_action(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        episode_first_frame: bool = False,
        unnorm_key: Optional[str] = None,
        depth_maps: Optional[Sequence[torch.Tensor]] = None,
        proprio=None,
        proprio_projector=None,
        use_film: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, MemoryActionPolicyOutput]:
        if episode_first_frame:
            self.reset_memory()

        device = input_ids.device
        timestep = torch.tensor(self.cur_timestep, device=device)
        self.cur_timestep += 1
        episode_ids = [0]
        timesteps = [timestep]

        action_input_ids, action_attention_mask, labels = self._prepare_action_prediction_inputs(input_ids, attention_mask)
        memory_inputs = self.encode_memory_inputs(
            input_ids=action_input_ids,
            attention_mask=action_attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            episode_ids=episode_ids,
            timesteps=timesteps,
            depth_maps=depth_maps,
            proprio=proprio,
            proprio_projector=proprio_projector,
            use_film=use_film,
        )
        conditioned_patch_embeddings = self._build_conditioned_patch_embeddings(
            fused_tokens=memory_inputs.fused_tokens,
            original_projected_patch_embeddings=memory_inputs.base_output.projector_features,
        )
        conditioned_output = self._forward_autoregressive_with_memory(
            input_ids=action_input_ids,
            attention_mask=action_attention_mask,
            labels=labels,
            conditioned_patch_embeddings=conditioned_patch_embeddings,
        )
        memory_inputs.base_output = conditioned_output
        num_prompt_tokens = input_ids.shape[-1] - 1
        num_condition_tokens = conditioned_patch_embeddings.shape[1]
        predicted_action_token_ids = (
            conditioned_output.logits[
                :,
                num_condition_tokens + num_prompt_tokens : num_condition_tokens + num_prompt_tokens + ACTION_DIM * NUM_ACTIONS_CHUNK,
            ]
            .argmax(dim=2)
            .cpu()
            .numpy()
        )
        discretized_actions = self.vla.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.vla.bin_centers.shape[0] - 1)
        normalized_actions = self.vla.bin_centers[discretized_actions].reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
        actions = self.vla._unnormalize_actions(normalized_actions, unnorm_key)
        return actions, normalized_actions, memory_inputs

    def encode_memory_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        episode_ids: Sequence[Any],
        timesteps: Sequence[Any],
        labels: Optional[torch.LongTensor] = None,
        depth_maps: Optional[Sequence[torch.Tensor]] = None,
        proprio=None,
        proprio_projector=None,
        use_film: bool = False,
    ) -> MemoryActionPolicyOutput:
        base_output = self._forward_base_vla(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            proprio=proprio,
            proprio_projector=proprio_projector,
            use_film=use_film,
        )
        projector_features = base_output.projector_features
        if projector_features is None:
            raise ValueError("Base VLA forward did not return projector_features.")

        cognition_tokens = self._extract_cognition_tokens(
            hidden_states=base_output.hidden_states[-1],
            attention_mask=attention_mask,
            num_condition_tokens=projector_features.shape[1],
        )
        perception_tokens = self.perception_projector(projector_features)
        depth_perception_tokens = self._build_depth_perception_tokens(
            depth_maps=depth_maps,
            target_num_tokens=perception_tokens.shape[1],
            output_device=perception_tokens.device,
            output_dtype=perception_tokens.dtype,
        )
        fused_tokens = self.spatial_mem_bank.get_action_expert_input_batch(
            perception_tokens=perception_tokens,
            episode_ids=episode_ids,
            timesteps=timesteps,
            depth_perception_tokens=depth_perception_tokens,
        )

        return MemoryActionPolicyOutput(
            loss=None,
            base_output=base_output,
            cognition_tokens=cognition_tokens,
            perception_tokens=perception_tokens,
            depth_perception_tokens=depth_perception_tokens,
            fused_tokens=fused_tokens,
        )

    def _extract_cognition_tokens(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        num_condition_tokens: int,
    ) -> torch.Tensor:
        text_hidden_states = torch.cat(
            (hidden_states[:, :1], hidden_states[:, num_condition_tokens + 1 :]),
            dim=1,
        )
        cumulative_sum = attention_mask.cumsum(dim=1)
        last_true_indices = (cumulative_sum == cumulative_sum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)
        expanded_indices = last_true_indices.unsqueeze(-1).expand(-1, text_hidden_states.size(-1))
        return text_hidden_states.gather(1, expanded_indices.unsqueeze(1))

    def _build_depth_perception_tokens(
        self,
        depth_maps: Optional[Sequence[torch.Tensor]],
        target_num_tokens: int,
        output_device: torch.device,
        output_dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if depth_maps is None:
            return None
        if len(self.depth_projectors_list) == 0:
            self.load_depth_projectors_from_checkpoint()

        projected_depth_tokens = []
        for depth_map, depth_projector in zip(depth_maps, self.depth_projectors_list):
            depth_tokens = process_depth_perception_tokens(
                depth_map=depth_map,
                depth_projector=depth_projector,
                depth_perception_projector=self.depth_perception_projector,
                output_device=output_device,
                output_dtype=output_dtype,
            )
            if depth_tokens is None:
                continue
            depth_tokens = self._match_token_count(depth_tokens, target_num_tokens)
            projected_depth_tokens.append(depth_tokens)

        if not projected_depth_tokens:
            return None

        return torch.stack(projected_depth_tokens, dim=0).mean(dim=0)

    def _build_conditioned_patch_embeddings(
        self,
        fused_tokens: torch.Tensor,
        original_projected_patch_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        memory_delta = self.memory_condition_projector(fused_tokens)
        return original_projected_patch_embeddings + memory_delta

    def _forward_autoregressive_with_memory(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.LongTensor],
        conditioned_patch_embeddings: torch.Tensor,
    ) -> PrismaticCausalLMOutputWithPast:
        input_embeddings = self.vla.get_input_embeddings()(input_ids)
        if labels is not None:
            all_actions_mask = self.vla._process_action_masks(labels)
            input_embeddings = input_embeddings * ~all_actions_mask.unsqueeze(-1)

        multimodal_embeddings, multimodal_attention_mask = self.vla._build_multimodal_attention(
            input_embeddings,
            conditioned_patch_embeddings,
            attention_mask,
        )
        multimodal_labels = self.vla._build_multimodal_labels(labels, conditioned_patch_embeddings)
        language_model_output = self.vla.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=multimodal_labels,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        return PrismaticCausalLMOutputWithPast(
            loss=language_model_output.loss,
            logits=language_model_output.logits,
            past_key_values=language_model_output.past_key_values,
            hidden_states=language_model_output.hidden_states,
            attentions=language_model_output.attentions,
            projector_features=conditioned_patch_embeddings,
        )

    @staticmethod
    def _match_token_count(tokens: torch.Tensor, target_num_tokens: int) -> torch.Tensor:
        if tokens.shape[1] == target_num_tokens:
            return tokens
        return F.adaptive_avg_pool1d(tokens.transpose(1, 2), target_num_tokens).transpose(1, 2)

    def _prepare_action_prediction_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.LongTensor]:
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.tensor([29871], device=input_ids.device).long(), dim=0)),
                dim=1,
            )
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), device=attention_mask.device, dtype=attention_mask.dtype),
                ),
                dim=1,
            )

        labels = input_ids.clone()
        labels[:] = -100
        input_ids, attention_mask = self.vla._prepare_input_for_action_prediction(input_ids, attention_mask)
        labels = self.vla._prepare_labels_for_action_prediction(labels, input_ids)
        return input_ids, attention_mask, labels


__all__ = [
    "MemoryActionPolicyOutput",
    "SpatialMemoryVLA",
    "VLMPerceptionProjector",
]
