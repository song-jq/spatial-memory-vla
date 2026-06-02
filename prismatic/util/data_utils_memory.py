"""
data_utils_memory.py

Memory-aware collation utilities for the parallel spatial-memory execution flow.
"""

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from prismatic.util.data_utils import IGNORE_INDEX


@dataclass
class PaddedCollatorForMemoryActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        dataset_names = [instance["dataset_name"] for instance in instances] if "dataset_name" in instances[0] else None

        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]
        attention_mask = input_ids.ne(self.pad_token_id)

        if isinstance(pixel_values[0], torch.Tensor):
            if "pixel_values_wrist" in instances[0]:
                pixel_values_wrist = [instance["pixel_values_wrist"] for instance in instances]
                pixel_values = torch.cat((torch.stack(pixel_values), torch.stack(pixel_values_wrist)), dim=1)
            else:
                pixel_values = torch.stack(pixel_values)
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        actions = [torch.from_numpy(np.copy(instance["actions"])) for instance in instances]
        actions = torch.stack(actions)

        output = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "actions": actions,
        }

        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        if "proprio" in instances[0]:
            proprio = [instance["proprio"] for instance in instances]
            output["proprio"] = torch.tensor(np.squeeze(np.stack(proprio)), dtype=torch.float32)
        if "depth_map" in instances[0]:
            output["depth_maps"] = torch.stack(
                [torch.tensor(instance["depth_map"], dtype=torch.float32) for instance in instances]
            )
        if "depth_map_wrist" in instances[0]:
            output["depth_maps_wrist"] = torch.stack(
                [torch.tensor(instance["depth_map_wrist"], dtype=torch.float32) for instance in instances]
            )
        if "timesteps" in instances[0]:
            output["timesteps"] = np.concatenate([instance["timesteps"] for instance in instances], axis=0)
        if "episode_ids" in instances[0]:
            output["episode_ids"] = np.concatenate([instance["episode_ids"] for instance in instances], axis=0)

        return output


__all__ = [
    "PaddedCollatorForMemoryActionPrediction",
]
