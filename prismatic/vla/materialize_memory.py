"""
materialize_memory.py

Factory helpers for memory-aware RLDS datasets and collators.
"""

from pathlib import Path
from typing import Tuple, Type

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils_memory import PaddedCollatorForMemoryActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.datasets_memory import (
    GroupMemoryRLDSDataset,
    MemoryRLDSBatchTransform,
    MemoryRLDSDataset,
    StreamMemoryRLDSDataset,
)


def get_memory_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    image_aug: bool = False,
    dataloader_type: str = "group",
    group_size: int = 16,
    use_wrist_image: bool = False,
    use_proprio: bool = False,
    use_depth: bool = False,
) -> Tuple[Dataset, ActionTokenizer, PaddedCollatorForMemoryActionPrediction]:
    action_tokenizer = ActionTokenizer(tokenizer)
    batch_transform = MemoryRLDSBatchTransform(
        action_tokenizer,
        tokenizer,
        image_transform,
        prompt_builder_fn,
        predict_stop_token=predict_stop_token,
        use_wrist_image=use_wrist_image,
        use_proprio=use_proprio,
        use_depth=use_depth,
    )
    collator = PaddedCollatorForMemoryActionPrediction(
        tokenizer.model_max_length,
        tokenizer.pad_token_id,
        padding_side=padding_side,
    )
    if dataloader_type == "group":
        if group_size <= 1:
            raise ValueError("group_size must be greater than 1 for grouped memory training")
        dataset_cls = GroupMemoryRLDSDataset
        dataset_kwargs = {"group_size": group_size}
    elif dataloader_type == "stream":
        dataset_cls = StreamMemoryRLDSDataset
        dataset_kwargs = {}
    elif dataloader_type == "normal":
        dataset_cls = MemoryRLDSDataset
        dataset_kwargs = {}
    else:
        raise NotImplementedError(f"Unsupported memory dataloader_type: {dataloader_type}")

    dataset = dataset_cls(
        data_root_dir,
        data_mix,
        batch_transform,
        resize_resolution=default_image_resolution[1:],
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        image_aug=image_aug,
        use_proprio=use_proprio,
        use_depth=use_depth,
        **dataset_kwargs,
    )
    return dataset, action_tokenizer, collator


__all__ = [
    "get_memory_vla_dataset_and_collator",
]
