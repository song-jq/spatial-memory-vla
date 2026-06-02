"""
datasets_memory.py

Parallel RLDS dataset wrappers for the spatial-memory execution flow.
"""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import dlimp as dl
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, IGNORE_INDEX, NUM_ACTIONS_CHUNK
from prismatic.vla.datasets.rlds import make_interleaved_dataset
from prismatic.vla.datasets.rlds.dataset import (
    apply_frame_transforms,
    apply_per_dataset_frame_transforms,
    apply_trajectory_transforms,
    make_dataset_from_rlds,
)
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import allocate_threads, pprint_data_mixture


@dataclass
class MemoryRLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False
    use_depth: bool = False

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        dataset_name, current_action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        actions = rlds_batch["action"]

        prompt_builder = self.prompt_builder_fn("openvla")
        future_actions = rlds_batch["action"][1:]
        future_actions_string = "".join(self.action_tokenizer(future_actions))
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
            "dataset_name": dataset_name,
            "actions": actions,
            "timesteps": np.array([rlds_batch["observation"].get("timestep", 0)]),
            "episode_ids": np.array([-1]),
        }

        if self.use_wrist_image:
            all_wrist_pixels = []
            for key in rlds_batch["observation"].keys():
                if "image_wrist" in key or "wrist_image" in key:
                    img_wrist = Image.fromarray(rlds_batch["observation"][key][0])
                    all_wrist_pixels.append(self.image_transform(img_wrist))
            if all_wrist_pixels:
                return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)

        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            return_dict["proprio"] = rlds_batch["observation"]["proprio"]

        if self.use_depth:
            if "depth_primary" in rlds_batch["observation"]:
                return_dict["depth_map"] = np.squeeze(rlds_batch["observation"]["depth_primary"][0])
            elif "depth_image" in rlds_batch["observation"]:
                return_dict["depth_map"] = np.squeeze(rlds_batch["observation"]["depth_image"][0])

            for key in rlds_batch["observation"].keys():
                if "depth_wrist" in key:
                    return_dict["depth_map_wrist"] = np.squeeze(rlds_batch["observation"][key][0])

        return return_dict


def make_interleaved_episodic_dataset(
    dataset_kwargs_list,
    sample_weights=None,
    *,
    train: bool,
    shuffle_buffer_size: int,
    traj_transform_kwargs=None,
    frame_transform_kwargs=None,
    batch_size=None,
    balance_weights: bool = False,
    traj_transform_threads=None,
    traj_read_threads=None,
    group_size: int = 0,
    use_optim_group_sample: bool = False,
):
    if not sample_weights:
        sample_weights = [1.0] * len(dataset_kwargs_list)

    if len(sample_weights) != len(dataset_kwargs_list):
        raise ValueError(f"sample_weights must be None or have length {len(dataset_kwargs_list)}.")
    if (traj_transform_kwargs is None) or (frame_transform_kwargs is None):
        raise ValueError("Missing `traj_transform_kwargs` and `frame_transform_kwargs`!")

    dataset_sizes, all_dataset_statistics = [], {}
    for dataset_kwargs in dataset_kwargs_list:
        data_kwargs = copy.deepcopy(dataset_kwargs)
        if "dataset_frame_transform_kwargs" in data_kwargs:
            data_kwargs.pop("dataset_frame_transform_kwargs")
        _, dataset_statistics = make_dataset_from_rlds(**data_kwargs, train=train)
        dataset_sizes.append(dataset_statistics["num_transitions"])
        all_dataset_statistics[dataset_kwargs["name"]] = dataset_statistics

    primary_dataset_indices = np.array([idx for idx in range(len(sample_weights)) if sample_weights[idx] == 1.0])
    if balance_weights:
        sample_weights = np.array(sample_weights) * np.array(dataset_sizes)
    sample_weights = np.array(sample_weights) / np.sum(sample_weights)
    pprint_data_mixture(dataset_kwargs_list, sample_weights)

    dataset_len = int((np.array(dataset_sizes) / sample_weights)[primary_dataset_indices].max())
    threads_per_dataset = allocate_threads(traj_transform_threads, sample_weights)
    reads_per_dataset = allocate_threads(traj_read_threads, sample_weights)

    datasets = []
    for dataset_kwargs, threads, reads in zip(dataset_kwargs_list, threads_per_dataset, reads_per_dataset):
        dataset_frame_transform_kwargs = (
            dataset_kwargs.pop("dataset_frame_transform_kwargs")
            if "dataset_frame_transform_kwargs" in dataset_kwargs
            else {}
        )
        dataset, _ = make_dataset_from_rlds(
            **dataset_kwargs,
            train=train,
            num_parallel_calls=threads,
            num_parallel_reads=reads,
            dataset_statistics=all_dataset_statistics[dataset_kwargs["name"]],
        )
        dataset = apply_trajectory_transforms(
            dataset.repeat(),
            **traj_transform_kwargs,
            num_parallel_calls=threads,
            train=train,
        )

        if use_optim_group_sample:
            def group_sample(traj):
                traj_len = tf.shape(traj["action"])[0]

                def pad_case():
                    return tf.concat([tf.range(traj_len), tf.fill([group_size - traj_len], traj_len - 1)], axis=0)

                def sample_case():
                    shuffled = tf.random.shuffle(tf.range(traj_len))
                    return tf.sort(shuffled[:group_size])

                indices = tf.cond(traj_len < group_size, pad_case, sample_case)
                return tf.nest.map_structure(lambda tensor: tf.gather(tensor, indices, axis=0), traj)

            dataset = dataset.map(lambda traj: group_sample(traj), num_parallel_calls=threads)

        dataset = apply_per_dataset_frame_transforms(dataset, **dataset_frame_transform_kwargs)
        datasets.append(dataset)

    dataset: dl.DLataset = dl.DLataset.sample_from_datasets(datasets, sample_weights)
    if not train:
        dataset = dataset.take(shuffle_buffer_size).cache()
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = apply_frame_transforms(dataset, **frame_transform_kwargs, train=train)

    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    dataset = dataset.with_ram_budget(1)
    dataset.sample_weights = sample_weights
    return dataset, dataset_len, all_dataset_statistics


class MemoryRLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: MemoryRLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        use_proprio: bool = False,
        use_depth: bool = False,
    ) -> None:
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            mixture_spec = [(self.data_mix, 1.0)]

        if "aloha" in self.data_mix:
            load_camera_views = ("primary", "left_wrist", "right_wrist")
        else:
            load_camera_views = ("primary", "wrist")

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=use_depth,
            load_proprio=use_proprio,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,
                future_action_window_size=NUM_ACTIONS_CHUNK - 1,
                skip_unlabeled=True,
                goal_relabeling_strategy="uniform",
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )
        if image_aug:
            rlds_config["frame_transform_kwargs"].update(
                {
                    "image_augment_kwargs": dict(
                        random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                        random_brightness=[0.2],
                        random_contrast=[0.8, 1.2],
                        random_saturation=[0.8, 1.2],
                        random_hue=[0.05],
                        augment_order=[
                            "random_resized_crop",
                            "random_brightness",
                            "random_contrast",
                            "random_saturation",
                            "random_hue",
                        ],
                    )
                }
            )

        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        episode_id = -1
        for rlds_batch in self.dataset.as_numpy_iterator():
            episode_id += 1
            frame = self.batch_transform(rlds_batch)
            frame["episode_ids"] = np.array([episode_id])
            yield frame

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class GroupMemoryRLDSDataset(MemoryRLDSDataset):
    def __init__(self, *args, group_size: int = 16, **kwargs) -> None:
        self.group_size = group_size
        super().__init__(*args, **kwargs)

    def __iter__(self) -> Dict[str, Any]:
        episode_id = -1
        for rlds_batch in self.dataset.as_numpy_iterator():
            episode_id += 1
            for idx in range(rlds_batch["action"].shape[0]):
                frame = self.batch_transform(tree_map(lambda x: x[idx], rlds_batch))
                frame["episode_ids"] = np.array([episode_id])
                yield frame

    def make_dataset(self, rlds_config):
        return make_interleaved_episodic_dataset(
            **rlds_config,
            group_size=self.group_size,
            use_optim_group_sample=True,
        )


class StreamMemoryRLDSDataset(MemoryRLDSDataset):
    def make_dataset(self, rlds_config):
        return make_interleaved_episodic_dataset(
            **rlds_config,
            use_optim_group_sample=False,
        )

    def __iter__(self) -> Dict[str, Any]:
        episode_id = -1
        for rlds_batch in self.dataset.as_numpy_iterator():
            episode_id += 1
            for idx in range(rlds_batch["action"].shape[0]):
                frame = self.batch_transform(tree_map(lambda x: x[idx], rlds_batch))
                frame["episode_ids"] = np.array([episode_id])
                yield frame


__all__ = [
    "make_interleaved_episodic_dataset",
    "MemoryRLDSBatchTransform",
    "MemoryRLDSDataset",
    "GroupMemoryRLDSDataset",
    "StreamMemoryRLDSDataset",
]
