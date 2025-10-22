# Copyright (c) 2025 The HuggingFace Team.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0 
#
# This file has been modified by Bytedance Ltd. and/or its affiliates on September 15, 2025.
#
# Original file was released under Apache License 2.0, with the full license text
# available at https://github.com/huggingface/finetrainers/blob/main/LICENSE.
#
# This modified file is released under the same license.


import pathlib
import random
from typing import Any, Dict, List, Optional, Tuple, Union, Literal
from collections import defaultdict
import math

import os
import json
import pandas as pd
import datasets
import datasets.data_files
import datasets.distributed
import datasets.exceptions
import huggingface_hub
import huggingface_hub.errors
import numpy as np
import cv2
import PIL.Image
import PIL.JpegImagePlugin
import torch
import torch.distributed.checkpoint.stateful
import torchvision
from diffusers.utils import load_image, load_video
from huggingface_hub import list_repo_files, repo_exists, snapshot_download
from tqdm.auto import tqdm

from finetrainers import constants
from finetrainers import functional as FF
from finetrainers.logging import get_logger
from finetrainers.utils import find_files
from finetrainers.utils.import_utils import is_datasets_version


import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger()


# fmt: off
MAX_PRECOMPUTABLE_ITEMS_LIMIT = 1024
COMMON_CAPTION_FILES = ["prompt.txt", "prompts.txt", "caption.txt", "captions.txt"]
COMMON_VIDEO_FILES = ["video.txt", "videos.txt"]
COMMON_IMAGE_FILES = ["image.txt", "images.txt"]
COMMON_WDS_CAPTION_COLUMN_NAMES = ["txt", "text", "caption", "captions", "short_caption", "long_caption", "prompt", "prompts", "short_prompt", "long_prompt", "description", "descriptions", "alt_text", "alt_texts", "alt_caption", "alt_captions", "alt_prompt", "alt_prompts", "alt_description", "alt_descriptions", "image_description", "image_descriptions", "image_caption", "image_captions", "image_prompt", "image_prompts", "image_alt_text", "image_alt_texts", "image_alt_caption", "image_alt_captions", "image_alt_prompt", "image_alt_prompts", "image_alt_description", "image_alt_descriptions", "video_description", "video_descriptions", "video_caption", "video_captions", "video_prompt", "video_prompts", "video_alt_text", "video_alt_texts", "video_alt_caption", "video_alt_captions", "video_alt_prompt", "video_alt_prompts", "video_alt_description"]
# fmt: on

def filter_and_update_refs(
    df: pd.DataFrame,
    alignment_score_threshold: float,
    ref_col: str = "ref_video_paths",
    video_path_col: str = "video_paths",
    vap_kind_col: str = "kind",
    valid_col: str = "_valid",
    align_score_col: str = "reference_alignment_score",
    random_state: Optional[int] = None,
) -> pd.DataFrame:

    rng = np.random.default_rng(random_state)

    required_cols = {video_path_col, vap_kind_col, align_score_col, ref_col}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"lack of columns: {missing}")

    filt_df = df.copy()
    if valid_col in filt_df.columns:
        filt_df = filt_df[filt_df[valid_col] > 0]

    filt_df = filt_df[filt_df[align_score_col] > alignment_score_threshold]

    if len(filt_df) == 0:
        return filt_df.reset_index(drop=True)

    alive_paths = set(filt_df[video_path_col].astype(str).tolist())
    pools_by_kind = (
        filt_df.groupby(vap_kind_col)[video_path_col]
        .apply(lambda s: s.astype(str).tolist())
        .to_dict()
    )
    global_pool = list(alive_paths)

    assert isinstance(filt_df[ref_col].values[0], list)

    def _refresh_refs(row: pd.Series) -> List[str]:
        self_path = str(row[video_path_col])
        kind = row[vap_kind_col]
        cur_refs = [p for p in row[ref_col] if p in alive_paths and p != self_path]
        if cur_refs:
            return cur_refs

        pool = [p for p in pools_by_kind.get(kind, []) if p != self_path]
        candidate_pool = pool
        if not candidate_pool:
            return [] 
        sampled = candidate_pool[rng.integers(0, len(candidate_pool))]
        return [sampled]

    filt_df[ref_col] = filt_df.apply(_refresh_refs, axis=1)

    filt_df = filt_df[filt_df[ref_col].apply(lambda x: len(x) > 0)]

    return filt_df.reset_index(drop=True)


def get_resample_indices(source_fps: float, target_fps: float, num_source_frames: int) -> np.ndarray:

    if source_fps == target_fps:
        return np.arange(num_source_frames)

    num_target_frames = int(num_source_frames * (target_fps / source_fps))
    
    if num_target_frames < 1:
        return np.array([num_source_frames // 2])

    indices = np.linspace(0, num_source_frames - 1, num=num_target_frames, dtype=float)
    indices = np.round(indices).astype(int)

    _, unique_indices_mask = np.unique(indices, return_index=True)
    return indices[np.sort(unique_indices_mask)]


class ImageCaptionFilePairDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    def __init__(self, root: str, infinite: bool = False) -> None:
        super().__init__()

        self.root = pathlib.Path(root)
        self.infinite = infinite

        data = []
        caption_files = sorted(find_files(self.root.as_posix(), "*.txt", depth=0))
        for caption_file in caption_files:
            data_file = self._find_data_file(caption_file)
            if data_file:
                data.append(
                    {
                        "caption": (self.root / caption_file).as_posix(),
                        "image": (self.root / data_file).as_posix(),
                    }
                )

        data = datasets.Dataset.from_list(data)
        data = data.cast_column("image", datasets.Image(mode="RGB"))

        self._data = data.to_iterable_dataset()
        self._sample_index = 0
        self._precomputable_once = len(data) <= MAX_PRECOMPUTABLE_ITEMS_LIMIT

    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)
        return iter(self._data.skip(self._sample_index))

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_index += 1
                sample["caption"] = _read_caption_from_file(sample["caption"])
                yield sample

            if not self.infinite:
                logger.warning(f"Dataset ({self.__class__.__name__}={self.root}) has run out of data")
                break
            else:
                self._sample_index = 0

    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]

    def state_dict(self):
        return {"sample_index": self._sample_index}

    def _find_data_file(self, caption_file: str) -> str:
        caption_file = pathlib.Path(caption_file)
        data_file = None
        found_data = 0

        for extension in constants.SUPPORTED_IMAGE_FILE_EXTENSIONS:
            image_filename = caption_file.with_suffix(f".{extension}")
            if image_filename.exists():
                found_data += 1
                data_file = image_filename

        if found_data == 0:
            return False
        elif found_data > 1:
            raise ValueError(
                f"Multiple data files found for caption file {caption_file}. Please ensure there is only one data "
                f"file per caption file. The following extensions are supported:\n"
                f"  - Images: {constants.SUPPORTED_IMAGE_FILE_EXTENSIONS}\n"
            )

        return data_file.as_posix()


class VideoCaptionFilePairDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    def __init__(self, root: str, infinite: bool = False) -> None:
        super().__init__()

        self.root = pathlib.Path(root)
        self.infinite = infinite

        data = []
        caption_files = sorted(find_files(self.root.as_posix(), "*.txt", depth=0))
        for caption_file in caption_files:
            data_file = self._find_data_file(caption_file)
            if data_file:
                data.append(
                    {
                        "caption": (self.root / caption_file).as_posix(),
                        "video": (self.root / data_file).as_posix(),
                    }
                )

        data = datasets.Dataset.from_list(data)
        data = data.cast_column("video", datasets.Video())

        self._data = data.to_iterable_dataset()
        self._sample_index = 0
        self._precomputable_once = len(data) <= MAX_PRECOMPUTABLE_ITEMS_LIMIT

    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)
        return iter(self._data.skip(self._sample_index))

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_index += 1
                sample["caption"] = _read_caption_from_file(sample["caption"])
                yield sample

            if not self.infinite:
                logger.warning(f"Dataset ({self.__class__.__name__}={self.root}) has run out of data")
                break
            else:
                self._sample_index = 0

    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]

    def state_dict(self):
        return {"sample_index": self._sample_index}

    def _find_data_file(self, caption_file: str) -> str:
        caption_file = pathlib.Path(caption_file)
        data_file = None
        found_data = 0

        for extension in constants.SUPPORTED_VIDEO_FILE_EXTENSIONS:
            video_filename = caption_file.with_suffix(f".{extension}")
            if video_filename.exists():
                found_data += 1
                data_file = video_filename

        if found_data == 0:
            return False
        elif found_data > 1:
            raise ValueError(
                f"Multiple data files found for caption file {caption_file}. Please ensure there is only one data "
                f"file per caption file. The following extensions are supported:\n"
                f"  - Videos: {constants.SUPPORTED_VIDEO_FILE_EXTENSIONS}\n"
            )

        return data_file.as_posix()


class ImageFileCaptionFileListDataset(
    torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful
):
    def __init__(self, root: str, infinite: bool = False) -> None:
        super().__init__()

        VALID_CAPTION_FILES = ["caption.txt", "captions.txt", "prompt.txt", "prompts.txt"]
        VALID_IMAGE_FILES = ["image.txt", "images.txt"]

        self.root = pathlib.Path(root)
        self.infinite = infinite

        data = []
        existing_caption_files = [file for file in VALID_CAPTION_FILES if (self.root / file).exists()]
        existing_image_files = [file for file in VALID_IMAGE_FILES if (self.root / file).exists()]

        if len(existing_caption_files) == 0:
            raise FileNotFoundError(
                f"No caption file found in {self.root}. Must have exactly one of {VALID_CAPTION_FILES}"
            )
        if len(existing_image_files) == 0:
            raise FileNotFoundError(
                f"No image file found in {self.root}. Must have exactly one of {VALID_IMAGE_FILES}"
            )
        if len(existing_caption_files) > 1:
            raise ValueError(
                f"Multiple caption files found in {self.root}. Must have exactly one of {VALID_CAPTION_FILES}"
            )
        if len(existing_image_files) > 1:
            raise ValueError(
                f"Multiple image files found in {self.root}. Must have exactly one of {VALID_IMAGE_FILES}"
            )

        caption_file = existing_caption_files[0]
        image_file = existing_image_files[0]

        with open((self.root / caption_file).as_posix(), "r") as f:
            captions = f.read().splitlines()
        with open((self.root / image_file).as_posix(), "r") as f:
            images = f.read().splitlines()
            images = [(self.root / image).as_posix() for image in images]

        if len(captions) != len(images):
            raise ValueError(f"Number of captions ({len(captions)}) must match number of images ({len(images)})")

        for caption, image in zip(captions, images):
            data.append({"caption": caption, "image": image})

        data = datasets.Dataset.from_list(data)
        data = data.cast_column("image", datasets.Image(mode="RGB"))

        self._data = data.to_iterable_dataset()
        self._sample_index = 0
        self._precomputable_once = len(data) <= MAX_PRECOMPUTABLE_ITEMS_LIMIT

    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)
        return iter(self._data.skip(self._sample_index))

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_index += 1
                yield sample

            if not self.infinite:
                logger.warning(f"Dataset ({self.__class__.__name__}={self.root}) has run out of data")
                break
            else:
                self._sample_index = 0

    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]

    def state_dict(self):
        return {"sample_index": self._sample_index}


class VideoFileCaptionFileListDataset(
    torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful
):
    def __init__(self, root: str, infinite: bool = False) -> None:
        super().__init__()

        VALID_CAPTION_FILES = ["caption.txt", "captions.txt", "prompt.txt", "prompts.txt"]
        VALID_VIDEO_FILES = ["video.txt", "videos.txt"]

        self.root = pathlib.Path(root)
        self.infinite = infinite

        data = []
        existing_caption_files = [file for file in VALID_CAPTION_FILES if (self.root / file).exists()]
        existing_video_files = [file for file in VALID_VIDEO_FILES if (self.root / file).exists()]

        if len(existing_caption_files) == 0:
            raise FileNotFoundError(
                f"No caption file found in {self.root}. Must have exactly one of {VALID_CAPTION_FILES}"
            )
        if len(existing_video_files) == 0:
            raise FileNotFoundError(
                f"No video file found in {self.root}. Must have exactly one of {VALID_VIDEO_FILES}"
            )
        if len(existing_caption_files) > 1:
            raise ValueError(
                f"Multiple caption files found in {self.root}. Must have exactly one of {VALID_CAPTION_FILES}"
            )
        if len(existing_video_files) > 1:
            raise ValueError(
                f"Multiple video files found in {self.root}. Must have exactly one of {VALID_VIDEO_FILES}"
            )

        caption_file = existing_caption_files[0]
        video_file = existing_video_files[0]

        with open((self.root / caption_file).as_posix(), "r") as f:
            captions = f.read().splitlines()
        with open((self.root / video_file).as_posix(), "r") as f:
            videos = f.read().splitlines()
            videos = [(self.root / video).as_posix() for video in videos]

        if len(captions) != len(videos):
            raise ValueError(f"Number of captions ({len(captions)}) must match number of videos ({len(videos)})")

        for caption, video in zip(captions, videos):
            data.append({"caption": caption, "video": video})

        data = datasets.Dataset.from_list(data)
        data = data.cast_column("video", datasets.Video())

        self._data = data.to_iterable_dataset()
        self._sample_index = 0
        self._precomputable_once = len(data) <= MAX_PRECOMPUTABLE_ITEMS_LIMIT

    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)
        return iter(self._data.skip(self._sample_index))

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_index += 1
                yield sample

            if not self.infinite:
                logger.warning(f"Dataset ({self.__class__.__name__}={self.root}) has run out of data")
                break
            else:
                self._sample_index = 0

    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]

    def state_dict(self):
        return {"sample_index": self._sample_index}


class ImageFolderDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    def __init__(self, root: str, infinite: bool = False) -> None:
        super().__init__()

        self.root = pathlib.Path(root)
        self.infinite = infinite

        data = datasets.load_dataset("imagefolder", data_dir=self.root.as_posix(), split="train")

        self._data = data.to_iterable_dataset()
        self._sample_index = 0
        self._precomputable_once = len(data) <= MAX_PRECOMPUTABLE_ITEMS_LIMIT

    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)
        return iter(self._data.skip(self._sample_index))

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_index += 1
                yield sample

            if not self.infinite:
                logger.warning(f"Dataset ({self.__class__.__name__}={self.root}) has run out of data")
                break
            else:
                self._sample_index = 0

    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]

    def state_dict(self):
        return {"sample_index": self._sample_index}


class VideoFolderDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    def __init__(self, root: str, infinite: bool = False) -> None:
        super().__init__()

        self.root = pathlib.Path(root)
        self.infinite = infinite

        data = datasets.load_dataset("videofolder", data_dir=self.root.as_posix(), split="train")

        self._data = data.to_iterable_dataset()
        self._sample_index = 0
        self._precomputable_once = len(data) <= MAX_PRECOMPUTABLE_ITEMS_LIMIT

    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)
        return iter(self._data.skip(self._sample_index))

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_index += 1
                yield sample

            if not self.infinite:
                logger.warning(f"Dataset ({self.__class__.__name__}={self.root}) has run out of data")
                break
            else:
                self._sample_index = 0

    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]

    def state_dict(self):
        return {"sample_index": self._sample_index}


class ImageWebDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    def __init__(
        self,
        dataset_name: str,
        infinite: bool = False,
        column_names: Union[str, List[str]] = "__auto__",
        weights: Dict[str, float] = -1,
        **kwargs,
    ) -> None:
        super().__init__()

        assert weights == -1 or isinstance(weights, dict), (
            "`weights` must be a dictionary of probabilities for each caption column"
        )

        self.dataset_name = dataset_name
        self.infinite = infinite

        data = datasets.load_dataset(dataset_name, split="train", streaming=True)

        if column_names == "__auto__":
            if weights == -1:
                caption_columns = [column for column in data.column_names if column in COMMON_WDS_CAPTION_COLUMN_NAMES]
                if len(caption_columns) == 0:
                    raise ValueError(
                        f"No common caption column found in the dataset. Supported columns are: {COMMON_WDS_CAPTION_COLUMN_NAMES}. "
                        f"Available columns are: {data.column_names}"
                    )
                weights = [1] * len(caption_columns)
            else:
                caption_columns = list(weights.keys())
                weights = list(weights.values())
                if not all(column in data.column_names for column in caption_columns):
                    raise ValueError(
                        f"Caption columns {caption_columns} not found in the dataset. Available columns are: {data.column_names}"
                    )
        else:
            if isinstance(column_names, str):
                if column_names not in data.column_names:
                    raise ValueError(
                        f"Caption column {column_names} not found in the dataset. Available columns are: {data.column_names}"
                    )
                caption_columns = [column_names]
                weights = [1] if weights == -1 else [weights.get(column_names)]
            elif isinstance(column_names, list):
                if not all(column in data.column_names for column in column_names):
                    raise ValueError(
                        f"Caption columns {column_names} not found in the dataset. Available columns are: {data.column_names}"
                    )
                caption_columns = column_names
                weights = [1] if weights == -1 else [weights.get(column) for column in column_names]
            else:
                raise ValueError(f"Unsupported type for column_name: {type(column_names)}")

        for column_names in constants.SUPPORTED_IMAGE_FILE_EXTENSIONS:
            if column_names in data.column_names:
                data = data.cast_column(column_names, datasets.Image(mode="RGB"))
                data = data.rename_column(column_names, "image")
                break

        self._data = data
        self._sample_index = 0
        self._precomputable_once = False
        self._caption_columns = caption_columns
        self._weights = weights

    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)
        return iter(self._data.skip(self._sample_index))

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_index += 1
                caption_column = random.choices(self._caption_columns, weights=self._weights, k=1)[0]
                sample["caption"] = sample[caption_column]
                yield sample

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_index = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")

    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]

    def state_dict(self):
        return {"sample_index": self._sample_index}


class VideoWebDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    def __init__(
        self,
        dataset_name: str,
        infinite: bool = False,
        column_names: Union[str, List[str]] = "__auto__",
        weights: Dict[str, float] = -1,
        **kwargs,
    ) -> None:
        super().__init__()

        assert weights == -1 or isinstance(weights, dict), (
            "`weights` must be a dictionary of probabilities for each caption column"
        )

        self.dataset_name = dataset_name
        self.infinite = infinite

        data = datasets.load_dataset(dataset_name, split="train", streaming=True)

        if column_names == "__auto__":
            if weights == -1:
                caption_columns = [column for column in data.column_names if column in COMMON_WDS_CAPTION_COLUMN_NAMES]
                if len(caption_columns) == 0:
                    raise ValueError(
                        f"No common caption column found in the dataset. Supported columns are: {COMMON_WDS_CAPTION_COLUMN_NAMES}"
                    )
                weights = [1] * len(caption_columns)
            else:
                caption_columns = list(weights.keys())
                weights = list(weights.values())
                if not all(column in data.column_names for column in caption_columns):
                    raise ValueError(
                        f"Caption columns {caption_columns} not found in the dataset. Available columns are: {data.column_names}"
                    )
        else:
            if isinstance(column_names, str):
                if column_names not in data.column_names:
                    raise ValueError(
                        f"Caption column {column_names} not found in the dataset. Available columns are: {data.column_names}"
                    )
                caption_columns = [column_names]
                weights = [1] if weights == -1 else [weights.get(column_names)]
            elif isinstance(column_names, list):
                if not all(column in data.column_names for column in column_names):
                    raise ValueError(
                        f"Caption columns {column_names} not found in the dataset. Available columns are: {data.column_names}"
                    )
                caption_columns = column_names
                weights = [1] if weights == -1 else [weights.get(column) for column in column_names]
            else:
                raise ValueError(f"Unsupported type for column_name: {type(column_names)}")

        for column_names in constants.SUPPORTED_VIDEO_FILE_EXTENSIONS:
            if column_names in data.column_names:
                data = data.cast_column(column_names, datasets.Video())
                data = data.rename_column(column_names, "video")
                break

        self._data = data
        self._sample_index = 0
        self._precomputable_once = False
        self._caption_columns = caption_columns
        self._weights = weights

    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)
        return iter(self._data.skip(self._sample_index))

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_index += 1
                caption_column = random.choices(self._caption_columns, weights=self._weights, k=1)[0]
                sample["caption"] = sample[caption_column]
                yield sample

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_index = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")

    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]

    def state_dict(self):
        return {"sample_index": self._sample_index}


class VideoAsPromptDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    def __init__(self, 
        root: str, 
        infinite: bool = False, 
        ref_videos_num: int = 5, 
        sample_ref_videos_num: int = 2, 
        mask_ref_ratio: float = 0.2, 
        mask_caption_ratio: float = 0.2, 
        training_dataset_kind: str = "vap_data", 
        meta_df_name: str = "", 
        ablation_scaling_data_num: int = 99999999, 
        baseline_single_condition: Optional[str] = None, 
        alignment_score_threshold: int = 5
    ) -> None:
        super().__init__()


        self.root = pathlib.Path(root)
        self.infinite = infinite
        
        self.ref_videos_num = ref_videos_num
        self.sample_ref_videos_num = sample_ref_videos_num
        self.mask_ref_ratio = mask_ref_ratio
        self.mask_caption_ratio = mask_caption_ratio
        self.training_dataset_kind = training_dataset_kind
        self.meta_df_name = meta_df_name
        self.ablation_scaling_data_num = ablation_scaling_data_num
        self.baseline_single_condition = baseline_single_condition

        train_set_csv = self.root / meta_df_name

        data_csv = pd.read_csv(train_set_csv).iloc[:]

        if self.baseline_single_condition is not None:
            data_csv = data_csv.loc[data_csv['kind'] == self.baseline_single_condition]


        data_csv['ref_video_paths'] = data_csv['ref_video_paths'].apply(lambda x: json.loads(x))
        data_csv["video"] = data_csv["video_paths"]

        self.caption_video_dict = {}
        for _, item in data_csv.iterrows():
            self.caption_video_dict[item['video_paths']] = item['tar_video_caption']
        self.vap_video_dict = {}
        for _, item in data_csv.iterrows():
            self.vap_video_dict[item['video_paths']] = item['kind']

        data_csv['video'] = data_csv['video'].apply(lambda x: (self.root / x).as_posix())
        data_csv = data_csv.loc[data_csv['split'] == 'train']

        data_csv = data_csv.sample(frac=1, random_state=42).reset_index(drop=True)
    
        data_csv = filter_and_update_refs(
            df=data_csv,
            alignment_score_threshold=alignment_score_threshold
        )

        data_csv = data_csv.iloc[:len(data_csv) - len(data_csv) % 48]
        if self.ablation_scaling_data_num < len(data_csv):
            data_csv = data_csv.iloc[:self.ablation_scaling_data_num]
        assert len(data_csv) % 48 == 0, "length of data_csv must be divided by 48"
        logger.info(f"data_csv:{len(data_csv)}\n{data_csv.head()}")
        logger.info(f"After Reference_alignment_score filtering {sorted(list(data_csv['kind'].unique()))}, {len(data_csv['kind'].unique())}")

        data = datasets.Dataset.from_pandas(data_csv)
        data = data.cast_column("video", datasets.Video())

        self._data = data.to_iterable_dataset()
        self._sample_index = 0
        self._precomputable_once = len(data) <= MAX_PRECOMPUTABLE_ITEMS_LIMIT
        self.fps = 16

    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)
        return iter(self._data.skip(self._sample_index))

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_index += 1

                sample["video"], sample['fps'] = _preprocess_video(sample["video"], return_fps=True)
            
                if sample['fps'] != self.fps:
                    resample_indices = get_resample_indices(
                        source_fps=sample['fps'],
                        target_fps=self.fps,
                        num_source_frames=len(sample["video"])
                    )
                    sample["video"] = sample["video"][resample_indices]
                    
                sample["ref_videos"] = []
                sample["caption_mot_ref"] = []
                sample["effect_types"] = []
                                
                for ref_video_name in random.sample(sample["ref_video_paths"], self.sample_ref_videos_num):
                    if random.random() < self.mask_ref_ratio and len(sample["ref_videos"]) > 0:
                        continue
                    # logger.warning(f"Training Ref Video Sample:{ref_video_name} - {self._sample_index}")
                    ref_video_path = (self.root / ref_video_name).as_posix()
                    ref_video = decord.VideoReader(ref_video_path)
                    ref_video, ref_fps = _preprocess_video(ref_video, return_fps=True)


                    if ref_fps != self.fps:
                        resample_indices = get_resample_indices(
                            source_fps=ref_fps,
                            target_fps=self.fps,
                            num_source_frames=len(ref_video)
                        )
                        ref_video = ref_video[resample_indices]

                    sample["ref_videos"].append(ref_video)
                    sample["caption_mot_ref"].append(self.caption_video_dict[ref_video_name].strip())
                    sample["effect_types"].append(self.vap_video_dict[ref_video_name])

                sample["caption"] = sample["tar_video_caption"]

                if random.random() < self.mask_caption_ratio:
                    sample["caption"] = ""
                    sample["caption_mot_ref"] = [""] * self.sample_ref_videos_num

                yield sample

            if not self.infinite:
                logger.warning(f"Dataset ({self.__class__.__name__}={self.root}) has run out of data")
                break
            else:
                self._sample_index = 0

    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]

    def state_dict(self):
        return {"sample_index": self._sample_index}


class VideoAsPromptDPOV2Dataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):

    def __init__(
        self,
        root: str,
        infinite: bool = False,
        ref_videos_num: int = 5,
        sample_ref_videos_num: int = 1,
        mask_ref_ratio: float = 0.2,
        mask_caption_ratio: float = 0.2,
        training_dataset_kind: str = "cake",
        meta_df_name: str = "",
        seed: int = 42,
        # dpo weight
        reweight: bool = True,
        metric: str = "reference_alignment_score",
        alpha: float = 1.0,
        beta: float = 0.02,
        prob_eps: float = 1e-8,
        # bin
        freq_bin_width: int = 10, 
        freq_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.root = pathlib.Path(root)
        self.infinite = infinite

        self.ref_videos_num = ref_videos_num
        self.sample_ref_videos_num = max(1, sample_ref_videos_num)
        self.mask_ref_ratio = mask_ref_ratio
        self.mask_caption_ratio = mask_caption_ratio
        self.training_dataset_kind = training_dataset_kind
        self.meta_df_name = meta_df_name

        self.fps = 16
        self._sample_index = 0
        self._num_rows = 0

        self.reweight = bool(reweight)
        self.metric = str(metric)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.prob_eps = float(prob_eps)

        assert 1 <= int(freq_bin_width) <= 100 and (100 % int(freq_bin_width) == 0), \
            f"freq_bin_width must divide 100; got {freq_bin_width}"
        self.freq_bin_width = int(freq_bin_width)
        self.freq_smoothing = float(freq_smoothing)

        self.stage_schedule = [
            {"epochs": 2, "pos_min": 90, "neg_max": 10},
            {"epochs": 10_000, "pos_min": 80, "neg_max": 30},
        ]
        self._stage_boundaries = self._build_stage_boundaries(self.stage_schedule)

        train_set_csv = self.root / meta_df_name
        data_csv = pd.read_csv(train_set_csv).iloc[:]
        if "ref_video_paths" in data_csv.columns:
            data_csv["ref_video_paths"] = data_csv["ref_video_paths"].apply(
                lambda x: json.loads(x) if isinstance(x, str) and len(x) > 0 else []
            )
        else:
            data_csv["ref_video_paths"] = [[] for _ in range(len(data_csv))]
        data_csv["video"] = data_csv["video_paths"]

        data_csv = filter_and_update_refs(
            df=data_csv,
            alignment_score_threshold=0
        )

        self.caption_video_dict: Dict[str, str] = {}
        self.vap_video_dict: Dict[str, str] = {}
        self.score_video_dict: Dict[str, float] = {}

        for _, row in data_csv.iterrows():
            vid = row["video_paths"]
            self.caption_video_dict[vid] = row["tar_video_caption"]
            self.vap_video_dict[vid] = row["kind"]
            self.score_video_dict[vid] = float(row["reference_alignment_score"])

        data_csv = data_csv.loc[data_csv['split'] == 'train']
        data_csv = data_csv.sample(frac=1, random_state=seed).reset_index(drop=True)
        if len(data_csv) >= 16:
            data_csv = data_csv.iloc[: len(data_csv) - (len(data_csv) % 48)]
        assert len(data_csv) == 0 or len(data_csv) % 48 == 0, "length of data_csv must be divided by 16"
        logger.info(f"data_csv:{len(data_csv)}\n{data_csv.head()}")

        self._num_rows = len(data_csv)
        self._data = datasets.Dataset.from_pandas(data_csv).to_iterable_dataset()
        self._precomputable_once = self._num_rows <= MAX_PRECOMPUTABLE_ITEMS_LIMIT

        self.by_kind_all: Dict[str, List[str]] = {}
        for vid, kind in self.vap_video_dict.items():
            self.by_kind_all.setdefault(kind, []).append(vid)

        self.by_kind_sorted_desc: Dict[str, List[str]] = {}
        self.by_kind_sorted_asc: Dict[str, List[str]] = {}
        for kind, vids in self.by_kind_all.items():
            vids_desc = sorted(vids, key=lambda x: self.score_video_dict[x], reverse=True)
            vids_asc = list(reversed(vids_desc))
            self.by_kind_sorted_desc[kind] = vids_desc
            self.by_kind_sorted_asc[kind] = vids_asc

        self.pos_pools: Dict[str, Dict[int, List[str]]] = {}  # kind -> {b: [vids score>=b]}
        self.neg_pools: Dict[str, Dict[int, List[str]]] = {}  # kind -> {b: [vids score<=b]}
        ten_bins = list(range(0, 101, 10))  # 0,10,...,100
        for kind in self.by_kind_all.keys():
            vids_desc = self.by_kind_sorted_desc[kind]
            vids_asc = self.by_kind_sorted_asc[kind]
            pos_dict: Dict[int, List[str]] = {}
            for b in ten_bins:
                pos_dict[b] = [v for v in vids_desc if self.score_video_dict[v] >= b]
            self.pos_pools[kind] = pos_dict
            neg_dict: Dict[int, List[str]] = {}
            for b in ten_bins:
                neg_dict[b] = [v for v in vids_asc if self.score_video_dict[v] <= b]
            self.neg_pools[kind] = neg_dict

        self.ref_pool_eq100: Dict[str, List[str]] = {}
        self.ref_pool_gt90: Dict[str, List[str]] = {}
        for kind, vids in self.by_kind_sorted_desc.items():
            eq100 = [v for v in vids if self.score_video_dict[v] == 100]
            gt90 = [v for v in vids if (self.score_video_dict[v] > 80 and self.score_video_dict[v] < 100)]
            self.ref_pool_eq100[kind] = eq100
            self.ref_pool_gt90[kind] = gt90

        self.video_eval_weights: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._score_prob_by_vid: Dict[str, float] = {}
        self._score_by_vid: Dict[str, float] = {}
        self._precompute_eval_weights()


    def _precompute_eval_weights(self):
        bin_w = self.freq_bin_width
        max_idx = 100 // bin_w

        def clamp_score(x: float) -> float:
            return float(min(max(x, 0.0), 100.0))

        def score_to_bin(score: float) -> int:
            s = clamp_score(score)
            si = int(s)
            if si >= 100:
                return max_idx
            return si // bin_w

        def bin_bounds(idx: int):
            if idx < max_idx:
                low = idx * bin_w
                up  = low + bin_w - 1
            else:
                low, up = 100, 100
            return low, up

        bin_count = defaultdict(float)
        vid_bin_idx: Dict[str, int] = {}
        vid_score: Dict[str, float] = {}

        for vid, score in self.score_video_dict.items():
            s = clamp_score(float(score))
            idx = score_to_bin(s)
            vid_bin_idx[vid] = idx
            vid_score[vid]   = s
            bin_count[idx]  += 1.0

        if self.freq_smoothing > 0.0:
            for idx in range(max_idx + 1):
                bin_count[idx] += self.freq_smoothing

        total = sum(bin_count.values()) if bin_count else 1.0
        bin_prob = {idx: (bin_count[idx] / total if total > 0 else 0.0) for idx in range(max_idx + 1)}
        bin_inv  = {idx: (total / bin_count[idx] if bin_count[idx] > 0 else float("inf"))
                    for idx in range(max_idx + 1)}

        self.video_eval_weights.clear()
        self._score_prob_by_vid.clear()
        self._score_by_vid.clear()

        for vid, idx in vid_bin_idx.items():
            low, up = bin_bounds(idx)
            p  = float(bin_prob[idx])
            inv = float(bin_inv[idx])
            s  = float(vid_score[vid])

            self.video_eval_weights.setdefault(vid, {})
            self.video_eval_weights[vid][self.metric] = {
                "frequency": p,
                "inverse_frequency": inv,
                "score": s,
                "bin_index": int(idx),
                "bin_lower": int(low),
                "bin_upper": int(up),
                "bin_width": int(bin_w),
            }
            self._score_prob_by_vid[vid] = p
            self._score_by_vid[vid] = s

    def _build_stage_boundaries(self, schedule: List[Dict[str, int]]) -> List[Tuple[int, int]]:
        bounds = []
        cur = 0
        for st in schedule:
            start = cur
            end = cur + st["epochs"]
            bounds.append((start, end))
            cur = end
        return bounds

    def _current_epoch(self) -> int:
        if self._num_rows <= 0:
            return 0
        return self._sample_index // self._num_rows

    def _current_stage(self) -> Dict[str, int]:
        e = self._current_epoch()
        for i, (start, end) in enumerate(self._stage_boundaries):
            if e >= start and e < end:
                return self.stage_schedule[i]
        return self.stage_schedule[-1]

    def _choose_ref_for_kind(self, kind: str, ban: set) -> Optional[str]:
        pool = [v for v in self.ref_pool_eq100.get(kind, []) if v not in ban]
        if pool:
            return random.choice(pool)
        pool = [v for v in self.ref_pool_gt90.get(kind, []) if v not in ban]
        if pool:
            return random.choice(pool)
        return None

    def resample_video(self, video, fps):
        resample_indices = get_resample_indices(
            source_fps=fps, target_fps=self.fps, num_source_frames=len(video)
        )
        return video[resample_indices]

    def __iter__(self):
        while True:
            it = self._get_data_iter()
            for row in it:
                stage = self._current_stage()
                pos_min = (stage["pos_min"] // 10) * 10
                neg_max = (stage["neg_max"] // 10) * 10
                pos_min = min(max(pos_min, 0), 100)
                neg_max = min(max(neg_max, 0), 100)

                kind = row["kind"]
                if kind not in self.by_kind_all:
                    self._sample_index += 1
                    continue

                pos_pool = self.pos_pools[kind].get(pos_min, [])
                neg_pool = self.neg_pools[kind].get(neg_max, [])
                if not pos_pool or not neg_pool:
                    self._sample_index += 1
                    continue

                win_video_name = random.choice(pos_pool)
                lose_candidates = [v for v in neg_pool if v != win_video_name]
                if not lose_candidates:
                    self._sample_index += 1
                    continue
                lose_video_name = random.choice(lose_candidates)

                ban = {win_video_name, lose_video_name}
                ref_video_name = self._choose_ref_for_kind(kind, ban)
                if ref_video_name is None:
                    self._sample_index += 1
                    continue

                try:
                    win_vr = decord.VideoReader((self.root / win_video_name).as_posix())
                    lose_vr = decord.VideoReader((self.root / lose_video_name).as_posix())
                    ref_vr = decord.VideoReader((self.root / ref_video_name).as_posix())

                    win_video, win_fps = _preprocess_video(win_vr, return_fps=True)
                    lose_video, lose_fps = _preprocess_video(lose_vr, return_fps=True)
                    ref_video, ref_fps = _preprocess_video(ref_vr, return_fps=True)

                    if win_fps != self.fps:
                        win_video = self.resample_video(win_video, win_fps)
                    if lose_fps != self.fps:
                        lose_video = self.resample_video(lose_video, lose_fps)
                    if ref_fps != self.fps:
                        ref_video = self.resample_video(ref_video, ref_fps)
                except Exception as e:
                    logger.warning(f"Video IO failed for kind={kind}: {e}")
                    self._sample_index += 1
                    continue

                s0 = self._score_by_vid.get(win_video_name, 0.0)
                s1 = self._score_by_vid.get(lose_video_name, 0.0)
                p0 = self._score_prob_by_vid.get(win_video_name, 1.0)
                p1 = self._score_prob_by_vid.get(lose_video_name, 1.0)
                prob = math.sqrt(max(p0, self.prob_eps) * max(p1, self.prob_eps))

                weight = 1.0
                if self.reweight:
                    diff = abs(s0 - s1)
                    weight = ((diff * self.beta) / max(prob, self.prob_eps)) ** self.alpha

                sample = dict(row)
                sample["video"] = [win_video, lose_video]
                sample["ref_videos"] = [ref_video]
                sample["caption"] = [
                    self.caption_video_dict.get(win_video_name, "").strip(),
                    self.caption_video_dict.get(lose_video_name, "").strip(),
                ]
                sample["caption_mot_ref"] = [self.caption_video_dict.get(ref_video_name, "").strip()]
                sample["effect_types"] = [kind]
                sample["fps"] = self.fps


                sample["dpo_weight"] = float(weight)

                if random.random() < self.mask_caption_ratio:
                    sample["caption"] = ["", ""]
                    sample["caption_mot_ref"] = [""]

                self._sample_index += 1
                yield sample

            if not self.infinite:
                logger.warning(f"Dataset ({self.__class__.__name__}={self.root}) has run out of data")
                break

    def _get_data_iter(self):
        if self._num_rows == 0:
            return iter(self._data)
        offset = self._sample_index % self._num_rows if self.infinite else min(self._sample_index, self._num_rows)
        return iter(self._data.skip(offset))

    def load_state_dict(self, state_dict):
        self._sample_index = int(state_dict.get("sample_index", 0))

    def state_dict(self):
        return {"sample_index": self._sample_index}


class ValidationDataset(torch.utils.data.IterableDataset):
    def __init__(self, filename: str):
        super().__init__()

        self.filename = pathlib.Path(filename)

        if not self.filename.exists():
            raise FileNotFoundError(f"File {self.filename.as_posix()} does not exist")

        if self.filename.suffix == ".csv":
            data = datasets.load_dataset("csv", data_files=self.filename.as_posix(), split="train")
        elif self.filename.suffix == ".json":
            data = datasets.load_dataset("json", data_files=self.filename.as_posix(), split="train", field="data")
        elif self.filename.suffix == ".parquet":
            data = datasets.load_dataset("parquet", data_files=self.filename.as_posix(), split="train")
        elif self.filename.suffix == ".arrow":
            data = datasets.load_dataset("arrow", data_files=self.filename.as_posix(), split="train")
        else:
            _SUPPORTED_FILE_FORMATS = [".csv", ".json", ".parquet", ".arrow"]
            raise ValueError(
                f"Unsupported file format {self.filename.suffix} for validation dataset. Supported formats are: {_SUPPORTED_FILE_FORMATS}"
            )

        self._data = data.to_iterable_dataset()

    def __iter__(self):
        for sample in self._data:
            # For consistency reasons, we mandate that "caption" is always present in the validation dataset.
            # However, since the model specifications use "prompt", we create an alias here.
            sample["prompt"] = sample["caption"]

            # Load image or video if the path is provided
            # TODO(aryan): need to handle custom columns here for control conditions
            sample["image"] = None
            sample["video"] = None

            if sample.get("image_path", None) is not None:
                image_path = sample["image_path"]
                if not pathlib.Path(image_path).is_file() and not image_path.startswith("http"):
                    logger.warning(f"Image file {image_path.as_posix()} does not exist.")
                else:
                    sample["image"] = load_image(sample["image_path"])

            if sample.get("video_path", None) is not None:
                video_path = sample["video_path"]
                if not pathlib.Path(video_path).is_file() and not video_path.startswith("http"):
                    logger.warning(f"Video file {video_path.as_posix()} does not exist.")
                else:
                    sample["video"] = load_video(sample["video_path"])

            if sample.get("control_image_path", None) is not None:
                control_image_path = sample["control_image_path"]
                if not pathlib.Path(control_image_path).is_file() and not control_image_path.startswith("http"):
                    logger.warning(f"Control Image file {control_image_path.as_posix()} does not exist.")
                else:
                    sample["control_image"] = load_image(sample["control_image_path"])

            if sample.get("control_video_path", None) is not None:
                control_video_path = sample["control_video_path"]
                if not pathlib.Path(control_video_path).is_file() and not control_video_path.startswith("http"):
                    logger.warning(f"Control Video file {control_video_path.as_posix()} does not exist.")
                else:
                    sample["control_video"] = load_video(sample["control_video_path"])

            sample = {k: v for k, v in sample.items() if v is not None}
            yield sample


class VideoAsPromptValidationDataset(torch.utils.data.IterableDataset):
    def __init__(self, filename: str, height: int = 480, width: int = 832, baseline_single_condition: Optional[str] = None, alignment_score_threshold: int = 70):
        super().__init__()

        self.height = height
        self.width = width
        self.fps = 16
        self.baseline_single_condition = baseline_single_condition

        with open(filename, "r") as file:
            data_config = json.load(file)
            self.root = data_config.get("root", None)
            self.val_set_csvs = data_config.get("val_data", None)
            self.id_token = data_config.get("id_token", None)
            self.ref_videos_num = data_config.get("ref_videos_num", 10)
            self.sample_ref_videos_num = data_config.get("sample_ref_videos_num", 1)
            self.mask_ref_ratio = data_config.get("mask_ref_ratio", 0.0)
            self.num_frames = data_config.get("num_frames", 49)
            self.height = data_config.get("height", height)
            self.width = data_config.get("width", width)
            self.frames_selection = data_config.get("frames_selection", "evenly")
        logger.info(f"Validation data_config: {data_config}")
        logger.info(f"Validation configured to use {len(self.val_set_csvs)} datasets")


        data_csv = pd.concat([pd.read_csv(os.path.join(self.root, val_set_csv)) for val_set_csv in self.val_set_csvs])

        if self.baseline_single_condition is not None:
            data_csv = data_csv.loc[data_csv['kind'] == self.baseline_single_condition]
            logger.info(f"Filter validation df with {self.baseline_single_condition}")

        self.caption_video_dict = {}
        for _, item in data_csv.iterrows():
            self.caption_video_dict[item['video_paths']] = item['tar_video_caption']
        self.vap_video_dict = {}
        for _, item in data_csv.iterrows():
            self.vap_video_dict[item['video_paths']] = item['kind']

        data_csv['ref_video_paths'] = data_csv['ref_video_paths'].apply(lambda x: json.loads(x))
        data_csv['video'] = data_csv['video_paths']
        data_csv = data_csv.sample(frac=1).reset_index(drop=True)
        
        data_csv = filter_and_update_refs(
            df=data_csv,
            alignment_score_threshold=alignment_score_threshold
        )

        data_csv = data_csv.iloc[:len(data_csv) - len(data_csv) % 48]
        logger.info(f"videoanimator validation data_csv: {len(data_csv)}\n{data_csv.head()}")

        self._data = datasets.Dataset.from_pandas(data_csv)

    def __iter__(self):
        for sample in self._data:
            
            sample["prompt"] = sample["tar_video_caption"]

            # Load image or video if the path is provided
            # TODO(aryan): need to handle custom columns here for control conditions
            sample["image"] = None

            if sample.get("video", None) is not None:
                sample["gt_video"] = sample["video"]
                video_path = os.path.join(self.root, sample["video"])
                if not os.path.isfile(video_path):
                    logger.warning(f"Video file {video_path} does not exist.")
                else:
                    sample["video"] = load_video(video_path)

                    sample["prompt_mot_ref"] = []
                    sample["effect_types"] = []


                    video_capture = cv2.VideoCapture(video_path)
                    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
                    video_capture.release()


                    resample_indices = get_resample_indices(
                        source_fps=fps,
                        target_fps=self.fps,
                        num_source_frames=len(sample["video"])
                    )
                    sample["video"] = [sample["video"][i] for i in resample_indices]
                    
                    sample["ref_videos"] = []


                    sample["gt_ref_videos"] = ""
                    for ref_video_name in random.sample(sample["ref_video_paths"], self.sample_ref_videos_num):
                        sample["gt_ref_videos"] += ref_video_name + "#"
                        ref_video = load_video(os.path.join(self.root, ref_video_name))

                        video_capture = cv2.VideoCapture(os.path.join(self.root, ref_video_name))
                        ref_fps = int(video_capture.get(cv2.CAP_PROP_FPS))
                        video_capture.release()


                        resample_indices = get_resample_indices(
                            source_fps=ref_fps,
                            target_fps=self.fps,
                            num_source_frames=len(ref_video)
                        )
                        ref_video = [ref_video[i] for i in resample_indices]


                        sample["ref_videos"].append(ref_video)
                        sample["prompt_mot_ref"].append(self.caption_video_dict[ref_video_name].strip())
                        sample["effect_types"].append(self.vap_video_dict[ref_video_name])

                    
                    sample["num_frames"] = self.num_frames
                    sample["frames_selection"] = self.frames_selection
                    
                    
                    
            sample = {k: v for k, v in sample.items() if v is not None}
            yield sample


class IterableDatasetPreprocessingWrapper(
    torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful
):
    def __init__(
        self,
        dataset: torch.utils.data.IterableDataset,
        dataset_type: str,
        id_token: Optional[str] = None,
        image_resolution_buckets: List[Tuple[int, int]] = None,
        video_resolution_buckets: List[Tuple[int, int, int]] = None,
        rename_columns: Optional[Dict[str, str]] = None,
        drop_columns: Optional[List[str]] = None,
        reshape_mode: str = "bicubic",
        remove_common_llm_caption_prefixes: bool = False,
        frames_selection: Literal["first", "evenly"] = "evenly",
        **kwargs,
    ):
        super().__init__()

        self.dataset = dataset
        self.dataset_type = dataset_type
        self.id_token = id_token
        self.image_resolution_buckets = image_resolution_buckets
        self.video_resolution_buckets = video_resolution_buckets
        self.rename_columns = rename_columns or {}
        self.drop_columns = drop_columns or []
        self.reshape_mode = reshape_mode
        self.remove_common_llm_caption_prefixes = remove_common_llm_caption_prefixes

        self.frames_selection = frames_selection

        logger.info(
            f"Initializing IterableDatasetPreprocessingWrapper for the dataset with the following configuration:\n"
            f"  - Dataset Type: {dataset_type}\n"
            f"  - ID Token: {id_token}\n"
            f"  - Image Resolution Buckets: {image_resolution_buckets}\n"
            f"  - Video Resolution Buckets: {video_resolution_buckets}\n"
            f"  - Rename Columns: {rename_columns}\n"
            f"  - Reshape Mode: {reshape_mode}\n"
            f"  - Remove Common LLM Caption Prefixes: {remove_common_llm_caption_prefixes}\n"
        )

    def __iter__(self):
        logger.info("Starting IterableDatasetPreprocessingWrapper for the dataset")
        for sample in iter(self.dataset):
            for column in self.drop_columns:
                sample.pop(column, None)

            sample = {self.rename_columns.get(k, k): v for k, v in sample.items()}

            for key in sample.keys():
                if isinstance(sample[key], PIL.Image.Image):
                    sample[key] = _preprocess_image(sample[key])
                elif isinstance(sample[key], (decord.VideoReader, torchvision.io.video_reader.VideoReader)):
                    sample[key] = _preprocess_video(sample[key])

            if self.dataset_type == "image":
                if self.image_resolution_buckets:
                    sample["_original_num_frames"] = 1
                    sample["_original_height"] = sample["image"].size(1)
                    sample["_original_width"] = sample["image"].size(2)
                    sample["image"] = FF.resize_to_nearest_bucket_image(
                        sample["image"], self.image_resolution_buckets, self.reshape_mode
                    )
            elif self.dataset_type == "video":
                if self.video_resolution_buckets:
                    
                    if isinstance(sample["video"], list):
                        sample["_original_num_frames"] = sample["video"][0].size(0)
                        sample["_original_height"] = sample["video"][0].size(2)
                        sample["_original_width"] = sample["video"][0].size(3)

                        sample["video"][0], _first_frame_only = FF.resize_to_nearest_bucket_video(
                            sample["video"][0], self.video_resolution_buckets, self.reshape_mode, self.frames_selection
                        )
                        sample["video"][1], _first_frame_only = FF.resize_to_nearest_bucket_video(
                            sample["video"][1], self.video_resolution_buckets, self.reshape_mode, self.frames_selection
                        )
                        sample["video"] = torch.stack(sample["video"], dim=0)
                        
                    else:
                        sample["_original_num_frames"] = sample["video"].size(0)
                        sample["_original_height"] = sample["video"].size(2)
                        sample["_original_width"] = sample["video"].size(3)
                        sample["video"], _first_frame_only = FF.resize_to_nearest_bucket_video(
                            sample["video"], self.video_resolution_buckets, self.reshape_mode, self.frames_selection
                        )
                    # logger.warning(f'data sample video: {sample["video"].shape}')
                    if "ref_videos" in sample:
                        sample["ref_videos"] = [ref_video.to(sample["video"].dtype) for ref_video in sample["ref_videos"]]
                        for i in range(len(sample["ref_videos"])):
                            sample["ref_videos"][i], _ = FF.resize_to_nearest_bucket_video(
                                sample["ref_videos"][i], self.video_resolution_buckets, self.reshape_mode, self.frames_selection
                            )
                        # logger.warning(f'data sample ref_video: {sample["ref_videos"][0].shape}')
                    if _first_frame_only:
                        msg = (
                            "The number of frames in the video is less than the minimum bucket size "
                            "specified. The first frame is being used as a single frame video. This "
                            "message is logged at the first occurence and for every 128th occurence "
                            "after that."
                        )
                        logger.log_freq("WARNING", "BUCKET_TEMPORAL_SIZE_UNAVAILABLE", msg, frequency=128)
                        sample["video"] = sample["video"][:1]

            caption = sample["caption"]
            if isinstance(caption, list) and len(caption) != sample["video"].shape[0]:
                caption = caption[0]
            if not isinstance(caption, list) and caption.startswith("b'") and caption.endswith("'"):
                caption = FF.convert_byte_str_to_str(caption)
            if self.remove_common_llm_caption_prefixes:
                if isinstance(caption, str):
                    caption = FF.remove_prefix(caption, constants.COMMON_LLM_START_PHRASES)
                elif isinstance(caption, list):
                    caption = [FF.remove_prefix(c, constants.COMMON_LLM_START_PHRASES) for c in caption]
                else:
                    raise ValueError(f"caption type {type(caption)} not supported")
            if not isinstance(caption, list) and self.id_token is not None:
                caption = f"{self.id_token} {caption}"
            sample["caption"] = caption

            yield sample

    def load_state_dict(self, state_dict):
        self.dataset.load_state_dict(state_dict["dataset"])

    def state_dict(self):
        return {"dataset": self.dataset.state_dict()}


class IterableCombinedDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    def __init__(self, datasets: List[torch.utils.data.IterableDataset], buffer_size: int, shuffle: bool = False):
        super().__init__()

        self.datasets = datasets
        self.buffer_size = buffer_size
        self.shuffle = shuffle

        logger.info(
            f"Initializing IterableCombinedDataset with the following configuration:\n"
            f"  - Number of Datasets: {len(datasets)}\n"
            f"  - Buffer Size: {buffer_size}\n"
            f"  - Shuffle: {shuffle}\n"
        )

    def __iter__(self):
        logger.info(f"Starting IterableCombinedDataset with {len(self.datasets)} datasets")
        iterators = [iter(dataset) for dataset in self.datasets]
        buffer = []
        per_iter = max(1, self.buffer_size // len(iterators))

        for index, it in enumerate(iterators):
            for _ in tqdm(range(per_iter), desc=f"Filling buffer from data iterator {index}"):
                try:
                    buffer.append((it, next(it)))
                except StopIteration:
                    continue

        while len(buffer) > 0:
            idx = 0
            if self.shuffle:
                idx = random.randint(0, len(buffer) - 1)
            current_it, sample = buffer.pop(idx)
            yield sample
            try:
                buffer.append((current_it, next(current_it)))
            except StopIteration:
                pass

    def load_state_dict(self, state_dict):
        for dataset, dataset_state_dict in zip(self.datasets, state_dict["datasets"]):
            dataset.load_state_dict(dataset_state_dict)

    def state_dict(self):
        return {"datasets": [dataset.state_dict() for dataset in self.datasets]}


# TODO(aryan): maybe write a test for this
def initialize_dataset(
    dataset_name_or_root: str,
    dataset_type: str = "video",
    streaming: bool = True,
    infinite: bool = False,
    *,
    _caption_options: Optional[Dict[str, Any]] = None,
) -> torch.utils.data.IterableDataset:
    assert dataset_type in ["image", "video"]

    try:
        does_repo_exist_on_hub = repo_exists(dataset_name_or_root, repo_type="dataset")
    except huggingface_hub.errors.HFValidationError:
        does_repo_exist_on_hub = False

    if does_repo_exist_on_hub:
        return _initialize_hub_dataset(dataset_name_or_root, dataset_type, infinite, _caption_options=_caption_options)
    else:
        return _initialize_local_dataset(
            dataset_name_or_root, dataset_type, infinite, _caption_options=_caption_options
        )

def initialize_videoasprompt_dataset(
    dataset_name_or_root: str,
    dataset_type: str = "video",
    streaming: bool = True,
    infinite: bool = False,
    # mot
    ref_videos_num: int = 10,
    sample_ref_videos_num: int = 1,
    mask_ref_ratio: float = 0.0,
    mask_caption_ratio: float = 0.0,
    training_dataset_kind: str = "openvap",
    meta_df_name: str = "",
    dpo: bool = False,
    ablation_scaling_data_num: int = 99999999,
    baseline_single_condition: Optional[str] = None,
    alignment_score_threshold: int = 0,
    *,
    _caption_options: Optional[Dict[str, Any]] = None,
) -> torch.utils.data.IterableDataset:

    root = pathlib.Path(dataset_name_or_root)
    if not dpo:
        dataset = VideoAsPromptDataset(
            root.as_posix(), 
            infinite=infinite, 
            ref_videos_num=ref_videos_num, 
            sample_ref_videos_num=sample_ref_videos_num, 
            mask_ref_ratio=mask_ref_ratio, 
            mask_caption_ratio=mask_caption_ratio, 
            training_dataset_kind=training_dataset_kind,
            meta_df_name=meta_df_name,
            ablation_scaling_data_num=ablation_scaling_data_num,
            baseline_single_condition=baseline_single_condition,
            alignment_score_threshold=alignment_score_threshold,
        )
    else:
        dataset = VideoAsPromptDPOV2Dataset(
            root.as_posix(), 
            infinite=infinite, 
            ref_videos_num=ref_videos_num, 
            sample_ref_videos_num=sample_ref_videos_num, 
            mask_ref_ratio=mask_ref_ratio, 
            mask_caption_ratio=mask_caption_ratio, 
            training_dataset_kind=training_dataset_kind,
            meta_df_name=meta_df_name,
        )
    return dataset

def combine_datasets(
    datasets: List[torch.utils.data.IterableDataset], buffer_size: int, shuffle: bool = False
) -> torch.utils.data.IterableDataset:
    return IterableCombinedDataset(datasets=datasets, buffer_size=buffer_size, shuffle=shuffle)


def wrap_iterable_dataset_for_preprocessing(
    dataset: torch.utils.data.IterableDataset, dataset_type: str, config: Dict[str, Any]
) -> torch.utils.data.IterableDataset:
    return IterableDatasetPreprocessingWrapper(dataset, dataset_type, **config)


def _initialize_local_dataset(
    dataset_name_or_root: str,
    dataset_type: str,
    infinite: bool = False,
    *,
    _caption_options: Optional[Dict[str, Any]] = None,
):
    root = pathlib.Path(dataset_name_or_root)
    supported_metadata_files = ["metadata.json", "metadata.jsonl", "metadata.csv"]
    metadata_files = [root / metadata_file for metadata_file in supported_metadata_files]
    metadata_files = [metadata_file for metadata_file in metadata_files if metadata_file.exists()]

    if len(metadata_files) > 1:
        raise ValueError("Found multiple metadata files. Please ensure there is only one metadata file.")

    if len(metadata_files) == 1:
        if dataset_type == "image":
            dataset = ImageFolderDataset(root.as_posix(), infinite=infinite)
        else:
            dataset = VideoFolderDataset(root.as_posix(), infinite=infinite)
        return dataset

    file_list = find_files(root.as_posix(), "*", depth=100)
    has_tar_or_parquet_files = any(file.endswith(".tar") or file.endswith(".parquet") for file in file_list)
    if has_tar_or_parquet_files:
        return _initialize_webdataset(root.as_posix(), dataset_type, infinite, _caption_options=_caption_options)

    if _has_data_caption_file_pairs(root, remote=False):
        if dataset_type == "image":
            dataset = ImageCaptionFilePairDataset(root.as_posix(), infinite=infinite)
        else:
            dataset = VideoCaptionFilePairDataset(root.as_posix(), infinite=infinite)
    elif _has_data_file_caption_file_lists(root, remote=False):
        if dataset_type == "image":
            dataset = ImageFileCaptionFileListDataset(root.as_posix(), infinite=infinite)
        else:
            dataset = VideoFileCaptionFileListDataset(root.as_posix(), infinite=infinite)
    else:
        raise ValueError(
            f"Could not find any supported dataset structure in the directory {root}. Please open an issue at "
            f"https://github.com/a-r-r-o-w/finetrainers with information about your dataset structure and we will "
            f"help you set it up."
        )

    return dataset


def _initialize_hub_dataset(
    dataset_name: str, dataset_type: str, infinite: bool = False, *, _caption_options: Optional[Dict[str, Any]] = None
):
    repo_file_list = list_repo_files(dataset_name, repo_type="dataset")
    if _has_data_caption_file_pairs(repo_file_list, remote=True):
        return _initialize_data_caption_file_dataset_from_hub(dataset_name, dataset_type, infinite)
    elif _has_data_file_caption_file_lists(repo_file_list, remote=True):
        return _initialize_data_file_caption_file_dataset_from_hub(dataset_name, dataset_type, infinite)

    has_tar_or_parquet_files = any(file.endswith(".tar") or file.endswith(".parquet") for file in repo_file_list)
    if has_tar_or_parquet_files:
        return _initialize_webdataset(dataset_name, dataset_type, infinite, _caption_options=_caption_options)

    # TODO(aryan): This should be improved
    caption_files = [pathlib.Path(file).name for file in repo_file_list if file.endswith(".txt")]
    if len(caption_files) < MAX_PRECOMPUTABLE_ITEMS_LIMIT:
        try:
            dataset_root = snapshot_download(dataset_name, repo_type="dataset")
            if dataset_type == "image":
                dataset = ImageFolderDataset(dataset_root, infinite=infinite)
            else:
                dataset = VideoFolderDataset(dataset_root, infinite=infinite)
            return dataset
        except Exception:
            pass

    raise ValueError(f"Could not load dataset {dataset_name} from the HF Hub")


def _initialize_data_caption_file_dataset_from_hub(
    dataset_name: str, dataset_type: str, infinite: bool = False
) -> torch.utils.data.IterableDataset:
    logger.info(f"Downloading dataset {dataset_name} from the HF Hub")
    dataset_root = snapshot_download(dataset_name, repo_type="dataset")
    if dataset_type == "image":
        return ImageCaptionFilePairDataset(dataset_root, infinite=infinite)
    else:
        return VideoCaptionFilePairDataset(dataset_root, infinite=infinite)


def _initialize_data_file_caption_file_dataset_from_hub(
    dataset_name: str, dataset_type: str, infinite: bool = False
) -> torch.utils.data.IterableDataset:
    logger.info(f"Downloading dataset {dataset_name} from the HF Hub")
    dataset_root = snapshot_download(dataset_name, repo_type="dataset")
    if dataset_type == "image":
        return ImageFileCaptionFileListDataset(dataset_root, infinite=infinite)
    else:
        return VideoFileCaptionFileListDataset(dataset_root, infinite=infinite)


def _initialize_webdataset(
    dataset_name: str, dataset_type: str, infinite: bool = False, _caption_options: Optional[Dict[str, Any]] = None
) -> torch.utils.data.IterableDataset:
    logger.info(f"Streaming webdataset {dataset_name} from the HF Hub")
    _caption_options = _caption_options or {}
    if dataset_type == "image":
        return ImageWebDataset(dataset_name, infinite=infinite, **_caption_options)
    else:
        return VideoWebDataset(dataset_name, infinite=infinite, **_caption_options)


def _has_data_caption_file_pairs(root: Union[pathlib.Path, List[str]], remote: bool = False) -> bool:
    # TODO(aryan): this logic can be improved
    if not remote:
        caption_files = find_files(root.as_posix(), "*.txt", depth=0)
        for caption_file in caption_files:
            caption_file = pathlib.Path(caption_file)
            for extension in [*constants.SUPPORTED_IMAGE_FILE_EXTENSIONS, *constants.SUPPORTED_VIDEO_FILE_EXTENSIONS]:
                data_filename = caption_file.with_suffix(f".{extension}")
                if data_filename.exists():
                    return True
        return False
    else:
        caption_files = [file for file in root if file.endswith(".txt")]
        for caption_file in caption_files:
            caption_file = pathlib.Path(caption_file)
            for extension in [*constants.SUPPORTED_IMAGE_FILE_EXTENSIONS, *constants.SUPPORTED_VIDEO_FILE_EXTENSIONS]:
                data_filename = caption_file.with_suffix(f".{extension}").name
                if data_filename in root:
                    return True
        return False


def _has_data_file_caption_file_lists(root: Union[pathlib.Path, List[str]], remote: bool = False) -> bool:
    # TODO(aryan): this logic can be improved
    if not remote:
        file_list = {x.name for x in root.iterdir()}
        has_caption_files = any(file in file_list for file in COMMON_CAPTION_FILES)
        has_video_files = any(file in file_list for file in COMMON_VIDEO_FILES)
        has_image_files = any(file in file_list for file in COMMON_IMAGE_FILES)
        return has_caption_files and (has_video_files or has_image_files)
    else:
        has_caption_files = any(file in root for file in COMMON_CAPTION_FILES)
        has_video_files = any(file in root for file in COMMON_VIDEO_FILES)
        has_image_files = any(file in root for file in COMMON_IMAGE_FILES)
        return has_caption_files and (has_video_files or has_image_files)


def _read_caption_from_file(filename: str) -> str:
    with open(filename, "r") as f:
        return f.read().strip()


def _preprocess_image(image: PIL.Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    image = np.array(image).astype(np.float32)
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1).contiguous() / 127.5 - 1.0
    return image


if is_datasets_version("<", "3.4.0"):

    def _preprocess_video(video: decord.VideoReader, start_frame: int=-1, end_frame: int=-1, return_fps: bool=False) -> Union[torch.Tensor, Tuple[torch.Tensor, float]]:
        fps = int(video.get_avg_fps()) if return_fps else None
        if start_frame != -1 and end_frame != -1:
            video = video.get_batch(list(range(start_frame, end_frame)))
        else:
            video = video.get_batch(list(range(len(video))))
        video = video.permute(0, 3, 1, 2).contiguous()
        video = video.float() / 127.5 - 1.0
        return (video, fps) if return_fps else video

else:
    # Hardcode max frames for now. Ideally, we should allow user to set this and handle it in IterableDatasetPreprocessingWrapper
    MAX_FRAMES = 4096

    def _preprocess_video(video: torchvision.io.video_reader.VideoReader, start_frame: int=-1, end_frame: int=-1, return_fps: bool=False) -> Union[torch.Tensor, Tuple[torch.Tensor, float]]:
        frames = []
        # Error driven data loading! torchvision does not expose length of video
        try:
            for _ in range(MAX_FRAMES):
                frames.append(next(video)["data"])
        except StopIteration:
            pass
        if start_frame != -1 and end_frame != -1:
            video = torch.stack(frames[start_frame:end_frame])
        else:
            video = torch.stack(frames)
        video = video.float() / 127.5 - 1.0
        return (video, fps) if return_fps else video
