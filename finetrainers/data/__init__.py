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

from ._artifact import ImageArtifact, VideoArtifact
from .dataloader import DPDataLoader
from .dataset import (
    ImageCaptionFilePairDataset,
    ImageFileCaptionFileListDataset,
    ImageFolderDataset,
    ImageWebDataset,
    ValidationDataset,
    VideoAsPromptValidationDataset,
    VideoCaptionFilePairDataset,
    VideoFileCaptionFileListDataset,
    VideoFolderDataset,
    VideoWebDataset,
    combine_datasets,
    initialize_dataset,
    initialize_videoasprompt_dataset,
    wrap_iterable_dataset_for_preprocessing,
)
from .precomputation import (
    InMemoryDataIterable,
    InMemoryDistributedDataPreprocessor,
    InMemoryOnceDataIterable,
    PrecomputedDataIterable,
    PrecomputedDistributedDataPreprocessor,
    PrecomputedOnceDataIterable,
    initialize_preprocessor,
)
from .sampler import ResolutionSampler
