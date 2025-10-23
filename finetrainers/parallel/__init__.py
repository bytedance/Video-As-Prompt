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

from enum import Enum
from typing import Union

from .accelerate import AccelerateParallelBackend
from .ptd import PytorchDTensorParallelBackend
from .utils import dist_max, dist_mean


ParallelBackendType = Union[AccelerateParallelBackend, PytorchDTensorParallelBackend]


class ParallelBackendEnum(str, Enum):
    ACCELERATE = "accelerate"
    PTD = "ptd"


def get_parallel_backend_cls(backend: ParallelBackendEnum) -> ParallelBackendType:
    if backend == ParallelBackendEnum.ACCELERATE:
        return AccelerateParallelBackend
    if backend == ParallelBackendEnum.PTD:
        return PytorchDTensorParallelBackend
    raise ValueError(f"Unknown parallel backend: {backend}")
