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

from typing import Union

from diffusers import CogVideoXDDIMScheduler, FlowMatchEulerDiscreteScheduler
from transformers import CLIPTokenizer, LlamaTokenizer, LlamaTokenizerFast, T5Tokenizer, T5TokenizerFast

from .data import ImageArtifact, VideoArtifact


ArtifactType = Union[ImageArtifact, VideoArtifact]
SchedulerType = Union[CogVideoXDDIMScheduler, FlowMatchEulerDiscreteScheduler]
TokenizerType = Union[CLIPTokenizer, T5Tokenizer, T5TokenizerFast, LlamaTokenizer, LlamaTokenizerFast]
