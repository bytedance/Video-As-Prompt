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

from typing import Any, Dict, List, Optional

from .base import ProcessorMixin
from .canny import CannyProcessor
from .clip import CLIPPooledProcessor
from .glm import CogView4GLMProcessor
from .llama import LlamaProcessor
from .t5 import T5Processor, T5ProcessorMOT
from .text import CaptionEmbeddingDropoutProcessor, CaptionTextDropoutProcessor


class CopyProcessor(ProcessorMixin):
    r"""Processor that copies the input data unconditionally to the output."""

    def __init__(self, output_names: List[str] = None, input_names: Optional[Dict[str, Any]] = None):
        super().__init__()

        self.output_names = output_names
        self.input_names = input_names
        assert len(output_names) == 1

    def forward(self, input: Any) -> Any:
        return {self.output_names[0]: input}
