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

import inspect
from typing import Any, Dict, List


class ProcessorMixin:
    def __init__(self) -> None:
        self._forward_parameter_names = inspect.signature(self.forward).parameters.keys()
        self.output_names: List[str] = None
        self.input_names: Dict[str, Any] = None

    def __call__(self, *args, **kwargs) -> Any:
        shallow_copy_kwargs = dict(kwargs.items())
        if self.input_names is not None:
            for k, v in self.input_names.items():
                if k in shallow_copy_kwargs:
                    shallow_copy_kwargs[v] = shallow_copy_kwargs.pop(k)
        acceptable_kwargs = {k: v for k, v in shallow_copy_kwargs.items() if k in self._forward_parameter_names}
        output = self.forward(*args, **acceptable_kwargs)
        if "__drop__" in output:
            output.pop("__drop__")
        return output

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("ProcessorMixin::forward method should be implemented by the subclass.")
