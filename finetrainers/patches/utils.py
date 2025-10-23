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

import torch


class DisableTensorToDtype:
    def __enter__(self):
        self.original_to = torch.Tensor.to

        def modified_to(tensor, *args, **kwargs):
            # remove dtype from args if present
            args = [arg if not isinstance(arg, torch.dtype) else None for arg in args]
            if "dtype" in kwargs:
                kwargs.pop("dtype")
            return self.original_to(tensor, *args, **kwargs)

        torch.Tensor.to = modified_to

    def __exit__(self, *args, **kwargs):
        torch.Tensor.to = self.original_to
