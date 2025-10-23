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

import functools

from peft.tuners.tuners_utils import BaseTunerLayer

from finetrainers.patches.utils import DisableTensorToDtype


def patch_peft_move_adapter_to_device_of_base_layer() -> None:
    _perform_patch_move_adapter_to_device_of_base_layer()


def _perform_patch_move_adapter_to_device_of_base_layer() -> None:
    BaseTunerLayer._move_adapter_to_device_of_base_layer = _patched_move_adapter_to_device_of_base_layer(
        BaseTunerLayer._move_adapter_to_device_of_base_layer
    )


def _patched_move_adapter_to_device_of_base_layer(func) -> None:
    # TODO(aryan): This is really unsafe probably and may break things. It works for now, but revisit and refactor.
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with DisableTensorToDtype():
            return func(self, *args, **kwargs)

    return wrapper
