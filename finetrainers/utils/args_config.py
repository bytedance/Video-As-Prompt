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

import argparse
from typing import TYPE_CHECKING, Any, Dict


if TYPE_CHECKING:
    from finetrainers.args import BaseArgs


class ArgsConfigMixin:
    def add_args(self, parser: argparse.ArgumentParser):
        raise NotImplementedError("ArgsConfigMixin::add_args should be implemented by subclasses.")

    def map_args(self, argparse_args: argparse.Namespace, mapped_args: "BaseArgs"):
        raise NotImplementedError("ArgsConfigMixin::map_args should be implemented by subclasses.")

    def validate_args(self, args: "BaseArgs"):
        raise NotImplementedError("ArgsConfigMixin::validate_args should be implemented by subclasses.")

    def to_dict(self) -> Dict[str, Any]:
        return {}
