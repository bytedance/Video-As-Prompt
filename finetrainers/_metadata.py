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

from dataclasses import dataclass, field
from typing import Dict, ForwardRef, List, Optional, Type, Union


ParamIdentifierType = ForwardRef("ParamIdentifier")
ContextParallelInputMetadataType = ForwardRef("ContextParallelInputMetadata")
ContextParallelOutputMetadataType = ForwardRef("ContextParallelOutputMetadata")

_ContextParallelInputType = Dict[
    ParamIdentifierType, Union[ContextParallelInputMetadataType, List[ContextParallelInputMetadataType]]
]
_ContextParallelOutputType = List[ContextParallelOutputMetadataType]
ContextParallelModelPlan = Union[_ContextParallelInputType, _ContextParallelOutputType]


@dataclass(frozen=True)
class ParamId:
    """
    A class to identify a parameter of a method.

    Atleast one of `name` or `index` must be provided.

    Attributes:
        name (`str`, *optional*):
            The name of the parameter.
        index (`int`, *optional*):
            The index of the parameter in the method signature. Indexing starts at 0 (ignore
            the `self` parameter for instance methods).
    """

    name: Optional[str] = None
    index: Optional[int] = None

    def __post_init__(self):
        if self.name is None and self.index is None:
            raise ValueError("At least one of `name` or `index` must be provided.")


@dataclass(frozen=True)
class CPInput:
    split_dim: int
    expected_dims: Optional[int] = None
    split_output: bool = False


@dataclass(frozen=True)
class CPOutput:
    gather_dim: int
    expected_dims: Optional[int] = None


@dataclass
class TransformerMetadata:
    # Mapping of FQN to mapping of input name to ContextParallelModelPlan
    cp_plan: Dict[str, ContextParallelModelPlan] = field(default_factory=dict)

    # tp_plan  # TODO(aryan)


class TransformerRegistry:
    _registry = {}

    @classmethod
    def register(cls, model_class: Type, metadata: TransformerMetadata):
        cls._registry[model_class] = metadata

    @classmethod
    def get(cls, model_class: Type) -> TransformerMetadata:
        if model_class not in cls._registry:
            raise ValueError(f"Model class {model_class} not registered.")
        return cls._registry[model_class]
