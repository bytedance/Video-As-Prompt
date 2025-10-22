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
from typing import TYPE_CHECKING, Any, Dict, List, Union

from finetrainers.utils import ArgsConfigMixin


if TYPE_CHECKING:
    from finetrainers.args import BaseArgs


class SFTLowRankConfig(ArgsConfigMixin):
    r"""
    Configuration class for SFT low rank training.

    Args:
        rank (int):
            Rank of the low rank approximation matrix.
        lora_alpha (int):
            The lora_alpha parameter to compute scaling factor (lora_alpha / rank) for low-rank matrices.
        target_modules (`str` or `List[str]`):
            Target modules for the low rank approximation matrices. Can be a regex string or a list of regex strings.
    """

    rank: int = 64
    lora_alpha: int = 64
    target_modules: Union[str, List[str]] = "(transformer_blocks|single_transformer_blocks).*(to_q|to_k|to_v|to_out.0)"

    def add_args(self, parser: argparse.ArgumentParser):
        parser.add_argument("--rank", type=int, default=64)
        parser.add_argument("--lora_alpha", type=int, default=64)
        parser.add_argument(
            "--target_modules",
            type=str,
            nargs="+",
            default=["(transformer_blocks|single_transformer_blocks).*(to_q|to_k|to_v|to_out.0)"],
        )

    def validate_args(self, args: "BaseArgs"):
        assert self.rank > 0, "Rank must be a positive integer."
        assert self.lora_alpha > 0, "lora_alpha must be a positive integer."

    def map_args(self, argparse_args: argparse.Namespace, mapped_args: "BaseArgs"):
        mapped_args.rank = argparse_args.rank
        mapped_args.lora_alpha = argparse_args.lora_alpha
        mapped_args.target_modules = (
            argparse_args.target_modules[0] if len(argparse_args.target_modules) == 1 else argparse_args.target_modules
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"rank": self.rank, "lora_alpha": self.lora_alpha, "target_modules": self.target_modules}


class SFTFullRankConfig(ArgsConfigMixin):
    r"""
    Configuration class for SFT full rank training.
    """

    def add_args(self, parser: argparse.ArgumentParser):
        pass

    def validate_args(self, args: "BaseArgs"):
        pass

    def map_args(self, argparse_args: argparse.Namespace, mapped_args: "BaseArgs"):
        pass


class VideoAsPromptMOTConfig(ArgsConfigMixin):
    rank: int = 64
    lora_alpha: int = 64
    target_modules: Union[str, List[str]] = "(transformer_blocks|single_transformer_blocks).*(to_q|to_k|to_v|to_out.0)"

    # mot
    videoasprompt_mot_name_or_path: str = None
    ref_videos_num: int = 10
    sample_ref_videos_num: int = 1
    mask_ref_ratio: float = 0.0
    mask_caption_ratio: float = 0.1
    training_dataset_kind: str = "ours"
    model_structure_config: str = ""
    caption_guidance_scale: float = 0.0
    reference_train_mode: str = None
    random_refer_noise: bool = False
    num_ref_diffusion_time_sigmas: int = 0
    ref_type: str = "continous_negative"
    dpo: bool = False
    alignment_score_threshold: int = 0

    # ablation
    ablation_single_branch: bool = False
    ablation_single_encoder: bool = False
    ablation_residual_addition: bool = False
    ablation_scaling_data_num: int = 99999999

    # baseline
    baseline_single_condition: str = None

    def add_args(self, parser: argparse.ArgumentParser):
        parser.add_argument("--rank", type=int, default=64)
        parser.add_argument("--lora_alpha", type=int, default=64)
        parser.add_argument(
            "--target_modules",
            type=str,
            nargs="+",
            default=["(transformer_blocks|single_transformer_blocks).*(to_q|to_k|to_v|to_out.0)"],
        )

        # mot
        parser.add_argument(
            "--videoasprompt_mot_name_or_path",
            type=str,
            default=None,
            help="Path to the Videoanimator weights.",
        )
        parser.add_argument(
            "--ref_videos_num",
            type=int,
            default=10,
            help="Number of reference videos.",
        )
        parser.add_argument(
            "--sample_ref_videos_num",
            type=int,
            default=1,
            help="Number of sampled reference videos.",
        )
        parser.add_argument(
            "--mask_ref_ratio",
            type=float,
            default=0.0,
            help="Mask ratio for reference videos.",
        )
        parser.add_argument(
            "--mask_caption_ratio",
            type=float,
            default=0.1,
            help="Mask ratio for caption.",
        )
        parser.add_argument(
            "--training_dataset_kind",
            type=str,
            default="openvap",
            help="Training dataset kind.",
        )

        parser.add_argument(
            "--model_structure_config",
            type=str,
            default="examples/training/sft/cogvideox/vap_mot/config_ori.json",
            help="Model structure config.",
        )
        parser.add_argument(
            "--caption_guidance_scale",
            type=float,
            default=0.0,
            help="Caption guidance scale.",
        )
        parser.add_argument(
            "--reference_train_mode",
            type=str,
            default=None,
            help="Reference train mode.",
        )
        parser.add_argument(
            '--random_refer_noise', 
            action='store_true', 
            help='Enable the random reference video noise'
        )
        parser.add_argument(
            "--num_ref_diffusion_time_sigmas",
            type=int,
            default=0,
            help="Number of reference diffusion time sigmas.",
        )
        parser.add_argument(
            "--ref_type",
            type=str,
            default="continous_negative",
            help="Reference RoPE type.",
        )
        parser.add_argument(
            "--dpo",
            action='store_true', 
            help='Enable the DPO training'
        )
        parser.add_argument(
            "--alignment_score_threshold",
            type=int,
            default=0,
            help="Alignment score threshold.",
        )

        # ablation
        parser.add_argument(
            "--ablation_single_branch",
            action='store_true', 
            help='Enable the ablation single branch training'
        )
        parser.add_argument(
            "--ablation_single_encoder",
            action='store_true', 
            help='Enable the ablation single encoder training'
        )
        parser.add_argument(
            "--ablation_residual_addition",
            action='store_true', 
            help='Enable the ablation residual addition training'
        )
        parser.add_argument(
            "--ablation_scaling_data_num",
            type=int,
            default=99999999,
            help="Ablation scaling data num.",
        )

        # baseline
        parser.add_argument(
            "--baseline_single_condition",
            type=str,
            default=None,
            help="Baseline single condition.",
        )

    def validate_args(self, args: "BaseArgs"):
        assert self.rank > 0, "Rank must be a positive integer."
        assert self.lora_alpha > 0, "lora_alpha must be a positive integer."
        assert self.ref_videos_num > 0, "ref_videos_num must be a positive integer."
        assert self.sample_ref_videos_num > 0, "sample_ref_videos_num must be a positive integer."
        assert 0.0 <= self.mask_ref_ratio <= 1.0, "mask_ref_ratio must be between 0 and 1."
        assert 0.0 <= self.mask_caption_ratio <= 1.0, "mask_caption_ratio must be between 0 and 1."
        assert 0.0 <= self.caption_guidance_scale, "caption_guidance_scale must be a positive float."
        assert 0 <= self.num_ref_diffusion_time_sigmas <= 999, "num_ref_diffusion_time_sigmas must be a positive integer."
        assert self.ablation_scaling_data_num > 0, "ablation_scaling_data_num must be a positive integer."

    def map_args(self, argparse_args: argparse.Namespace, mapped_args: "BaseArgs"):
        mapped_args.rank = argparse_args.rank
        mapped_args.lora_alpha = argparse_args.lora_alpha
        mapped_args.target_modules = (
            argparse_args.target_modules[0] if len(argparse_args.target_modules) == 1 else argparse_args.target_modules
        )

        # mot
        mapped_args.videoasprompt_mot_name_or_path = argparse_args.videoasprompt_mot_name_or_path
        mapped_args.ref_videos_num = argparse_args.ref_videos_num
        mapped_args.sample_ref_videos_num = argparse_args.sample_ref_videos_num
        mapped_args.mask_ref_ratio = argparse_args.mask_ref_ratio
        mapped_args.mask_caption_ratio = argparse_args.mask_caption_ratio
        mapped_args.training_dataset_kind = argparse_args.training_dataset_kind
        mapped_args.model_structure_config = argparse_args.model_structure_config
        mapped_args.caption_guidance_scale = argparse_args.caption_guidance_scale
        mapped_args.reference_train_mode = argparse_args.reference_train_mode
        mapped_args.random_refer_noise = argparse_args.random_refer_noise
        mapped_args.num_ref_diffusion_time_sigmas = argparse_args.num_ref_diffusion_time_sigmas
        mapped_args.ref_type = argparse_args.ref_type
        mapped_args.dpo = argparse_args.dpo
        mapped_args.alignment_score_threshold = argparse_args.alignment_score_threshold

        # ablation
        mapped_args.ablation_single_branch = argparse_args.ablation_single_branch
        mapped_args.ablation_single_encoder = argparse_args.ablation_single_encoder
        mapped_args.ablation_residual_addition = argparse_args.ablation_residual_addition
        mapped_args.ablation_scaling_data_num = argparse_args.ablation_scaling_data_num

        # baseline
        mapped_args.baseline_single_condition = argparse_args.baseline_single_condition

    def to_dict(self) -> Dict[str, Any]:
        return {"rank": self.rank, "lora_alpha": self.lora_alpha, "target_modules": self.target_modules}
