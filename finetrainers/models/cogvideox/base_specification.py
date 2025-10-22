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
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import json

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDDIMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXPipeline,
    CogVideoXImageToVideoMOTPipeline,
    CogVideoXTransformer3DModel,
    CogVideoXTransformer3DMOTModel,
)
import PIL
from PIL.Image import Image
import cv2
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer

from finetrainers.data import VideoArtifact
from finetrainers.logging import get_logger
from finetrainers.models.modeling_utils import ModelSpecification
from finetrainers.models.utils import DiagonalGaussianDistribution
from finetrainers.processors import ProcessorMixin, T5Processor, T5ProcessorMOT
from finetrainers.typing import ArtifactType, SchedulerType
from finetrainers.utils import _enable_vae_memory_optimizations, get_non_null_items, safetensors_torch_save_function

from .utils import prepare_rotary_positional_embeddings


logger = get_logger()

from enum import Enum
class TrainingType(str, Enum):
    # SFT
    LORA = "lora"
    FULL_FINETUNE = "full-finetune"

    # Control
    CONTROL_LORA = "control-lora"
    CONTROL_FULL_FINETUNE = "control-full-finetune"

    # mot
    VIDEO_AS_PROMPT_MOT = "video-as-prompt-mot"


class CogVideoXLatentEncodeProcessor(ProcessorMixin):
    r"""
    Processor to encode image/video into latents using the CogVideoX VAE.

    Args:
        output_names (`List[str]`):
            The names of the outputs that the processor returns. The outputs are in the following order:
            - latents: The latents of the input image/video.
    """

    def __init__(self, output_names: List[str]):
        super().__init__()
        self.output_names = output_names
        assert len(self.output_names) == 1

    def forward(
        self,
        vae: AutoencoderKLCogVideoX,
        video: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
    ) -> Dict[str, torch.Tensor]:
        device = vae.device
        dtype = vae.dtype
        
        if video.ndim == 6:
            video = video.squeeze(0)

        assert video.ndim == 5, f"Expected 5D tensor, got {video.ndim}D tensor with {video.shape}"
        video = video.to(device=device, dtype=vae.dtype)
        video = video.permute(0, 2, 1, 3, 4).contiguous()  # [B, F, C, H, W] -> [B, C, F, H, W]

        if compute_posterior:
            latents = vae.encode(video).latent_dist.sample(generator=generator)
            latents = latents.to(dtype=dtype)
        else:
            if vae.use_slicing and video.shape[0] > 1:
                encoded_slices = [vae._encode(x_slice) for x_slice in video.split(1)]
                moments = torch.cat(encoded_slices)
            else:
                moments = vae._encode(video)
            latents = moments.to(dtype=dtype)

        latents = latents.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W] -> [B, F, C, H, W]
        return {self.output_names[0]: latents}

class CogVideoXLatentEncodeProcessorMOT(ProcessorMixin):
    r"""
    Processor to encode image/video into latents using the CogVideoX VAE.

    Args:
        output_names (`List[str]`):
            The names of the outputs that the processor returns. The outputs are in the following order:
            - latents: The latents of the input image/video.
    """

    def __init__(self, output_names: List[str]):
        super().__init__()
        self.output_names = output_names
        assert len(self.output_names) == 1

    def forward(
        self,
        vae: AutoencoderKLCogVideoX,
        ref_videos: Optional[List[torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
    ) -> Dict[str, List[torch.Tensor]]:
        device = vae.device
        dtype = vae.dtype

        ref_videos_latents = []
        for ref_video in ref_videos:

            assert ref_video.ndim == 5, f"Expected 5D tensor, got {ref_video.ndim}D tensor"
            ref_video = ref_video.to(device=device, dtype=vae.dtype)
            ref_video = ref_video.permute(0, 2, 1, 3, 4).contiguous()  # [B, F, C, H, W] -> [B, C, F, H, W]

            if compute_posterior:
                latents = vae.encode(ref_video).latent_dist.sample(generator=generator)
                latents = latents.to(dtype=dtype)
            else:
                if vae.use_slicing and ref_video.shape[0] > 1:
                    encoded_slices = [vae._encode(x_slice) for x_slice in ref_video.split(1)]
                    moments = torch.cat(encoded_slices)
                else:
                    moments = vae._encode(ref_video)
                latents = moments.to(dtype=dtype)

            latents = latents.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W] -> [B, F, C, H, W]
            ref_videos_latents.append(latents)
        return {self.output_names[0]: ref_videos_latents}

class CogVideoXImageConditioningLatentEncodeProcessor(ProcessorMixin):
    r"""
    Processor to encode image into latents using the CogVideoX VAE.

    Args:
        output_names (`List[str]`):
            The names of the outputs that the processor returns. The outputs are in the following order:
            - image_latents: The latents of the input image.
    """

    def __init__(self, output_names: List[str]):
        super().__init__()
        self.output_names = output_names
        assert len(self.output_names) == 1

    def forward(
        self,
        vae: AutoencoderKLCogVideoX,
        video: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
    ) -> Dict[str, torch.Tensor]:
        device = vae.device
        dtype = vae.dtype

        if video.ndim == 6:
            video = video.squeeze(0)

        assert video.ndim == 5, f"Expected 5D tensor, got {video.ndim}D tensor"
        video = video.to(device=device, dtype=vae.dtype)
        video = video.permute(0, 2, 1, 3, 4).contiguous()  # [B, F, C, H, W] -> [B, C, F, H, W]
        video = video[:, :, :1, :, :]


        if compute_posterior:
            latents = vae.encode(video).latent_dist.sample(generator=generator)
            image_latents = latents.to(dtype=dtype)
        else:
            if vae.use_slicing and video.shape[0] > 1:
                encoded_slices = [vae._encode(x_slice) for x_slice in video.split(1)]
                moments = torch.cat(encoded_slices)
            else:
                moments = vae._encode(video)
            image_latents = moments.to(dtype=dtype)

        image_latents = image_latents.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W] -> [B, F, C, H, W]

        return {self.output_names[0]: image_latents}

class CogVideoXImageConditioningLatentEncodeProcessorMOT(ProcessorMixin):
    r"""
    Processor to encode image into latents using the CogVideoX VAE.

    Args:
        output_names (`List[str]`):
            The names of the outputs that the processor returns. The outputs are in the following order:
            - image_latents: The latents of the input image.
    """

    def __init__(self, output_names: List[str]):
        super().__init__()
        self.output_names = output_names
        assert len(self.output_names) == 1

    def forward(
        self,
        vae: AutoencoderKLCogVideoX,
        ref_videos: Optional[List[torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
    ) -> Dict[str, List[torch.Tensor]]:
        device = vae.device
        dtype = vae.dtype

        ref_videos_image_latents = []
        for ref_video in ref_videos:

            assert ref_video.ndim == 5, f"Expected 5D tensor, got {ref_video.ndim}D tensor"
            ref_video = ref_video.to(device=device, dtype=vae.dtype)
            ref_video = ref_video.permute(0, 2, 1, 3, 4).contiguous()  # [B, F, C, H, W] -> [B, C, F, H, W]
            ref_video = ref_video[:, :, :1, :, :]

            if compute_posterior:
                latents = vae.encode(ref_video).latent_dist.sample(generator=generator)
                image_latents = latents.to(dtype=dtype)
            else:
                if vae.use_slicing and ref_video.shape[0] > 1:
                    encoded_slices = [vae._encode(x_slice) for x_slice in ref_video.split(1)]
                    moments = torch.cat(encoded_slices)
                else:
                    moments = vae._encode(ref_video)
                image_latents = moments.to(dtype=dtype)

            image_latents = image_latents.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W] -> [B, F, C, H, W]

            ref_videos_image_latents.append(image_latents)

        return {self.output_names[0]: ref_videos_image_latents}

class CogVideoXModelSpecification(ModelSpecification):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "THUDM/CogVideoX-5b",
        tokenizer_id: Optional[str] = None,
        text_encoder_id: Optional[str] = None,
        transformer_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        text_encoder_dtype: torch.dtype = torch.bfloat16,
        transformer_dtype: torch.dtype = torch.bfloat16,
        vae_dtype: torch.dtype = torch.bfloat16,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        condition_model_processors: List[ProcessorMixin] = None,
        latent_model_processors: List[ProcessorMixin] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            tokenizer_id=tokenizer_id,
            text_encoder_id=text_encoder_id,
            transformer_id=transformer_id,
            vae_id=vae_id,
            text_encoder_dtype=text_encoder_dtype,
            transformer_dtype=transformer_dtype,
            vae_dtype=vae_dtype,
            revision=revision,
            cache_dir=cache_dir,
        )
        self.training_type = kwargs.get("training_type", None)

        if condition_model_processors is None:
            condition_model_processors = [T5Processor(["encoder_hidden_states", "prompt_attention_mask"])]
        if latent_model_processors is None:
            latent_model_processors = [CogVideoXLatentEncodeProcessor(["latents"])]

        if self.transformer_config.get("in_channels", 16) == 32:
            latent_model_processors.append(
                CogVideoXImageConditioningLatentEncodeProcessor(["image_latents"])
            )

        if self.training_type == TrainingType.VIDEO_AS_PROMPT_MOT:
            condition_model_processors.append(T5ProcessorMOT(["encoder_hidden_states_mot_ref", "prompt_attention_mask_mot_ref"]))
            latent_model_processors.append(CogVideoXLatentEncodeProcessorMOT(["latents_mot_ref"]))
            latent_model_processors.append(CogVideoXImageConditioningLatentEncodeProcessorMOT(["image_latents_mot_ref"]))

        self.condition_model_processors = condition_model_processors
        self.latent_model_processors = latent_model_processors

    @property
    def _resolution_dim_keys(self):
        return {"latents": (1, 3, 4)}

    def load_condition_models(self) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        if self.tokenizer_id is not None:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id, **common_kwargs)
        else:
            tokenizer = T5Tokenizer.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="tokenizer", **common_kwargs
            )

        if self.text_encoder_id is not None:
            text_encoder = AutoModel.from_pretrained(
                self.text_encoder_id, torch_dtype=self.text_encoder_dtype, **common_kwargs
            )
        else:
            text_encoder = T5EncoderModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="text_encoder",
                torch_dtype=self.text_encoder_dtype,
                **common_kwargs,
            )

        return {"tokenizer": tokenizer, "text_encoder": text_encoder}

    def load_latent_models(self) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        if self.vae_id is not None:
            vae = AutoencoderKLCogVideoX.from_pretrained(self.vae_id, torch_dtype=self.vae_dtype, **common_kwargs)
        else:
            vae = AutoencoderKLCogVideoX.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="vae", torch_dtype=self.vae_dtype, **common_kwargs
            )

        return {"vae": vae}

    def load_diffusion_models(self, videoasprompt_mot_name_or_path) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        if self.transformer_id is not None:
            transformer = CogVideoXTransformer3DModel.from_pretrained(
                self.transformer_id, torch_dtype=self.transformer_dtype, **common_kwargs
            )
        elif videoasprompt_mot_name_or_path is not None:
            logger.info(f"Load from a finetuned model: {videoasprompt_mot_name_or_path}")
            transformer = CogVideoXTransformer3DModel.from_pretrained(
                videoasprompt_mot_name_or_path, torch_dtype=self.transformer_dtype, **common_kwargs
            )
        else:
            transformer = CogVideoXTransformer3DModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="transformer",
                torch_dtype=self.transformer_dtype,
                **common_kwargs,
            )

        scheduler = CogVideoXDDIMScheduler.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="scheduler", **common_kwargs
        )

        return {"transformer": transformer, "scheduler": scheduler}

    def load_videoasprompt_mot_models(self, model_structure_config_path, videoasprompt_mot_name_or_path, reference_train_mode=None, ablation_single_encoder=False, ablation_residual_addition=False) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        assert model_structure_config_path != "", "model_structure_config must be provided for training type VIDEO_AS_PROMPT_MOT"

        with open(model_structure_config_path, "r") as f:
            model_structure_config = json.load(f)

        model_structure_config["reference_train_mode"] = reference_train_mode
        model_structure_config["ablation_single_encoder"] = ablation_single_encoder
        model_structure_config["ablation_residual_addition"] = ablation_residual_addition

        logger.info(f"model_structure_config: {model_structure_config}")

        if videoasprompt_mot_name_or_path is not None:
            logger.info(f"Load from a finetuned model: {videoasprompt_mot_name_or_path}")
            transformer = CogVideoXTransformer3DMOTModel.from_pretrained(
                videoasprompt_mot_name_or_path, torch_dtype=self.transformer_dtype, **common_kwargs
            )
        else:
            logger.info(f"Load from a pretrained model: {self.pretrained_model_name_or_path}")
            transformer = CogVideoXTransformer3DMOTModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="transformer",
                torch_dtype=self.transformer_dtype,
                **model_structure_config,
                **common_kwargs,
            )
            state_dict = transformer.state_dict()

            for name, param in state_dict.items():
                if "_mot_ref" in name:
                    base_name = name.replace("_mot_ref", "")
                    if base_name in state_dict and "config_ori" in model_structure_config_path:
                        state_dict[name] = state_dict[base_name]
                    elif base_name in state_dict and "config_ori" not in model_structure_config_path:
                        param_device = param.device if param.device.type != 'meta' else 'cuda'
                        if param.shape == state_dict[base_name].shape:
                            state_dict[name] = state_dict[base_name]
                        else:
                            if param.ndim > 1:
                                new_param = nn.Parameter(torch.empty_like(param, dtype=param.dtype, device=param_device))
                                init.xavier_uniform_(new_param)
                            else:
                                if ".weight" in name and "norm" in name:
                                    new_param = nn.Parameter(torch.ones_like(param, dtype=param.dtype, device=param_device) + torch.randn_like(param, dtype=param.dtype, device=param_device) * 0.02)
                                elif "bias" in name:
                                    new_param = nn.Parameter(torch.zeros_like(param, dtype=param.dtype, device=param_device))
                                else:
                                    raise ValueError(f"warning: No base parameter found for {name} with shape {param.shape}")
                            state_dict[name] = new_param
                    else:
                        raise ValueError(f"warning: No base parameter found for {name}")

            for name, param in state_dict.items():
                if "effect_embeddings" in name or "ref_embeddings" in name:
                    if param.device.type == 'meta':
                        param_device = 'cuda'
                        new_param = nn.Parameter(torch.zeros(param.shape, dtype=param.dtype, device=param_device))
                        state_dict[name] = new_param
                    else:
                        raise ValueError(f"{name}'device = {param.device.type}")

            transformer.load_state_dict(state_dict, strict=True, assign=True)


        scheduler = CogVideoXDDIMScheduler.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="scheduler", **common_kwargs
        )

        return {"transformer": transformer, "scheduler": scheduler}

    def load_pipeline(
        self,
        tokenizer: Optional[T5Tokenizer] = None,
        text_encoder: Optional[T5EncoderModel] = None,
        transformer: Optional[CogVideoXTransformer3DModel] = None,
        vae: Optional[AutoencoderKLCogVideoX] = None,
        scheduler: Optional[CogVideoXDDIMScheduler] = None,
        enable_slicing: bool = False,
        enable_tiling: bool = False,
        enable_model_cpu_offload: bool = False,
        training: bool = False,
        **kwargs,
    ) -> Union[CogVideoXPipeline, CogVideoXImageToVideoPipeline, CogVideoXImageToVideoMOTPipeline]:
        components = {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
        }
        components = get_non_null_items(components)

        if self.training_type == TrainingType.VIDEO_AS_PROMPT_MOT:
            if self.transformer_config.get("in_channels", 16) == 32:
                pipe = CogVideoXImageToVideoMOTPipeline.from_pretrained(
                    self.pretrained_model_name_or_path, **components, revision=self.revision, cache_dir=self.cache_dir
                )
            else:
                raise NotImplementedError("CogVideoXMOTPipeline is not implemented")
        else:
            if self.transformer_config.get("in_channels", 16) == 32:
                pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                    self.pretrained_model_name_or_path, **components, revision=self.revision, cache_dir=self.cache_dir
                )
            else:
                pipe = CogVideoXPipeline.from_pretrained(
                    self.pretrained_model_name_or_path, **components, revision=self.revision, cache_dir=self.cache_dir
                )

        pipe.text_encoder.to(self.text_encoder_dtype)
        pipe.vae.to(self.vae_dtype)

        _enable_vae_memory_optimizations(pipe.vae, enable_slicing, enable_tiling)
        if not training:
            pipe.transformer.to(self.transformer_dtype)
        if enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()
        return pipe

    @torch.no_grad()
    def prepare_conditions(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        caption: str,
        max_sequence_length: int = 226,
        effect_types: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        conditions = {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "caption": caption,
            "max_sequence_length": max_sequence_length,
            **kwargs,
        }
        input_keys = set(conditions.keys())
        conditions = super().prepare_conditions(**conditions)
        conditions = {k: v for k, v in conditions.items() if k not in input_keys}
        conditions.pop("prompt_attention_mask", None)
        if self.training_type == TrainingType.VIDEO_AS_PROMPT_MOT:
            conditions.pop("prompt_attention_mask_mot_ref", None)
            conditions["effect_types"] = effect_types
        return conditions

    @torch.no_grad()
    def prepare_latents(
        self,
        vae: AutoencoderKLCogVideoX,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
        ref_videos: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        conditions = {
            "vae": vae,
            "image": image,
            "video": video,
            "generator": generator,
            "compute_posterior": compute_posterior,
            "ref_videos": ref_videos,
            **kwargs,
        }
        input_keys = set(conditions.keys())
        conditions = super().prepare_latents(**conditions)
        if "dpo_weight" in conditions:
            dpo_weight = conditions.pop("dpo_weight")
        else:
            dpo_weight = None
        conditions = {k: v for k, v in conditions.items() if k not in input_keys}
        if dpo_weight is not None:
            conditions["dpo_weight"] = dpo_weight
        return conditions

    def forward(
        self,
        transformer: CogVideoXTransformer3DModel,
        scheduler: CogVideoXDDIMScheduler,
        condition_model_conditions: Dict[str, torch.Tensor],
        latent_model_conditions: Dict[str, torch.Tensor],
        sigmas: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
        sigmas_mot_ref_list: Optional[List[torch.Tensor]] = None,
        reference_train_mode: Optional[str] = None,
        random_refer_noise: bool = False,
        ref_type: Optional[str] = "continous_negative",
        ablation_single_branch: bool = False,
        baseline_single_condition: Optional[str] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        if "noisy_latents" not in latent_model_conditions:
            # Just hardcode for now. In Diffusers, we will refactor such that RoPE would be handled within the model itself.
            VAE_SPATIAL_SCALE_FACTOR = 8
            rope_base_height = self.transformer_config.sample_height * VAE_SPATIAL_SCALE_FACTOR
            rope_base_width = self.transformer_config.sample_width * VAE_SPATIAL_SCALE_FACTOR
            patch_size = self.transformer_config.patch_size
            patch_size_t = getattr(self.transformer_config, "patch_size_t", None)

            if compute_posterior:
                latents = latent_model_conditions.pop("latents")
                if self.training_type == TrainingType.VIDEO_AS_PROMPT_MOT:
                    latents_mot_ref = latent_model_conditions.pop("latents_mot_ref")[0]
            else:
                posterior = DiagonalGaussianDistribution(latent_model_conditions.pop("latents"), _dim=2)
                latents = posterior.sample(generator=generator)
                del posterior
                if self.training_type == TrainingType.VIDEO_AS_PROMPT_MOT:
                    tmp_latents_mot_ref = latent_model_conditions.pop("latents_mot_ref")[0]
                    latents_mot_ref = []
                    for latents_mot_ref_i in tmp_latents_mot_ref:
                        posterior = DiagonalGaussianDistribution(latents_mot_ref_i, _dim=2)
                        latents_mot_ref.append(posterior.sample(generator=generator))
                        del posterior

            # # BUG: no multiply for video latents previously
            # if not getattr(self.vae_config, "invert_scale_latents", False):
            #     latents = latents * self.vae_config.scaling_factor
            #     if self.training_type == TrainingType.VIDEO_AS_PROMPT_MOT:
            #         latents_mot_ref = [latents_mot_ref_i * self.vae_config.scaling_factor for latents_mot_ref_i in latents_mot_ref]
            latents = latents * self.vae_config.scaling_factor
            if self.training_type == TrainingType.VIDEO_AS_PROMPT_MOT:
                latents_mot_ref = [latents_mot_ref_i * self.vae_config.scaling_factor for latents_mot_ref_i in latents_mot_ref]

            if patch_size_t is not None:
                latents = self._pad_frames(latents, patch_size_t)
                if self.training_type == TrainingType.VIDEO_AS_PROMPT_MOT:
                    latents_mot_ref = [self._pad_frames(latents_mot_ref_i, patch_size_t) for latents_mot_ref_i in latents_mot_ref]

            timesteps = (sigmas.flatten() * 1000.0).long()

            noise = torch.zeros_like(latents).normal_(generator=generator)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            # logger.warning(f"Forward noisy_latents shape: {noisy_latents.shape}")

            if sigmas_mot_ref_list is not None and self.training_type == TrainingType.VIDEO_AS_PROMPT_MOT:
                ref_timesteps_list = []
                ref_noisy_latents_list = []
                for ref_sigmas_idx, ref_sigmas in enumerate(sigmas_mot_ref_list):
                    ref_timesteps = (ref_sigmas.flatten() * 1000.0).long()
                    ref_timesteps_list.append(ref_timesteps)

                    noise_mot_ref = torch.zeros_like(latents_mot_ref[ref_sigmas_idx]).normal_(generator=generator)
                    ref_noisy_latents = scheduler.add_noise(latents_mot_ref[ref_sigmas_idx], noise_mot_ref, ref_timesteps)
                    ref_noisy_latents_list.append(ref_noisy_latents)

            if self.transformer_config.get("in_channels", 16) == 32:
                

                if compute_posterior:
                    image_latents = latent_model_conditions.pop("image_latents")
                else:
                    posterior = DiagonalGaussianDistribution(latent_model_conditions.pop("image_latents"), _dim=2)
                    image_latents = posterior.sample(generator=generator)
                    del posterior
                # logger.warning(f"Forward image_latents shape: {image_latents.shape}")
                
                if not getattr(self.vae_config, "invert_scale_latents", False):
                    image_latents = image_latents * self.vae_config.scaling_factor

                batch_size, num_frames, num_channels_latents, height, width = noisy_latents.shape
                
                padding_shape = (
                    batch_size,
                    num_frames - 1,
                    num_channels_latents,
                    image_latents.shape[3],
                    image_latents.shape[4],
                )

                latent_padding = torch.zeros(padding_shape, device=noisy_latents.device, dtype=noisy_latents.dtype)
                image_latents = torch.cat([image_latents, latent_padding], dim=1)

                # Select the first frame along the second dimension
                if patch_size_t is not None:
                    first_frame = image_latents[:, : image_latents.size(1) % patch_size_t, ...]
                    image_latents = torch.cat([first_frame, image_latents], dim=1)

                if self.training_type == TrainingType.VIDEO_AS_PROMPT_MOT:
                    tmp_image_latents_mot_ref = latent_model_conditions.pop("image_latents_mot_ref")[0]
                    image_latents_mot_ref = []
                    for image_latents_mot_ref_i in tmp_image_latents_mot_ref:

                        if compute_posterior:
                            pass
                        else:
                            posterior = DiagonalGaussianDistribution(image_latents_mot_ref_i, _dim=2)
                            image_latents_mot_ref_i = posterior.sample(generator=generator)
                            del posterior
                        
                        if not getattr(self.vae_config, "invert_scale_latents", False):
                            image_latents_mot_ref_i = image_latents_mot_ref_i * self.vae_config.scaling_factor

                        image_latents_mot_ref_i = torch.cat([image_latents_mot_ref_i, latent_padding[:image_latents_mot_ref_i.shape[0]]], dim=1)

                        if patch_size_t is not None:
                            first_frame = image_latents_mot_ref_i[:, : image_latents_mot_ref_i.size(1) % patch_size_t, ...]
                            image_latents_mot_ref_i = torch.cat([first_frame, image_latents_mot_ref_i], dim=1)

                        image_latents_mot_ref.append(image_latents_mot_ref_i)

            batch_size, num_frames, num_channels, height, width = latents.shape
            ofs_emb = (
                None
                if getattr(self.transformer_config, "ofs_embed_dim", None) is None
                else latents.new_full((batch_size,), fill_value=2.0)
            )

            image_rotary_emb = (
                prepare_rotary_positional_embeddings(
                    height=height * VAE_SPATIAL_SCALE_FACTOR,
                    width=width * VAE_SPATIAL_SCALE_FACTOR,
                    num_frames=num_frames,
                    vae_scale_factor_spatial=VAE_SPATIAL_SCALE_FACTOR,
                    patch_size=patch_size,
                    patch_size_t=patch_size_t,
                    attention_head_dim=self.transformer_config.attention_head_dim,
                    device=transformer.device,
                    base_height=rope_base_height,
                    base_width=rope_base_width,
                )
                if self.transformer_config.use_rotary_positional_embeddings
                else None
            )
            # logger.warning(f"Forward image_rotary_emb shape: {len(image_rotary_emb)} {image_rotary_emb[0].shape}")

            if self.training_type == TrainingType.VIDEO_AS_PROMPT_MOT:
                image_rotary_emb_mot_ref = (
                    prepare_rotary_positional_embeddings(
                        height=height * VAE_SPATIAL_SCALE_FACTOR,
                        width=width * VAE_SPATIAL_SCALE_FACTOR,
                        num_frames=num_frames,
                        vae_scale_factor_spatial=VAE_SPATIAL_SCALE_FACTOR,
                        patch_size=patch_size,
                        patch_size_t=patch_size_t,
                        attention_head_dim=self.transformer_config.attention_head_dim,
                        device=transformer.device,
                        base_height=rope_base_height,
                        base_width=rope_base_width,
                        mot_num=len(latents_mot_ref),
                        ref_type=ref_type,
                    )
                    if self.transformer_config.use_rotary_positional_embeddings
                    else None
                )
                latent_model_conditions["image_rotary_emb_mot_ref"] = image_rotary_emb_mot_ref

                condition_model_conditions["effect_types"] = [effect_type_item[0] for effect_type_item in condition_model_conditions["effect_types"][0]]


            if self.transformer_config.get("in_channels", 16) == 32:
                latent_model_conditions["hidden_states"] = torch.cat([noisy_latents, image_latents], dim=2)
                if self.training_type == TrainingType.VIDEO_AS_PROMPT_MOT:
                    if reference_train_mode not in ["reference_independent"]:
                        latent_model_conditions["hidden_states_mot_ref"] = [
                            torch.cat([latents_mot_ref[i], image_latents_mot_ref[i]], dim=2) for i in range(len(latents_mot_ref))
                        ]
                    else:
                        latent_model_conditions["hidden_states_mot_ref"] = [
                            torch.cat([ref_noisy_latents_list[i], image_latents_mot_ref[i]], dim=2) for i in range(len(ref_noisy_latents_list))
                        ]
                        latent_model_conditions["reference_train_mode"] = reference_train_mode
                    latent_model_conditions["hidden_states_mot_ref"] = torch.cat(latent_model_conditions["hidden_states_mot_ref"], dim=1)

                    latent_model_conditions["num_mot_ref"] = int(latent_model_conditions["hidden_states_mot_ref"].shape[1] // latent_model_conditions["hidden_states"].shape[1])

                    condition_model_conditions["encoder_hidden_states_mot_ref"] = torch.cat(condition_model_conditions["encoder_hidden_states_mot_ref"][0], dim=1)
                    # HACK: for dpo
                    if latent_model_conditions["hidden_states"].shape[0] == 2:
                        latent_model_conditions["hidden_states_mot_ref"] = latent_model_conditions["hidden_states_mot_ref"].unsqueeze(1).expand(-1, 2, -1, -1, -1, -1).reshape(-1, num_frames, num_channels*2, height, width)
                        condition_model_conditions["encoder_hidden_states_mot_ref"] = condition_model_conditions["encoder_hidden_states_mot_ref"].unsqueeze(1).expand(-1, 2, -1, -1).reshape(-1, condition_model_conditions["encoder_hidden_states_mot_ref"].shape[1], condition_model_conditions["encoder_hidden_states_mot_ref"].shape[2])

            else:
                latent_model_conditions["hidden_states"] = noisy_latents.to(latents)
                if self.training_type == TrainingType.VIDEO_AS_PROMPT_MOT:
                    raise NotImplementedError("CogVideoX T2V with MoT hasn't been supported")
            latent_model_conditions["image_rotary_emb"] = image_rotary_emb
            latent_model_conditions["ofs"] = ofs_emb
        else:
            # logger.info(f"DPO Ref Inference!")
            noisy_latents = latent_model_conditions.pop("noisy_latents")
            latents = latent_model_conditions.pop("latents")
            timesteps = latent_model_conditions.pop("timesteps")
        if reference_train_mode is None:
            if ablation_single_branch and not baseline_single_condition:
                latent_model_conditions['hidden_states'] = torch.cat([latent_model_conditions['hidden_states'], latent_model_conditions.pop('hidden_states_mot_ref')], dim=1)
                latent_model_conditions['image_rotary_emb'] = (torch.cat([latent_model_conditions['image_rotary_emb'][0], latent_model_conditions['image_rotary_emb_mot_ref'][0]], dim=0), torch.cat([latent_model_conditions['image_rotary_emb'][1], latent_model_conditions['image_rotary_emb_mot_ref'][1]], dim=0))
                
                latent_model_conditions.pop(f'num_mot_ref')
                latent_model_conditions.pop('image_rotary_emb_mot_ref')
                condition_model_conditions.pop(f'effect_types')
                condition_model_conditions.pop('encoder_hidden_states_mot_ref')
                latent_model_conditions['ablation_single_branch'] = ablation_single_branch

            if ablation_single_branch and baseline_single_condition is not None:
                latent_model_conditions.pop(f'num_mot_ref')
                latent_model_conditions.pop('image_rotary_emb_mot_ref')
                latent_model_conditions.pop(f'hidden_states_mot_ref')
                condition_model_conditions.pop(f'effect_types')
                condition_model_conditions.pop('encoder_hidden_states_mot_ref')
                latent_model_conditions['ablation_single_branch'] = False

            velocity = transformer(
                **latent_model_conditions,
                **condition_model_conditions,
                timestep=timesteps,
                return_dict=False,
            )[0]
            if ablation_single_branch:
                velocity = velocity[:, :noisy_latents.shape[1], :, :, :]
            # For CogVideoX, the transformer predicts the velocity. The denoised output is calculated by applying the same
            # code paths as scheduler.get_velocity(), which can be confusing to understand.
            pred = scheduler.get_velocity(velocity, noisy_latents, timesteps)
            target = latents
            latent_model_conditions['noisy_latents'] = noisy_latents
            latent_model_conditions['latents'] = latents
            latent_model_conditions['timesteps'] = timesteps
            # logger.warning(f"Forward velocity shape: {velocity.shape}")
            # logger.warning(f"Forward pred shape: {pred.shape}")
            # logger.warning(f"Forward target shape: {target.shape}")
            return pred, target, sigmas
        elif reference_train_mode in ["reference_independent"]:
            velocity, velocity_mot_ref = transformer(
                **latent_model_conditions,
                **condition_model_conditions,
                timestep=timesteps,
                timestep_list_mot_ref=ref_timesteps_list if random_refer_noise else None,
                return_dict=False,
            )
            # For CogVideoX, the transformer predicts the velocity. The denoised output is calculated by applying the same
            # code paths as scheduler.get_velocity(), which can be confusing to understand.
            pred = scheduler.get_velocity(velocity, noisy_latents, timesteps)
            target = latents

            velocity_mot_ref_list = list(torch.chunk(velocity_mot_ref, latent_model_conditions["num_mot_ref"], dim=1))
            pred_mot_ref_list = []
            target_mot_ref_list = []
            for i in range(latent_model_conditions["num_mot_ref"]):
                pred_mot_ref_list.append(scheduler.get_velocity(velocity_mot_ref_list[i], ref_noisy_latents_list[i], ref_timesteps_list[i]))
                target_mot_ref_list.append(latents_mot_ref[i])

            

            return pred, target, sigmas, pred_mot_ref_list, target_mot_ref_list, sigmas_mot_ref_list
        else:
            raise ValueError(f"sigmas_mot_ref_list: {sigmas_mot_ref_list}, reference_train_mode: {reference_train_mode}")

    def validation(
        self,
        pipeline: CogVideoXPipeline,
        prompt: str,
        image: Optional[Image] = None,
        video: Optional[List[Image]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> List[ArtifactType]:
        # TODO(aryan): add support for more parameters
        if image is not None and self.training_type != TrainingType.VIDEO_AS_PROMPT_MOT:
            pipeline = CogVideoXImageToVideoPipeline.from_pipe(pipeline)

        generation_kwargs = {
            "prompt": prompt,
            "image": image,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
            "return_dict": True,
            "output_type": "pil",
        }
        if self.transformer_config.get("in_channels", 16) == 32:
            if image is None and video is None:
                raise ValueError("Either image or video must be provided for CogVideoX I2V validation.")
            image = image if image is not None else video[0]
            generation_kwargs["image"] = image
        if self.training_type == TrainingType.VIDEO_AS_PROMPT_MOT:
            generation_kwargs["ref_videos"] = kwargs.get("ref_videos", None)
            generation_kwargs["prompt_mot_ref"] = kwargs.get("prompt_mot_ref", None)
            generation_kwargs["negative_prompt"] = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
            generation_kwargs["negative_prompt_mot_ref"] = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
            generation_kwargs["empty_caption_mot_ref"] = kwargs.get("empty_caption_mot_ref", "")
            generation_kwargs["caption_guidance_scale"] = kwargs.get("caption_guidance_scale", 0)
            generation_kwargs["effect_types"] = kwargs.get("effect_types", None)
            generation_kwargs["reference_train_mode"] = kwargs.get("reference_train_mode", None)
            generation_kwargs["random_refer_noise"] = kwargs.get("random_refer_noise", False)
            generation_kwargs["frames_selection"] = kwargs.get("frames_selection", "evenly")
            generation_kwargs["ref_type"] = kwargs.get("ref_type", "continous_negative")
            # ablation
            generation_kwargs["ablation_single_branch"] = kwargs.get("ablation_single_branch", False)
            generation_kwargs["baseline_single_condition"] = kwargs.get("baseline_single_condition", None)
            

        original_video = video
        generation_kwargs = get_non_null_items(generation_kwargs)
        video = pipeline(**generation_kwargs).frames[0]

        if self.training_type == TrainingType.VIDEO_AS_PROMPT_MOT:
            concat_frames = []

            generation_kwargs["video"] = original_video
            
            for i in range(len(video)):
                width, height = video[i].size
                resized_orig = generation_kwargs["video"][i].resize((width, height), PIL.Image.LANCZOS)
                resized_refs = []
                for j in range(len(generation_kwargs["ref_videos"])):
                    resized_refs.append(generation_kwargs["ref_videos"][j][i].resize((width, height), PIL.Image.LANCZOS))
                
                concat_img = PIL.Image.new('RGB', (width * (2+len(resized_refs)), height))
                for j in range(len(resized_refs)):
                    resized_refs[j] = resized_refs[j].convert("RGB")
                    concat_img.paste(resized_refs[j], (width * j, 0))
                concat_img.paste(resized_orig, (width * len(resized_refs), 0))
                concat_img.paste(video[i], (width * (len(resized_refs)+1), 0))
                
                concat_frames.append(concat_img)
            
            video = concat_frames


        return [VideoArtifact(value=video)]

    def _save_lora_weights(
        self,
        directory: str,
        transformer_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        scheduler: Optional[SchedulerType] = None,
        metadata: Optional[Dict[str, str]] = None,
        *args,
        **kwargs,
    ) -> None:
        pipeline_cls = (
            CogVideoXImageToVideoPipeline if self.transformer_config.get("in_channels", 16) == 32 else CogVideoXPipeline
        )
        # TODO(aryan): this needs refactoring
        if transformer_state_dict is not None:
            pipeline_cls.save_lora_weights(
                directory,
                transformer_state_dict,
                save_function=functools.partial(safetensors_torch_save_function, metadata=metadata),
                safe_serialization=True,
            )
        if scheduler is not None:
            scheduler.save_pretrained(os.path.join(directory, "scheduler"))

    def _save_model(
        self,
        directory: str,
        transformer: CogVideoXTransformer3DModel,
        transformer_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        scheduler: Optional[SchedulerType] = None,
    ) -> None:
        # TODO(aryan): this needs refactoring
        if transformer_state_dict is not None:
            with init_empty_weights():
                transformer_copy = CogVideoXTransformer3DModel.from_config(transformer.config)
            transformer_copy.load_state_dict(transformer_state_dict, strict=True, assign=True)
            transformer_copy.save_pretrained(os.path.join(directory, "transformer"))
        if scheduler is not None:
            scheduler.save_pretrained(os.path.join(directory, "scheduler"))

    def _save_model_videoasprompt_mot(
        self,
        directory: str,
        transformer: CogVideoXTransformer3DMOTModel,
        transformer_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        scheduler: Optional[SchedulerType] = None,
    ) -> None:
        # TODO(aryan): this needs refactoring
        if transformer_state_dict is not None:
            with init_empty_weights():
                transformer_copy = CogVideoXTransformer3DMOTModel.from_config(transformer.config)
            transformer_copy.load_state_dict(transformer_state_dict, strict=True, assign=True)
            transformer_copy.save_pretrained(os.path.join(directory, "transformer"))
        if scheduler is not None:
            scheduler.save_pretrained(os.path.join(directory, "scheduler"))

    @staticmethod
    def _pad_frames(latents: torch.Tensor, patch_size_t: int) -> torch.Tensor:
        num_frames = latents.size(1)
        additional_frames = patch_size_t - (num_frames % patch_size_t)
        if additional_frames > 0:
            last_frame = latents[:, -1:]
            padding_frames = last_frame.expand(-1, additional_frames, -1, -1, -1)
            latents = torch.cat([latents, padding_frames], dim=1)
        return latents
