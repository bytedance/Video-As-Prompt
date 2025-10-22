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

import PIL.Image
import torch
from accelerate import init_empty_weights
from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    WanImageToVideoPipeline,
    WanImageToVideoMOTPipeline,
    WanPipeline,
    WanTransformer3DModel,
    WanTransformer3DMOTModel,
)
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel
import PIL
from PIL.Image import Image
import cv2
import json

import finetrainers.functional as FF
from finetrainers.data import VideoArtifact
from finetrainers.logging import get_logger
from finetrainers.models.modeling_utils import ModelSpecification
from finetrainers.parallel import ParallelBackendEnum
from torch.distributed.tensor import DTensor, Replicate
from finetrainers.processors import ProcessorMixin, T5Processor, T5ProcessorMOT
from finetrainers.typing import ArtifactType, SchedulerType
from finetrainers.utils import get_non_null_items, safetensors_torch_save_function


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


class WanLatentEncodeProcessor(ProcessorMixin):
    r"""
    Processor to encode image/video into latents using the Wan VAE.

    Args:
        output_names (`List[str]`):
            The names of the outputs that the processor returns. The outputs are in the following order:
            - latents: The latents of the input image/video.
            - latents_mean: The channel-wise mean of the latent space.
            - latents_std: The channel-wise standard deviation of the latent space.
    """

    def __init__(self, output_names: List[str]):
        super().__init__()
        self.output_names = output_names
        assert len(self.output_names) == 3

    def forward(
        self,
        vae: AutoencoderKLWan,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
    ) -> Dict[str, torch.Tensor]:
        device = vae.device
        dtype = vae.dtype

        if image is not None:
            video = image.unsqueeze(1)

        assert video.ndim == 5, f"Expected 5D tensor, got {video.ndim}D tensor"
        video = video.to(device=device, dtype=dtype)
        video = video.permute(0, 2, 1, 3, 4).contiguous()  # [B, F, C, H, W] -> [B, C, F, H, W]

        if compute_posterior:
            latents = vae.encode(video).latent_dist.sample(generator=generator)
            latents = latents.to(dtype=dtype)
        else:
            # TODO(aryan): refactor in diffusers to have use_slicing attribute
            # if vae.use_slicing and video.shape[0] > 1:
            #     encoded_slices = [vae._encode(x_slice) for x_slice in video.split(1)]
            #     moments = torch.cat(encoded_slices)
            # else:
            #     moments = vae._encode(video)
            moments = vae._encode(video)
            latents = moments.to(dtype=dtype)

        latents_mean = torch.tensor(vae.config.latents_mean)
        latents_std = 1.0 / torch.tensor(vae.config.latents_std)

        return {self.output_names[0]: latents, self.output_names[1]: latents_mean, self.output_names[2]: latents_std}


class WanLatentEncodeProcessorMOT(ProcessorMixin):
    r"""
    Processor to encode image/video into latents using the Wan VAE.

    Args:
        output_names (`List[str]`):
            The names of the outputs that the processor returns. The outputs are in the following order:
            - latents: The latents of the input image/video.
            - latents_mean: The channel-wise mean of the latent space.
            - latents_std: The channel-wise standard deviation of the latent space.
    """

    def __init__(self, output_names: List[str]):
        super().__init__()
        self.output_names = output_names
        assert len(self.output_names) == 1

    def forward(
        self,
        vae: AutoencoderKLWan,
        ref_videos: Optional[List[torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
    ) -> Dict[str, torch.Tensor]:
        device = vae.device
        dtype = vae.dtype

        ref_videos_latents = []
        for ref_video in ref_videos:

            assert ref_video.ndim == 5, f"Expected 5D tensor, got {ref_video.ndim}D tensor"
            ref_video = ref_video.to(device=device, dtype=dtype)
            ref_video = ref_video.permute(0, 2, 1, 3, 4).contiguous()  # [B, F, C, H, W] -> [B, C, F, H, W]

            if compute_posterior:
                latents = vae.encode(ref_video).latent_dist.sample(generator=generator)
                latents = latents.to(dtype=dtype)
            else:
                # TODO(aryan): refactor in diffusers to have use_slicing attribute
                # if vae.use_slicing and video.shape[0] > 1:
                #     encoded_slices = [vae._encode(x_slice) for x_slice in video.split(1)]
                #     moments = torch.cat(encoded_slices)
                # else:
                #     moments = vae._encode(video)
                moments = vae._encode(ref_video)
                latents = moments.to(dtype=dtype)
            
            ref_videos_latents.append(latents)
        return {self.output_names[0]: ref_videos_latents}


class WanImageConditioningLatentEncodeProcessor(ProcessorMixin):
    r"""
    Processor to encode image/video into latents using the Wan VAE.

    Args:
        output_names (`List[str]`):
            The names of the outputs that the processor returns. The outputs are in the following order:
            - latents: The latents of the input image/video.
            - latents_mean: The channel-wise mean of the latent space.
            - latents_std: The channel-wise standard deviation of the latent space.
            - mask: The conditioning frame mask for the input image/video.
    """

    def __init__(self, output_names: List[str], *, use_last_frame: bool = False):
        super().__init__()
        self.output_names = output_names
        self.use_last_frame = use_last_frame
        assert len(self.output_names) == 4

    def forward(
        self,
        vae: AutoencoderKLWan,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        compute_posterior: bool = True,
    ) -> Dict[str, torch.Tensor]:
        device = vae.device
        dtype = vae.dtype

        if image is not None:
            video = image.unsqueeze(1)

        assert video.ndim == 5, f"Expected 5D tensor, got {video.ndim}D tensor"
        video = video.to(device=device, dtype=dtype)
        video = video.permute(0, 2, 1, 3, 4).contiguous()  # [B, F, C, H, W] -> [B, C, F, H, W]

        num_frames = video.size(2)
        if not self.use_last_frame:
            first_frame, remaining_frames = video[:, :, :1], video[:, :, 1:]
            video = torch.cat([first_frame, torch.zeros_like(remaining_frames)], dim=2)
        else:
            first_frame, remaining_frames, last_frame = video[:, :, :1], video[:, :, 1:-1], video[:, :, -1:]
            video = torch.cat([first_frame, torch.zeros_like(remaining_frames), last_frame], dim=2)

        # Image conditioning uses argmax sampling, so we use "mode" here
        if compute_posterior:
            latents = vae.encode(video).latent_dist.mode()
            latents = latents.to(dtype=dtype)
        else:
            # TODO(aryan): refactor in diffusers to have use_slicing attribute
            # if vae.use_slicing and video.shape[0] > 1:
            #     encoded_slices = [vae._encode(x_slice) for x_slice in video.split(1)]
            #     moments = torch.cat(encoded_slices)
            # else:
            #     moments = vae._encode(video)
            moments = vae._encode(video)
            latents = moments.to(dtype=dtype)

        latents_mean = torch.tensor(vae.config.latents_mean)
        latents_std = 1.0 / torch.tensor(vae.config.latents_std)

        temporal_downsample = 2 ** sum(vae.temperal_downsample) if getattr(self, "vae", None) else 4
        mask = latents.new_ones(latents.shape[0], 1, num_frames, latents.shape[3], latents.shape[4])
        if not self.use_last_frame:
            mask[:, :, 1:] = 0
        else:
            mask[:, :, 1:-1] = 0
        first_frame_mask = mask[:, :, :1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=temporal_downsample)
        mask = torch.cat([first_frame_mask, mask[:, :, 1:]], dim=2)
        mask = mask.view(latents.shape[0], -1, temporal_downsample, latents.shape[3], latents.shape[4])
        mask = mask.transpose(1, 2)

        return {
            self.output_names[0]: latents,
            self.output_names[1]: latents_mean,
            self.output_names[2]: latents_std,
            self.output_names[3]: mask,
        }


class WanImageConditioningLatentEncodeProcessorMOT(ProcessorMixin):
    r"""
    Processor to encode image/video into latents using the Wan VAE.

    Args:
        output_names (`List[str]`):
            The names of the outputs that the processor returns. The outputs are in the following order:
            - latents: The latents of the input image/video.
            - latents_mean: The channel-wise mean of the latent space.
            - latents_std: The channel-wise standard deviation of the latent space.
            - mask: The conditioning frame mask for the input image/video.
    """

    def __init__(self, output_names: List[str], *, use_last_frame: bool = False):
        super().__init__()
        self.output_names = output_names
        self.use_last_frame = use_last_frame
        assert len(self.output_names) == 2

    def forward(
        self,
        vae: AutoencoderKLWan,
        ref_videos: Optional[List[torch.Tensor]] = None,
        compute_posterior: bool = True,
    ) -> Dict[str, torch.Tensor]:
        device = vae.device
        dtype = vae.dtype

        ref_videos_image_latents = []
        ref_videos_image_latents_masks = []
        for ref_video in ref_videos:

            assert ref_video.ndim == 5, f"Expected 5D tensor, got {ref_video.ndim}D tensor"
            ref_video = ref_video.to(device=device, dtype=dtype)
            ref_video = ref_video.permute(0, 2, 1, 3, 4).contiguous()  # [B, F, C, H, W] -> [B, C, F, H, W]

            num_frames = ref_video.size(2)
            if not self.use_last_frame:
                first_frame, remaining_frames = ref_video[:, :, :1], ref_video[:, :, 1:]
                ref_video = torch.cat([first_frame, torch.zeros_like(remaining_frames)], dim=2)
            else:
                first_frame, remaining_frames, last_frame = ref_video[:, :, :1], ref_video[:, :, 1:-1], ref_video[:, :, -1:]
                ref_video = torch.cat([first_frame, torch.zeros_like(remaining_frames), last_frame], dim=2)

            # Image conditioning uses argmax sampling, so we use "mode" here
            if compute_posterior:
                latents = vae.encode(ref_video).latent_dist.mode()
                latents = latents.to(dtype=dtype)
            else:
                # TODO(aryan): refactor in diffusers to have use_slicing attribute
                # if vae.use_slicing and ref_video.shape[0] > 1:
                #     encoded_slices = [vae._encode(x_slice) for x_slice in ref_video.split(1)]
                #     moments = torch.cat(encoded_slices)
                # else:
                #     moments = vae._encode(ref_video)
                moments = vae._encode(ref_video)
                latents = moments.to(dtype=dtype)


            temporal_downsample = 2 ** sum(vae.temperal_downsample) if getattr(self, "vae", None) else 4
            mask = latents.new_ones(latents.shape[0], 1, num_frames, latents.shape[3], latents.shape[4])
            if not self.use_last_frame:
                mask[:, :, 1:] = 0
            else:
                mask[:, :, 1:-1] = 0
            first_frame_mask = mask[:, :, :1]
            first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=temporal_downsample)
            mask = torch.cat([first_frame_mask, mask[:, :, 1:]], dim=2)
            mask = mask.view(latents.shape[0], -1, temporal_downsample, latents.shape[3], latents.shape[4])
            mask = mask.transpose(1, 2)

            ref_videos_image_latents.append(latents)
            ref_videos_image_latents_masks.append(mask)

        return {
            self.output_names[0]: ref_videos_image_latents,
            self.output_names[1]: ref_videos_image_latents_masks,
        }


class WanImageEncodeProcessor(ProcessorMixin):
    r"""
    Processor to encoding image conditioning for Wan I2V training.

    Args:
        output_names (`List[str]`):
            The names of the outputs that the processor returns. The outputs are in the following order:
            - image_embeds: The CLIP vision model image embeddings of the input image.
    """

    def __init__(self, output_names: List[str], *, use_last_frame: bool = False):
        super().__init__()
        self.output_names = output_names
        self.use_last_frame = use_last_frame
        assert len(self.output_names) == 1

    def forward(
        self,
        image_encoder: CLIPVisionModel,
        image_processor: CLIPImageProcessor,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = image_encoder.device
        dtype = image_encoder.dtype
        last_image = None

        # We know the image here is in the range [-1, 1] (probably a little overshot if using bilinear interpolation), but
        # the processor expects it to be in the range [0, 1].
        image = image if video is None else video[:, 0]  # [B, F, C, H, W] -> [B, C, H, W] (take first frame)
        image = FF.normalize(image, min=0.0, max=1.0, dim=1)
        assert image.ndim == 4, f"Expected 4D tensor, got {image.ndim}D tensor"

        if self.use_last_frame:
            last_image = image if video is None else video[:, -1]
            last_image = FF.normalize(last_image, min=0.0, max=1.0, dim=1)
            image = torch.stack([image, last_image], dim=0)

        image = image_processor(images=image.float(), do_rescale=False, do_convert_rgb=False, return_tensors="pt")
        image = image.to(device=device, dtype=dtype)
        image_embeds = image_encoder(**image, output_hidden_states=True)
        image_embeds = image_embeds.hidden_states[-2]
        return {self.output_names[0]: image_embeds}


class WanImageEncodeProcessorMOT(ProcessorMixin):
    r"""
    Processor to encoding image conditioning for Wan I2V training.

    Args:
        output_names (`List[str]`):
            The names of the outputs that the processor returns. The outputs are in the following order:
            - image_embeds: The CLIP vision model image embeddings of the input image.
    """

    def __init__(self, output_names: List[str], *, use_last_frame: bool = False):
        super().__init__()
        self.output_names = output_names
        self.use_last_frame = use_last_frame
        assert len(self.output_names) == 1

    def forward(
        self,
        image_encoder: CLIPVisionModel,
        image_processor: CLIPImageProcessor,
        ref_videos: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        device = image_encoder.device
        dtype = image_encoder.dtype

        ref_videos_image_clip_latents = []

        for ref_video in ref_videos:

            last_image = None

            # We know the image here is in the range [-1, 1] (probably a little overshot if using bilinear interpolation), but
            # the processor expects it to be in the range [0, 1].
            image = image if ref_video is None else ref_video[:, 0]  # [B, F, C, H, W] -> [B, C, H, W] (take first frame)
            image = FF.normalize(image, min=0.0, max=1.0, dim=1)
            assert image.ndim == 4, f"Expected 4D tensor, got {image.ndim}D tensor"

            if self.use_last_frame:
                last_image = image if ref_video is None else ref_video[:, -1]
                last_image = FF.normalize(last_image, min=0.0, max=1.0, dim=1)
                image = torch.stack([image, last_image], dim=0)

            image = image_processor(images=image.float(), do_rescale=False, do_convert_rgb=False, return_tensors="pt")
            image = image.to(device=device, dtype=dtype)
            image_embeds = image_encoder(**image, output_hidden_states=True)
            image_embeds = image_embeds.hidden_states[-2]

            ref_videos_image_clip_latents.append(image_embeds)

        return {self.output_names[0]: ref_videos_image_clip_latents}


class WanModelSpecification(ModelSpecification):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
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

        use_last_frame = self.transformer_config.get("pos_embed_seq_len", None) is not None

        if condition_model_processors is None:
            condition_model_processors = [T5Processor(["encoder_hidden_states", "__drop__"])]
        if latent_model_processors is None:
            latent_model_processors = [WanLatentEncodeProcessor(["latents", "latents_mean", "latents_std"])]

        if self.transformer_config.get("image_dim", None) is not None:
            latent_model_processors.append(
                WanImageConditioningLatentEncodeProcessor(
                    ["latent_condition", "__drop__", "__drop__", "latent_condition_mask"],
                    use_last_frame=use_last_frame,
                )
            )
            latent_model_processors.append(
                WanImageEncodeProcessor(["encoder_hidden_states_image"], use_last_frame=use_last_frame)
            )

        if self.training_type == TrainingType.VIDEO_AS_PROMPT_MOT:
            condition_model_processors.append(T5ProcessorMOT(["encoder_hidden_states_mot_ref", "__drop__"]))

            latent_model_processors.append(WanLatentEncodeProcessorMOT(["latents_mot_ref"]))

            if self.transformer_config.get("image_dim", None) is not None:
                latent_model_processors.append(
                    WanImageConditioningLatentEncodeProcessorMOT(
                        ["latent_condition_mot_ref", "latent_condition_mask_mot_ref"],
                        use_last_frame=use_last_frame,
                    )
                )
                latent_model_processors.append(
                    WanImageEncodeProcessorMOT(["encoder_hidden_states_image_mot_ref"], use_last_frame=use_last_frame)
                )

        self.condition_model_processors = condition_model_processors
        self.latent_model_processors = latent_model_processors

    @property
    def _resolution_dim_keys(self):
        return {"latents": (2, 3, 4)}

    def load_condition_models(self) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        if self.tokenizer_id is not None:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id, **common_kwargs)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="tokenizer", **common_kwargs
            )

        if self.text_encoder_id is not None:
            text_encoder = AutoModel.from_pretrained(
                self.text_encoder_id, torch_dtype=self.text_encoder_dtype, **common_kwargs
            )
        else:
            text_encoder = UMT5EncoderModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="text_encoder",
                torch_dtype=self.text_encoder_dtype,
                **common_kwargs,
            )

        return {"tokenizer": tokenizer, "text_encoder": text_encoder}

    def load_latent_models(self) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        if self.vae_id is not None:
            vae = AutoencoderKLWan.from_pretrained(self.vae_id, torch_dtype=self.vae_dtype, **common_kwargs)
        else:
            vae = AutoencoderKLWan.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="vae", torch_dtype=self.vae_dtype, **common_kwargs
            )

        models = {"vae": vae}
        if self.transformer_config.get("image_dim", None) is not None:
            # TODO(aryan): refactor the trainer to be able to support these extra models from CLI args more easily
            image_encoder = CLIPVisionModel.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="image_encoder", torch_dtype=torch.bfloat16
            )
            image_processor = CLIPImageProcessor.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="image_processor"
            )
            models["image_encoder"] = image_encoder
            models["image_processor"] = image_processor

        return models

    def load_diffusion_models(self, videoasprompt_mot_name_or_path) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        if self.transformer_id is not None:
            transformer = WanTransformer3DModel.from_pretrained(
                self.transformer_id, torch_dtype=self.transformer_dtype, **common_kwargs
            )
        elif videoasprompt_mot_name_or_path is not None:
            logger.info(f"Load from a finetuned model: {videoasprompt_mot_name_or_path}")
            transformer = WanTransformer3DModel.from_pretrained(
                videoasprompt_mot_name_or_path, torch_dtype=self.transformer_dtype, **common_kwargs
            )
        else:
            transformer = WanTransformer3DModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="transformer",
                torch_dtype=self.transformer_dtype,
                **common_kwargs,
            )

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {"transformer": transformer, "scheduler": scheduler}

    def load_videoasprompt_mot_models(self, model_structure_config_path, videoasprompt_mot_name_or_path, reference_train_mode=None, ablation_single_encoder=False, ablation_residual_addition=False) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        assert model_structure_config_path != "", "model_structure_config must be provided for training type VIDEO_AS_PROMPT_MOT"

        with open(model_structure_config_path, "r") as f:
            model_structure_config = json.load(f)

        model_structure_config["reference_train_mode"] = reference_train_mode

        logger.info(f"model_structure_config: {model_structure_config}")

        if videoasprompt_mot_name_or_path is not None:
            logger.info(f"Load from a finetuned model: {videoasprompt_mot_name_or_path}")

            transformer = WanTransformer3DMOTModel.from_pretrained(
                videoasprompt_mot_name_or_path, torch_dtype=self.transformer_dtype, **common_kwargs
            )
        else:
            logger.info(f"Load from a pretrained model: {self.pretrained_model_name_or_path}")

            transformer = WanTransformer3DMOTModel.from_pretrained(
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

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {"transformer": transformer, "scheduler": scheduler}

    def load_pipeline(
        self,
        tokenizer: Optional[AutoTokenizer] = None,
        text_encoder: Optional[UMT5EncoderModel] = None,
        transformer: Optional[WanTransformer3DModel] = None,
        vae: Optional[AutoencoderKLWan] = None,
        scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None,
        image_encoder: Optional[CLIPVisionModel] = None,
        image_processor: Optional[CLIPImageProcessor] = None,
        enable_slicing: bool = False,
        enable_tiling: bool = False,
        enable_model_cpu_offload: bool = False,
        training: bool = False,
        **kwargs,
    ) -> Union[WanPipeline, WanImageToVideoPipeline, WanImageToVideoMOTPipeline]:
        components = {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "image_encoder": image_encoder,
            "image_processor": image_processor,
        }
        components = get_non_null_items(components)

        if self.transformer_config.get("image_dim", None) is None:
            pipe = WanPipeline.from_pretrained(
                self.pretrained_model_name_or_path, **components, revision=self.revision, cache_dir=self.cache_dir
            )
        else:
            if self.training_type == TrainingType.VIDEO_AS_PROMPT_MOT:
                pipe = WanImageToVideoMOTPipeline.from_pretrained(
                    self.pretrained_model_name_or_path, **components, revision=self.revision, cache_dir=self.cache_dir
                )
            else:
                pipe = WanImageToVideoPipeline.from_pretrained(
                    self.pretrained_model_name_or_path, **components, revision=self.revision, cache_dir=self.cache_dir
                )
        pipe.text_encoder.to(self.text_encoder_dtype)
        pipe.vae.to(self.vae_dtype)

        if not training:
            pipe.transformer.to(self.transformer_dtype)

        # TODO(aryan): add support in diffusers
        # if enable_slicing:
        #     pipe.vae.enable_slicing()
        # if enable_tiling:
        #     pipe.vae.enable_tiling()
        if enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()

        return pipe

    @torch.no_grad()
    def prepare_conditions(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        caption: str,
        max_sequence_length: int = 512,
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
        return conditions

    @torch.no_grad()
    def prepare_latents(
        self,
        vae: AutoencoderKLWan,
        image_encoder: Optional[CLIPVisionModel] = None,
        image_processor: Optional[CLIPImageProcessor] = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        conditions = {
            "vae": vae,
            "image_encoder": image_encoder,
            "image_processor": image_processor,
            "image": image,
            "video": video,
            "generator": generator,
            # We must force this to False because the latent normalization should be done before
            # the posterior is computed. The VAE does not handle this any more:
            # https://github.com/huggingface/diffusers/pull/10998
            "compute_posterior": False,
            **kwargs,
        }
        input_keys = set(conditions.keys())
        conditions = super().prepare_latents(**conditions)
        conditions = {k: v for k, v in conditions.items() if k not in input_keys}
        return conditions

    def forward(
        self,
        transformer: WanTransformer3DModel,
        condition_model_conditions: Dict[str, torch.Tensor],
        latent_model_conditions: Dict[str, torch.Tensor],
        sigmas: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
        # mot
        sigmas_mot_ref_list: Optional[List[torch.Tensor]] = None,
        reference_train_mode: Optional[str] = None,
        random_refer_noise: bool = False,
        ablation_single_branch: bool = False,
        baseline_single_condition: Optional[str] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        compute_posterior = False  # See explanation in prepare_latents
        latent_condition = latent_condition_mask = None

        if compute_posterior:
            latents = latent_model_conditions.pop("latents")
            latent_condition = latent_model_conditions.pop("latent_condition", None)
            latent_condition_mask = latent_model_conditions.pop("latent_condition_mask", None)
        else:
            latents = latent_model_conditions.pop("latents")
            latents_mean = latent_model_conditions.pop("latents_mean")
            latents_std = latent_model_conditions.pop("latents_std")
            latent_condition = latent_model_conditions.pop("latent_condition", None)
            latent_condition_mask = latent_model_conditions.pop("latent_condition_mask", None)

            mu, logvar = torch.chunk(latents, 2, dim=1)
            mu = self._normalize_latents(mu, latents_mean, latents_std)
            logvar = self._normalize_latents(logvar, latents_mean, latents_std)
            latents = torch.cat([mu, logvar], dim=1)

            posterior = DiagonalGaussianDistribution(latents)
            latents = posterior.sample(generator=generator)

            if latent_condition is not None:
                mu, logvar = torch.chunk(latent_condition, 2, dim=1)
                mu = self._normalize_latents(mu, latents_mean, latents_std)
                logvar = self._normalize_latents(logvar, latents_mean, latents_std)
                latent_condition = torch.cat([mu, logvar], dim=1)

                posterior = DiagonalGaussianDistribution(latent_condition)
                latent_condition = posterior.mode()

            del posterior

            if self.training_type == TrainingType.VIDEO_AS_PROMPT_MOT:
                tmp_latents_mot_ref = latent_model_conditions.pop("latents_mot_ref", None)[0]
                tmp_latent_condition_mot_ref = latent_model_conditions.pop("latent_condition_mot_ref", None)[0]
                latent_condition_mask_mot_ref = latent_model_conditions.pop("latent_condition_mask_mot_ref", None)[0]


                latents_mot_ref = []
                latent_condition_mot_ref = []
                for latents_mot_ref_i, latent_condition_mot_ref_i in zip(tmp_latents_mot_ref, tmp_latent_condition_mot_ref):
                    # latents_mot_ref
                    mu, logvar = torch.chunk(latents_mot_ref_i, 2, dim=1)
                    mu = self._normalize_latents(mu, latents_mean, latents_std)
                    logvar = self._normalize_latents(logvar, latents_mean, latents_std)
                    latents_mot_ref_i = torch.cat([mu, logvar], dim=1)

                    posterior = DiagonalGaussianDistribution(latents_mot_ref_i)
                    latents_mot_ref_i = posterior.sample(generator=generator)
                    
                    latents_mot_ref.append(latents_mot_ref_i)

                    # latent_condition_mot_ref
                    mu, logvar = torch.chunk(latent_condition_mot_ref_i, 2, dim=1)
                    mu = self._normalize_latents(mu, latents_mean, latents_std)
                    logvar = self._normalize_latents(logvar, latents_mean, latents_std)
                    latent_condition_mot_ref_i = torch.cat([mu, logvar], dim=1)

                    posterior = DiagonalGaussianDistribution(latent_condition_mot_ref_i)
                    latent_condition_mot_ref_i = posterior.mode()

                    latent_condition_mot_ref.append(latent_condition_mot_ref_i)
                    

                del posterior

        noise = torch.zeros_like(latents).normal_(generator=generator)
        noisy_latents = FF.flow_match_xt(latents, noise, sigmas)
        timesteps = (sigmas.flatten() * 1000.0).long()

        if sigmas_mot_ref_list is not None and self.training_type == TrainingType.VIDEO_AS_PROMPT_MOT:
            
            ref_noise_list = []
            noisy_latents_wref_list = []
            ref_timesteps_list = []
            
            for ref_sigmas_idx, ref_sigmas in enumerate(sigmas_mot_ref_list):
                
                # Do not override noise here, otherwise we will train a wrong model
                noise_mot_ref = torch.zeros_like(latents_mot_ref[ref_sigmas_idx]).normal_(generator=generator)
                ref_noise_list.append(noise_mot_ref)

                noisy_latents_wref = FF.flow_match_xt(latents_mot_ref[ref_sigmas_idx], noise_mot_ref, ref_sigmas)
                noisy_latents_wref_list.append(noisy_latents_wref)


                ref_timesteps = (ref_sigmas.flatten() * 1000.0).long()
                ref_timesteps_list.append(ref_timesteps)

                

        if self.transformer_config.get("image_dim", None) is not None:
            noisy_latents = torch.cat([noisy_latents, latent_condition_mask, latent_condition], dim=1)

            if self.training_type == TrainingType.VIDEO_AS_PROMPT_MOT:
                if reference_train_mode not in ["reference_independent"]:
                    noisy_latents_wref = [
                        torch.cat([latents_mot_ref[i], latent_condition_mask_mot_ref[i], latent_condition_mot_ref[i]], dim=1) for i in range(len(latents_mot_ref))
                    ]
                else:
                    noisy_latents_wref = [
                        torch.cat([noisy_latents_wref_list[i], latent_condition_mask_mot_ref[i], latent_condition_mot_ref[i]], dim=1) for i in range(len(latents_mot_ref))
                    ]
                    latent_model_conditions["reference_train_mode"] = reference_train_mode

        latent_model_conditions["hidden_states"] = noisy_latents.to(latents)

        if self.training_type == TrainingType.VIDEO_AS_PROMPT_MOT:
            latent_model_conditions["hidden_states_mot_ref"] = torch.cat(noisy_latents_wref, dim=2)

            latent_model_conditions["num_mot_ref"] = int(latent_model_conditions["hidden_states_mot_ref"].shape[2] // latent_model_conditions["hidden_states"].shape[2])

            condition_model_conditions["encoder_hidden_states_mot_ref"] = torch.cat(condition_model_conditions["encoder_hidden_states_mot_ref"][0], dim=1)

            latent_model_conditions["encoder_hidden_states_image_mot_ref"] = torch.cat(latent_model_conditions["encoder_hidden_states_image_mot_ref"][0], dim=1)
            

        if reference_train_mode is None:
            if ablation_single_branch and baseline_single_condition is not None:
                latent_model_conditions.pop(f'num_mot_ref')
                latent_model_conditions.pop(f'hidden_states_mot_ref')
                condition_model_conditions.pop('encoder_hidden_states_mot_ref')
                latent_model_conditions.pop('encoder_hidden_states_image_mot_ref')
                pred = transformer(
                    **latent_model_conditions,
                    **condition_model_conditions,
                    timestep=timesteps,
                    return_dict=False,
                )[0]
            else:
                pred = transformer(
                    **latent_model_conditions,
                    **condition_model_conditions,
                    timestep=timesteps,
                    timestep_list_mot_ref=ref_timesteps_list,
                    return_dict=False,
                )[0]
            target = FF.flow_match_target(noise, latents)
            # logger.warning(f"pred: {pred.shape}")
            # logger.warning(f"target: {target.shape}")
            # logger.warning(f"timesteps: {timesteps}; ref_timesteps_list: {ref_timesteps_list}")

            return pred, target, sigmas
        elif reference_train_mode in ["reference_independent"]:
            pred, pred_mot_ref = transformer(
                **latent_model_conditions,
                **condition_model_conditions,
                timestep=timesteps,
                timestep_list_mot_ref=ref_timesteps_list,
                return_dict=False,
            )
            target = FF.flow_match_target(noise, latents)

            pred_mot_ref_list = list(torch.chunk(pred_mot_ref, latent_model_conditions["num_mot_ref"], dim=2))

            target_mot_ref_list = []
            for i in range(latent_model_conditions["num_mot_ref"]):
                target_mot_ref_list.append(FF.flow_match_target(ref_noise_list[i], latents_mot_ref[i]))

            return pred, target, sigmas, pred_mot_ref_list, target_mot_ref_list, sigmas_mot_ref_list
        else:
            raise ValueError(f"sigmas_mot_ref_list: {sigmas_mot_ref_list}, reference_train_mode: {reference_train_mode}")

    def validation(
        self,
        pipeline: Union[WanPipeline, WanImageToVideoPipeline, WanImageToVideoMOTPipeline],
        prompt: str,
        image: Optional[PIL.Image.Image] = None,
        last_image: Optional[PIL.Image.Image] = None,
        video: Optional[List[PIL.Image.Image]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> List[ArtifactType]:
        generation_kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
            "return_dict": True,
            "output_type": "pil",
        }
        if self.transformer_config.get("image_dim", None) is not None:
            if image is None and video is None:
                raise ValueError("Either image or video must be provided for Wan I2V validation.")
            image = image if image is not None else video[0]
            generation_kwargs["image"] = image
        if self.transformer_config.get("pos_embed_seq_len", None) is not None:
            last_image = last_image if last_image is not None else image if video is None else video[-1]
            generation_kwargs["last_image"] = last_image
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
            WanImageToVideoPipeline if self.transformer_config.get("image_dim", None) is not None else WanPipeline
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
        transformer: WanTransformer3DModel,
        transformer_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        scheduler: Optional[SchedulerType] = None,
    ) -> None:
        # TODO(aryan): this needs refactoring
        if transformer_state_dict is not None:
            with init_empty_weights():
                transformer_copy = WanTransformer3DModel.from_config(transformer.config)
            transformer_copy.load_state_dict(transformer_state_dict, strict=True, assign=True)
            transformer_copy.save_pretrained(os.path.join(directory, "transformer"))
        if scheduler is not None:
            scheduler.save_pretrained(os.path.join(directory, "scheduler"))

    def _save_model_videoasprompt_mot(
        self,
        directory: str,
        transformer: WanTransformer3DMOTModel,
        transformer_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        scheduler: Optional[SchedulerType] = None,
    ) -> None:
        # TODO(aryan): this needs refactoring
        if transformer_state_dict is not None:
            with init_empty_weights():
                transformer_copy = WanTransformer3DMOTModel.from_config(transformer.config)
            transformer_copy.load_state_dict(transformer_state_dict, strict=True, assign=True)
            transformer_copy.save_pretrained(os.path.join(directory, "transformer"))
        if scheduler is not None:
            scheduler.save_pretrained(os.path.join(directory, "scheduler"))

    def apply_tensor_parallel(
        self,
        backend: ParallelBackendEnum,
        device_mesh: torch.distributed.DeviceMesh,
        transformer: WanTransformer3DMOTModel,
        **kwargs,
    ) -> None:
        if backend == ParallelBackendEnum.PTD:
            _apply_tensor_parallel_ptd(device_mesh, transformer)
        else:
            raise NotImplementedError(f"Parallel backend {backend} is not supported for LTXVideoModelSpecification")
    
    
    @staticmethod
    def _normalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

def _apply_tensor_parallel_ptd(
    device_mesh: torch.distributed.device_mesh.DeviceMesh, transformer: WanTransformer3DMOTModel
    ) -> None:
    from torch.distributed.tensor.parallel import parallelize_module
    from torch.distributed.tensor.parallel.style import ColwiseParallel, RowwiseParallel

    transformer_plan = {
    }

    for block in transformer.blocks:
        block_plan = {
            # "attn1_mot_ref.to_q": ColwiseParallel(output_layouts=Replicate()),
            # "attn1_mot_ref.to_k": ColwiseParallel(output_layouts=Replicate()),
            # "attn1_mot_ref.to_v": ColwiseParallel(output_layouts=Replicate()),
            # "attn1_mot_ref.to_out.0": RowwiseParallel(input_layouts=Replicate()),
            # "attn2_mot_ref.to_q": ColwiseParallel(output_layouts=Replicate()),
            # "attn2_mot_ref.to_k": ColwiseParallel(output_layouts=Replicate()),
            # "attn2_mot_ref.to_v": ColwiseParallel(output_layouts=Replicate()),
            # "attn2_mot_ref.to_out.0": RowwiseParallel(input_layouts=Replicate()),
            "ffn_mot_ref.net.0.proj": ColwiseParallel(),
            "ffn_mot_ref.net.2": RowwiseParallel(),

            # "attn1.to_q": ColwiseParallel(output_layouts=Replicate()),
            # "attn1.to_k": ColwiseParallel(output_layouts=Replicate()),
            # "attn1.to_v": ColwiseParallel(output_layouts=Replicate()),
            # "attn1.to_out.0": RowwiseParallel(input_layouts=Replicate()),
            # "attn2.to_q": ColwiseParallel(output_layouts=Replicate()),
            # "attn2.to_k": ColwiseParallel(output_layouts=Replicate()),
            # "attn2.to_v": ColwiseParallel(output_layouts=Replicate()),
            # "attn2.to_out.0": RowwiseParallel(input_layouts=Replicate()),
            "ffn.net.0.proj": ColwiseParallel(),
            "ffn.net.2": RowwiseParallel(),
        }

        parallelize_module(block, device_mesh, block_plan)

    parallelize_module(transformer, device_mesh, transformer_plan)
