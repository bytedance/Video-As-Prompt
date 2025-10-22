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


import warnings
warnings.filterwarnings('ignore')
import functools
import json
import os, shutil, uuid, tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


import datasets.distributed
import torch
import torch.nn.functional as F
import wandb
from diffusers import DiffusionPipeline
from diffusers.hooks import apply_layerwise_casting
from diffusers.training_utils import cast_training_params
from diffusers.utils import export_to_video
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, get_peft_model_state_dict
from tqdm import tqdm
import gc
import copy


from finetrainers import data, logging, models, optimizer, parallel, utils
from finetrainers.args import BaseArgsType
from finetrainers.config import TrainingType
from finetrainers.state import TrainState

from ..base import Trainer
from .config import SFTFullRankConfig, SFTLowRankConfig, VideoAsPromptMOTConfig


ArgsType = Union[BaseArgsType, SFTFullRankConfig, SFTLowRankConfig, VideoAsPromptMOTConfig]

logger = logging.get_logger()


class SFTTrainer(Trainer):
    def __init__(self, args: ArgsType, model_specification: models.ModelSpecification) -> None:
        super().__init__(args)

        # Tokenizers
        self.tokenizer = None
        self.tokenizer_2 = None
        self.tokenizer_3 = None

        # Text encoders
        self.text_encoder = None
        self.text_encoder_2 = None
        self.text_encoder_3 = None

        # Image encoders
        self.image_encoder = None
        self.image_processor = None

        # Denoisers
        self.transformer = None
        self.unet = None
        self.transformer_ref = None

        # Autoencoders
        self.vae = None

        # Scheduler
        self.scheduler = None

        # Optimizer & LR scheduler
        self.optimizer = None
        self.lr_scheduler = None

        # Checkpoint manager
        self.checkpointer = None

        self.model_specification = model_specification
        self._are_condition_models_loaded = False

    def run(self) -> None:
        try:
            self._init_logging()
            self._init_trackers()
            self._init_directories_and_repositories()
            self._prepare_models()
            self._prepare_trainable_parameters()
            self._prepare_for_training()
            self._prepare_dataset()
            self._prepare_checkpointing()
            self._train()
        except Exception as e:
            logger.error(f"Error during training: {e}")
            self.state.parallel_backend.destroy()
            raise e

    def _prepare_models(self) -> None:
        logger.info("Initializing models")

        if self.args.training_type == TrainingType.VIDEO_AS_PROMPT_MOT:
            if not self.args.ablation_single_branch:
                videoasprompt_mot_components = self.model_specification.load_videoasprompt_mot_models(
                    model_structure_config_path=self.args.model_structure_config,
                    videoasprompt_mot_name_or_path=self.args.videoasprompt_mot_name_or_path,
                    reference_train_mode=self.args.reference_train_mode,
                    ablation_single_encoder=self.args.ablation_single_encoder,
                    ablation_residual_addition=self.args.ablation_residual_addition,
                )
            if self.args.dpo:
                videoasprompt_mot_components["transformer_ref"] = copy.deepcopy(videoasprompt_mot_components["transformer"])
                videoasprompt_mot_components["transformer_ref"].requires_grad_(False)
                videoasprompt_mot_components["transformer_ref"].eval()
            if self.args.ablation_single_branch:
                videoasprompt_mot_components = self.model_specification.load_diffusion_models(videoasprompt_mot_name_or_path=self.args.videoasprompt_mot_name_or_path)
            self._set_components(videoasprompt_mot_components)
        else:
            diffusion_components = self.model_specification.load_diffusion_models()
            self._set_components(diffusion_components)

        if self.state.parallel_backend.pipeline_parallel_enabled:
            raise NotImplementedError(
                "Pipeline parallelism is not supported yet. This will be supported in the future."
            )

    def _prepare_trainable_parameters(self) -> None:
        logger.info("Initializing trainable parameters")

        parallel_backend = self.state.parallel_backend

        if self.args.training_type == TrainingType.FULL_FINETUNE or (self.args.ablation_single_branch and self.args.training_type == TrainingType.VIDEO_AS_PROMPT_MOT and self.args.target_modules == "none"):
            logger.info("Finetuning transformer with no additional parameters")
            utils.set_requires_grad([self.transformer], True)
        elif self.args.training_type == TrainingType.LORA or (self.args.ablation_single_branch and self.args.training_type == TrainingType.VIDEO_AS_PROMPT_MOT and self.args.target_modules != "none"):
            logger.info("Finetuning transformer with PEFT parameters")
            utils.set_requires_grad([self.transformer], False)
        elif self.args.training_type == TrainingType.VIDEO_AS_PROMPT_MOT:
            logger.info("Finetuning transformer in MOT mode")
            utils.set_requires_grad([self.transformer], False)
            
            mot_ref_params = 0
            total_params = 0
            all_params_list = []
            mot_params_list = []
            for name, param in self.transformer.named_parameters():
                total_params += 1
                if "_mot_ref" in name or "effect_embeddings" in name or "ref_embeddings" in name:
                    if not self.args.dpo:
                        utils.set_requires_grad([param], True)
                    else:
                        utils.set_requires_grad([param], False)
                    mot_ref_params += 1

                    mot_params_list.append(name)
                all_params_list.append(name)

            if parallel_backend.is_main_process:
                with open(os.path.join(self.args.output_dir, "all_params_list.json"), "w", encoding="utf-8") as f:
                    json.dump(all_params_list, f, ensure_ascii=False, indent=2)
                with open(os.path.join(self.args.output_dir, "mot_params_list.json"), "w", encoding="utf-8") as f:
                    json.dump(mot_params_list, f, ensure_ascii=False, indent=2)
                
            logger.info(f"The number of mot_ref parameters: {mot_ref_params}/{total_params}")

            if self.args.dpo:
                transformer_requires_grad = [p.requires_grad for p in self.transformer_ref.parameters()]
                logger.info(f"Transformer for dpo reference parameters requiring gradient: {sum(transformer_requires_grad)}/{len(transformer_requires_grad)}")
        else:
            raise ValueError(f"Training type {self.args.training_type} is not supported")
        transformer_requires_grad = [p.requires_grad for p in self.transformer.parameters()]
        logger.info(f"Transformer parameters requiring gradient: {sum(transformer_requires_grad)}/{len(transformer_requires_grad)}")
        # Layerwise upcasting must be applied before adding the LoRA adapter.
        # If we don't perform this before moving to device, we might OOM on the GPU. So, best to do it on
        # CPU for now, before support is added in Diffusers for loading and enabling layerwise upcasting directly.
        if self.args.training_type == TrainingType.LORA and "transformer" in self.args.layerwise_upcasting_modules:
            apply_layerwise_casting(
                self.transformer,
                storage_dtype=self.args.layerwise_upcasting_storage_dtype,
                compute_dtype=self.args.transformer_dtype,
                skip_modules_pattern=self.args.layerwise_upcasting_skip_modules_pattern,
                non_blocking=True,
            )

        transformer_lora_config = None
        if self.args.training_type == TrainingType.LORA or (self.args.training_type == TrainingType.VIDEO_AS_PROMPT_MOT and self.args.dpo) or (self.args.ablation_single_branch and self.args.training_type == TrainingType.VIDEO_AS_PROMPT_MOT and self.args.target_modules != "none"):
            transformer_lora_config = LoraConfig(
                r=self.args.rank,
                lora_alpha=self.args.lora_alpha,
                init_lora_weights=True,
                target_modules=self.args.target_modules,
            )
            self.transformer.add_adapter(transformer_lora_config)

            transformer_requires_grad = [p.requires_grad for p in self.transformer.parameters()]
            logger.info(f"Transformer after adding lora parameters requiring gradient: {sum(transformer_requires_grad)}/{len(transformer_requires_grad)}")

            if self.args.dpo:
                transformer_requires_grad = [p.requires_grad for p in self.transformer_ref.parameters()]
                logger.info(f"Transformer_ref after lora parameters requiring gradient: {sum(transformer_requires_grad)}/{len(transformer_requires_grad)}")

                assert id(self.transformer) != id(self.transformer_ref)

        # Make sure the trainable params are in float32 if data sharding is not enabled. For FSDP, we need all
        # parameters to be of the same dtype.
        if parallel_backend.data_sharding_enabled:
            self.transformer.to(dtype=self.args.transformer_dtype)
        else:
            if self.args.training_type == TrainingType.LORA or (self.args.training_type == TrainingType.VIDEO_AS_PROMPT_MOT and self.args.dpo) or (self.args.ablation_single_branch and self.args.training_type == TrainingType.VIDEO_AS_PROMPT_MOT and self.args.target_modules != "none"):
                cast_training_params([self.transformer], dtype=torch.float32)
            elif self.args.training_type == TrainingType.VIDEO_AS_PROMPT_MOT:
                # BUG: to be determined
                # cast_training_params([self.transformer], dtype=torch.float32)
                self.transformer.to(dtype=self.args.transformer_dtype)
        
        if self.args.dpo:
            self.transformer_ref.to(dtype=self.args.transformer_dtype)

    def _prepare_for_training(self) -> None:
        # 1. Apply parallelism
        parallel_backend = self.state.parallel_backend
        model_specification = self.model_specification

        if parallel_backend.context_parallel_enabled:
            parallel_backend.apply_context_parallel(self.transformer, parallel_backend.get_mesh()["cp"])

        if parallel_backend.tensor_parallel_enabled:
            # TODO(aryan): handle fp8 from TorchAO here
            model_specification.apply_tensor_parallel(
                backend=parallel.ParallelBackendEnum.PTD,
                device_mesh=parallel_backend.get_mesh()["tp"],
                transformer=self.transformer,
            )

        # Enable gradient checkpointing
        if self.args.gradient_checkpointing:
            # TODO(aryan): support other checkpointing types
            utils.apply_activation_checkpointing(self.transformer, checkpointing_type="full")

        # Apply torch.compile
        self._maybe_torch_compile()

        # Enable DDP, FSDP or HSDP
        if parallel_backend.data_sharding_enabled:
            # TODO(aryan): remove this when supported
            if self.args.parallel_backend == "accelerate":
                raise NotImplementedError("Data sharding is not supported with Accelerate yet.")

            dp_method = "HSDP" if parallel_backend.data_replication_enabled else "FSDP"
            logger.info(f"Applying {dp_method} on the model")

            if parallel_backend.data_replication_enabled or parallel_backend.context_parallel_enabled:
                dp_mesh_names = ("dp_replicate", "dp_shard_cp")
            else:
                dp_mesh_names = ("dp_shard_cp",)

            parallel_backend.apply_fsdp2(
                model=self.transformer,
                param_dtype=self.args.transformer_dtype,
                reduce_dtype=torch.float32,
                output_dtype=None,
                pp_enabled=parallel_backend.pipeline_parallel_enabled,
                cpu_offload=False,  # TODO(aryan): needs to be tested and allowed for enabling later
                device_mesh=parallel_backend.get_mesh()[dp_mesh_names],
            )
        elif parallel_backend.data_replication_enabled:
            if parallel_backend.get_mesh().ndim > 1:
                raise ValueError("DDP not supported for > 1D parallelism")
            logger.info("Applying DDP to the model")
            parallel_backend.apply_ddp(self.transformer, parallel_backend.get_mesh())
        else:
            parallel_backend.prepare_model(self.transformer)
            if self.args.dpo:
                parallel_backend.prepare_model(self.transformer_ref)

        self._move_components_to_device()

        # 2. Prepare optimizer and lr scheduler
        # For training LoRAs, we can be a little more optimal. Currently, the OptimizerWrapper only accepts torch::nn::Module.
        # This causes us to loop over all the parameters (even ones that don't require gradients, as in LoRA) at each optimizer
        # step. This is OK (see https://github.com/pytorch/pytorch/blob/2f40f789dafeaa62c4e4b90dbf4a900ff6da2ca4/torch/optim/sgd.py#L85-L99)
        # but can be optimized a bit by maybe creating a simple wrapper module encompassing the actual parameters that require
        # gradients. TODO(aryan): look into it in the future.
        model_parts = [self.transformer]
        self.state.num_trainable_parameters = sum(
            p.numel() for m in model_parts for p in m.parameters() if p.requires_grad
        )

        # Setup distributed optimizer and lr scheduler
        logger.info("Initializing optimizer and lr scheduler")
        self.state.train_state = TrainState()
        self.optimizer = optimizer.get_optimizer(
            parallel_backend=self.args.parallel_backend,
            name=self.args.optimizer,
            model_parts=model_parts,
            learning_rate=self.args.lr,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            fused=False,
        )
        self.lr_scheduler = optimizer.get_lr_scheduler(
            parallel_backend=self.args.parallel_backend,
            name=self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.lr_warmup_steps,
            num_training_steps=self.args.train_steps,
            # TODO(aryan): handle last_epoch
        )
        self.optimizer, self.lr_scheduler = parallel_backend.prepare_optimizer(self.optimizer, self.lr_scheduler)

        # # 3. Initialize trackers, directories and repositories
        # self._init_logging()
        # self._init_trackers()
        # self._init_directories_and_repositories()

    def _prepare_dataset(self) -> None:
        logger.info("Initializing dataset and dataloader")

        with open(self.args.dataset_config, "r") as file:
            dataset_configs = json.load(file)["datasets"]
        logger.info(f"Training configured to use {len(dataset_configs)} datasets")
        logger.info(f"Dataset configs: {dataset_configs}")

        datasets = []
        for config in dataset_configs:
            data_root = config.pop("data_root", None)
            dataset_file = config.pop("dataset_file", None)
            dataset_type = config.pop("dataset_type")
            caption_options = config.pop("caption_options", {})

            if data_root is not None and dataset_file is not None:
                raise ValueError("Both data_root and dataset_file cannot be provided in the same dataset config.")

            dataset_name_or_root = data_root or dataset_file
            if self.args.training_type in [TrainingType.LORA, TrainingType.FULL_FINETUNE, ]:
                dataset = data.initialize_dataset(
                    dataset_name_or_root, dataset_type, streaming=True, infinite=True, _caption_options=caption_options
                )
            elif self.args.training_type in [TrainingType.VIDEO_AS_PROMPT_MOT]:
                meta_df_name = config.pop("meta_df_name", "")
                dataset = data.initialize_videoasprompt_dataset(
                    dataset_name_or_root, 
                    dataset_type, 
                    streaming=True, 
                    infinite=True,
                    _caption_options=caption_options, 
                    ref_videos_num=self.args.ref_videos_num, 
                    sample_ref_videos_num=self.args.sample_ref_videos_num,
                    mask_ref_ratio=self.args.mask_ref_ratio, 
                    mask_caption_ratio=self.args.mask_caption_ratio,
                    training_dataset_kind=self.args.training_dataset_kind,
                    meta_df_name=meta_df_name,
                    dpo=self.args.dpo,
                    ablation_scaling_data_num=self.args.ablation_scaling_data_num,
                    baseline_single_condition=self.args.baseline_single_condition,
                    alignment_score_threshold=self.args.alignment_score_threshold,
                )
            else:
                raise ValueError(f"Training type {self.args.training_type} is not supported")

            if not dataset._precomputable_once and self.args.precomputation_once:
                raise ValueError(
                    f"Dataset {dataset_name_or_root} does not support precomputing all embeddings at once."
                )

            logger.info(f"Initialized dataset: {dataset_name_or_root}")
            dataset = self.state.parallel_backend.prepare_dataset(dataset)
            dataset = data.wrap_iterable_dataset_for_preprocessing(dataset, dataset_type, config)
            datasets.append(dataset)

        dataset = data.combine_datasets(datasets, buffer_size=self.args.dataset_shuffle_buffer_size, shuffle=True)
        dataloader = self.state.parallel_backend.prepare_dataloader(
            dataset, batch_size=1, num_workers=self.args.dataloader_num_workers, pin_memory=self.args.pin_memory
        )

        self.dataset = dataset
        self.dataloader = dataloader

    def _prepare_checkpointing(self) -> None:
        parallel_backend = self.state.parallel_backend

        def save_model_hook(state_dict: Dict[str, Any]) -> None:
            state_dict = utils.get_unwrapped_model_state_dict(state_dict)
            if parallel_backend.is_main_process:
                if self.args.training_type == TrainingType.LORA or (self.args.training_type == TrainingType.VIDEO_AS_PROMPT_MOT and self.args.dpo) or (self.args.ablation_single_branch and self.args.training_type == TrainingType.VIDEO_AS_PROMPT_MOT and self.args.target_modules != "none"):
                    state_dict = get_peft_model_state_dict(self.transformer, state_dict)
                    # fmt: off
                    metadata = {
                        "r": self.args.rank,
                        "lora_alpha": self.args.lora_alpha,
                        "init_lora_weights": True,
                        "target_modules": self.args.target_modules,
                    }
                    metadata = {"lora_config": json.dumps(metadata, indent=4)}
                    # fmt: on
                    self.model_specification._save_lora_weights(
                        os.path.join(self.args.output_dir, "lora_weights", f"{self.state.train_state.step:06d}"),
                        state_dict,
                        self.scheduler,
                        metadata,
                    )
                elif self.args.training_type == TrainingType.FULL_FINETUNE or (self.args.ablation_single_branch and self.args.training_type == TrainingType.VIDEO_AS_PROMPT_MOT and self.args.target_modules == "none"):
                    self.model_specification._save_model(
                        os.path.join(self.args.output_dir, "model_weights", f"{self.state.train_state.step:06d}"),
                        self.transformer,
                        state_dict,
                        self.scheduler,
                    )
                elif self.args.training_type == TrainingType.VIDEO_AS_PROMPT_MOT:
                    self.model_specification._save_model_videoasprompt_mot(
                        os.path.join(self.args.output_dir, "model_weights", f"{self.state.train_state.step:06d}"),
                        self.transformer,
                        state_dict,
                        self.scheduler,
                    )
            parallel_backend.wait_for_everyone()

        enable_state_checkpointing = self.args.checkpointing_steps > 0
        self.checkpointer = parallel_backend.get_checkpointer(
            dataloader=self.dataloader,
            model_parts=[self.transformer],
            optimizers=self.optimizer,
            schedulers=self.lr_scheduler,
            states={"train_state": self.state.train_state},
            checkpointing_steps=self.args.checkpointing_steps,
            checkpointing_limit=self.args.checkpointing_limit,
            output_dir=self.args.output_dir,
            enable=enable_state_checkpointing,
            _callback_fn=save_model_hook,
        )

        resume_from_checkpoint = self.args.resume_from_checkpoint
        if resume_from_checkpoint == "latest":
            resume_from_checkpoint = -1
        if resume_from_checkpoint is not None:
            self.checkpointer.load(resume_from_checkpoint)

    def _train(self) -> None:
        logger.info("Starting training")

        parallel_backend = self.state.parallel_backend
        train_state = self.state.train_state
        device = parallel_backend.device
        dtype = self.args.transformer_dtype

        memory_statistics = utils.get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        global_batch_size = self.args.batch_size * parallel_backend._dp_degree
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "train steps": self.args.train_steps,
            "per-replica batch size": self.args.batch_size,
            "global batch size": global_batch_size,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        progress_bar = tqdm(
            range(0, self.args.train_steps),
            initial=train_state.step,
            desc="Training steps",
            disable=not parallel_backend.is_local_main_process,
        )

        generator = torch.Generator(device=device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        scheduler_sigmas = utils.get_scheduler_sigmas(self.scheduler)
        scheduler_sigmas = (
            scheduler_sigmas.to(device=device, dtype=torch.float32) if scheduler_sigmas is not None else None
        )
        scheduler_alphas = utils.get_scheduler_alphas(self.scheduler)
        scheduler_alphas = (
            scheduler_alphas.to(device=device, dtype=torch.float32) if scheduler_alphas is not None else None
        )
        # timesteps_buffer = []

        self.transformer.train()
        if self.args.dpo:
            self.transformer_ref.eval()
        data_iterator = iter(self.dataloader)

        compute_posterior = False if self.args.enable_precomputation else (not self.args.precomputation_once)
        preprocessor = data.initialize_preprocessor(
            rank=parallel_backend.rank,
            world_size=parallel_backend.world_size,
            num_items=self.args.precomputation_items if self.args.enable_precomputation else 1,
            processor_fn={
                "condition": self.model_specification.prepare_conditions,
                "latent": functools.partial(
                    self.model_specification.prepare_latents, compute_posterior=compute_posterior
                ),
            },
            save_dir=self.args.precomputation_dir,
            enable_precomputation=self.args.enable_precomputation,
            enable_reuse=self.args.precomputation_reuse,
        )
        condition_iterator: Iterable[Dict[str, Any]] = None
        latent_iterator: Iterable[Dict[str, Any]] = None
        sampler = data.ResolutionSampler(
            batch_size=self.args.batch_size, dim_keys=self.model_specification._resolution_dim_keys
        )
        requires_gradient_step = True
        accumulated_loss = 0.0
        accumulated_loss_wref = 0.0

        while (
            train_state.step < self.args.train_steps and train_state.observed_data_samples < self.args.max_data_samples
        ):
            if train_state.step == 0 and (self.args.training_type in [TrainingType.VIDEO_AS_PROMPT_MOT]):
                save_train_data_sample = next(data_iterator)

                video = save_train_data_sample['video']
                if 'ref_videos' in save_train_data_sample:
                    ref_videos = save_train_data_sample['ref_videos']
                
                def normalize_tensor(tensor):
                    if tensor.min() < 0:
                        tensor = (tensor + 1) / 2
                    return tensor
                
                norm_video = normalize_tensor(video).clamp(0, 1)
                if 'ref_videos' in save_train_data_sample:
                    norm_ref_videos = [normalize_tensor(ref_video).clamp(0, 1) for ref_video in ref_videos]

                norm_video = norm_video.squeeze(0)       # [T, 3, H, W]
                if 'ref_videos' in save_train_data_sample:
                    norm_ref_videos = [norm_ref_video.squeeze(0) for norm_ref_video in norm_ref_videos]  # [T, 3, H, W]
                
                if 'ref_videos' in save_train_data_sample:
                    if norm_video.dim() == 4:
                        concat_video = torch.cat([*norm_ref_videos, norm_video], dim=3)  # [T, 3, H, 2*W]
                    elif norm_video.dim() == 5:
                        concat_video = torch.cat([*norm_ref_videos, norm_video[0], norm_video[1]], dim=3)  # [T, 3, H, 3*W]
                else:
                    concat_video = norm_video
                
                concat_video = concat_video.permute(0, 2, 3, 1).cpu().numpy()

                logger.debug(f"concat_video: {concat_video.shape}, {concat_video.max()}, {concat_video.min()}, {concat_video.dtype}")

                filename = "0-training_sample.mp4"
                output_path = os.path.join(self.args.output_dir, filename)


                # export_to_video(concat_video, output_path, fps=16)
                # hdfs adaption
                tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.mp4")
                export_to_video(concat_video, tmp_path, fps=16)
                shutil.move(tmp_path, output_path)
                
                

                if isinstance(save_train_data_sample['caption'][0], tuple):
                    caption = "Win Video Caption:\n" + save_train_data_sample['caption'][0][0] + "\n\nLose Video Caption:\n" + save_train_data_sample['caption'][1][0] + "\n\n\n"
                else:
                    caption = "Target Video Caption:\n" + "\n".join(save_train_data_sample['caption']) + "\n\n\n"
                if type(save_train_data_sample['caption_mot_ref'][0]) == str:
                    caption += "Ref Video Captions:\n" + "\n".join(save_train_data_sample['caption_mot_ref'])
                elif type(save_train_data_sample['caption_mot_ref'][0]) in [list, tuple]:
                    for caption_mot_ref_item_idx, caption_mot_ref_item in enumerate(save_train_data_sample['caption_mot_ref']):
                        caption += f"\n\nRef Video Captions-{caption_mot_ref_item_idx+1}:\n" + caption_mot_ref_item[0]
                else:
                    raise ValueError(f"type(save_train_data_sample['caption_mot_ref'][0]): {type(save_train_data_sample['caption_mot_ref'][0])}")
                with open(os.path.join(self.args.output_dir, filename.replace(".mp4", ".txt")), "w") as f:
                    f.write(caption)
            # 1. Load & preprocess data if required
            if preprocessor.requires_data:
                condition_iterator, latent_iterator = self._prepare_data(preprocessor, data_iterator)

            # 2. Prepare batch
            with self.tracker.timed("timing/batch_preparation"):
                try:
                    condition_item = next(condition_iterator)
                    latent_item = next(latent_iterator)
                    sampler.consume(condition_item, latent_item)
                except StopIteration:
                    if requires_gradient_step:
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        requires_gradient_step = False
                    logger.info("Data exhausted. Exiting training loop.")
                    break

                if sampler.is_ready:
                    condition_batch, latent_batch = sampler.get_batch()
                    condition_model_conditions = self.model_specification.collate_conditions(condition_batch)
                    latent_model_conditions = self.model_specification.collate_latents(latent_batch)
                else:
                    continue

            train_state.step += 1
            train_state.observed_data_samples += self.args.batch_size * parallel_backend._dp_degree

            logger.debug(f"Starting training step ({train_state.step}/{self.args.train_steps})")

            latent_model_conditions = utils.align_device_and_dtype(latent_model_conditions, device, dtype)
            condition_model_conditions = utils.align_device_and_dtype(condition_model_conditions, device, dtype)
            latent_model_conditions = utils.make_contiguous(latent_model_conditions)
            condition_model_conditions = utils.make_contiguous(condition_model_conditions)

            # 3. Forward pass
            sigmas = utils.prepare_sigmas(
                scheduler=self.scheduler,
                sigmas=scheduler_sigmas,
                batch_size=self.args.batch_size,
                num_train_timesteps=self.scheduler.config.num_train_timesteps,
                flow_weighting_scheme=self.args.flow_weighting_scheme,
                flow_logit_mean=self.args.flow_logit_mean,
                flow_logit_std=self.args.flow_logit_std,
                flow_mode_scale=self.args.flow_mode_scale,
                device=device,
                generator=self.state.generator,
            )
            sigmas = utils.expand_tensor_dims(sigmas, latent_model_conditions["latents"].ndim)

            if self.args.reference_train_mode in ["reference_independent"] and self.args.training_type in [TrainingType.VIDEO_AS_PROMPT_MOT]:
                ref_sigmas_list = []
                num_mot_ref = len(latent_model_conditions["latents_mot_ref"][0])
                for num_mot_ref_idx in range(num_mot_ref):
                    # logger.warning(f"scheduler_sigmas: {self.args.num_ref_diffusion_time_sigmas}/{len(scheduler_sigmas)} - {scheduler_sigmas[-self.args.num_ref_diffusion_time_sigmas:]}")
                    ref_sigmas_i = utils.prepare_sigmas(
                        scheduler=self.scheduler,
                        sigmas=scheduler_sigmas[-self.args.num_ref_diffusion_time_sigmas:],
                        batch_size=self.args.batch_size,
                        num_train_timesteps=self.args.num_ref_diffusion_time_sigmas,
                        flow_weighting_scheme=self.args.flow_weighting_scheme,
                        flow_logit_mean=self.args.flow_logit_mean,
                        flow_logit_std=self.args.flow_logit_std,
                        flow_mode_scale=self.args.flow_mode_scale,
                        device=device,
                        generator=self.state.generator,
                    )
                    ref_sigmas_i = utils.expand_tensor_dims(ref_sigmas_i, latent_model_conditions["latents"].ndim)
                    ref_sigmas_list.append(ref_sigmas_i)

            elif self.args.reference_train_mode is None and self.args.training_type in [TrainingType.VIDEO_AS_PROMPT_MOT]:
                ref_sigmas_list = []
                num_mot_ref = len(latent_model_conditions["latents_mot_ref"][0])
                for num_mot_ref_idx in range(num_mot_ref):
                    ref_sigmas_i = utils.prepare_sigmas(
                        scheduler=self.scheduler,
                        sigmas=scheduler_sigmas[-1:],
                        batch_size=self.args.batch_size,
                        num_train_timesteps=1,
                        flow_weighting_scheme=self.args.flow_weighting_scheme,
                        flow_logit_mean=self.args.flow_logit_mean,
                        flow_logit_std=self.args.flow_logit_std,
                        flow_mode_scale=self.args.flow_mode_scale,
                        device=device,
                        generator=self.state.generator,
                    )
                    ref_sigmas_i = utils.expand_tensor_dims(ref_sigmas_i, latent_model_conditions["latents"].ndim)
                    ref_sigmas_list.append(ref_sigmas_i)                
            else:
                raise ValueError(f"Unknown reference_train_mode: {self.args.reference_train_mode}")

            # NOTE: for planned refactor, make sure that forward and backward pass run under the context.
            # If only forward runs under context, backward will most likely fail when using activation checkpointing
            with self.attention_provider_ctx(training=True):
                if self.args.reference_train_mode not in ["reference_independent"]:
                    with self.tracker.timed("timing/forward"):
                        if "dpo_weight" in latent_model_conditions:
                            batch_dpo_weight = latent_model_conditions.pop("dpo_weight")
                        else:
                            batch_dpo_weight = 1.0
                        pred, target, sigmas = self.model_specification.forward(
                            transformer=self.transformer,
                            scheduler=self.scheduler,
                            condition_model_conditions=condition_model_conditions,
                            latent_model_conditions=latent_model_conditions,
                            sigmas=sigmas,
                            compute_posterior=compute_posterior,
                            sigmas_mot_ref_list=ref_sigmas_list,
                            ref_type=self.args.ref_type,
                            ablation_single_branch=self.args.ablation_single_branch,
                            baseline_single_condition=self.args.baseline_single_condition,
                        )

                    timesteps = (sigmas * 1000.0).long()
                    weights = utils.prepare_loss_weights(
                        scheduler=self.scheduler,
                        alphas=scheduler_alphas[timesteps] if scheduler_alphas is not None else None,
                        sigmas=sigmas,
                        flow_weighting_scheme=self.args.flow_weighting_scheme,
                    )
                    weights = utils.expand_tensor_dims(weights, pred.ndim)

                    # 4. Compute loss & backward pass
                    with self.tracker.timed("timing/backward"):
                        tmp_loss = weights.float() * (pred.float() - target.float()).pow(2)
                        # Average loss across all but batch dimension (for per-batch debugging in case needed)
                        tmp_loss = tmp_loss.mean(list(range(1, tmp_loss.ndim)))
                        
                        if not self.args.dpo:
                            # Average loss across batch dimension
                            loss = tmp_loss.mean()
                            if self.args.gradient_accumulation_steps > 1:
                                loss = loss / self.args.gradient_accumulation_steps
                            loss.backward()
                        else:
                            pass
                
                else:
                    with self.tracker.timed("timing/forward"):
                        pred, target, sigmas, pred_mot_ref_list, target_mot_ref_list, sigmas_mot_ref_list = self.model_specification.forward(
                            transformer=self.transformer,
                            scheduler=self.scheduler,
                            condition_model_conditions=condition_model_conditions,
                            latent_model_conditions=latent_model_conditions,
                            sigmas=sigmas,
                            compute_posterior=compute_posterior,
                            sigmas_mot_ref_list=ref_sigmas_list,
                            reference_train_mode=self.args.reference_train_mode,
                            random_refer_noise=self.args.random_refer_noise,
                            ref_type=self.args.ref_type,
                        )

                    timesteps = (sigmas * 1000.0).long()
                    weights = utils.prepare_loss_weights(
                        scheduler=self.scheduler,
                        alphas=scheduler_alphas[timesteps] if scheduler_alphas is not None else None,
                        sigmas=sigmas,
                        flow_weighting_scheme=self.args.flow_weighting_scheme,
                    )
                    weights = utils.expand_tensor_dims(weights, pred.ndim)

                    weights_mot_ref_list = []
                    for pred_mot_ref, target_mot_ref, sigmas_mot_ref in zip(pred_mot_ref_list, target_mot_ref_list, sigmas_mot_ref_list):
                        timesteps_mot_ref = (sigmas_mot_ref * 1000.0).long()
                        weights_mot_ref = utils.prepare_loss_weights(
                            scheduler=self.scheduler,
                            alphas=scheduler_alphas[timesteps_mot_ref] if scheduler_alphas is not None else None,
                            sigmas=sigmas_mot_ref,
                            flow_weighting_scheme=self.args.flow_weighting_scheme,
                        )
                        weights_mot_ref = utils.expand_tensor_dims(weights_mot_ref, pred_mot_ref.ndim)
                        weights_mot_ref_list.append(weights_mot_ref)
                    
                    # 4. Compute loss & backward pass
                    with self.tracker.timed("timing/backward"):
                        loss = weights.float() * (pred.float() - target.float()).pow(2)
                        # Average loss across all but batch dimension (for per-batch debugging in case needed)
                        loss = loss.mean(list(range(1, loss.ndim)))
                        # Average loss across batch dimension
                        loss = loss.mean()
                        for pred_mot_ref, target_mot_ref, weights_mot_ref in zip(pred_mot_ref_list, target_mot_ref_list, weights_mot_ref_list):
                            loss_mot_ref = weights_mot_ref.float() * (pred_mot_ref.float() - target_mot_ref.float()).pow(2)
                            # Average loss across all but batch dimension (for per-batch debugging in case needed)
                            loss_mot_ref = loss_mot_ref.mean(list(range(1, loss_mot_ref.ndim)))
                            # Average loss across batch dimension
                            loss_mot_ref = loss_mot_ref.mean() * 0.6 / self.args.sample_ref_videos_num
                            loss += loss_mot_ref
                            accumulated_loss_wref += loss_mot_ref.detach().item()
                        
                        if self.args.gradient_accumulation_steps > 1:
                            loss = loss / self.args.gradient_accumulation_steps
                        loss.backward()

            if self.args.dpo:
                loss_w, loss_l = tmp_loss.chunk(2)
                model_diff = loss_w - loss_l
                # logger.warning(f"loss: {loss.shape}")
                # logger.warning(f"loss_w: {loss_w.shape}; loss_l: {loss_l.shape}")
                # logger.warning(f"model_diff: {model_diff.shape}")
                # logger.warning(f'latent_model_conditions: {latent_model_conditions.keys()}')
                # logger.warning(f'condition_model_conditions: {condition_model_conditions.keys()}')
                with torch.no_grad():
                    pred_ref, target_ref, sigmas_ref = self.model_specification.forward(
                        transformer=self.transformer_ref,
                        scheduler=self.scheduler,
                        condition_model_conditions=condition_model_conditions,
                        latent_model_conditions=latent_model_conditions,
                        sigmas=sigmas,
                        compute_posterior=compute_posterior,
                        sigmas_mot_ref_list=ref_sigmas_list,
                        ref_type=self.args.ref_type,
                    )

                    timesteps = (sigmas_ref * 1000.0).long()
                    weights = utils.prepare_loss_weights(
                        scheduler=self.scheduler,
                        alphas=scheduler_alphas[timesteps] if scheduler_alphas is not None else None,
                        sigmas=sigmas_ref,
                        flow_weighting_scheme=self.args.flow_weighting_scheme,
                    )
                    weights = utils.expand_tensor_dims(weights, pred_ref.ndim)

                    tmp_loss_ref = weights.float() * (pred_ref.float() - target_ref.float()).pow(2)
                    # Average loss across all but batch dimension (for per-batch debugging in case needed)
                    tmp_loss_ref = tmp_loss_ref.mean(list(range(1, tmp_loss_ref.ndim)))
                    
                    loss_w_ref, loss_l_ref = tmp_loss_ref.chunk(2)
                    ref_diff = loss_w_ref - loss_l_ref


                scale_term = -0.5 * 500
                inside_term = scale_term * (model_diff - ref_diff)
                # implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
                loss = -1 * F.logsigmoid(inside_term).mean()

                loss *= batch_dpo_weight.item()

                logger.info(f'loss: {loss}, model_diff: {model_diff.item()}, ref_diff: {ref_diff.item()}, loss/batch_dpo_weight: {loss / batch_dpo_weight.item()}')
                
                dpo_reweight = 1.0
                loss *= dpo_reweight

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                loss.backward()
            accumulated_loss += loss.detach().item()
            requires_gradient_step = True

            # 5. Clip gradients
            model_parts = [self.transformer]
            grad_norm = utils.torch._clip_grad_norm_while_handling_failing_dtensor_cases(
                [p for m in model_parts for p in m.parameters()],
                self.args.max_grad_norm,
                foreach=True,
                pp_mesh=parallel_backend.get_mesh()["pp"] if parallel_backend.pipeline_parallel_enabled else None,
            )

            # 6. Step optimizer & log metrics
            logs = {}

            if train_state.step % self.args.gradient_accumulation_steps == 0:
                # TODO(aryan): revisit no_sync() for FSDP
                with self.tracker.timed("timing/optimizer_step"):
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                if grad_norm is not None:
                    grad_norm = grad_norm if isinstance(grad_norm, float) else grad_norm.detach().item()
                if (
                    parallel_backend.data_replication_enabled
                    or parallel_backend.data_sharding_enabled
                    or parallel_backend.context_parallel_enabled
                ):
                    dp_cp_mesh = parallel_backend.get_mesh()["dp_cp"]
                    if grad_norm is not None:
                        grad_norm = parallel.dist_mean(torch.tensor([grad_norm], device=device), dp_cp_mesh)
                    global_avg_loss, global_max_loss = (
                        parallel.dist_mean(torch.tensor([accumulated_loss], device=device), dp_cp_mesh),
                        parallel.dist_max(torch.tensor([accumulated_loss], device=device), dp_cp_mesh),
                    )
                    if self.args.reference_train_mode in ["reference_independent"]:
                        global_avg_loss_wref, global_max_loss_wref = (
                            parallel.dist_mean(torch.tensor([accumulated_loss_wref], device=device), dp_cp_mesh),
                            parallel.dist_max(torch.tensor([accumulated_loss_wref], device=device), dp_cp_mesh),
                        )
                    else:
                        global_avg_loss_wref = global_max_loss_wref = 0
                else:
                    global_avg_loss = global_max_loss = accumulated_loss
                    global_avg_loss_wref = global_max_loss_wref = 0

                logs["train/global_avg_loss"] = global_avg_loss
                # logs["train/global_avg_loss_wref"] = global_avg_loss_wref
                logs["train/global_max_loss"] = global_max_loss
                # logs["train/global_max_loss_wref"] = global_max_loss_wref

                if grad_norm is not None:
                    logs["train/grad_norm"] = grad_norm
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)
                accumulated_loss = 0.0
                accumulated_loss_wref = 0.0
                requires_gradient_step = False

            progress_bar.update(1)
            progress_bar.set_postfix(logs)

            # timesteps_buffer.extend([(train_state.step, t) for t in timesteps.detach().cpu().numpy().tolist()])

            if train_state.step % self.args.logging_steps == 0:
                # TODO(aryan): handle non-SchedulerWrapper schedulers (probably not required eventually) since they might not be dicts
                # TODO(aryan): causes NCCL hang for some reason. look into later
                # logs.update(self.lr_scheduler.get_last_lr())

                # timesteps_table = wandb.Table(data=timesteps_buffer, columns=["step", "timesteps"])
                # logs["timesteps"] = wandb.plot.scatter(
                #     timesteps_table, "step", "timesteps", title="Timesteps distribution"
                # )
                # timesteps_buffer = []

                logs["train/observed_data_samples"] = train_state.observed_data_samples
                logs["train/lr"] = self.lr_scheduler.get_last_lr()['lr_0'][0]

                parallel_backend.log(logs, step=train_state.step)
                train_state.log_steps.append(train_state.step)

            # 7. Save checkpoint if required
            with self.tracker.timed("timing/checkpoint"):
                self.checkpointer.save(
                    step=train_state.step, _device=device, _is_main_process=parallel_backend.is_main_process
                )

            # 8. Perform validation if required
            if train_state.step % self.args.validation_steps == 0 or train_state.step == 1:
                self._validate(step=train_state.step, final_validation=False)

        # 9. Final checkpoint, validation & cleanup
        self.checkpointer.save(
            train_state.step, force=True, _device=device, _is_main_process=parallel_backend.is_main_process
        )
        parallel_backend.wait_for_everyone()
        self._validate(step=train_state.step, final_validation=True)

        self._delete_components()
        memory_statistics = utils.get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        # 10. Upload artifacts to hub
        if parallel_backend.is_main_process and self.args.push_to_hub:
            upload_folder(
                repo_id=self.state.repo_id,
                folder_path=self.args.output_dir,
                ignore_patterns=[f"{self.checkpointer._prefix}_*"],
            )

        parallel_backend.destroy()

    def _validate(self, step: int, final_validation: bool = False) -> None:
        if self.args.validation_dataset_file is None:
            return

        logger.info("Starting validation")

        # 1. Load validation dataset
        parallel_backend = self.state.parallel_backend
        if self.args.training_type in [TrainingType.LORA, TrainingType.FULL_FINETUNE]:
            dataset = data.ValidationDataset(self.args.validation_dataset_file)
        elif self.args.training_type in [TrainingType.VIDEO_AS_PROMPT_MOT]:
            dataset = data.VideoAsPromptValidationDataset(
                filename=self.args.validation_dataset_file,
                baseline_single_condition=self.args.baseline_single_condition,
            )
        else:
            raise ValueError(f"Training type {self.args.training_type} is not supported")

        # Hack to make accelerate work. TODO(aryan): refactor
        if parallel_backend._dp_degree > 1:
            dp_mesh = parallel_backend.get_mesh()["dp"]
            dp_local_rank, dp_world_size = dp_mesh.get_local_rank(), dp_mesh.size()
            dataset._data = datasets.distributed.split_dataset_by_node(dataset._data, dp_local_rank, dp_world_size)
        else:
            dp_mesh = None
            dp_local_rank, dp_world_size = parallel_backend.local_rank, 1

        validation_dataloader = data.DPDataLoader(
            dp_local_rank,
            dataset,
            batch_size=1,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=lambda items: items,
        )
        data_iterator = iter(validation_dataloader)
        main_process_prompts_to_filenames = {}  # Used to save model card
        all_processes_artifacts = []  # Used to gather artifacts from all processes

        memory_statistics = utils.get_memory_statistics()
        logger.info(f"Memory before validation start: {json.dumps(memory_statistics, indent=4)}")

        seed = self.args.seed if self.args.seed is not None else 0
        generator = torch.Generator(device=parallel_backend.device).manual_seed(seed)
        pipeline = self._init_pipeline(final_validation=final_validation)

        # 2. Run validation
        # TODO(aryan): when running validation with FSDP, if the number of data points is not divisible by dp_shards, we
        # will hang indefinitely. Either pad the dataset or raise an error early on during initialization if the dataset
        # size is not divisible by dp_shards.
        self.transformer.eval()
        num_validation_steps = 0
        while True:
            if num_validation_steps >= 1:
                break
            else:
                num_validation_steps += 1
            validation_data = next(data_iterator, None)
            if validation_data is None:
                break

            validation_data = validation_data[0]

            if self.args.training_type == TrainingType.VIDEO_AS_PROMPT_MOT:
                with open(self.args.validation_dataset_file, "r") as file:
                    data_config = json.load(file)
                    frames_selection = data_config.get("frames_selection", "evenly")
                    num_inference_steps = data_config.get("num_inference_steps", 50)
                
                validation_data["frames_selection"] = frames_selection
                validation_data["caption_guidance_scale"] = self.args.caption_guidance_scale
                validation_data["reference_train_mode"] = self.args.reference_train_mode
                validation_data["random_refer_noise"] = self.args.random_refer_noise
                validation_data["num_inference_steps"] = num_inference_steps
                validation_data["ref_type"] = self.args.ref_type
                validation_data["ablation_single_branch"] = self.args.ablation_single_branch
                validation_data["baseline_single_condition"] = self.args.baseline_single_condition
            
            with self.attention_provider_ctx(training=False):
                validation_artifacts = self.model_specification.validation(
                    pipeline=pipeline, generator=generator, **validation_data
                )

            if dp_local_rank != 0:
                continue

            PROMPT = validation_data["prompt"]
            IMAGE = validation_data.get("image", None)
            VIDEO = validation_data.get("video", None)
            EXPORT_FPS = validation_data.get("export_fps", 16)

            # 2.1. If there are any initial images or videos, they will be logged to keep track of them as
            # conditioning for generation.
            prompt_filename = utils.string_to_filename(PROMPT)[:25]
            if "gt_ref_videos" in validation_data and "gt_video" in validation_data:
                prompt_filename = f"{validation_data['gt_ref_videos'].replace('/', '#')}++{validation_data['gt_video'].replace('/', '#')}"
            artifacts = {
                "input_image": data.ImageArtifact(value=IMAGE),
                "input_video": data.VideoArtifact(value=VIDEO),
            }

            # 2.2. Track the artifacts generated from validation
            for i, validation_artifact in enumerate(validation_artifacts):
                if validation_artifact.value is None:
                    continue
                artifacts.update({f"artifact_{i}": validation_artifact})

            # 2.3. Save the artifacts to the output directory and create appropriate logging objects
            # TODO(aryan): Currently, we only support WandB so we've hardcoded it here. Needs to be revisited.
            for index, (key, artifact) in enumerate(list(artifacts.items())):
                assert isinstance(artifact, (data.ImageArtifact, data.VideoArtifact))
                if artifact.value is None:
                    continue
                
                if key == "input_video":
                    continue

                time_, rank, ext = int(time.time()), parallel_backend.rank, artifact.file_extension
                filename = "validation-" if not final_validation else "final-"
                filename += f"{step}-{rank}-{index}-{prompt_filename}-{time_}.{ext}"
                output_filename = os.path.join(self.args.output_dir, filename)

                if parallel_backend.is_main_process and ext in ["mp4", "jpg", "jpeg", "png"]:
                    main_process_prompts_to_filenames[PROMPT] = filename

                if isinstance(artifact, data.ImageArtifact):
                    artifact.value.save(output_filename)
                    all_processes_artifacts.append(wandb.Image(output_filename, caption=PROMPT))
                elif isinstance(artifact, data.VideoArtifact):
                    # export_to_video(artifact.value, output_filename, fps=EXPORT_FPS)
                    # hdfs adaption
                    # 1) 先写到本地临时盘（/tmp）
                    tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.mp4")
                    export_to_video(artifact.value, tmp_path, fps=16)
                    # 2) 再移动到最终目录（跨盘会自动走拷贝+删源）
                    shutil.move(tmp_path, output_filename)
                    if "hdfs" not in output_filename:
                        all_processes_artifacts.append(wandb.Video(output_filename, caption=PROMPT))

        # 3. Cleanup & log artifacts
        parallel_backend.wait_for_everyone()

        memory_statistics = utils.get_memory_statistics()
        logger.info(f"Memory after validation end: {json.dumps(memory_statistics, indent=4)}")

        # Remove all hooks that might have been added during pipeline initialization to the models
        pipeline.remove_all_hooks()
        del pipeline
        module_names = ["text_encoder", "text_encoder_2", "text_encoder_3", "image_encoder", "image_processor", "vae"]
        if self.args.enable_precomputation:
            self._delete_components(module_names)
        torch.cuda.reset_peak_memory_stats(parallel_backend.device)

        # Gather artifacts from all processes. We also need to flatten them since each process returns a list of artifacts.
        all_artifacts = [None] * dp_world_size
        if dp_world_size > 1:
            torch.distributed.all_gather_object(all_artifacts, all_processes_artifacts)
            # all_artifacts = [all_processes_artifacts]
        else:
            all_artifacts = [all_processes_artifacts]
        all_artifacts = [artifact for artifacts in all_artifacts for artifact in artifacts]

        if parallel_backend.is_main_process and "hdfs" not in output_filename:
            tracker_key = "final" if final_validation else "validation"
            artifact_log_dict = {}

            image_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Image)]
            if len(image_artifacts) > 0:
                artifact_log_dict["images"] = image_artifacts
            video_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Video)]
            if len(video_artifacts) > 0:
                artifact_log_dict["videos"] = video_artifacts
            parallel_backend.log({tracker_key: artifact_log_dict}, step=step)

            if self.args.push_to_hub and final_validation:
                video_filenames = list(main_process_prompts_to_filenames.values())
                prompts = list(main_process_prompts_to_filenames.keys())
                utils.save_model_card(
                    args=self.args, repo_id=self.state.repo_id, videos=video_filenames, validation_prompts=prompts
                )

        parallel_backend.wait_for_everyone()
        if not final_validation:
            self._move_components_to_device()
            self.transformer.train()
            if self.args.dpo:
                self.transformer_ref.eval()

    def _evaluate(self) -> None:
        raise NotImplementedError("Evaluation has not been implemented yet.")

    def _init_directories_and_repositories(self) -> None:
        if self.state.parallel_backend.is_main_process:
            self.args.output_dir = Path(self.args.output_dir)
            self.args.output_dir.mkdir(parents=True, exist_ok=True)
            self.state.output_dir = Path(self.args.output_dir)

            if self.args.push_to_hub:
                repo_id = self.args.hub_model_id or Path(self.args.output_dir).name
                self.state.repo_id = create_repo(token=self.args.hub_token, repo_id=repo_id, exist_ok=True).repo_id

    def _move_components_to_device(
        self, components: Optional[List[torch.nn.Module]] = None, device: Optional[Union[str, torch.device]] = None
    ) -> None:
        if device is None:
            device = self.state.parallel_backend.device
        if components is None:
            components = [
                self.text_encoder,
                self.text_encoder_2,
                self.text_encoder_3,
                self.image_encoder,
                self.transformer,
                self.transformer_ref,
                self.vae,
            ]
        components = utils.get_non_null_items(components)
        components = list(filter(lambda x: hasattr(x, "to"), components))
        for component in components:
            component.to(device)

    def _set_components(self, components: Dict[str, Any]) -> None:
        for component_name in self._all_component_names:
            existing_component = getattr(self, component_name, None)
            new_component = components.get(component_name, existing_component)
            setattr(self, component_name, new_component)

    def _delete_components(self, component_names: Optional[List[str]] = None) -> None:
        if component_names is None:
            component_names = self._all_component_names
        for component_name in component_names:
            setattr(self, component_name, None)
        utils.free_memory()
        utils.synchronize_device()

    def _init_pipeline(self, final_validation: bool = False) -> DiffusionPipeline:
        module_names = ["text_encoder", "text_encoder_2", "text_encoder_3", "image_encoder", "transformer", "vae"]

        if not final_validation:
            module_names.remove("transformer")
            pipeline = self.model_specification.load_pipeline(
                tokenizer=self.tokenizer,
                tokenizer_2=self.tokenizer_2,
                tokenizer_3=self.tokenizer_3,
                text_encoder=self.text_encoder,
                text_encoder_2=self.text_encoder_2,
                text_encoder_3=self.text_encoder_3,
                image_encoder=self.image_encoder,
                image_processor=self.image_processor,
                # TODO(aryan): handle unwrapping for compiled modules
                # transformer=utils.unwrap_model(accelerator, self.transformer),
                transformer=self.transformer,
                vae=self.vae,
                enable_slicing=self.args.enable_slicing,
                enable_tiling=self.args.enable_tiling,
                enable_model_cpu_offload=self.args.enable_model_cpu_offload,
                training=True,
            )
        else:
            self._delete_components()

            # Load the transformer weights from the final checkpoint if performing full-finetune
            transformer = None
            if self.args.training_type == TrainingType.FULL_FINETUNE:
                transformer = self.model_specification.load_diffusion_models()["transformer"]

            pipeline = self.model_specification.load_pipeline(
                transformer=transformer,
                enable_slicing=self.args.enable_slicing,
                enable_tiling=self.args.enable_tiling,
                enable_model_cpu_offload=self.args.enable_model_cpu_offload,
                training=False,
            )

            # Load the LoRA weights if performing LoRA finetuning
            if self.args.training_type == TrainingType.LORA:
                pipeline.load_lora_weights(
                    os.path.join(self.args.output_dir, "lora_weights", f"{self.state.train_state.step:06d}")
                )

        components = {module_name: getattr(pipeline, module_name, None) for module_name in module_names}
        self._set_components(components)
        if not self.args.enable_model_cpu_offload:
            self._move_components_to_device(list(components.values()))
        self._maybe_torch_compile()
        return pipeline

    def _prepare_data(
        self,
        preprocessor: Union[data.InMemoryDistributedDataPreprocessor, data.PrecomputedDistributedDataPreprocessor],
        data_iterator,
    ):
        if not self.args.enable_precomputation:
            if not self._are_condition_models_loaded:
                logger.info(
                    "Precomputation disabled. Loading in-memory data loaders. All components will be loaded on GPUs."
                )
                condition_components = self.model_specification.load_condition_models()
                latent_components = self.model_specification.load_latent_models()
                all_components = {**condition_components, **latent_components}
                self._set_components(all_components)
                self._move_components_to_device(list(all_components.values()))
                utils._enable_vae_memory_optimizations(self.vae, self.args.enable_slicing, self.args.enable_tiling)
                self._maybe_torch_compile()
            else:
                condition_components = {k: v for k in self._condition_component_names if (v := getattr(self, k, None))}
                latent_components = {k: v for k in self._latent_component_names if (v := getattr(self, k, None))}

            condition_iterator = preprocessor.consume(
                "condition",
                components=condition_components,
                data_iterator=data_iterator,
                generator=self.state.generator,
                cache_samples=True,
            )
            latent_iterator = preprocessor.consume(
                "latent",
                components=latent_components,
                data_iterator=data_iterator,
                generator=self.state.generator,
                use_cached_samples=True,
                drop_samples=True,
            )

            self._are_condition_models_loaded = True
        else:
            logger.info("Precomputed condition & latent data exhausted. Loading & preprocessing new data.")

            parallel_backend = self.state.parallel_backend
            if parallel_backend.world_size == 1:
                self._move_components_to_device([self.transformer], "cpu")
                utils.free_memory()
                utils.synchronize_device()
                torch.cuda.reset_peak_memory_stats(parallel_backend.device)

            consume_fn = preprocessor.consume_once if self.args.precomputation_once else preprocessor.consume

            # Prepare condition iterators
            condition_components, component_names, component_modules = {}, [], []
            if not self.args.precomputation_reuse:
                condition_components = self.model_specification.load_condition_models()
                component_names = list(condition_components.keys())
                component_modules = list(condition_components.values())
                self._set_components(condition_components)
                self._move_components_to_device(component_modules)
                self._maybe_torch_compile()
            condition_iterator = consume_fn(
                "condition",
                components=condition_components,
                data_iterator=data_iterator,
                generator=self.state.generator,
                cache_samples=True,
            )
            self._delete_components(component_names)
            del condition_components, component_names, component_modules

            # Prepare latent iterators
            latent_components, component_names, component_modules = {}, [], []
            if not self.args.precomputation_reuse:
                latent_components = self.model_specification.load_latent_models()
                utils._enable_vae_memory_optimizations(self.vae, self.args.enable_slicing, self.args.enable_tiling)
                component_names = list(latent_components.keys())
                component_modules = list(latent_components.values())
                self._set_components(latent_components)
                self._move_components_to_device(component_modules)
                self._maybe_torch_compile()
            latent_iterator = consume_fn(
                "latent",
                components=latent_components,
                data_iterator=data_iterator,
                generator=self.state.generator,
                use_cached_samples=True,
                drop_samples=True,
            )
            self._delete_components(component_names)
            del latent_components, component_names, component_modules

            if parallel_backend.world_size == 1:
                self._move_components_to_device([self.transformer])

        return condition_iterator, latent_iterator

    def _maybe_torch_compile(self):
        for model_name, compile_scope in zip(self.args.compile_modules, self.args.compile_scopes):
            model = getattr(self, model_name, None)
            if model is not None:
                logger.info(f"Applying torch.compile to '{model_name}' with scope '{compile_scope}'.")
                compiled_model = utils.apply_compile(model, compile_scope)
                setattr(self, model_name, compiled_model)

    def _get_training_info(self) -> Dict[str, Any]:
        info = self.args.to_dict()

        # Removing flow matching arguments when not using flow-matching objective
        diffusion_args = info.get("diffusion_arguments", {})
        scheduler_name = self.scheduler.__class__.__name__ if self.scheduler is not None else ""
        if scheduler_name != "FlowMatchEulerDiscreteScheduler":
            filtered_diffusion_args = {k: v for k, v in diffusion_args.items() if "flow" not in k}
        else:
            filtered_diffusion_args = diffusion_args

        info.update({"diffusion_arguments": filtered_diffusion_args})
        return info

    # fmt: off
    _all_component_names = ["tokenizer", "tokenizer_2", "tokenizer_3", "text_encoder", "text_encoder_2", "text_encoder_3", "image_encoder", "image_processor", "transformer", "unet", "vae", "scheduler",
    # dpo
    "transformer_ref"]
    _condition_component_names = ["tokenizer", "tokenizer_2", "tokenizer_3", "text_encoder", "text_encoder_2", "text_encoder_3"]
    _latent_component_names = ["image_encoder", "image_processor", "vae"]
    _diffusion_component_names = ["transformer", "unet", "scheduler"]
    # fmt: on
