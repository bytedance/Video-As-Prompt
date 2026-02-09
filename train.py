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


import sys
import traceback

from finetrainers import BaseArgs, ControlTrainer, SFTTrainer, TrainingType, get_logger
from finetrainers.config import _get_model_specifiction_cls
from finetrainers.trainer.control_trainer.config import ControlFullRankConfig, ControlLowRankConfig
from finetrainers.trainer.sft_trainer.config import SFTFullRankConfig, SFTLowRankConfig, VideoAsPromptMOTConfig


logger = get_logger()


def main():
    try:
        import multiprocessing

        multiprocessing.set_start_method("fork")
    except Exception as e:
        logger.error(
            f'Failed to set multiprocessing start method to "fork". This can lead to poor performance, high memory usage, or crashes. '
            f"See: https://pytorch.org/docs/stable/notes/multiprocessing.html\n"
            f"Error: {e}"
        )

    try:
        args = BaseArgs()

        argv = [y.strip() for x in sys.argv for y in x.split()]
        try:
            training_type_index = argv.index("--training_type")
        except ValueError:
            raise ValueError("Training type not provided in command line arguments. Use --training_type <type>")

        if training_type_index + 1 >= len(argv):
            raise ValueError("Training type value is missing after --training_type argument.")

        training_type = argv[training_type_index + 1]
        training_cls = None
        if training_type == TrainingType.LORA:
            training_cls = SFTLowRankConfig
        elif training_type == TrainingType.FULL_FINETUNE:
            training_cls = SFTFullRankConfig
        elif training_type == TrainingType.CONTROL_LORA:
            training_cls = ControlLowRankConfig
        elif training_type == TrainingType.CONTROL_FULL_FINETUNE:
            training_cls = ControlFullRankConfig
        elif training_type == TrainingType.VIDEO_AS_PROMPT_MOT:
            training_cls = VideoAsPromptMOTConfig
        else:
            raise ValueError(f"Training type {training_type} not supported.")

        args.register_args(training_cls())
        args = args.parse_args()

        model_specification_cls = _get_model_specifiction_cls(args.model_name, args.training_type)
        model_specification = model_specification_cls(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            tokenizer_id=args.tokenizer_id,
            tokenizer_2_id=args.tokenizer_2_id,
            tokenizer_3_id=args.tokenizer_3_id,
            text_encoder_id=args.text_encoder_id,
            text_encoder_2_id=args.text_encoder_2_id,
            text_encoder_3_id=args.text_encoder_3_id,
            transformer_id=args.transformer_id,
            vae_id=args.vae_id,
            text_encoder_dtype=args.text_encoder_dtype,
            text_encoder_2_dtype=args.text_encoder_2_dtype,
            text_encoder_3_dtype=args.text_encoder_3_dtype,
            transformer_dtype=args.transformer_dtype,
            vae_dtype=args.vae_dtype,
            revision=args.revision,
            cache_dir=args.cache_dir,
            training_type=args.training_type,
        )

        if args.training_type in [TrainingType.LORA, TrainingType.FULL_FINETUNE, TrainingType.VIDEO_AS_PROMPT_MOT]:
            trainer = SFTTrainer(args, model_specification)
        elif args.training_type in [TrainingType.CONTROL_LORA, TrainingType.CONTROL_FULL_FINETUNE]:
            trainer = ControlTrainer(args, model_specification)
        else:
            raise ValueError(f"Training type {args.training_type} not supported.")

        trainer.run()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Exiting...")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
