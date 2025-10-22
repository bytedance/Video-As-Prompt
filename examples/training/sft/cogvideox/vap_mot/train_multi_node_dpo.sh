#!/bin/bash

set -e -x

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 MASTER_ADDR NODE_RANK"
    exit 1
fi

MASTER_ADDR=$1
NODE_RANK=$2
NNODES=6
MASTER_PORT=9998 

# export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
# export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG="ERROR"
export TORCH_NCCL_ENABLE_MONITORING=0
export FINETRAINERS_LOG_LEVEL="INFO"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Finetrainers supports multiple backends for distributed training. Select your favourite and benchmark the differences!
# BACKEND="accelerate"
BACKEND="ptd"

# In this setting, I'm using 2 GPUs on a 4-GPU node for training
NUM_GPUS=8
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# Check the JSON files for the expected JSON format
TRAINING_DATASET_CONFIG="examples/training/sft/cogvideox/vap_mot/training.json"
VALIDATION_DATASET_FILE="examples/training/sft/cogvideox/vap_mot/validation.json"

# Depending on how many GPUs you have available, choose your degree of parallelism and technique!
DDP_1="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 1 --dp_shards 1 --cp_degree 1 --tp_degree 1"
DDP_2="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 2 --dp_shards 1 --cp_degree 1 --tp_degree 1"
DDP_4="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 4 --dp_shards 1 --cp_degree 1 --tp_degree 1"
DDP_8="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 8 --dp_shards 1 --cp_degree 1 --tp_degree 1"
FSDP_2="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 1 --dp_shards 2 --cp_degree 1 --tp_degree 1"
FSDP_4="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 1 --dp_shards 4 --cp_degree 1 --tp_degree 1"
HSDP_2_2="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 2 --dp_shards 2 --cp_degree 1 --tp_degree 1"
HSDP_4_2="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 4 --dp_shards 2 --cp_degree 1 --tp_degree 1"
HSDP_8_2="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 8 --dp_shards 2 --cp_degree 1 --tp_degree 1"
HSDP_24_2="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 24 --dp_shards 2 --cp_degree 1 --tp_degree 1"

# Parallel arguments
parallel_cmd=(
  $HSDP_24_2
)

# Model arguments
model_cmd=(
  --model_name "cogvideox"
  --pretrained_model_name_or_path "ckpts/CogVideoX-5b-I2V"
)

# videoanimator arguments
video_animator_cmd=(
  --ref_videos_num 10
  --sample_ref_videos_num 1
  --mask_ref_ratio 0.0
  --mask_caption_ratio 0.02
  --training_dataset_kind "vap-data"
  --model_structure_config "examples/training/sft/cogvideox/vap_mot/config_ori.json"
  --dpo
)

# Dataset arguments
dataset_cmd=(
  --dataset_config $TRAINING_DATASET_CONFIG
)

# Dataloader arguments
dataloader_cmd=(
  --dataloader_num_workers 0
)

# Diffusion arguments
diffusion_cmd=(
  --flow_weighting_scheme "logit_normal"
)

# Training arguments
# We target just the attention projections layers for LoRA training here.
# You can modify as you please and target any layer (regex is supported)
training_cmd=(
  --training_type "video-as-prompt-mot"
  --seed 42
  --batch_size 1
  --train_steps 60000
  --rank 32
  --lora_alpha 32
  --target_modules "(transformer_blocks|single_transformer_blocks).*(to_q|to_k|to_v|to_out.0)"
  --gradient_accumulation_steps 1
  --gradient_checkpointing
  --checkpointing_steps 501
  --checkpointing_limit 1
  --enable_slicing
  --enable_tiling
)

# Optimizer arguments
optimizer_cmd=(
  --optimizer "adamw"
  --lr 5e-5
  --lr_scheduler "constant"
  --lr_num_cycles 1
  --beta1 0.9
  --beta2 0.99
  --weight_decay 1e-4
  --epsilon 1e-8
  --max_grad_norm 1.0
)

# Validation arguments
validation_cmd=(
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 100
)

# Miscellaneous arguments
miscellaneous_cmd=(
  --tracker_name "VideoAsPrompt"
  --output_dir "outputs/train_multi_node_dpo"
  --runs_name "train_multi_node_dpo"
  --init_timeout 3600
  --nccl_timeout 3600
  --report_to "wandb"
)

# Execute the training script
if [ "$BACKEND" == "ptd" ]; then

  export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
  
  torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NUM_GPUS \
    --node_rank=$NODE_RANK \
    --rdzv_backend c10d \
    --rdzv_id=my_job_123 \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    train.py \
      "${parallel_cmd[@]}" \
      "${model_cmd[@]}" \
      "${video_animator_cmd[@]}" \
      "${dataset_cmd[@]}" \
      "${dataloader_cmd[@]}" \
      "${diffusion_cmd[@]}" \
      "${training_cmd[@]}" \
      "${optimizer_cmd[@]}" \
      "${validation_cmd[@]}" \
      "${miscellaneous_cmd[@]}"
fi

echo -ne "-------------------- Finished executing script --------------------\n\n"
