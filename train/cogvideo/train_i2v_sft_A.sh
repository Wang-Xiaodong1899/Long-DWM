#!/bin/bash

accelerate launch --multi-gpu --config_file /workspace/wxd/diffusers/examples/cogvideo/dp.yaml sft.py \
  --mixed_precision="fp16" \
  --pretrained_model_name_or_path /volsparse2/wxd/models/CogVideoX-2b \
  --cache_dir /volsparse2/wxd/cache \
  --instance_data_root /volsparse1/wxd/data/nuscenes \
  --dataset_name nuscenes \
  --validation_prompt "turn left. Overcast. Daytime. A curving road with black and white bollards on the sides, surrounded by greenery and a few buildings. The road, bollards, trees, and buildings. :::wait.  Overcast. Daytime. A curving road with black and white bollards on the sides, surrounded by greenery and a few buildings. The road, bollards, trees, and buildings. :::turn right.  Overcast. Daytime. A curving road with black and white bollards on the sides, surrounded by greenery and a few buildings. The road, bollards, trees, and buildings.:::go straight.  Overcast. Daytime. A curving road with black and white bollards on the sides, surrounded by greenery and a few buildings. The road, bollards, trees, and buildings.:::slow down.  Overcast. Daytime. A curving road with black and white bollards on the sides, surrounded by greenery and a few buildings. The road, bollards, trees, and buildings." \
  --validation_images "/workspace/wxd/diffusers/examples/cogvideo/n015-2018-07-18-11-41-49+0800__CAM_FRONT__1531885320012463.jpg" \
  --validation_prompt_separator ::: \
  --num_validation_videos 1 \
  --validation_epochs 5 \
  --seed 42 \
  --tracker_name cogvideo-A4-clean-image-sft \
  --output_dir /volsparse2/wxd/ckpt/cogvideo/cogvideox-A4-clean-image-sft-bs2 \
  --height 480 --width 720 --fps 8 --max_num_frames 33 \
  --train_batch_size 2 \
  --num_train_epochs 1000 \
  --checkpointing_steps 500 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 100 \
  --lr_num_cycles 1 \
  --enable_slicing \
  --enable_tiling \
  --optimizer Adam \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --allow_tf32 \
  --report_to wandb \
  --gradient_checkpointing \
  --denoised_image \
  # --dataloader_num_workers 16 \