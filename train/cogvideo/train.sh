# /home/user/wuzhirong/Projects/diffusers/wzr_distill_gan/sft_inject_key_fbf_fps10_distill_gan_fps1.py \

# --validation_prompt "go straight. Overcast. Daytime. Urban, with tall buildings on both sides of the street. Vehicles, traffic lights, pedestrian crossings, and road markings. ::: wait. Overcast. Daytime. Urban, with tall buildings on both sides of the street. Vehicles, traffic lights, pedestrian crossings, and road markings. ::: slow down. Overcast. Daytime. Urban, with tall buildings on both sides of the street. Vehicles, traffic lights, pedestrian crossings, and road markings. " \
# --validation_images /data/wuzhirong/datasets/Nuscenes/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151671612404.jpg \

# --validation_prompt "go straight. Overcast. Daytime. Urban, with tall buildings on both sides of the street. Vehicles, traffic lights, pedestrian crossings, and road markings. " \
# --validation_images /data/wuzhirong/datasets/Nuscenes/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151671612404.jpg \

# --validation_prompt "go straight. Sunny. Daytime. Urban, with buildings on both sides of the street, traffic lights at intersections, and a clear sky. Buildings, traffic lights, vehicles, road markings. " \
# --validation_images /data/wangxd/val_fps1/scene-0521/n008-2018-08-30-10-33-52-0400__CAM_FRONT__1535639678662404.jpg \

# --validation_prompt "go straight. Sunny. Daytime. Urban street with trees, buildings, and a clear road. A black SUV, a white delivery truck with advertisements on its side, and traffic lights. " \
# --validation_images /data/wangxd/val_fps1/scene-0560/n008-2018-08-31-11-37-23-0400__CAM_FRONT__1535730420912404.jpg \

WANDB__SERVICE_WAIT=300 accelerate launch \
    --main_process_port 25340 \
    --config_file /home/user/wangxd/diffusers/examples/cogvideo/debug.yaml \
    /home/user/wangxd/diffusers/examples/cogvideo/sft_inject_key_fbf_fps10_distill_gan_fps1.py \
        --mixed_precision=bf16 --pretrained_model_name_or_path /home/user/wangxd/diffusers/CogVideoX-2b \
        --instance_data_root /data/wangxd/nuscenes/ \
        --dataset_name nuscenes \
        --validation_prompt "go straight. Overcast. Daytime. Urban, with tall buildings on both sides of the street. Vehicles, traffic lights, pedestrian crossings, and road markings.::: slow down. Overcast. Daytime. Urban, with tall buildings on both sides of the street. Vehicles, traffic lights, pedestrian crossings, and road markings. " \
        --validation_images /home/user/wangxd/diffusers/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151671612404.jpg \
        --validation_prompt_separator ::: --num_validation_videos 1 \
        --validation_epochs 1000 \
        --seed 42 \
        --tracker_name cogvideox-A4-clean-image-distill-gan \
        --output_dir /data/wangxd/ckpt/cogvideox-A4-clean-image-input-fft-distill-gan-0110 \
        --height 480 --width 720 --fps 8 \
        --max_num_frames 145 \
        --train_batch_size 1 \
        --num_train_epochs 500 \
        --checkpointing_steps 100 \
        --gradient_accumulation_steps 2 \
        --learning_rate 1e-6 --lr_scheduler cosine_with_restarts --lr_warmup_steps 10 --lr_num_cycles 1 \
        --optimizer Adam --adam_beta1 0.9 --adam_beta2 0.95 --max_grad_norm 1.0 --allow_tf32 --report_to wandb \
        --gradient_checkpointing \
        --denoised_image \
        --generator_update_ratio 500000 \
        --distill_loss_weight 1.0 \
        --gen_cls_loss_weight 0.1 \
        --diffusion_loss_weight 1.0 \
        --gan_loss_weight 0.1 \
        --explicit_distill_loss_weight 0.0 \
        # --enable_slicing --enable_tiling \