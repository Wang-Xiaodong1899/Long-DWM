import os
import sys
sys.path.append("/home/user/wangxd/diffusers/src")

import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXTransformer3DModel,
)

from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video_inject_fbf import CogVideoXImageToVideoPipeline


from diffusers.utils import load_image, export_to_video
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

from safetensors.torch import load_file

pretrained_model_name_or_path = "/home/user/wangxd/diffusers/CogVideoX-2b"

tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer", revision=None, 
        torch_dtype=torch.float16,
    )

text_encoder = T5EncoderModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder", revision=None,
    torch_dtype=torch.float16,
)

transformer = CogVideoXTransformer3DModel.from_pretrained(
        # "/data/wangxd/ckpt/cogvideox-A4-clean-image-inject-f13-fps1-0102-fbf-noaug-cond4/checkpoint-1000/",
        "/data/wangxd/ckpt/cogvideox-A4-clean-image-inject-f13-fps1-0108-fbf-noaug-cond8-inject8-fft/checkpoint-500",
        subfolder="transformer",
        # "/root/autodl-fs/CogVidx-2b-I2V-base-transfomer",
        torch_dtype=torch.float16,
        revision=None,
        variant=None,
        in_channels=32, low_cpu_mem_usage=False, ignore_mismatched_sizes=True
    )

vae = AutoencoderKLCogVideoX.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", revision=None, variant=None,
        torch_dtype=torch.float16,
    )

scheduler = CogVideoXDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler",)

components = {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }

pipe = CogVideoXImageToVideoPipeline(**components, inject=True)
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

print('Pipeline loaded!')

rollout = input("rollout: ")
rollout = int(rollout)

given_images = input("Given Images: ")
given_images = int(given_images)

actions = ["turn left", "turn left sharply", "turn right", "turn right sharply"]

global_prompt = "Cloudy. Daytime. The road is a two-lane street with a yellow dividing line, surrounded by sidewalks, buildings, and trees. There are utility poles and power lines running parallel to the road. A bus stop shelter, and various signs and banners attached to the fences along the sidewalk."

while True:
    # image_path = input("image_path: ")
    image_dir = input("image dir: ")
    image_paths = list(os.listdir(image_dir))
    image_paths = [a for a in image_paths if a.endswith(".png") or a.endswith(".jpg")]
    image_paths.sort()
    
    images = [load_image(os.path.join(image_dir, im)) for im in image_paths]
    images = images[:given_images]
    
    validation_prompt = input("prompt: ")
    guidance_scale = input("cfg: ") # 6
    total_frames = []
    for r in range(rollout):
        pipeline_args = {
            "image": images if r==0 else frames[-given_images:],
            "prompt": validation_prompt,
            "guidance_scale": int(guidance_scale),
            "use_dynamic_cfg": True,
            "height": 480,
            "width": 720,
            "num_frames": 13,
        }
        frames = pipe(**pipeline_args).frames[0]
        total_frames.extend(frames if r==(rollout-1) else frames[:-given_images])
    name_prefix = validation_prompt.replace(" ", "_").strip()[:40]
    export_to_video(total_frames, os.path.join(image_dir, f"{name_prefix}_cfg_{guidance_scale}_inject_fbf_roll_{rollout}_multiple_inject{given_images}_500.mp4"), fps=8)