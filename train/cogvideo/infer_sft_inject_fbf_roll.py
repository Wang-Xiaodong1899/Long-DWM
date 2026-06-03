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
        "/data/wangxd/ckpt/cogvideox-A4-clean-image-inject-f13-fps1-0109-fbf-fft/checkpoint-1200", # coarse fft
        # "/data/wangxd/ckpt/cogvideox-A4-clean-image-inject-f13-fps1-1219-fbf-noaug/checkpoint-6000", # coarse
        # "/data/wangxd/ckpt/cogvideox-A4-clean-image-inject-f13-fps10-1222-fbf-noaug/checkpoint-5000", # fine
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

actions = ["turn left", "turn left sharply", "turn right", "turn right sharply"]

global_prompt = "Cloudy. Daytime. The road is a two-lane street with a yellow dividing line, surrounded by sidewalks, buildings, and trees. There are utility poles and power lines running parallel to the road. A bus stop shelter, and various signs and banners attached to the fences along the sidewalk."


# a SUV
# Go straight. Sunny. Daytime. a SUV. trees.

while True:
    image_path = input("image_path: ")
    validation_prompt = input("prompt: ")
    guidance_scale = input("cfg: ") # 6
    total_frames = []
    for r in range(rollout):
        pipeline_args = {
            "image": load_image(image_path) if r==0 else frames[-1],
            "prompt": validation_prompt,
            "guidance_scale": int(guidance_scale),
            "use_dynamic_cfg": True,
            "height": 480,
            "width": 720,
            "num_frames": 13
        }
        frames = pipe(**pipeline_args).frames[0]
        total_frames.extend(frames if r==(rollout-1) else frames[:-1])
    name_prefix = validation_prompt.replace(" ", "_").strip()[:40]
    dir = os.path.dirname(image_path)
    # export_to_video(total_frames, f"{name_prefix}_cfg_{guidance_scale}_inject_fbf_roll_{rollout}_noaug_2k.mp4", fps=8)
    # export_to_video(total_frames, os.path.join(dir, f"{name_prefix}_cfg_{guidance_scale}_inject_fbf_roll_{rollout}_fps1-1219-fbf-6k.mp4"), fps=8)
    export_to_video(total_frames, os.path.join(dir, f"{name_prefix}_cfg_{guidance_scale}_inject_fbf_roll_{rollout}_fps1-0110-fbf-fft-1200.mp4"), fps=8)

# image_path = "/home/user/wangxd/diffusers/val_fps1/scene-0104/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151671612404.jpg"
# validation_prompt = "go straight. Overcast. Daytime. Urban, with tall buildings on both sides of the street. Vehicles, traffic lights, pedestrian crossings, and road markings"
