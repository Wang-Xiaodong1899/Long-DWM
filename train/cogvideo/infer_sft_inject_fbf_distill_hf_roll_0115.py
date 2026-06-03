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
        # "/data/wangxd/ckpt/cogvideox-A4-clean-image-distill-explicit-L2-0105/checkpoint-50",
        # "/data/wangxd/ckpt/cogvideox-A4-clean-image-fft-ckpt-distill-explicit-L2-0111/checkpoint-100",
        "/data/wangxd/ckpt/cogvideox-A4-clean-image-fft-ckpt-distill-explicit-L2-hf-loss-0111/checkpoint-100",
        subfolder="transformer",
        torch_dtype=torch.float16,
        revision=None,
        variant=None,
        in_channels=32, low_cpu_mem_usage=False, ignore_mismatched_sizes=True
    )
transformer.requires_grad_(True)
transformer.train()

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

seed = 42

while True:
    image_path = input("image_path: ")
    validation_prompt = input("prompt: ")
    guidance_scale = 6 # input("cfg: ") # 6
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
        generator = torch.Generator(device="cuda").manual_seed(seed)
        frames = pipe(**pipeline_args, generator=generator).frames[0]
        total_frames.extend(frames if r==(rollout-1) else frames[:-1])
    name_prefix = validation_prompt.replace(" ", "_").strip()[:40]
    dir = os.path.dirname(image_path)
    export_to_video(total_frames, os.path.join(dir, f"{name_prefix}_cfg_{guidance_scale}_inject_fbf_roll_{rollout}_fft-ckpt-distill-explicit-L2-hf-loss-0111-100-seed-{seed}.mp4"), fps=8)

# image_path = "/home/user/wangxd/diffusers/val_fps1/scene-0332/n008-2018-08-22-15-53-49-0400__CAM_FRONT__1534968257412404.jpg"
# validation_prompt = "Go straight. The sky is partly cloudy with some clouds scattered across the blue expanse. It appears to be daytime, as indicated by the natural light and shadows present in the video. The road is a two-lane street with yellow dividing lines. On both sides of the street, there are parked cars, buildings, and trees lining the sidewalks. The critical objects include the parked cars, buildings, trees, and the yellow dividing lines on the road."

# image_path = "/home/user/wangxd/diffusers/val_fps1/scene-0035/n015-2018-07-24-10-42-41+0800__CAM_FRONT__1532400189162460.jpg"
# validation_prompt = "Go straight. Sunny. Daytime. Urban area with trees, buildings, and a clear road. Trees, buildings, road markings, vehicles, and traffic signs."

# image_path = "/home/user/wangxd/diffusers/val_fps1/scene-0104/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151671612404.jpg"
# validation_prompt = "go straight. Overcast. Daytime. Urban, with tall buildings on both sides of the street. Vehicles, traffic lights, pedestrian crossings, and road markings. "
