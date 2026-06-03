import os
import sys
sys.path.append("/home/user/wangxd/diffusers/src")

import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
)

from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video_inject_fbf import CogVideoXImageToVideoPipeline


from diffusers.utils import load_image, export_to_video
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

from safetensors.torch import load_file

pretrained_model_name_or_path = "/data/wuzhirong/hf-models/CogVideoX-2b"

tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer", revision=None, 
        # torch_dtype=torch.float16,
    )

text_encoder = T5EncoderModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder", revision=None,
    # torch_dtype=torch.float16,
).to(torch.bfloat16)

transformer = CogVideoXTransformer3DModel.from_pretrained(
        # "/data/wangxd/ckpt/cogvideox-A4-clean-image-distill-gan-explicit-L2-0106-2/checkpoint-200/fake_unet",
        # "/data/wangxd/ckpt/cogvideox-A4-clean-image-distill-gan-0110/checkpoint-100/fake_unet",
        "/data/wangxd/ckpt/cogvideox-A4-clean-image-distill-gan-explicit-loss-0111/checkpoint-200/fake_unet",
        # "/data/wangxd/ckpt/cogvideox-A4-clean-image-distill-gan-explicit-loss-0110/checkpoint-100/fake_unet",
        torch_dtype=torch.bfloat16,
        # revision=None,
        # variant=None,
        # in_channels=32, low_cpu_mem_usage=False, ignore_mismatched_sizes=True
    )
# transformer.train()
# transformer.enable_gradient_checkpointing()

vae = AutoencoderKLCogVideoX.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", revision=None, variant=None,
        # torch_dtype=torch.float16,
    ).to(torch.bfloat16)

scheduler = CogVideoXDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler",)

components = {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }

pipe = CogVideoXImageToVideoPipeline(**components, inject=True)

# pipe.enable_model_cpu_offload()
# pipe.vae.enable_slicing()
# pipe.vae.enable_tiling()
pipe.to("cuda")

print('Pipeline loaded!')


# image_path = "/home/user/wangxd/diffusers/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151671612404.jpg"
# validation_prompt = "go straight. Overcast. Daytime. Urban, with tall buildings on both sides of the street. Vehicles, traffic lights, pedestrian crossings, and road markings"


# image_path = "/home/user/wangxd/diffusers/val_fps1/scene-0802/n015-2018-10-02-10-50-40+0800__CAM_FRONT__1538448887162460.jpg"
# image_path = "/home/user/wangxd/diffusers/val_fps1/scene-0035/n015-2018-07-24-10-42-41+0800__CAM_FRONT__1532400189162460.jpg"


# image_path = "/home/user/wangxd/diffusers/val_fps1/scene-0332/n008-2018-08-22-15-53-49-0400__CAM_FRONT__1534968257412404.jpg"
# validation_prompt = "Go straight. The sky is partly cloudy with some clouds scattered across the blue expanse. It appears to be daytime, as indicated by the natural light and shadows present in the video. The road is a two-lane street with yellow dividing lines. On both sides of the street, there are parked cars, buildings, and trees lining the sidewalks. The critical objects include the parked cars, buildings, trees, and the yellow dividing lines on the road."


save_dir = "./infer_results/gan-explicit-loss-0111-ckpt_200"
os.makedirs(save_dir, exist_ok=True)

# while True:
# image_path = input("Image path: ")
# validation_prompt = input("Prompt: ")
# guidance_scale = input("cfg: ") # 6

# validation_prompt = "go straight. Sunny. Daytime. Urban street with trees, buildings, and a clear road. A black SUV, a white delivery truck with advertisements on its side, and traffic lights. " 
# image_path = "/data/wangxd/val_fps1/scene-0560/n008-2018-08-31-11-37-23-0400__CAM_FRONT__1535730420912404.jpg"


image_path = "/home/user/wangxd/diffusers/val_fps1/scene-0104/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151671612404.jpg"
validation_prompt = "go straight. Overcast. Daytime. Urban, with tall buildings on both sides of the street. Vehicles, traffic lights, pedestrian crossings, and road markings. "


# validation_prompt = "Go straigt. Sunny. Daytime. The road is a two-lane street with a white van driving ahead. There are trees on both sides of the road, and buildings can be seen in the distance. The white van, trees, and buildings."

# validation_prompt = "Go straight. Sunny. Daytime. The road is lined with trees and buildings, featuring a crosswalk, traffic signs, and a fire hydrant. There are parked cars on the side of the road. Crosswalk, traffic signs, fire hydrant, parked cars, trees, buildings."

# image_path = "/data/wuzhirong/datasets/Nuscenes/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151671612404.jpg"
guidance_scale = 6

rollouts = 1


pipeline_args = {
    "image": load_image(image_path),
    "prompt": validation_prompt,
    "guidance_scale": int(guidance_scale),
    "use_dynamic_cfg": False,
    "height": 480,
    "width": 720,
    "num_frames": 13,
}

seed = 42

all_frames = []

for rollouts_idx in range(rollouts):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    frames = pipe(**pipeline_args, generator=generator).frames[0]
    if rollouts_idx == 0:
        all_frames += frames
    else:
        all_frames += frames[1:]
    pipeline_args['image'] = frames[-1]

# frames = pipe(**pipeline_args).frames[0]

name_prefix = validation_prompt.replace(" ", "_").strip()[:40]
name_end = image_path.replace("/", "_").strip()
export_to_video(all_frames, os.path.join(save_dir, f"seed_{seed}_roll_{rollouts}_{name_prefix}_0111-gan-explicit-loss-200_cfg_{guidance_scale}_{name_end}.mp4"), fps=8)