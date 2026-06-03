import os
import sys
sys.path.append("/home/user/wangxd/diffusers/src")

import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
)

from diffusers.models.transformers.cogvideox_transformer_2d import CogVideoXTransformer2DModel
# from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from diffusers.pipelines.cogvideo.pipeline_cogvideox_clip2image import CogVideoXImageToVideoPipeline


from diffusers.utils import load_image, export_to_video
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer, CLIPModel, CLIPImageProcessor


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

transformer = CogVideoXTransformer2DModel.from_pretrained(
        "/data/wangxd/ckpt/cogvideox-2d-clip2image-continue/checkpoint-5000",
        subfolder="transformer",
        torch_dtype=torch.float16,
        revision=None,
        variant=None,
        in_channels=16, low_cpu_mem_usage=False, ignore_mismatched_sizes=True
    )

vae = AutoencoderKLCogVideoX.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", revision=None, variant=None,
        torch_dtype=torch.float16,
    )

scheduler = CogVideoXDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler",)

clip_model = CLIPModel.from_pretrained(
            "/data/wangxd/models/CLIP-ViT-H-14-laion2B-s32B-b79K", torch_dtype=torch.float16)

feature_extractor = CLIPImageProcessor.from_pretrained('/home/user/wangxd/SVD/smodels/image-keyframes-s448-ep100-resumefrom50', subfolder="feature_extractor")

components = {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "feature_extractor": feature_extractor,
            "clip_model": clip_model
        }

pipe = CogVideoXImageToVideoPipeline(**components)
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

print('Pipeline loaded!')

while True:
    image_path = input("image_path: ")
    validation_prompt = input("prompt: ")
    guidance_scale = input("cfg: ") # 6
    pipeline_args = {
        "image": load_image(image_path),
        "prompt": validation_prompt,
        "guidance_scale": int(guidance_scale),
        "use_dynamic_cfg": True,
        "height": 480,
        "width": 720,
        "num_frames": 33
    }
    name_prefix = validation_prompt.replace(" ", "_").strip()[:40]
    frames = pipe(**pipeline_args).frames[0]
    frames[0].save("2d_img_5k.jpg")