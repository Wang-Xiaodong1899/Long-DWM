import os
import sys
sys.path.append("/root/autodl-tmp/diffusers/src")

import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.utils import export_to_video
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

from safetensors.torch import load_file
transformer_patch_embed_proj = load_file(os.path.join("/root/autodl-fs/ckpt/cogvideo/cogvideox-lora/checkpoint-100","transformer_patch_embed_proj.safetensors"))

tokenizer = AutoTokenizer.from_pretrained(
        "/root/autodl-fs/models/CogVideoX-2b", subfolder="tokenizer", revision=None, 
    )

text_encoder = T5EncoderModel.from_pretrained(
    "/root/autodl-fs/models/CogVideoX-2b", subfolder="text_encoder", revision=None, 
)

transformer = CogVideoXTransformer3DModel.from_pretrained(
        "/root/autodl-fs/models/CogVideoX-2b",
        subfolder="transformer",
        # "/root/autodl-fs/CogVidx-2b-I2V-base-transfomer",
        torch_dtype=torch.float16,
        revision=None,
        variant=None,
        in_channels=32, low_cpu_mem_usage=False, ignore_mismatched_sizes=True
    )
transformer.patch_embed.proj.weight.data = transformer_patch_embed_proj['transformer.patch_embed.proj.weight']
transformer.patch_embed.proj.bias.data = transformer_patch_embed_proj['transformer.patch_embed.proj.bias']

vae = AutoencoderKLCogVideoX.from_pretrained(
        "/root/autodl-fs/models/CogVideoX-2b", subfolder="vae", revision=None, variant=None, 
    )

scheduler = CogVideoXDPMScheduler.from_pretrained("/root/autodl-fs/models/CogVideoX-2b", subfolder="scheduler",)

components = {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }

pipe = CogVideoXImageToVideoPipeline(**components)
# pipe.load_lora_weights("/path/to/lora/weights", adapter_name="cogvideox-lora") # Or,
pipe.load_lora_weights("my-awesome-hf-username/my-awesome-lora-name", adapter_name="cogvideox-lora") # If loading from the HF Hub
pipe.to("cuda")

# Assuming lora_alpha=32 and rank=64 for training. If different, set accordingly
pipe.set_adapters(["cogvideox-lora"], [32 / 64])

prompt = (
    "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The "
    "panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other "
    "pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, "
    "casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. "
    "The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical "
    "atmosphere of this unique musical performance"
)
frames = pipe(prompt, guidance_scale=6, use_dynamic_cfg=True).frames[0]
export_to_video(frames, "output.mp4", fps=8)