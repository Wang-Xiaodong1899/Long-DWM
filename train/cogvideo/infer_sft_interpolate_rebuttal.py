import os
import sys
sys.path.append("/home/user/wangxd/diffusers/src")

import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
)

from diffusers.models.transformers.cogvideox_transformer_3d_interpolate import CogVideoXTransformer3DModel
# from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video_interpolate import CogVideoXImageToVideoPipeline


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
        "/data/wangxd/ckpt/cogvideox-A4-clean-image-inject-fps10-f13-1202-inherit1022/checkpoint-1000",
        subfolder="transformer",
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

# dirs = ["validation_792_left", "validation_795_go_straight", "validation_793_wait"]
# prompts = ["turn left", "go straight", "wait"]


while True:
    # for idx in range(len(dirs)):
    image_dir = input("image dir: ")
    # image_dir = dirs[idx]
    
    # image_paths = list(os.listdir(image_dir))
    # image_paths = [a for a in image_paths if a.endswith(".png") or a.endswith(".jpg")]
    # image_paths.sort()
    
    image_paths = ['frame_00000.png', 'frame_00001.png', 'frame_00002.png', 'frame_00003.png', 'frame_00004.png', 'frame_00005.png', 'frame_00006.png', 'frame_00007.png', 'frame_00008.png', 'frame_00009.png', 'frame_00010.png', 'frame_00011.png', 'frame_00012.png', 'frame_00013.png', 'frame_00014.png', 'frame_00015.png', 'frame_00016.png', 'frame_00017.png', 'frame_00018.png', 'frame_00019.png', 'frame_00020.png', 'frame_00021.png', 'frame_00022.png', 'frame_00023.png', 'frame_00024.png']
    
    validation_prompt = input("prompt: ")
    # validation_prompt = prompts[idx]
    guidance_scale = 6
    
    # guidance_scale = input("cfg: ") # 6
    
    total_frames = []
    for idx in range(len(image_paths)-1):
        image_path = image_paths[idx]
        last_image_path = image_paths[idx+1]
        
        print(f"interpolation ...  {image_path} {last_image_path}")
        
        pipeline_args = {
            "image": load_image(os.path.join(image_dir, image_path)),
            "last_image": load_image(os.path.join(image_dir, last_image_path)),
            "prompt": validation_prompt,
            "guidance_scale": int(guidance_scale),
            "use_dynamic_cfg": True,
            "height": 480,
            "width": 720,
            "num_frames": 13
        }
        frames = pipe(**pipeline_args).frames[0]
        # import pdb; pdb.set_trace()
        print(len(frames))
        if idx == 0:
            total_frames.extend(frames[3:-1])
        elif idx == (len(image_paths)-2):
            total_frames.extend(frames[3:])
        else:
            total_frames.extend(frames[4:])
    name_prefix = validation_prompt.replace(" ", "_").strip()[:40]
    print(len(total_frames))
    export_to_video(total_frames, os.path.join(image_dir, f"{name_prefix}_cfg_{guidance_scale}_interpolate_test_1k.mp4"), fps=8)