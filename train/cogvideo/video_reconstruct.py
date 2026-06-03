import sys
sys.path.append("/home/user/wangxd/diffusers/src")

import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
import fire

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

import torch
from diffusers import AutoencoderKLCogVideoX

from diffusers.utils import load_image, export_to_video
from tqdm import tqdm

from diffusers.video_processor import VideoProcessor
video_processor = VideoProcessor(vae_scale_factor=8)

pretrained_model_name_or_path = "/data/wuzhirong/hf-models/CogVideoX-2b"


from nuscenes_dataset_for_cogvidx import NuscenesDatasetAllframesFPS10OneByOneForValidatePath


def main(
    eval_frames=25,
    val_s = 0,
    val_e = 10,
    device="cuda:0",
):
    vae = AutoencoderKLCogVideoX.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", revision=None, variant=None,
        torch_dtype=torch.float16,
    )
    vae.to(device)
    # vae.enable_slicing()
    # vae.enable_tiling()
    print('Pipeline loaded!')
    
    root_dir = "/data/wangxd/ckpt/cogvideox-fbf-rec/"
    os.makedirs(root_dir, exist_ok=True)
    real_dataset = NuscenesDatasetAllframesFPS10OneByOneForValidatePath(
        data_root="/data/wangxd/nuscenes/",
        height=480,
        width=720,
        max_num_frames=25,
        encode_video=None,
        encode_prompt=None,
    )
    
    val_array = [(val_s, val_e)]
    
    real_frames_paths = []
    for (val_s, val_e) in val_array:
        global_key_idx = 0
        tgt_dir = os.path.join(root_dir, f"s{val_s}-e{val_e}")
        os.makedirs(tgt_dir, exist_ok=True)
        for i in tqdm(range(val_s, val_e)):
            item = real_dataset[i]
            for key_idx in tqdm(range(len(item))):
                key_frame = item[key_idx]
                pil_videos_path = key_frame["instance_video"]
                cur_frames = pil_videos_path[:eval_frames]
                
                # read
                imgs = [load_image(cur_path) for cur_path in cur_frames]
                imgs = video_processor.preprocess(imgs, height=480, width=720).to(device=device,dtype=torch.float16) # [F,C,H,W]
                imgs = imgs.unsqueeze(0).permute(0, 2, 1 ,3,4) # [B, C, F, H, W]
                
                # reconstruction
                with torch.no_grad():
                    latents = []
                    for i in range(len(cur_frames)):
                        latent = vae.encode(imgs[:,:,i:i+1]).latent_dist.sample()
                        latents.append(latent)
                    latents = torch.cat(latents, dim=2)
                
                    rec_imgs = []
                    for i in range(len(cur_frames)):
                        rec_img = vae.decode(latents[:,:,i:i+1]).sample
                        rec_imgs.append(rec_img)
                    rec_imgs = torch.cat(rec_imgs,dim=2)
                
                # print(rec_imgs.shape)
                
                # save
                video = video_processor.postprocess_video(video=rec_imgs, output_type="pil")[0]
                
                for fidx, img in enumerate(video):
                    img.save(os.path.join(tgt_dir, f"{global_key_idx}_{fidx}.png"))
                
                global_key_idx = global_key_idx + 1
                
                # import pdb; pdb.set_trace()
        
if __name__ == "__main__":
    fire.Fire(main)