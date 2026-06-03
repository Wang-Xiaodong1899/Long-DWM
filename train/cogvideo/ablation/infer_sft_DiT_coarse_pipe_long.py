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

from nuscenes_dataset_for_cogvidx import NuscenesDatasetAllframesFPS10OneByOneForValidate

from tqdm import tqdm

from diffusers.utils import load_image, export_to_video
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

from safetensors.torch import load_file
import fire

import cv2
import argparse

def extract_frames(mp4_path, output_dir, fps):
    """
    Extract frames from an MP4 file at a specified FPS and save them to an output directory.

    Args:
        mp4_path (str): Path to the MP4 file.
        output_dir (str): Directory where extracted frames will be saved.
        fps (int): Frames per second to extract.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    video = cv2.VideoCapture(mp4_path)
    if not video.isOpened():
        print(f"Error: Unable to open video file {mp4_path}")
        return

    # Get the original video's frame rate
    video_fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Save every nth frame based on the desired FPS
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    video.release()
    # print(f"Extracted {saved_count} frames to {output_dir}")

def main(val_s: int=0, val_e: int=10, rollout: int=1):

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
            "/data/wangxd/ckpt/cogvideox-A4-clean-image-inject-f13-fps1-1219-fbf-noaug/checkpoint-6000", # coarse
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

    train_dataset = NuscenesDatasetAllframesFPS10OneByOneForValidate(
            data_root="/data/wangxd/nuscenes/",
            height=480,
            width=720,
            max_num_frames=2,
            encode_video=None,
            encode_prompt=None,
        )
    
    root_dir = "/data/wangxd/IJCAI25/Ablation/DiT_coarse"
    
    os.makedirs(root_dir, exist_ok=True)

    for i in tqdm(range(val_s, val_e)): # each scene
        item = train_dataset[i] # total samples in a scene
        # if have 5 samples, cur_item_nums = 5
        key_indexs = range(5)
        for key_idx in tqdm(key_indexs):
            key_frame = item[key_idx]
            pil_videos = key_frame["instance_video"]
            validation_prompt = key_frame["instance_prompt"]
            
            first_frame_path = pil_videos[0]
            
            # original: interpolation: 0-1, 1-2 -> 25帧
            # interpolation-f3-
            tgt_dir = os.path.join(root_dir, f"{i}") # scene
            
            os.makedirs(tgt_dir, exist_ok=True)
            
            prefix = (
                validation_prompt[:25]
                .replace(" ", "_")
                .replace(" ", "_")
                .replace("'", "_")
                .replace('"', "_")
                .replace("/", "_")
            )
            
            guidance_scale = 6
            
            total_frames = []
            for ridx in tqdm(range(rollout)):
                pipeline_args = {
                    "image": load_image(first_frame_path) if ridx==0 else total_frames[-1],
                    "prompt": validation_prompt,
                    "guidance_scale": int(guidance_scale),
                    "use_dynamic_cfg": True,
                    "height": 480,
                    "width": 720,
                    "num_frames": 13
                }
                frames = pipe(**pipeline_args).frames[0]
                total_frames.extend(frames)
            export_to_video(total_frames, os.path.join(tgt_dir, f"{key_idx:04d}.mp4"), fps=8)
            

if __name__ == "__main__":
    fire.Fire(main)