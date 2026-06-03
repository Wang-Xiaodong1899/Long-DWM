import os
import sys
sys.path.append("/home/user/wangxd/diffusers/src")


from tqdm import tqdm

from diffusers.utils import load_image, export_to_video

import fire
from PIL import Image

def main(
    val_s=0,
    val_e=10,
):
    tgt_dir = "/data/wangxd/ckpt/cogvideox-fbf-rec"
    
    tgt_dir = os.path.join(tgt_dir, f"s{val_s}-e{val_e}")
    
    for i in tqdm(range(400)):
        first_img_path = os.path.join(tgt_dir, f"{i}_0.png")
        if not os.path.exists(first_img_path):
            continue
        
        total_frames = []
        for id in range(25):
            total_frames.append(Image.open(os.path.join(tgt_dir, f"{i}_{id}.png")))
        
        export_to_video(total_frames, os.path.join(tgt_dir, f"{i}_rec.mp4"), fps=8)

if __name__ == '__main__':
    fire.Fire(main)
