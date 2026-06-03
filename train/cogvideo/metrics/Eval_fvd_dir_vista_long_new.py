import sys
sys.path.append("/home/user/wuzhirong/WXD/diffusers/examples/cogvideo")


from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import json
import random
import torch
import transformers
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import copy
import os
from tqdm import tqdm
from einops import repeat
import fire
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection


# validate FVD on Nuscene val 150 samples
# default 8 frames
# default image size 256x256

# NOTE:
# to ensure accurate
# we load the video following the time steps

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from common import image2arr, pil2arr, mp4toarr, image2pil, json2data, preprocess_image


from fvd import load_fvd_model, compute_fvd, compute_fvd_1

from nuscenes_dataset_for_cogvidx import NuscenesDatasetAllframesFPS10OneByOneForValidate

from fvd_torch import FrechetVideoDistance

import json

NUSCENES_ROOT = "/data/wangxd/nuscenes/"


def main(
        root_dir= '/data/wuzhirong/exp/IJCAI25/long/vista',
        version = None,
        device="cuda:0",
        eval_frames=69,
        val_array = [
                    (0,10)
                     ,(10, 20), (20,30), (30, 40), (40, 50), (50,60), (60, 70),(70, 80), (80,90), (90, 100), (100, 110), (110, 120), (120, 130), (130, 140), (140, 150)
                     ],
):
    print(f'eval videos at {root_dir}')

    fvd_pipe = FrechetVideoDistance().to(device)

    fvd_pipe_0 = FrechetVideoDistance().to(device)
    fvd_pipe_1 = FrechetVideoDistance().to(device)
    fvd_pipe_2 = FrechetVideoDistance().to(device)

    batch_size = 48
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    import torchvision.transforms as TT
    resize_transform = TT.Compose([
        TT.Resize((480, 720)),
    ])

    key_indics = [0,1,2,3,4,5,9,13,17,21,25,29,33]

    video_tensor, video_tensor_0, video_tensor_1, video_tensor_2 = [],[],[],[]
    video_nums,video_nums_0,video_nums_1,video_nums_2=0,0,0,0

    for tgt_scene_idx in tqdm(range(150)):
        for key_idx in key_indics:
            video_arr = mp4toarr(os.path.join(root_dir, f"{tgt_scene_idx}/{key_idx:04d}.mp4"), resize=False, convert=True)
            pil_videos = []
            for im_arr in video_arr:
                pil_im = preprocess_image(Image.fromarray(im_arr))
                pil_videos.append(pil_im)
            videotensor = [transform(im) for im in pil_videos[:eval_frames]]
            videotensor = torch.stack(videotensor) # N C H W
            if videotensor.shape[0] >= 23:
                video_tensor_0.append(videotensor[:23]) 
                video_nums_0+=1
            if videotensor.shape[0] >= 46:
                video_tensor_1.append(videotensor[23:46])
                video_nums_1+=1 
            if videotensor.shape[0] == 69:
                video_tensor_2.append(videotensor[46:])
                video_nums_2+=1
                video_tensor.append(videotensor) 
                video_nums+=1

            if len(video_tensor) == batch_size:
                video_tensor = torch.stack(video_tensor) # B N C H W
                fvd_pipe.update(video_tensor.to(device), False)
                video_tensor = []

            if len(video_tensor_0) == batch_size:
                video_tensor_0 = torch.stack(video_tensor_0) # B N C H W
                fvd_pipe_0.update(video_tensor_0.to(device), False)
                video_tensor_0 = []

            if len(video_tensor_1) == batch_size:
                video_tensor_1 = torch.stack(video_tensor_1) # B N C H W
                fvd_pipe_1.update(video_tensor_1.to(device), False)
                video_tensor_1 = []

            if len(video_tensor_2) == batch_size:
                video_tensor_2 = torch.stack(video_tensor_2) # B N C H W
                fvd_pipe_2.update(video_tensor_2.to(device), False)
                video_tensor_2 = []

    if len(video_tensor) > 0:
        video_tensor = torch.stack(video_tensor) # B N C H W
        fvd_pipe.update(video_tensor.to(device), False)
        video_tensor = []

    if len(video_tensor_0) > 0:
        video_tensor_0 = torch.stack(video_tensor_0) # B N C H W
        fvd_pipe_0.update(video_tensor_0.to(device), False)
        video_tensor_0 = []

    if len(video_tensor_1) > 0:
        video_tensor_1 = torch.stack(video_tensor_1) # B N C H W
        fvd_pipe_1.update(video_tensor_1.to(device), False)
        video_tensor_1 = []

    if len(video_tensor_2) > 0:
        video_tensor_2 = torch.stack(video_tensor_2) # B N C H W
        fvd_pipe_2.update(video_tensor_2.to(device), False)
        video_tensor_2 = []
    
    print(f"sync video num: {video_nums},{video_nums_0},{video_nums_1},{video_nums_2}")

    video_tensor, video_tensor_0, video_tensor_1, video_tensor_2 = [],[],[],[]
    video_nums,video_nums_0,video_nums_1,video_nums_2=0,0,0,0

    for (val_s, val_e) in val_array:
        for i in tqdm(range(val_s, val_e)):
            for key_idx in key_indics:
                videotensor = torch.load(f"/data/wuzhirong/exp/IJCAI25/preload/ours/{i:04d}/{key_idx:04d}.pt",weights_only=True)
                videotensor = videotensor[:eval_frames]
                if videotensor.shape[0] >= 23:
                    video_tensor_0.append(videotensor[:23]) 
                    video_nums_0+=1
                if videotensor.shape[0] >= 46:
                    video_tensor_1.append(videotensor[23:46])
                    video_nums_1+=1 
                if videotensor.shape[0] == 69:
                    video_tensor_2.append(videotensor[46:])
                    video_nums_2+=1
                    video_tensor.append(videotensor) 
                    video_nums+=1

                if len(video_tensor) == batch_size:
                    video_tensor = torch.stack(video_tensor) # B N C H W
                    fvd_pipe.update(video_tensor.to(device), True)
                    video_tensor = []

                if len(video_tensor_0) == batch_size:
                    video_tensor_0 = torch.stack(video_tensor_0) # B N C H W
                    fvd_pipe_0.update(video_tensor_0.to(device), True)
                    video_tensor_0 = []

                if len(video_tensor_1) == batch_size:
                    video_tensor_1 = torch.stack(video_tensor_1) # B N C H W
                    fvd_pipe_1.update(video_tensor_1.to(device), True)
                    video_tensor_1 = []

                if len(video_tensor_2) == batch_size:
                    video_tensor_2 = torch.stack(video_tensor_2) # B N C H W
                    fvd_pipe_2.update(video_tensor_2.to(device), True)
                    video_tensor_2 = []

    if len(video_tensor) > 0:
        video_tensor = torch.stack(video_tensor) # B N C H W
        fvd_pipe.update(video_tensor.to(device), True)
        video_tensor = []

    if len(video_tensor_0) > 0:
        video_tensor_0 = torch.stack(video_tensor_0) # B N C H W
        fvd_pipe_0.update(video_tensor_0.to(device), True)
        video_tensor_0 = []

    if len(video_tensor_1) > 0:
        video_tensor_1 = torch.stack(video_tensor_1) # B N C H W
        fvd_pipe_1.update(video_tensor_1.to(device), True)
        video_tensor_1 = []

    if len(video_tensor_2) > 0:
        video_tensor_2 = torch.stack(video_tensor_2) # B N C H W
        fvd_pipe_2.update(video_tensor_2.to(device), True)
        video_tensor_2 = []
    
    print(f"real video num: {video_nums},{video_nums_0},{video_nums_1},{video_nums_2}")

    fvd_score = fvd_pipe.compute()
    fvd_score_0 = fvd_pipe_0.compute()
    fvd_score_1 = fvd_pipe_1.compute()
    fvd_score_2 = fvd_pipe_2.compute()

    print(f'FVD: {fvd_score}, {fvd_score_0}, {fvd_score_1}, {fvd_score_2}')

if __name__ == '__main__':
    fire.Fire(main)