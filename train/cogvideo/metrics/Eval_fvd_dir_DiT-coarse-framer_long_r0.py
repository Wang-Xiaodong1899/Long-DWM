import sys
sys.path.append("/home/user/wangxd/diffusers/examples/cogvideo")


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

# DATAROOT = '/mnt/lustrenew/wangxiaodong/data/nuscene'

import json


def main(
        tgt_dir='/data/wangxd/IJCAI25/Ablation/framer',
        version = None,
        split='val',
        num_frames=8,
        i3d_device='cuda:2',
        device="cuda:2",
        eval_frames=24,
        skip_gt = 0,
        skip_pred = 0,
        val_array = [(0,10)
                     ,(10, 20), (20,30), (30, 40), (40, 50), (50,60), (60, 70),(70, 80), (80,90), (90, 100), (100, 110), (110, 120), (120, 130), (130, 140), (140, 150)
                     ],
        samples_per_scene=1,
        gt_image_start=0,
):
    # [40, 50], [60, 70]

    # if version is None:
    #     version = os.path.basename(tgt_dir)+f'_F{num_frames}'
    
    print(f'eval videos at {tgt_dir}')
    # load fvd model
    # i3d_model = load_fvd_model(i3d_device)
    # print('loaded i3d_model')
    fvd_pipe = FrechetVideoDistance().to(device)

    # read real video
    # meta_data = json2data('/mnt/lustrenew/wangxiaodong/data/nuscene/samples_group_sort_val.json')
    # files_dir = '/mnt/lustrenew/wangxiaodong/data/nuscene/val_group'
    
    # 10 scenes * ~35 keyframes
    real_dataset = NuscenesDatasetAllframesFPS10OneByOneForValidate(
        data_root="/data/wangxd/nuscenes/",
        height=480,
        width=720,
        max_num_frames=125,
        encode_video=None,
        encode_prompt=None,
    )

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    import torchvision.transforms as TT
    resize_transform = TT.Compose([
        TT.Resize((480, 720)),
    ])

    count_sync = 0

    # read syn videos
    syn_videos = []
    scenes = range(150)


    for (scene_start, scene_end) in tqdm(val_array):
        # 5 samples
        cur_dir = os.path.join(tgt_dir, f"cogvideox-A4-clean-image-fft-ckpt-distill-explicit-L2-hf-loss-0111-infer-step100-FVD-s{scene_start}-e{scene_end}") # append
        sample_paths = os.listdir(cur_dir) # NOTE all samples

        video_tensor = []
        sync_video_num = 0
        for sample_path in sample_paths: # 0,1,...350
        # for file in tqdm(video_files):
            video_dir = os.path.join(cur_dir, sample_path, "generate")
            # NOTE first short video, only the 25 frames
            # first file: 0.mp4
            video_arr = mp4toarr(os.path.join(video_dir, "0.mp4"), resize=False, convert=True)
            pil_videos = []
            for im_arr in video_arr:
                pil_im = preprocess_image(Image.fromarray(im_arr))
                pil_videos.append(pil_im)
            
            # second file: 1.mp4
            video_arr = mp4toarr(os.path.join(video_dir, "1.mp4"), resize=False, convert=True)
            for im_arr in video_arr[1:]:
                pil_im = preprocess_image(Image.fromarray(im_arr))
                pil_videos.append(pil_im)
            
            # otherwise, only evaluate last 4 rollouts
            # print(f"frames per video {len(pil_videos)}")
            
            # split 13+12+12+12 into 24+24
            new_pil_videos = []
            new_pil_videos.extend(pil_videos[gt_image_start: gt_image_start+eval_frames])
            # new_pil_videos.extend(pil_videos[gt_image_start+eval_frames: gt_image_start+eval_frames*2])
            
            # videotensor = [transform(im) for im in pil_videos[:eval_frames]]
            videotensor = [transform(im) for im in new_pil_videos[:eval_frames]]
            videotensor = torch.stack(videotensor) # N C H W

            video_tensor.append(videotensor) # [, ,]

            if len(video_tensor) == 16:
                video_tensor = torch.stack(video_tensor) # B N C H W
                fvd_pipe.update(video_tensor.to(device), False)
                video_tensor = []
            # syn_videos.append(video_arr[:eval_frames]) # only load eval_frames

            count_sync = count_sync + 1

        # NOTE fix bug
        if len(video_tensor) > 0:
            video_tensor = torch.stack(video_tensor) # B N C H W
            fvd_pipe.update(video_tensor.to(device), False)
    
    
    print(f"sync video num: {count_sync}")
    # syn_videos = np.array(syn_videos)
    # print(f'syn shape {syn_videos.shape}')
    
    real_num = 0
    real_videos = []
    for (val_s, val_e) in tqdm(val_array):
        for i in (range(val_s, val_e)):
            video_tensor = []
            item = real_dataset[i]
            for key_idx in (range(len(item))):
                key_frame = item[key_idx]
                pil_videos = key_frame["instance_video"]
                
                # import pdb; pdb.set_trace()
                
                pil_videos = [preprocess_image(resize_transform(im)) for im in pil_videos]
                
                # pil_videos = [preprocess_image(im) for im in pil_videos]
                
                # print(f"frames per video {len(pil_videos[gt_image_start: gt_image_start+2*eval_frames])}")
                
                # split 48 to 24+24
                # new_pil_videos = []
                # new_pil_videos.extend(pil_videos[gt_image_start: gt_image_start+eval_frames])

                videotensor = [transform(im) for im in pil_videos[gt_image_start: gt_image_start+eval_frames]]
                videotensor = torch.stack(videotensor) # N C H W

                video_tensor.append(videotensor) # [, ,]

                if len(video_tensor) == 16:
                    video_tensor = torch.stack(video_tensor) # B N C H W
                    fvd_pipe.update(video_tensor.to(device), True)
                    video_tensor = []
                
                real_num = real_num + 1

                # real_videos.append(pil_videos[:eval_frames])
            
            # NOTE Fix bug, need calculate all
            if len(video_tensor) > 0:
                video_tensor = torch.stack(video_tensor) # B N C H W
                fvd_pipe.update(video_tensor.to(device), True)
                video_tensor = []
            
    
    print(f"real video num: {real_num}")
    # real_videos = []
    # for item in tqdm(meta_data):
    #     sce = item['scene']
    #     files = os.listdir(os.path.join(gt_dir, sce))
    #     for file in files:
    #         if file.split('.')[-1] == 'mp4':
    #             video_arr = mp4toarr(os.path.join(gt_dir, sce, file))
    #             sample_frames = video_arr[::skip_gt+1]
    #             # if len(sample_frames)<eval_frames:
    #             #     sample_frames = list(sample_frames) + [sample_frames[-1]] * (eval_frames-len(sample_frames))
    #             #     print(len(sample_frames), eval_frames-len(sample_frames))
    #             real_videos.append(sample_frames[:eval_frames]) # only load eval_frames

    # N T H W C
    # real_videos = np.array(real_videos)
    # print(f'real shape {real_videos.shape}')

    # fvd_score = compute_fvd(real_videos, syn_videos, i3d_model, i3d_device)
    # fvd_score = compute_fvd_1(real_videos, syn_videos, i3d_model, i3d_device)

    fvd_score = fvd_pipe.compute()
    
    # print('Save Done!')

    print(f'{version} AVG FVD: {fvd_score}')

if __name__ == '__main__':
    fire.Fire(main)