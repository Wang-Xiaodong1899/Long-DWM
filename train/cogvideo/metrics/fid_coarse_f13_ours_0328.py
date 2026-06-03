import sys
sys.path.append("/home/user/wangxd/diffusers/examples/cogvideo")

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

from common import image2arr, pil2arr, mp4toarr, image2pil, json2data, preprocess_image

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from inception import InceptionV3

from torchmetrics.image.fid import FrechetInceptionDistance
# NOTE normalize=true

from nuscenes_dataset_for_cogvidx import NuscenesDatasetAllframesFPS10OneByOneForValidatePath



IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

import torchvision.transforms as TT
resize_transform = TT.Compose([
    TT.Resize((480, 720)),
])

NUSCENES_ROOT = "/data/wangxd/nuscenes/"

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        # path -> frame path
        # img = resize_transform(Image.open(path).convert('RGB'))
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img

class VideoPathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None, frames=25, start=0):
        self.files = files
        self.transforms = transforms
        self.frames = frames
        self.start = start

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        # path -> video path

        video_arr = mp4toarr(path)
        video = video_arr[self.start:self.frames]
        images = []
        for frame in video:
            img = preprocess_image(Image.fromarray(frame))
            img = Image.fromarray(frame)
            if self.transforms is not None:
                img = self.transforms(img)
            images.append(img)
        images = torch.stack(images, 0)
        return images


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def main(
    tgt_dirs=[
            '/data/wangxd/ckpt/cogvideox-A4-clean-image-fft-ckpt-distill-explicit-L2-hf-loss-0111-infer-step100-FVD/interpolation',
            # '/data/wangxd/ckpt/cogvideox-A4-clean-image-fft-ckpt-distill-explicit-L2-hf-loss-0111-infer-step100-FVD-s10-e20/interpolation',
            # '/data/wangxd/ckpt/cogvideox-A4-clean-image-fft-ckpt-distill-explicit-L2-hf-loss-0111-infer-step100-FVD-s20-e30/interpolation',
            # '/data/wangxd/ckpt/cogvideox-A4-clean-image-fft-ckpt-distill-explicit-L2-hf-loss-0111-infer-step100-FVD-s30-e40/interpolation',
            # '/data/wangxd/ckpt/cogvideox-A4-clean-image-fft-ckpt-distill-explicit-L2-hf-loss-0111-infer-step100-FVD-s40-e50/interpolation',
            # '/data/wangxd/ckpt/cogvideox-A4-clean-image-fft-ckpt-distill-explicit-L2-hf-loss-0111-infer-step100-FVD-s50-e60/interpolation',
            # '/data/wangxd/ckpt/cogvideox-A4-clean-image-fft-ckpt-distill-explicit-L2-hf-loss-0111-infer-step100-FVD-s60-e70/interpolation',
            # '/data/wangxd/ckpt/cogvideox-A4-clean-image-fft-ckpt-distill-explicit-L2-hf-loss-0111-infer-step100-FVD-s70-e80/interpolation',
            # '/data/wangxd/ckpt/cogvideox-A4-clean-image-fft-ckpt-distill-explicit-L2-hf-loss-0111-infer-step100-FVD-s80-e90/interpolation',
            # '/data/wangxd/ckpt/interpolation-s90-e150/s90-e100','/data/wangxd/ckpt/interpolation-s90-e150/s100-e110',
            # '/data/wangxd/ckpt/interpolation-s90-e150/s110-e120','/data/wangxd/ckpt/interpolation-s90-e150/s120-e130',
            # '/data/wangxd/ckpt/interpolation-s90-e150/s130-e140','/data/wangxd/ckpt/interpolation-s90-e150/s140-e150'
            ],
    version = None,
    num_frames=8,
    eval_frames=25,
    batch_size=16, 
    device='cuda:1', 
    dims=2048,
    frame_start=1,
    num_workers=4,
    val_array = [(0,10),
                #  (10, 20), (20,30), (30, 40), (40, 50), (50,60), (60, 70),(70,80), (80,90), (90, 100), (100, 110), (110, 120), (120, 130), (130, 140), (140, 150)
                ],
):

    # if version is None:
    #     version = os.path.basename(tgt_dir)+f'_F{num_frames}'
    
    print(f'eval videos at {tgt_dirs} with {eval_frames} frames')

    # block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    # model = InceptionV3([block_idx]).to(device)
    # print('loaded iception model')
    fider = FrechetInceptionDistance(normalize=True).to(device)
    
    real_dataset = NuscenesDatasetAllframesFPS10OneByOneForValidatePath(
        data_root="/data/wangxd/nuscenes/",
        height=480,
        width=720,
        max_num_frames=25,
        encode_video=None,
        encode_prompt=None,
    )

    # read real video
    # meta_data = json2data('/mnt/lustrenew/wangxiaodong/data/nuscene/samples_group_sort_val.json')
    # files_dir = '/mnt/lustrenew/wangxiaodong/data/nuscene/val_group'
    
    syn_videos_paths = []
    for tgt_dir in tgt_dirs: # scenes
        video_files = os.listdir(tgt_dir)
        for file in tqdm(video_files):
            if file.split('.')[-1] == 'mp4':
                syn_videos_paths.append(os.path.join(tgt_dir, file)) # only load eval_frames
    
    print(f'syn length {len(syn_videos_paths)}')
    
    real_frames_paths = []
    
    for (val_s, val_e) in val_array:
        for i in tqdm(range(val_s, val_e)):
            item = real_dataset[i]
            for key_idx in tqdm(range(len(item))):
                key_frame = item[key_idx]
                pil_videos_path = key_frame["instance_video"]
                real_frames_paths.extend(pil_videos_path[frame_start:eval_frames])
    
    print(f'GT length {len(real_frames_paths)}')
    
    
    # resize_transform = TT.Compose([
    #     TT.Resize((480, 720)),
    #     TF.ToTensor()
    # ]) # -> FID=32 for 10 scenes
    resize_transform = TT.Compose([
        TT.Resize((256, 448)),
        TF.ToTensor()
    ]) # -> FID=30 for 10 scenes

    
    # video dataset
    video_dataset = VideoPathDataset(syn_videos_paths, transforms=resize_transform, frames=eval_frames, start=frame_start)
    video_dataloader = torch.utils.data.DataLoader(video_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)
    
    frame_dataset = ImagePathDataset(real_frames_paths, transforms=resize_transform)
    frame_dataloader = torch.utils.data.DataLoader(frame_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)
    # video
    pred_arr = np.empty((len(syn_videos_paths)*(eval_frames-frame_start), dims))
    print(f'pred arr shape: {pred_arr.shape}')

    start_idx = 0

    # 24 frames if skip condition
    for batch in tqdm(video_dataloader):
        batch = batch.to(device)
        batch = batch.flatten(0,1)    
        fider.update(batch.to(device), real=False)
    
    # frames
    print(f'gt arr shape: {len(real_frames_paths)}')

    for batch in tqdm(frame_dataloader):
        batch = batch.to(device)
        fider.update(batch, real=True)

    fid_value = fider.compute()

    print('FID: ', fid_value)


if __name__ == '__main__':
    fire.Fire(main)
