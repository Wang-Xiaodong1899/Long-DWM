
from torch.utils.data import Dataset
import json
import random
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as TT
from torchvision.transforms.functional import InterpolationMode, resize
import os
from tqdm import tqdm
import transformers

import math

# from nuscenes.nuscenes import NuScenes
# from nuscenes.utils.splits import create_splits_scenes

import random

plain_caption_dict = {
    0: "Go straight.",
    1: "Pass the intersection.",
    2: "Turn left.",
    3: "Turn right.",
    4: "Change to the left lane.",
    5: "Change to the right lane.",
    6: "Go to the left lane branch.",
    7: "Go to the right lane branch.",
    8: "Pass the crosswalk.",
    9: "Pass the railroad.",
    10: "Merge.",
    11: "Make a U-turn.",
    12: "Stop.",
    13: "Deviate."
}

diverse_caption_dict = {
    0: [
        "Move forward.",
        "Move steady.",
        "Go forward.",
        "Go straight.",
        "Proceed.",
        "Drive forward.",
        "Drive straight.",
        "Drive steady.",
        "Keep the direction.",
        "Maintain the direction.",
    ],
    1: [
        "Pass the intersection.",
        "Cross the intersection.",
        "Traverse the intersection.",
        "Drive through the intersection.",
        "Move past the intersection.",
        "Pass the junction.",
        "Cross the junction.",
        "Traverse the junction.",
        "Drive through the junction.",
        "Move past the junction.",
        "Pass the crossroad.",
        "Cross the crossroad.",
        "Traverse the crossroad.",
        "Drive through the crossroad.",
        "Move past the crossroad.",
    ],
    2: [
        "Turn left.",
        "Turn to the left.",
        "Make a left turn.",
        "Take a left turn.",
        "Turn to the left.",
        "Left turn.",
        "Steer left.",
        "Steer to the left.",
    ],
    3: [
        "Turn right.",
        "Turn to the right.",
        "Make a right turn.",
        "Take a right turn.",
        "Turn to the right.",
        "Right turn.",
        "Steer right.",
        "Steer to the right.",
    ],
    4: [
        "Make a left lane change.",
        "Change to the left lane.",
        "Switch to the left lane.",
        "Shift to the left lane.",
        "Move to the left lane.",
    ],
    5: [
        "Make a right lane change.",
        "Change to the right lane.",
        "Switch to the right lane.",
        "Shift to the right lane.",
        "Move to the right lane.",
    ],
    6: [
        "Go to the left lane branch.",
        "Take the left lane branch.",
        "Move into the left lane branch.",
        "Follow the left lane branch.",
        "Follow the left side road.",
    ],
    7: [
        "Go to the right lane branch.",
        "Take the right lane branch.",
        "Move into the right lane branch.",
        "Follow the right lane branch.",
        "Follow the right side road.",
    ],
    8: [
        "Pass the crosswalk.",
        "Cross the crosswalk.",
        "Traverse the crosswalk.",
        "Drive through the crosswalk.",
        "Move past the crosswalk.",
        "Pass the crossing area.",
        "Cross the crossing area.",
        "Traverse the crossing area.",
        "Drive through the crossing area.",
        "Move past the crossing area.",
    ],
    9: [
        "Pass the railroad.",
        "Cross the railroad.",
        "Traverse the railroad.",
        "Drive through the railroad.",
        "Move past the railroad.",
        "Pass the railway.",
        "Cross the railway.",
        "Traverse the railway.",
        "Drive through the railway.",
        "Move past the railway.",
    ],
    10: [
        "Merge.",
        "Merge traffic.",
        "Merge into traffic.",
        "Merge into the traffic.",
        "Join the traffic.",
        "Merge into the traffic flow.",
        "Join the traffic flow.",
        "Merge into the traffic stream.",
        "Join the traffic stream.",
        "Merge into the lane.",
    ],
    11: [
        "Make a U-turn.",
        "Make a 180-degree turn.",
        "Turn 180 degree.",
        "Turn around.",
        "Drive in a U-turn.",
    ],
    12: [
        "Stop.",
        "Halt.",
        "Decelerate.",
        "Slow down.",
        "Brake.",
    ],
    13: [
        "Deviate.",
        "Deviate from the path.",
        "Deviate from the lane.",
        "Change the direction.",
        "Shift the direction.",
    ]
}


def map_category_to_caption(category_index, diverse=True):
    if diverse:
        return random.choice(diverse_caption_dict[category_index])
    else:
        return plain_caption_dict[category_index]

def image2pil(filename):
    return Image.open(filename)

def image2arr(filename):
    pil = image2pil(filename)
    return pil2arr(pil)

def pil2arr(pil):
    if isinstance(pil, list):
        arr = np.array(
            [np.array(e.convert('RGB').getdata(), dtype=np.uint8).reshape(e.size[1], e.size[0], 3) for e in pil])
    else:
        arr = np.array(pil)
    return arr

def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out



class NuscenesDatasetForCogvidx(Dataset):
    def __init__(
        self,
        data_root: str = "/data/wangxd/nuscenes",
        height: int = 480,
        width: int = 720,
        max_num_frames: int = 1, # must be (4k+1)
        split: str = "train",
        encode_prompt = None,
        encode_video = None,
        max_samples: int = 700,
        preload_all_data: bool = True,
        preprocessed_data_path = "/root/autodl-fs/dataset_700_samples_fix"
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.height = height
        self.width = width
        self.max_num_frames = max_num_frames
        self.split = split
        self.encode_prompt=encode_prompt
        self.encode_video=encode_video
        self.preload_all_data = preload_all_data
        self.preprocessed_data_path = preprocessed_data_path

        if preload_all_data and preprocessed_data_path is not None:
            prompts = torch.load(os.path.join(preprocessed_data_path,"prompts.npy"))
            videos = torch.load(os.path.join(preprocessed_data_path,"videos.npy"))
            images = torch.load(os.path.join(preprocessed_data_path,"images.npy"))
            progress_dataset_bar = tqdm(
                range(0, prompts.shape[0]),
                desc="Loading preprocessed prompts and videos",
            )
            self.instance_prompts = []
            self.instance_videos =[]
            for i in range(prompts.shape[0]):
                progress_dataset_bar.update(1)
                self.instance_prompts.append(prompts[i:i+1])
                self.instance_videos.append((videos[i:i+1],images[i:i+1]))
        else:

            self.nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_root, verbose=True)

            self.splits = create_splits_scenes()

            # training samples
            self.samples_groups = self.group_sample_by_scene(split)
            
            self.scenes = list(self.samples_groups.keys())

            if len(self.scenes) > max_samples:
                self.scenes = self.scenes[:max_samples]
            
            self.frames_group = {} # (scene, image_paths)
            
            for my_scene in self.scenes:
                f = self.get_paths_from_scene(my_scene)
                if len(f) >= max_num_frames :
                    self.frames_group[my_scene] = self.get_paths_from_scene(my_scene)
            self.scenes = [k for k,v in self.frames_group.items()]

            print('Total samples: %d' % len(self.scenes))

            # search annotations
            json_path = f'{data_root}/nusc_video_{split}_8_ov-7b_dict.json'
            with open(json_path, 'r') as f:
                self.annotations = json.load(f)
            
            if preload_all_data:
                self.preload()

    
    def __len__(self):
        if self.preload_all_data:
            return len(self.instance_prompts)
        return len(self.scenes)

    def augmentation(self, frame, transform, state):
        torch.set_rng_state(state)
        return transform(frame)
    
    def get_samples(self, split='train'):
        scenes = self.splits[split]
        samples = [samp for samp in self.nusc.sample if
                   self.nusc.get('scene', samp['scene_token'])['name'] in scenes]
        return samples
    
    def group_sample_by_scene(self, split='train'):
        scenes = self.splits[split]
        samples_dict = {}
        for sce in scenes:
            samples_dict[sce] = [] # empty list
        for samp in self.nusc.sample:
            scene_token = samp['scene_token']
            scene = self.nusc.get('scene', scene_token)
            tmp_sce = scene['name']
            if tmp_sce in scenes:
                samples_dict[tmp_sce].append(samp)
        return samples_dict
    
    def get_image_path_from_sample(self, my_sample):
        sample_data = self.nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
        file_path = sample_data['filename']
        image_path = os.path.join(self.data_root, file_path)
        return image_path
    
    def get_paths_from_scene(self, my_scene):
        samples = self.samples_groups.get(my_scene, [])
        paths = [self.get_image_path_from_sample(sam) for sam in samples]
        paths.sort()
        return paths
    
    def load_sample(self, index):
        my_scene = self.scenes[index]
        all_frames = self.frames_group[my_scene]
        
        seek_start = random.randint(0, len(all_frames) - self.max_num_frames)
        seek_path = all_frames[seek_start: seek_start + self.max_num_frames]            

        scene_annotation = self.annotations[my_scene]

        seek_id = seek_start//4*4
        search_id = f"{seek_id}"
        timestep = search_id if search_id in scene_annotation else str((int(search_id)//4-1)*4)

        annotation = scene_annotation[timestep]
        
        caption = ""
        selected_attr = ["Weather", "Time", "Road environment", "Critical objects"]
        for attr in selected_attr:
            anno = annotation.get(attr, "")
            if anno == "":
                continue
            if anno[-1] == ".":
                anno = anno[:-1] 
            caption = caption + anno + ". "
        
        driving_prompt = annotation.get("Driving action", "").strip()
        # print(driving_prompt)

        frames = torch.Tensor(np.stack([image2arr(path) for path in seek_path]))

        selected_num_frames = frames.shape[0]

        assert (selected_num_frames - 1) % 4 == 0

        # Training transforms
        frames = (frames - 127.5) / 127.5
        frames = frames.permute(0, 3, 1, 2) # [F, C, H, W]
        frames = resize(frames,size=[self.height, self.width],interpolation=InterpolationMode.BICUBIC)

        return driving_prompt, frames

    def encode(self, prompt, video):
        prompt = self.encode_prompt(prompt).to("cpu")
        v1,v2 = self.encode_video(video)
        video = (v1.sample().to("cpu"),v2.sample().to("cpu"))
        return prompt, video

    def preload(self):
        self.instance_prompts = []
        self.instance_videos =[]

        progress_dataset_bar = tqdm(
            range(0, len(self.scenes)),
            desc="Loading prompts and videos",
        )

        for index in range(len(self.scenes)):
            progress_dataset_bar.update(1)

            prompt, video = self.load_sample(index)
            prompt, video = self.encode(prompt, video)

            self.instance_prompts.append(prompt)
            self.instance_videos.append(video)
        
    def __getitem__(self, index):
        if self.preload_all_data:
            return {
                "instance_prompt": self.instance_prompts[index],
                "instance_video": self.instance_videos[index],
            }
        else:
            prompt, video = self.load_sample(index)
            prompt, video = self.encode(prompt, video)
            return {
                "instance_prompt": prompt,
                "instance_video": video,
            }


class NuscenesDatasetCLIPForCogvidx(Dataset):
    def __init__(
        self,
        data_root: str = "/data/wangxd/nuscenes",
        height: int = 480,
        width: int = 720,
        max_num_frames: int = 1, # must be (4k+1)
        split: str = "train",
        encode_prompt = None,
        encode_video = None,
        max_samples: int = 700,
        preload_all_data: bool = False,
        preprocessed_data_path = "/root/autodl-fs/dataset_700_samples_fix"
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.height = height
        self.width = width
        self.max_num_frames = max_num_frames
        self.split = split
        self.encode_prompt=encode_prompt
        self.encode_video=encode_video
        self.preload_all_data = preload_all_data
        self.preprocessed_data_path = preprocessed_data_path

        if preload_all_data and preprocessed_data_path is not None:
            prompts = torch.load(os.path.join(preprocessed_data_path,"prompts.npy"))
            videos = torch.load(os.path.join(preprocessed_data_path,"videos.npy"))
            images = torch.load(os.path.join(preprocessed_data_path,"images.npy"))
            progress_dataset_bar = tqdm(
                range(0, prompts.shape[0]),
                desc="Loading preprocessed prompts and videos",
            )
            self.instance_prompts = []
            self.instance_videos =[]
            for i in range(prompts.shape[0]):
                progress_dataset_bar.update(1)
                self.instance_prompts.append(prompts[i:i+1])
                self.instance_videos.append((videos[i:i+1],images[i:i+1]))
        else:

            self.nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_root, verbose=True)

            self.splits = create_splits_scenes()

            # training samples
            self.samples_groups = self.group_sample_by_scene(split)
            
            self.scenes = list(self.samples_groups.keys())

            if len(self.scenes) > max_samples:
                self.scenes = self.scenes[:max_samples]
            
            self.frames_group = {} # (scene, image_paths)
            
            for my_scene in self.scenes:
                f = self.get_paths_from_scene(my_scene)
                if len(f) >= max_num_frames :
                    self.frames_group[my_scene] = self.get_paths_from_scene(my_scene)
            self.scenes = [k for k,v in self.frames_group.items()]

            print('Total samples: %d' % len(self.scenes))

            # search annotations
            json_path = f'{data_root}/nusc_video_{split}_8_ov-7b_dict.json'
            with open(json_path, 'r') as f:
                self.annotations = json.load(f)
            
            if preload_all_data:
                self.preload()
            
            self.clip_transform = TT.Normalize(
                transformers.image_utils.OPENAI_CLIP_MEAN,
                transformers.image_utils.OPENAI_CLIP_STD)

    
    def __len__(self):
        if self.preload_all_data:
            return len(self.instance_prompts)
        return len(self.scenes)

    def augmentation(self, frame, transform, state):
        torch.set_rng_state(state)
        return transform(frame)
    
    def get_samples(self, split='train'):
        scenes = self.splits[split]
        samples = [samp for samp in self.nusc.sample if
                   self.nusc.get('scene', samp['scene_token'])['name'] in scenes]
        return samples
    
    def group_sample_by_scene(self, split='train'):
        scenes = self.splits[split]
        samples_dict = {}
        for sce in scenes:
            samples_dict[sce] = [] # empty list
        for samp in self.nusc.sample:
            scene_token = samp['scene_token']
            scene = self.nusc.get('scene', scene_token)
            tmp_sce = scene['name']
            if tmp_sce in scenes:
                samples_dict[tmp_sce].append(samp)
        return samples_dict
    
    def get_image_path_from_sample(self, my_sample):
        sample_data = self.nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
        file_path = sample_data['filename']
        image_path = os.path.join(self.data_root, file_path)
        return image_path
    
    def get_paths_from_scene(self, my_scene):
        samples = self.samples_groups.get(my_scene, [])
        paths = [self.get_image_path_from_sample(sam) for sam in samples]
        paths.sort()
        return paths
    
    def load_sample(self, index):
        my_scene = self.scenes[index]
        all_frames = self.frames_group[my_scene]
        
        seek_start = random.randint(0, len(all_frames) - self.max_num_frames)
        seek_path = all_frames[seek_start: seek_start + self.max_num_frames]
        
        driving_prompt = ""

        frames = torch.Tensor(np.stack([image2arr(path) for path in seek_path]))

        selected_num_frames = frames.shape[0]

        assert (selected_num_frames - 1) % 4 == 0

        # Training transforms
        frames = (frames - 127.5) / 127.5
        frames = frames.permute(0, 3, 1, 2) # [F, C, H, W]
        frames = resize(frames,size=[self.height, self.width],interpolation=InterpolationMode.BICUBIC)

        clip_video = frames # n c h w
        clip_video = _resize_with_antialiasing(clip_video, (224, 224))
        clip_video = (clip_video + 1.0) / 2.0 # -> (0, 1)
        clip_video = self.clip_transform(clip_video)

        return driving_prompt, frames, clip_video

    def encode(self, prompt, video):
        prompt = self.encode_prompt(prompt).to("cpu")
        v1,v2 = self.encode_video(video)
        video = (v1.sample().to("cpu"),v2.sample().to("cpu"))
        return prompt, video

    def preload(self):
        self.instance_prompts = []
        self.instance_videos =[]

        progress_dataset_bar = tqdm(
            range(0, len(self.scenes)),
            desc="Loading prompts and videos",
        )

        for index in range(len(self.scenes)):
            progress_dataset_bar.update(1)

            prompt, video = self.load_sample(index)
            prompt, video = self.encode(prompt, video)

            self.instance_prompts.append(prompt)
            self.instance_videos.append(video)
        
    def __getitem__(self, index):
        if self.preload_all_data:
            return {
                "instance_prompt": self.instance_prompts[index],
                "instance_video": self.instance_videos[index],
            }
        else:
            prompt, video, clip_video = self.load_sample(index)
            prompt, video = self.encode(prompt, video)
            return {
                "instance_prompt": prompt,
                "instance_video": video,
                "clip_video": clip_video
            }


class NuscenesDatasetAllframesForCogvidx(Dataset):
    def __init__(
        self,
        data_root: str = "/data/wangxd/nuscenes",
        height: int = 480,
        width: int = 720,
        max_num_frames: int = 49, # 8 * N + 1 (where N ≤ 6)
        split: str = "train",
        encode_prompt = None,
        encode_video = None,
        max_samples: int = 700,
        preload_all_data: bool = False,
        preprocessed_data_path = "/root/autodl-fs/dataset_700_samples_fix"
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.height = height
        self.width = width
        self.max_num_frames = max_num_frames
        self.split = split
        self.encode_prompt=encode_prompt
        self.encode_video=encode_video
        self.preload_all_data = preload_all_data
        self.preprocessed_data_path = preprocessed_data_path

        if preload_all_data and preprocessed_data_path is not None:
            pass
        else:

            self.nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_root, verbose=True)

            self.splits = create_splits_scenes()

            # training samples
            self.samples = self.get_samples(split)

            print('Total samples: %d' % len(self.samples))

            # search annotations
            json_path = f'{data_root}/nusc_video_{split}_8_ov-7b_dict.json'
            with open(json_path, 'r') as f:
                self.annotations = json.load(f)
            
            command_path = f'{data_root}/nusc_action_{split}.json'
            with open(command_path, 'r') as f:
                self.command_dict = json.load(f)
            
            # image transform
            self.transform = TT.Compose([
                    TT.ToPILImage('RGB'),
                    TT.RandomResizedCrop((height, width), scale=(0.5, 1.), ratio=(1., 1.75)),
                    # transforms.RandomHorizontalFlip(p=0.1),
                    TT.ColorJitter(brightness=0.05, contrast=0.15, saturation=0.15),
                    TT.ToTensor(),
                    TT.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True),
                ])
            
            if preload_all_data:
                self.preload()

    
    def __len__(self):
        return len(self.samples)

    def augmentation(self, frame, transform, state):
        torch.set_rng_state(state)
        return transform(frame)
    
    def get_samples(self, split='train'):
        selected_scenes = self.splits[split] # all scenes
        all_scenes = self.nusc.scene
        selected_scenes_meta = []
        for sce in all_scenes:
            if sce['name'] in selected_scenes:
                selected_scenes_meta.append(sce)
        
        samples_group_by_scene = []
        for scene in selected_scenes_meta:
            samples_group_by_scene.append(
                {
                'scene': scene['name'],
                'video': self.get_all_frames_from_scene(scene)
                }
            )
        
        return samples_group_by_scene

    def get_all_frames_from_scene(self, scene):
        # get all frames (keyframes, sweeps)
        first_sample_token = scene['first_sample_token']
        my_sample = self.nusc.get('sample', first_sample_token)
        sensor = "CAM_FRONT"
        cam_front_data = self.nusc.get('sample_data', my_sample['data'][sensor]) # first frame sensor token
        # frames = 0
        all_frames_dict = [] # len() -> frame number
        while True:
            all_frames_dict.append(cam_front_data)
            # filename = cam_front_data['filename']  # current-frame filename
            next_sample_data_token = cam_front_data['next']  # next-frame sensor token
            if not next_sample_data_token: # ''
                break
            cam_front_data = self.nusc.get('sample_data', next_sample_data_token)
        
        return all_frames_dict
    
    def group_sample_by_scene(self, split='train'):
        scenes = self.splits[split]
        samples_dict = {}
        for sce in scenes:
            samples_dict[sce] = [] # empty list
        for samp in self.nusc.sample:
            scene_token = samp['scene_token']
            scene = self.nusc.get('scene', scene_token)
            tmp_sce = scene['name']
            if tmp_sce in scenes:
                samples_dict[tmp_sce].append(samp)
        return samples_dict
    
    def get_image_path_from_sample(self, my_sample):
        sample_data = self.nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
        file_path = sample_data['filename']
        image_path = os.path.join(self.data_root, file_path)
        return image_path
    
    def get_paths_from_scene(self, my_scene):
        samples = self.samples_groups.get(my_scene, [])
        paths = [self.get_image_path_from_sample(sam) for sam in samples]
        paths.sort()
        return paths
    
    def load_sample(self, index):
        my_scene = self.samples[index]
        my_scene_name = my_scene['scene']
        my_sample_list = my_scene["video"]
        inner_frame_len = len(my_sample_list)
        
        key_frame_idx = []
        for seek_idx in range(inner_frame_len):
            if my_sample_list[seek_idx]['is_key_frame']:
                key_frame_idx.append(seek_idx)
        
        seek_sample = None
        while seek_sample is None or not seek_sample['is_key_frame']:
            seek_start = random.randint(0, inner_frame_len - 51)
            seek_sample = my_sample_list[seek_start]

        seek_samples = my_sample_list[seek_start: seek_start + self.max_num_frames]
        
        # all frames
        frame_paths = []
        for my_sample in seek_samples:
            file_path = my_sample["filename"]
            image_path = os.path.join(self.data_root, file_path)
            frame_paths.append(image_path)

        scene_annotation = self.annotations[my_scene_name]
        
        seek_key_frame_idx = key_frame_idx.index(seek_start)

        seek_id = seek_key_frame_idx//4*4
                
        search_id = f"{seek_id}"
        timestep = search_id if search_id in scene_annotation else str((int(search_id)//4-1)*4)

        annotation = scene_annotation[timestep]
        
        caption = ""
        selected_attr = ["Weather", "Time", "Road environment", "Critical objects"]
        for attr in selected_attr:
            anno = annotation.get(attr, "")
            if anno == "":
                continue
            if anno[-1] == ".":
                anno = anno[:-1] 
            caption = caption + anno + ". "
        
        command_idx = str(seek_key_frame_idx)
        
        # import pdb; pdb.set_trace()
        
        command_miss_scenes = ["scene-0161", "scene-0162", "scene-0163", "scene-0164",
            "scene-0165", "scene-0166", "scene-0167", "scene-0168", "scene-0170",
            "scene-0171", "scene-0172", "scene-0173", "scene-0174",
            "scene-0175", "scene-0176", "scene-0419"
            ]
        
        if my_scene_name in command_miss_scenes:
            driving_prompt = ""
        else:  
            try:
                driving_prompt = self.command_dict[my_scene_name][command_idx]
            except:
                # import pdb; pdb.set_trace()
                driving_prompt = ""
                pass
        driving_prompt = driving_prompt + ". " + caption
        # print(driving_prompt)

        # frames = torch.Tensor(np.stack([image2arr(path) for path in frame_paths]))
        frames = np.stack([image2arr(path) for path in frame_paths])

        selected_num_frames = frames.shape[0]

        assert (selected_num_frames - 1) % 4 == 0

        # Training transforms
        
        # no aug
        # frames = (frames - 127.5) / 127.5
        # frames = frames.permute(0, 3, 1, 2) # [F, C, H, W]
        # frames = resize(frames,size=[self.height, self.width],interpolation=InterpolationMode.BICUBIC)
        # NOTE aug, pil -> tensor
        state = torch.get_rng_state()
        frames = torch.stack([self.augmentation(v, self.transform, state) for v in frames], dim=0)

        return driving_prompt, frames

    def encode(self, prompt, video):
        prompt = self.encode_prompt(prompt).to("cpu")
        v1,v2 = self.encode_video(video)
        video = (v1.sample().to("cpu"),v2.sample().to("cpu"))
        return prompt, video

    def preload(self):
        self.instance_prompts = []
        self.instance_videos =[]

        progress_dataset_bar = tqdm(
            range(0, len(self.scenes)),
            desc="Loading prompts and videos",
        )

        for index in range(len(self.scenes)):
            progress_dataset_bar.update(1)

            prompt, video = self.load_sample(index)
            prompt, video = self.encode(prompt, video)

            self.instance_prompts.append(prompt)
            self.instance_videos.append(video)
        
    def __getitem__(self, index):
        try:
            if self.preload_all_data:
                return {
                    "instance_prompt": self.instance_prompts[index],
                    "instance_video": self.instance_videos[index],
                }
            else:
                prompt, video = self.load_sample(index)
                prompt, video = self.encode(prompt, video)
                return {
                    "instance_prompt": prompt,
                    "instance_video": video,
                }
        except Exception as e:
            print('Bad idx %s skipped because of %s' % (index, e))
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))


class NuscenesDatasetKeyframesForCogvidx(Dataset):
    def __init__(
        self,
        data_root: str = "/data/wangxd/nuscenes",
        height: int = 480,
        width: int = 720,
        max_num_frames: int = 49, # 8 * N + 1 (where N ≤ 6)
        split: str = "train",
        encode_prompt = None,
        encode_video = None,
        max_samples: int = 700,
        preload_all_data: bool = False,
        preprocessed_data_path = "/root/autodl-fs/dataset_700_samples_fix"
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.height = height
        self.width = width
        self.max_num_frames = max_num_frames
        self.split = split
        self.encode_prompt=encode_prompt
        self.encode_video=encode_video
        self.preload_all_data = preload_all_data
        self.preprocessed_data_path = preprocessed_data_path

        if preload_all_data and preprocessed_data_path is not None:
            pass
        else:

            self.nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_root, verbose=True)

            self.splits = create_splits_scenes()

            # training samples
            self.samples_groups = self.group_sample_by_scene(split)
            
            self.scenes = list(self.samples_groups.keys())

            if len(self.scenes) > max_samples:
                self.scenes = self.scenes[:max_samples]
            
            self.frames_group = {} # (scene, image_paths)
            
            for my_scene in self.scenes:
                f = self.get_paths_from_scene(my_scene)
                if len(f) >= max_num_frames :
                    self.frames_group[my_scene] = self.get_paths_from_scene(my_scene)
            self.scenes = [k for k,v in self.frames_group.items()]

            print('Total samples: %d' % len(self.scenes))

            # search annotations
            json_path = f'{data_root}/nusc_video_{split}_8_ov-7b_dict.json'
            with open(json_path, 'r') as f:
                self.annotations = json.load(f)
            
            command_path = f'{data_root}/nusc_action_{split}.json'
            with open(command_path, 'r') as f:
                self.command_dict = json.load(f)
            
            # image transform
            self.transform = TT.Compose([
                    TT.ToPILImage('RGB'),
                    TT.RandomResizedCrop((height, width), scale=(0.5, 1.), ratio=(1., 1.75)),
                    # transforms.RandomHorizontalFlip(p=0.1),
                    TT.ColorJitter(brightness=0.05, contrast=0.15, saturation=0.15),
                    TT.ToTensor(),
                    TT.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True),
                ])
            
            
            
            if preload_all_data:
                self.preload()

    
    def __len__(self):
        return len(self.scenes)

    def augmentation(self, frame, transform, state):
        torch.set_rng_state(state)
        return transform(frame)
    
    def get_samples(self, split='train'):
        scenes = self.splits[split]
        samples = [samp for samp in self.nusc.sample if
                   self.nusc.get('scene', samp['scene_token'])['name'] in scenes]
        return samples
    
    def group_sample_by_scene(self, split='train'):
        scenes = self.splits[split]
        samples_dict = {}
        for sce in scenes:
            samples_dict[sce] = [] # empty list
        for samp in self.nusc.sample:
            scene_token = samp['scene_token']
            scene = self.nusc.get('scene', scene_token)
            tmp_sce = scene['name']
            if tmp_sce in scenes:
                samples_dict[tmp_sce].append(samp)
        return samples_dict
    
    def get_image_path_from_sample(self, my_sample):
        sample_data = self.nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
        file_path = sample_data['filename']
        image_path = os.path.join(self.data_root, file_path)
        return image_path
    
    def get_paths_from_scene(self, my_scene):
        samples = self.samples_groups.get(my_scene, [])
        paths = [self.get_image_path_from_sample(sam) for sam in samples]
        paths.sort()
        return paths
    
    def load_sample(self, index):
        # my_scene = self.samples[index]
        # my_scene_name = my_scene['scene']
        # my_sample_list = my_scene["video"]
        # inner_frame_len = len(my_sample_list)
        
        my_scene_name = self.scenes[index]
        all_frames = self.frames_group[my_scene_name]
        
        seek_start = random.randint(0, len(all_frames) - self.max_num_frames)
        seek_path = all_frames[seek_start: seek_start + self.max_num_frames]

        scene_annotation = self.annotations[my_scene_name]
        
        seek_key_frame_idx = seek_start

        seek_id = seek_key_frame_idx//4*4
                
        search_id = f"{seek_id}"
        timestep = search_id if search_id in scene_annotation else str((int(search_id)//4-1)*4)

        annotation = scene_annotation[timestep]
        
        caption = ""
        selected_attr = ["Weather", "Time", "Road environment", "Critical objects"]
        for attr in selected_attr:
            anno = annotation.get(attr, "")
            if anno == "":
                continue
            if anno[-1] == ".":
                anno = anno[:-1] 
            caption = caption + anno + ". "
        
        command_idx = str(seek_key_frame_idx)
        
        # import pdb; pdb.set_trace()
        
        command_miss_scenes = ["scene-0161", "scene-0162", "scene-0163", "scene-0164",
            "scene-0165", "scene-0166", "scene-0167", "scene-0168", "scene-0170",
            "scene-0171", "scene-0172", "scene-0173", "scene-0174",
            "scene-0175", "scene-0176", "scene-0419"
            ]
        
        if my_scene_name in command_miss_scenes:
            driving_prompt = ""
        else:  
            try:
                driving_prompt = self.command_dict[my_scene_name][command_idx]
            except:
                # import pdb; pdb.set_trace()
                driving_prompt = ""
                pass
        driving_prompt = driving_prompt + ". " + caption
        # print(driving_prompt)

        # frames = torch.Tensor(np.stack([image2arr(path) for path in frame_paths]))
        frames = np.stack([image2arr(path) for path in seek_path])

        selected_num_frames = frames.shape[0]

        assert (selected_num_frames - 1) % 4 == 0

        # Training transforms
        
        # no aug
        # frames = (frames - 127.5) / 127.5
        # frames = frames.permute(0, 3, 1, 2) # [F, C, H, W]
        # frames = resize(frames,size=[self.height, self.width],interpolation=InterpolationMode.BICUBIC)
        # NOTE aug, pil -> tensor
        state = torch.get_rng_state()
        frames = torch.stack([self.augmentation(v, self.transform, state) for v in frames], dim=0)

        return driving_prompt, frames

    def encode(self, prompt, video):
        prompt = self.encode_prompt(prompt).to("cpu")
        v1,v2 = self.encode_video(video)
        video = (v1.sample().to("cpu"),v2.sample().to("cpu"))
        return prompt, video

    def preload(self):
        self.instance_prompts = []
        self.instance_videos =[]

        progress_dataset_bar = tqdm(
            range(0, len(self.scenes)),
            desc="Loading prompts and videos",
        )

        for index in range(len(self.scenes)):
            progress_dataset_bar.update(1)

            prompt, video = self.load_sample(index)
            prompt, video = self.encode(prompt, video)

            self.instance_prompts.append(prompt)
            self.instance_videos.append(video)
        
    def __getitem__(self, index):
        try:
            if self.preload_all_data:
                return {
                    "instance_prompt": self.instance_prompts[index],
                    "instance_video": self.instance_videos[index],
                }
            else:
                prompt, video = self.load_sample(index)
                prompt, video = self.encode(prompt, video)
                return {
                    "instance_prompt": prompt,
                    "instance_video": video,
                }
        except Exception as e:
            print('Bad idx %s skipped because of %s' % (index, e))
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))


class OVkeyInterframeVideo(Dataset):
    def __init__(
        self,
        data_root: str = "/data/wangxd/nuscenes",
        height: int = 480,
        width: int = 720,
        max_num_frames: int = 49, # 8 * N + 1 (where N ≤ 6)
        split: str = "train",
        encode_prompt = None,
        encode_video = None,
        max_samples: int = 700,
        preload_all_data: bool = False,
        preprocessed_data_path = "/root/autodl-fs/dataset_700_samples_fix"
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.height = height
        self.width = width
        self.max_num_frames = max_num_frames
        self.split = split
        self.encode_prompt=encode_prompt
        self.encode_video=encode_video
        self.preload_all_data = preload_all_data
        self.preprocessed_data_path = preprocessed_data_path
        
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_root, verbose=True)
        self.splits = create_splits_scenes()

        # training samples
        self.samples = self.get_samples(split)

        print('Total samples: %d' % len(self.samples))

        # search annotations
        json_path = f'{data_root}/nusc_video_{split}_8_ov-7b_dict.json'
        with open(json_path, 'r') as f:
            self.annotations = json.load(f)
        
        command_path = f'{data_root}/nusc_action_{split}.json'
        with open(command_path, 'r') as f:
            self.command_dict = json.load(f)
        
        # image transform
        self.transform = TT.Compose([
                TT.ToPILImage('RGB'),
                TT.RandomResizedCrop((height, width), scale=(0.5, 1.), ratio=(1., 1.75)),
                # transforms.RandomHorizontalFlip(p=0.1),
                TT.ColorJitter(brightness=0.05, contrast=0.15, saturation=0.15),
                TT.ToTensor(),
                TT.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True),
            ])
    
    def __len__(self):
        return len(self.samples)

    def augmentation(self, frame, transform, state):
        torch.set_rng_state(state)
        return transform(frame)
    
    def get_samples_old(self, split='train'):
        scenes = self.splits[split]
        samples = [samp for samp in self.nusc.sample if
                   self.nusc.get('scene', samp['scene_token'])['name'] in scenes]
        return samples

    # collect all frames for all scenes
    def get_samples(self, split='train'):
        selected_scenes = self.splits[split] # all scenes
        all_scenes = self.nusc.scene
        selected_scenes_meta = []
        for sce in all_scenes:
            if sce['name'] in selected_scenes:
                selected_scenes_meta.append(sce)
        
        samples_group_by_scene = []
        for scene in selected_scenes_meta:
            samples_group_by_scene.append(
                {
                'scene': scene['name'],
                'video': self.get_all_frames_from_scene(scene)
                }
            )
        
        return samples_group_by_scene
    
    def group_sample_by_scene(self, split='train'):
        scenes = self.splits[split]
        samples_dict = {}
        for sce in scenes:
            samples_dict[sce] = [] # empty list
        for samp in self.nusc.sample:
            scene_token = samp['scene_token']
            scene = self.nusc.get('scene', scene_token)
            tmp_sce = scene['name']
            if tmp_sce in scenes:
                samples_dict[tmp_sce].append(samp)
        return samples_dict
    
    def get_image_path_from_sample(self, my_sample):
        sample_data = self.nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
        file_path = sample_data['filename']
        image_path = os.path.join(self.data_root, file_path)
        return image_path
    
    def get_paths_from_scene(self, my_scene):
        samples = self.samples_groups.get(my_scene, [])
        paths = [self.get_image_path_from_sample(sam) for sam in samples]
        paths.sort()
        return paths
    
    def get_all_frames_from_scene(self, scene):
        # get all frames (keyframes, sweeps)
        first_sample_token = scene['first_sample_token']
        my_sample = self.nusc.get('sample', first_sample_token)
        sensor = "CAM_FRONT"
        cam_front_data = self.nusc.get('sample_data', my_sample['data'][sensor]) # first frame sensor token
        # frames = 0
        all_frames_dict = [] # len() -> frame number
        while True:
            all_frames_dict.append(cam_front_data)
            # filename = cam_front_data['filename']  # current-frame filename
            next_sample_data_token = cam_front_data['next']  # next-frame sensor token
            if not next_sample_data_token: # ''
                break
            cam_front_data = self.nusc.get('sample_data', next_sample_data_token)
        
        return all_frames_dict
    
    def load_sample(self, index):
        my_scene = self.samples[index]
        my_scene_name = my_scene['scene']
        my_sample_list = my_scene["video"]
        inner_frame_len = len(my_sample_list)
        # print(f'{my_scene_name}: {inner_frame_len}')
        key_frame_idx = []
        for seek_idx in range(inner_frame_len):
            if my_sample_list[seek_idx]['is_key_frame']:
                key_frame_idx.append(seek_idx)
        
        seek_sample = None
        while seek_sample is None or not seek_sample['is_key_frame']:
            seek_start = random.randint(0, inner_frame_len - 51)
            seek_sample = my_sample_list[seek_start]
        
        seek_samples = my_sample_list[seek_start:]
        keyframe_count = 0
        selected_samples = []
        
        for i in range(0, len(seek_samples)):
            selected_samples.append(seek_samples[i])
            if seek_samples[i]['is_key_frame']:
                keyframe_count += 1
            # XXX hard code 8 key-frames
            if keyframe_count == 8:
                break
        
        filter_samples = []
        key_indices = [i for i, x in enumerate(selected_samples) if x['is_key_frame']]
        # print(key_indices)
        # import pdb; pdb.set_trace()
        for i in range(len(key_indices) - 1):
            start = key_indices[i]
            end = key_indices[i + 1]
            filter_samples.append(selected_samples[start])
            
            if i==0:
                #XXX 22 -> 23, add one
                filter_samples.append(selected_samples[start+1])
            
            non_key_section = selected_samples[start + 1:end]
            # if len(non_key_section) >= 5:
            mid1_index = len(non_key_section) // 3
            mid2_index = 2 * len(non_key_section) // 3
            filter_samples.append(non_key_section[mid1_index])
            filter_samples.append(non_key_section[mid2_index])
        
        filter_samples.append(selected_samples[key_indices[-1]])
        
        # the seek_sample is the random choosed key-frame
        frame_paths = []
        for my_sample in filter_samples:
            file_path = my_sample["filename"]
            image_path = os.path.join(self.data_root, file_path)
            frame_paths.append(image_path)

        scene_annotation = self.annotations[my_scene_name]
        
        # idxs belong to start index
        seek_key_frame_idx = key_frame_idx.index(seek_start)

        seek_id = seek_key_frame_idx//4*4
                
        search_id = f"{seek_id}"
        timestep = search_id if search_id in scene_annotation else str((int(search_id)//4-1)*4)

        # key-frame num <= 8, ok
        annotation = scene_annotation[timestep]
    
        caption = ""
        selected_attr = ["Weather", "Time", "Road environment", "Critical objects"]
        for attr in selected_attr:
            anno = annotation.get(attr, "")
            if anno == "":
                continue
            if anno[-1] == ".":
                anno = anno[:-1] 
            caption = caption + anno + ". "
        
        command_idx = str(seek_key_frame_idx)
        
        command_miss_scenes = ["scene-0161", "scene-0162", "scene-0163", "scene-0164",
            "scene-0165", "scene-0166", "scene-0167", "scene-0168", "scene-0170",
            "scene-0171", "scene-0172", "scene-0173", "scene-0174",
            "scene-0175", "scene-0176", "scene-0419"
            ]
        
        if my_scene_name in command_miss_scenes:
            driving_prompt = ""
        else:  
            try:
                driving_prompt = self.command_dict[my_scene_name][command_idx]
            except:
                # import pdb; pdb.set_trace()
                driving_prompt = ""
                pass
        driving_prompt = driving_prompt + ". " + caption
        
        frames = np.stack([image2arr(path) for path in frame_paths])

        selected_num_frames = frames.shape[0]

        assert (selected_num_frames - 1) % 4 == 0
        
        # NOTE aug, pil -> tensor
        state = torch.get_rng_state()
        frames = torch.stack([self.augmentation(v, self.transform, state) for v in frames], dim=0)

        if len(frames) != 23:
            raise ValueError(f"Error: 'label_imgs' length is {len(frames)}, but it must be 23.")

        return driving_prompt, frames

    def encode(self, prompt, video):
        prompt = self.encode_prompt(prompt).to("cpu")
        v1,v2 = self.encode_video(video)
        video = (v1.sample().to("cpu"),v2.sample().to("cpu"))
        return prompt, video

    def preload(self):
        self.instance_prompts = []
        self.instance_videos =[]

        progress_dataset_bar = tqdm(
            range(0, len(self.scenes)),
            desc="Loading prompts and videos",
        )

        for index in range(len(self.scenes)):
            progress_dataset_bar.update(1)

            prompt, video = self.load_sample(index)
            prompt, video = self.encode(prompt, video)

            self.instance_prompts.append(prompt)
            self.instance_videos.append(video)
        
    def __getitem__(self, index):
        try:
            if self.preload_all_data:
                return {
                    "instance_prompt": self.instance_prompts[index],
                    "instance_video": self.instance_videos[index],
                }
            else:
                prompt, video = self.load_sample(index)
                prompt, video = self.encode(prompt, video)
                return {
                    "instance_prompt": prompt,
                    "instance_video": video,
                }
        except Exception as e:
            print('Bad idx %s skipped because of %s' % (index, e))
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))



class NuscenesDatasetFPS1ForCogvidx(Dataset):
    def __init__(
        self,
        data_root: str = "/data/wangxd/nuscenes",
        height: int = 480,
        width: int = 720,
        max_num_frames: int = 13, # 8 * N + 1 (where N ≤ 6)
        split: str = "train",
        encode_prompt = None,
        encode_video = None,
        max_samples: int = 700,
        preload_all_data: bool = False,
        preprocessed_data_path = "/root/autodl-fs/dataset_700_samples_fix"
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.height = height
        self.width = width
        self.max_num_frames = max_num_frames
        self.split = split
        self.encode_prompt=encode_prompt
        self.encode_video=encode_video
        self.preload_all_data = preload_all_data
        self.preprocessed_data_path = preprocessed_data_path

        if preload_all_data and preprocessed_data_path is not None:
            pass
        else:

            self.nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_root, verbose=True)

            self.splits = create_splits_scenes()

            # training samples
            self.samples_groups = self.group_sample_by_scene(split)
            
            self.scenes = list(self.samples_groups.keys())

            if len(self.scenes) > max_samples:
                self.scenes = self.scenes[:max_samples]
            
            self.frames_group = {} # (scene, image_paths)
            
            for my_scene in self.scenes:
                f = self.get_paths_from_scene(my_scene)
                if len(f) >= max_num_frames :
                    self.frames_group[my_scene] = self.get_paths_from_scene(my_scene)
            self.scenes = [k for k,v in self.frames_group.items()]

            print('Total samples: %d' % len(self.scenes))

            # search annotations
            json_path = f'{data_root}/nusc_video_{split}_8_ov-7b_dict.json'
            with open(json_path, 'r') as f:
                self.annotations = json.load(f)
            
            command_path = f'{data_root}/nusc_action_26keyframe_all_{split}.json'
            with open(command_path, 'r') as f:
                self.command_dict = json.load(f)
            
            # image transform
            self.transform = TT.Compose([
                    TT.ToPILImage('RGB'),
                    TT.RandomResizedCrop((height, width), scale=(0.5, 1.), ratio=(1., 1.75)),
                    # transforms.RandomHorizontalFlip(p=0.1),
                    TT.ColorJitter(brightness=0.05, contrast=0.15, saturation=0.15),
                    TT.ToTensor(),
                    TT.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True),
                ])
            
            
            
            if preload_all_data:
                self.preload()

    
    def __len__(self):
        return len(self.scenes)

    def augmentation(self, frame, transform, state):
        torch.set_rng_state(state)
        return transform(frame)
    
    def get_samples(self, split='train'):
        scenes = self.splits[split]
        samples = [samp for samp in self.nusc.sample if
                   self.nusc.get('scene', samp['scene_token'])['name'] in scenes]
        return samples
    
    def group_sample_by_scene(self, split='train'):
        scenes = self.splits[split]
        samples_dict = {}
        for sce in scenes:
            samples_dict[sce] = [] # empty list
        for samp in self.nusc.sample:
            scene_token = samp['scene_token']
            scene = self.nusc.get('scene', scene_token)
            tmp_sce = scene['name']
            if tmp_sce in scenes:
                samples_dict[tmp_sce].append(samp)
        return samples_dict
    
    def get_image_path_from_sample(self, my_sample):
        sample_data = self.nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
        file_path = sample_data['filename']
        image_path = os.path.join(self.data_root, file_path)
        return image_path
    
    def get_paths_from_scene(self, my_scene):
        samples = self.samples_groups.get(my_scene, [])
        paths = [self.get_image_path_from_sample(sam) for sam in samples]
        paths.sort()
        return paths
    
    def load_sample(self, index):
        # my_scene = self.samples[index]
        # my_scene_name = my_scene['scene']
        # my_sample_list = my_scene["video"]
        # inner_frame_len = len(my_sample_list)
        
        my_scene_name = self.scenes[index]
        all_frames = self.frames_group[my_scene_name]
        
        # suppose 13*2
        seek_start = random.randint(0, len(all_frames) - self.max_num_frames*2)
        seek_path = all_frames[seek_start: seek_start + self.max_num_frames*2]
        
        seek_path = seek_path[::2] # 2 fps -> 1 fps

        scene_annotation = self.annotations[my_scene_name]
        
        seek_key_frame_idx = seek_start

        seek_id = seek_key_frame_idx//4*4
                
        search_id = f"{seek_id}"
        timestep = search_id if search_id in scene_annotation else str((int(search_id)//4-1)*4)

        annotation = scene_annotation[timestep]
        
        caption = ""
        selected_attr = ["Weather", "Time", "Road environment", "Critical objects"]
        for attr in selected_attr:
            anno = annotation.get(attr, "")
            if anno == "":
                continue
            if anno[-1] == ".":
                anno = anno[:-1] 
            caption = caption + anno + ". "
        
        # NOTE add more detailed captions
        addtional_timestep = [10, 20]
        for add_timestep in addtional_timestep:
            a_timestep = seek_key_frame_idx//4*4 + add_timestep
            a_timestep = f"{a_timestep}"
            a_timestep = a_timestep if a_timestep in scene_annotation else str((int(a_timestep)//4-1)*4)
            annotation = scene_annotation[a_timestep]
            selected_attr = ["Road environment", "Critical objects"]
            caption = caption + "The scene then changes: "
            for attr in selected_attr:
                anno = annotation.get(attr, "")
                if anno == "":
                    continue
                if anno[-1] == ".":
                    anno = anno[:-1] 
                caption = caption + anno + ". "
        
        command_idx = str(seek_key_frame_idx) # maybe 0-31, each denote 8 keyframes
        
        # import pdb; pdb.set_trace()
        
        command_miss_scenes = ["scene-0161", "scene-0162", "scene-0163", "scene-0164",
            "scene-0165", "scene-0166", "scene-0167", "scene-0168", "scene-0170",
            "scene-0171", "scene-0172", "scene-0173", "scene-0174",
            "scene-0175", "scene-0176", "scene-0419"
            ]
        
        if my_scene_name in command_miss_scenes:
            driving_prompt = ""
        else:  
            try:
                driving_prompt = self.command_dict[my_scene_name][command_idx]
            except:
                # import pdb; pdb.set_trace()
                driving_prompt = ""
                pass
        driving_prompt = driving_prompt + ". " + caption
        
        # import pdb; pdb.set_trace()
        # print(driving_prompt)

        # frames = torch.Tensor(np.stack([image2arr(path) for path in frame_paths]))
        frames = np.stack([image2arr(path) for path in seek_path])

        selected_num_frames = frames.shape[0]

        assert (selected_num_frames - 1) % 4 == 0

        # Training transforms
        
        # no aug
        # frames = (frames - 127.5) / 127.5
        # frames = frames.permute(0, 3, 1, 2) # [F, C, H, W]
        # frames = resize(frames,size=[self.height, self.width],interpolation=InterpolationMode.BICUBIC)
        # NOTE aug, pil -> tensor
        state = torch.get_rng_state()
        frames = torch.stack([self.augmentation(v, self.transform, state) for v in frames], dim=0)

        return driving_prompt, frames

    def encode(self, prompt, video):
        prompt = self.encode_prompt(prompt).to("cpu")
        v1,v2 = self.encode_video(video)
        video = (v1.sample().to("cpu"),v2.sample().to("cpu"))
        return prompt, video

    def preload(self):
        self.instance_prompts = []
        self.instance_videos =[]

        progress_dataset_bar = tqdm(
            range(0, len(self.scenes)),
            desc="Loading prompts and videos",
        )

        for index in range(len(self.scenes)):
            progress_dataset_bar.update(1)

            prompt, video = self.load_sample(index)
            prompt, video = self.encode(prompt, video)

            self.instance_prompts.append(prompt)
            self.instance_videos.append(video)
        
    def __getitem__(self, index):
        try:
            if self.preload_all_data:
                return {
                    "instance_prompt": self.instance_prompts[index],
                    "instance_video": self.instance_videos[index],
                }
            else:
                prompt, video = self.load_sample(index)
                prompt, video = self.encode(prompt, video)
                return {
                    "instance_prompt": prompt,
                    "instance_video": video,
                }
        except Exception as e:
            print('Bad idx %s skipped because of %s' % (index, e))
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))



class NuscenesDatasetAllframesFPS10ForCogvidx(Dataset):
    def __init__(
        self,
        data_root: str = "/data/wangxd/nuscenes",
        height: int = 480,
        width: int = 720,
        max_num_frames: int = 13, # 8 * N + 1 (where N ≤ 6)
        split: str = "train",
        encode_prompt = None,
        encode_video = None,
        max_samples: int = 700,
        preload_all_data: bool = False,
        preprocessed_data_path = "/root/autodl-fs/dataset_700_samples_fix"
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.height = height
        self.width = width
        self.max_num_frames = max_num_frames
        self.split = split
        self.encode_prompt=encode_prompt
        self.encode_video=encode_video
        self.preload_all_data = preload_all_data
        self.preprocessed_data_path = preprocessed_data_path

        if preload_all_data and preprocessed_data_path is not None:
            pass
        else:

            self.nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_root, verbose=True)

            self.splits = create_splits_scenes()

            # training samples
            self.samples = self.get_samples(split)

            print('Total samples: %d' % len(self.samples))

            # search annotations
            json_path = f'{data_root}/nusc_video_{split}_8_ov-7b_dict.json'
            with open(json_path, 'r') as f:
                self.annotations = json.load(f)
            
            command_path = f'{data_root}/nusc_action_{split}.json'
            with open(command_path, 'r') as f:
                self.command_dict = json.load(f)
            
            # image transform
            self.transform = TT.Compose([
                    TT.ToPILImage('RGB'),
                    TT.RandomResizedCrop((height, width), scale=(0.5, 1.), ratio=(1., 1.75)),
                    # transforms.RandomHorizontalFlip(p=0.1),
                    TT.ColorJitter(brightness=0.05, contrast=0.15, saturation=0.15),
                    TT.ToTensor(),
                    TT.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True),
                ])
            
            if preload_all_data:
                self.preload()

    
    def __len__(self):
        return len(self.samples)

    def augmentation(self, frame, transform, state):
        torch.set_rng_state(state)
        return transform(frame)
    
    def get_samples(self, split='train'):
        selected_scenes = self.splits[split] # all scenes
        all_scenes = self.nusc.scene
        selected_scenes_meta = []
        for sce in all_scenes:
            if sce['name'] in selected_scenes:
                selected_scenes_meta.append(sce)
        
        samples_group_by_scene = []
        for scene in selected_scenes_meta:
            samples_group_by_scene.append(
                {
                'scene': scene['name'],
                'video': self.get_all_frames_from_scene(scene)
                }
            )
        
        return samples_group_by_scene

    def get_all_frames_from_scene(self, scene):
        # get all frames (keyframes, sweeps)
        first_sample_token = scene['first_sample_token']
        my_sample = self.nusc.get('sample', first_sample_token)
        sensor = "CAM_FRONT"
        cam_front_data = self.nusc.get('sample_data', my_sample['data'][sensor]) # first frame sensor token
        # frames = 0
        all_frames_dict = [] # len() -> frame number
        while True:
            all_frames_dict.append(cam_front_data)
            # filename = cam_front_data['filename']  # current-frame filename
            next_sample_data_token = cam_front_data['next']  # next-frame sensor token
            if not next_sample_data_token: # ''
                break
            cam_front_data = self.nusc.get('sample_data', next_sample_data_token)
        
        return all_frames_dict
    
    def group_sample_by_scene(self, split='train'):
        scenes = self.splits[split]
        samples_dict = {}
        for sce in scenes:
            samples_dict[sce] = [] # empty list
        for samp in self.nusc.sample:
            scene_token = samp['scene_token']
            scene = self.nusc.get('scene', scene_token)
            tmp_sce = scene['name']
            if tmp_sce in scenes:
                samples_dict[tmp_sce].append(samp)
        return samples_dict
    
    def get_image_path_from_sample(self, my_sample):
        sample_data = self.nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
        file_path = sample_data['filename']
        image_path = os.path.join(self.data_root, file_path)
        return image_path
    
    def get_paths_from_scene(self, my_scene):
        samples = self.samples_groups.get(my_scene, [])
        paths = [self.get_image_path_from_sample(sam) for sam in samples]
        paths.sort()
        return paths
    
    def load_sample(self, index):
        my_scene = self.samples[index]
        my_scene_name = my_scene['scene']
        my_sample_list = my_scene["video"]
        inner_frame_len = len(my_sample_list)
        
        key_frame_idx = []
        for seek_idx in range(inner_frame_len):
            if my_sample_list[seek_idx]['is_key_frame']:
                key_frame_idx.append(seek_idx)
        
        seek_sample = None
        while seek_sample is None or not seek_sample['is_key_frame']:
            seek_start = random.randint(0, inner_frame_len - 51)
            seek_sample = my_sample_list[seek_start]

        seek_samples = my_sample_list[seek_start: seek_start + self.max_num_frames]
        
        # all frames
        frame_paths = []
        for my_sample in seek_samples:
            file_path = my_sample["filename"]
            image_path = os.path.join(self.data_root, file_path)
            frame_paths.append(image_path)

        scene_annotation = self.annotations[my_scene_name]
        
        seek_key_frame_idx = key_frame_idx.index(seek_start)

        seek_id = seek_key_frame_idx//4*4
                
        search_id = f"{seek_id}"
        timestep = search_id if search_id in scene_annotation else str((int(search_id)//4-1)*4)

        annotation = scene_annotation[timestep]
        
        caption = ""
        selected_attr = ["Weather", "Time", "Road environment", "Critical objects"]
        for attr in selected_attr:
            anno = annotation.get(attr, "")
            if anno == "":
                continue
            if anno[-1] == ".":
                anno = anno[:-1] 
            caption = caption + anno + ". "
        
        command_idx = str(seek_key_frame_idx)
        
        # import pdb; pdb.set_trace()
        
        command_miss_scenes = ["scene-0161", "scene-0162", "scene-0163", "scene-0164",
            "scene-0165", "scene-0166", "scene-0167", "scene-0168", "scene-0170",
            "scene-0171", "scene-0172", "scene-0173", "scene-0174",
            "scene-0175", "scene-0176", "scene-0419"
            ]
        
        if my_scene_name in command_miss_scenes:
            driving_prompt = ""
        else:  
            try:
                driving_prompt = self.command_dict[my_scene_name][command_idx]
            except:
                # import pdb; pdb.set_trace()
                driving_prompt = ""
                pass
        driving_prompt = driving_prompt + ". " + caption
        # print(driving_prompt)

        # frames = torch.Tensor(np.stack([image2arr(path) for path in frame_paths]))
        frames = np.stack([image2arr(path) for path in frame_paths])

        selected_num_frames = frames.shape[0]

        assert (selected_num_frames - 1) % 4 == 0

        # Training transforms
        
        # no aug
        # frames = (frames - 127.5) / 127.5
        # frames = frames.permute(0, 3, 1, 2) # [F, C, H, W]
        # frames = resize(frames,size=[self.height, self.width],interpolation=InterpolationMode.BICUBIC)
        # NOTE aug, pil -> tensor
        state = torch.get_rng_state()
        frames = torch.stack([self.augmentation(v, self.transform, state) for v in frames], dim=0)

        return driving_prompt, frames

    def encode(self, prompt, video):
        prompt = self.encode_prompt(prompt).to("cpu")
        v1, v2, v3 = self.encode_video(video)
        video = (v1.sample().to("cpu"), v2.sample().to("cpu"), v3.sample().to("cpu"))
        return prompt, video

    def preload(self):
        self.instance_prompts = []
        self.instance_videos =[]

        progress_dataset_bar = tqdm(
            range(0, len(self.scenes)),
            desc="Loading prompts and videos",
        )

        for index in range(len(self.scenes)):
            progress_dataset_bar.update(1)

            prompt, video = self.load_sample(index)
            prompt, video = self.encode(prompt, video)

            self.instance_prompts.append(prompt)
            self.instance_videos.append(video)
        
    def __getitem__(self, index):
        try:
            if self.preload_all_data:
                return {
                    "instance_prompt": self.instance_prompts[index],
                    "instance_video": self.instance_videos[index],
                }
            else:
                prompt, video = self.load_sample(index)
                prompt, video = self.encode(prompt, video)
                return {
                    "instance_prompt": prompt,
                    "instance_video": video,
                }
        except Exception as e:
            print('Bad idx %s skipped because of %s' % (index, e))
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))



class NuscenesDatasetKeyframesOneByOneForCogvidx(Dataset):
    def __init__(
        self,
        data_root: str = "/data/wangxd/nuscenes",
        height: int = 480,
        width: int = 720,
        max_num_frames: int = 49, # 8 * N + 1 (where N ≤ 6)
        split: str = "train",
        encode_prompt = None,
        encode_video = None,
        max_samples: int = 700,
        preload_all_data: bool = False,
        preprocessed_data_path = "/root/autodl-fs/dataset_700_samples_fix"
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.height = height
        self.width = width
        self.max_num_frames = max_num_frames
        self.split = split
        self.encode_prompt=encode_prompt
        self.encode_video=encode_video
        self.preload_all_data = preload_all_data
        self.preprocessed_data_path = preprocessed_data_path

        if preload_all_data and preprocessed_data_path is not None:
            pass
        else:

            self.nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_root, verbose=True)

            self.splits = create_splits_scenes()

            # training samples
            self.samples_groups = self.group_sample_by_scene(split)
            
            self.scenes = list(self.samples_groups.keys())

            if len(self.scenes) > max_samples:
                self.scenes = self.scenes[:max_samples]
            
            self.frames_group = {} # (scene, image_paths)
            
            for my_scene in self.scenes:
                f = self.get_paths_from_scene(my_scene)
                if len(f) >= max_num_frames :
                    self.frames_group[my_scene] = self.get_paths_from_scene(my_scene)
            self.scenes = [k for k,v in self.frames_group.items()]

            print('Total samples: %d' % len(self.scenes))

            # search annotations
            json_path = f'{data_root}/nusc_video_{split}_8_ov-7b_dict.json'
            with open(json_path, 'r') as f:
                self.annotations = json.load(f)
            
            command_path = f'{data_root}/nusc_action_{split}.json'
            with open(command_path, 'r') as f:
                self.command_dict = json.load(f)
            
            # image transform
            self.transform = TT.Compose([
                    TT.ToPILImage('RGB'),
                    TT.RandomResizedCrop((height, width), scale=(0.5, 1.), ratio=(1., 1.75)),
                    # transforms.RandomHorizontalFlip(p=0.1),
                    TT.ColorJitter(brightness=0.05, contrast=0.15, saturation=0.15),
                    TT.ToTensor(),
                    TT.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True),
                ])
            
            
            
            if preload_all_data:
                self.preload()

    
    def __len__(self):
        return len(self.scenes)

    def augmentation(self, frame, transform, state):
        torch.set_rng_state(state)
        return transform(frame)
    
    def get_samples(self, split='train'):
        scenes = self.splits[split]
        samples = [samp for samp in self.nusc.sample if
                   self.nusc.get('scene', samp['scene_token'])['name'] in scenes]
        return samples
    
    def group_sample_by_scene(self, split='train'):
        scenes = self.splits[split]
        samples_dict = {}
        for sce in scenes:
            samples_dict[sce] = [] # empty list
        for samp in self.nusc.sample:
            scene_token = samp['scene_token']
            scene = self.nusc.get('scene', scene_token)
            tmp_sce = scene['name']
            if tmp_sce in scenes:
                samples_dict[tmp_sce].append(samp)
        return samples_dict
    
    def get_image_path_from_sample(self, my_sample):
        sample_data = self.nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
        file_path = sample_data['filename']
        image_path = os.path.join(self.data_root, file_path)
        return image_path
    
    def get_paths_from_scene(self, my_scene):
        samples = self.samples_groups.get(my_scene, [])
        paths = [self.get_image_path_from_sample(sam) for sam in samples]
        paths.sort()
        return paths
    
    def load_sample(self, index):
        # my_scene = self.samples[index]
        # my_scene_name = my_scene['scene']
        # my_sample_list = my_scene["video"]
        # inner_frame_len = len(my_sample_list)
        
        my_scene_name = self.scenes[index]
        all_frames = self.frames_group[my_scene_name]
        
        seek_start = random.randint(0, len(all_frames) - self.max_num_frames)
        seek_path = all_frames[seek_start: seek_start + self.max_num_frames]

        scene_annotation = self.annotations[my_scene_name]
        
        seek_key_frame_idx = seek_start

        seek_id = seek_key_frame_idx//4*4
                
        search_id = f"{seek_id}"
        timestep = search_id if search_id in scene_annotation else str((int(search_id)//4-1)*4)

        annotation = scene_annotation[timestep]
        
        caption = ""
        selected_attr = ["Weather", "Time", "Road environment", "Critical objects"]
        for attr in selected_attr:
            anno = annotation.get(attr, "")
            if anno == "":
                continue
            if anno[-1] == ".":
                anno = anno[:-1] 
            caption = caption + anno + ". "
        
        command_idx = str(seek_key_frame_idx)
        
        # import pdb; pdb.set_trace()
        
        command_miss_scenes = ["scene-0161", "scene-0162", "scene-0163", "scene-0164",
            "scene-0165", "scene-0166", "scene-0167", "scene-0168", "scene-0170",
            "scene-0171", "scene-0172", "scene-0173", "scene-0174",
            "scene-0175", "scene-0176", "scene-0419"
            ]
        
        if my_scene_name in command_miss_scenes:
            driving_prompt = ""
        else:  
            try:
                driving_prompt = self.command_dict[my_scene_name][command_idx]
            except:
                # import pdb; pdb.set_trace()
                driving_prompt = ""
                pass
        driving_prompt = driving_prompt + ". " + caption
        # print(driving_prompt)

        # frames = torch.Tensor(np.stack([image2arr(path) for path in frame_paths]))
        frames = np.stack([image2arr(path) for path in seek_path])

        selected_num_frames = frames.shape[0]

        assert (selected_num_frames - 1) % 4 == 0

        # Training transforms
        
        # no aug
        # frames = (frames - 127.5) / 127.5
        # frames = frames.permute(0, 3, 1, 2) # [F, C, H, W]
        # frames = resize(frames,size=[self.height, self.width],interpolation=InterpolationMode.BICUBIC)
        # NOTE aug, pil -> tensor
        state = torch.get_rng_state()
        frames = torch.stack([self.augmentation(v, self.transform, state) for v in frames], dim=0)

        return driving_prompt, frames

    def encode(self, prompt, video):
        prompt = self.encode_prompt(prompt).to("cpu")
        v1,v2 = self.encode_video(video)
        video = (v1,v2.sample().to("cpu"))
        return prompt, video

    def preload(self):
        self.instance_prompts = []
        self.instance_videos =[]

        progress_dataset_bar = tqdm(
            range(0, len(self.scenes)),
            desc="Loading prompts and videos",
        )

        for index in range(len(self.scenes)):
            progress_dataset_bar.update(1)

            prompt, video = self.load_sample(index)
            prompt, video = self.encode(prompt, video)

            self.instance_prompts.append(prompt)
            self.instance_videos.append(video)
        
    def __getitem__(self, index):
        try:
            if self.preload_all_data:
                return {
                    "instance_prompt": self.instance_prompts[index],
                    "instance_video": self.instance_videos[index],
                }
            else:
                prompt, video = self.load_sample(index)
                prompt, video = self.encode(prompt, video)
                return {
                    "instance_prompt": prompt,
                    "instance_video": video,
                }
        except Exception as e:
            print('Bad idx %s skipped because of %s' % (index, e))
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))



class NuscenesDatasetFPS1OneByOneForCogvidx(Dataset):
    def __init__(
        self,
        data_root: str = "/data/wangxd/nuscenes",
        height: int = 480,
        width: int = 720,
        max_num_frames: int = 13, # 8 * N + 1 (where N ≤ 6)
        split: str = "train",
        encode_prompt = None,
        encode_video = None,
        max_samples: int = 700,
        preload_all_data: bool = False,
        preprocessed_data_path = "/root/autodl-fs/dataset_700_samples_fix"
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.height = height
        self.width = width
        self.max_num_frames = max_num_frames
        self.split = split
        self.encode_prompt=encode_prompt
        self.encode_video=encode_video
        self.preload_all_data = preload_all_data
        self.preprocessed_data_path = preprocessed_data_path

        if preload_all_data and preprocessed_data_path is not None:
            pass
        else:

            self.nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_root, verbose=True)

            self.splits = create_splits_scenes()

            # training samples
            self.samples_groups = self.group_sample_by_scene(split)
            
            self.scenes = list(self.samples_groups.keys())

            if len(self.scenes) > max_samples:
                self.scenes = self.scenes[:max_samples]
            
            self.frames_group = {} # (scene, image_paths)
            
            for my_scene in self.scenes:
                f = self.get_paths_from_scene(my_scene)
                if len(f) >= max_num_frames :
                    self.frames_group[my_scene] = self.get_paths_from_scene(my_scene)
            self.scenes = [k for k,v in self.frames_group.items()]

            print('Total samples: %d' % len(self.scenes))

            # search annotations
            json_path = f'{data_root}/nusc_video_{split}_8_ov-7b_dict.json'
            with open(json_path, 'r') as f:
                self.annotations = json.load(f)
            
            command_path = f'{data_root}/nusc_action_26keyframe_all_{split}.json'
            with open(command_path, 'r') as f:
                self.command_dict = json.load(f)
            
            # image transform
            self.transform = TT.Compose([
                    TT.ToPILImage('RGB'),
                    TT.RandomResizedCrop((height, width), scale=(0.5, 1.), ratio=(1., 1.75)),
                    # transforms.RandomHorizontalFlip(p=0.1),
                    TT.ColorJitter(brightness=0.05, contrast=0.15, saturation=0.15),
                    TT.ToTensor(),
                    TT.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True),
                ])
            
            
            
            if preload_all_data:
                self.preload()

    
    def __len__(self):
        return len(self.scenes)

    def augmentation(self, frame, transform, state):
        torch.set_rng_state(state)
        return transform(frame)
    
    def get_samples(self, split='train'):
        scenes = self.splits[split]
        samples = [samp for samp in self.nusc.sample if
                   self.nusc.get('scene', samp['scene_token'])['name'] in scenes]
        return samples
    
    def group_sample_by_scene(self, split='train'):
        scenes = self.splits[split]
        samples_dict = {}
        for sce in scenes:
            samples_dict[sce] = [] # empty list
        for samp in self.nusc.sample:
            scene_token = samp['scene_token']
            scene = self.nusc.get('scene', scene_token)
            tmp_sce = scene['name']
            if tmp_sce in scenes:
                samples_dict[tmp_sce].append(samp)
        return samples_dict
    
    def get_image_path_from_sample(self, my_sample):
        sample_data = self.nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
        file_path = sample_data['filename']
        image_path = os.path.join(self.data_root, file_path)
        return image_path
    
    def get_paths_from_scene(self, my_scene):
        samples = self.samples_groups.get(my_scene, [])
        paths = [self.get_image_path_from_sample(sam) for sam in samples]
        paths.sort()
        return paths
    
    def load_sample(self, index):
        # my_scene = self.samples[index]
        # my_scene_name = my_scene['scene']
        # my_sample_list = my_scene["video"]
        # inner_frame_len = len(my_sample_list)
        
        my_scene_name = self.scenes[index]
        all_frames = self.frames_group[my_scene_name]
        
        # suppose 13*2
        seek_start = random.randint(0, len(all_frames) - self.max_num_frames*2)
        seek_path = all_frames[seek_start: seek_start + self.max_num_frames*2]
        
        seek_path = seek_path[::2] # 2 fps -> 1 fps

        scene_annotation = self.annotations[my_scene_name]
        
        seek_key_frame_idx = seek_start

        seek_id = seek_key_frame_idx//4*4
                
        search_id = f"{seek_id}"
        timestep = search_id if search_id in scene_annotation else str((int(search_id)//4-1)*4)

        annotation = scene_annotation[timestep]
        
        caption = ""
        selected_attr = ["Weather", "Time", "Road environment", "Critical objects"]
        for attr in selected_attr:
            anno = annotation.get(attr, "")
            if anno == "":
                continue
            if anno[-1] == ".":
                anno = anno[:-1] 
            caption = caption + anno + ". "
        
        # NOTE add more detailed captions
        addtional_timestep = [10, 20]
        for add_timestep in addtional_timestep:
            a_timestep = seek_key_frame_idx//4*4 + add_timestep
            a_timestep = f"{a_timestep}"
            a_timestep = a_timestep if a_timestep in scene_annotation else str((int(a_timestep)//4-1)*4)
            annotation = scene_annotation[a_timestep]
            selected_attr = ["Road environment", "Critical objects"]
            caption = caption + "The scene then changes: "
            for attr in selected_attr:
                anno = annotation.get(attr, "")
                if anno == "":
                    continue
                if anno[-1] == ".":
                    anno = anno[:-1] 
                caption = caption + anno + ". "
        
        command_idx = str(seek_key_frame_idx) # maybe 0-31, each denote 8 keyframes
        
        # import pdb; pdb.set_trace()
        
        command_miss_scenes = ["scene-0161", "scene-0162", "scene-0163", "scene-0164",
            "scene-0165", "scene-0166", "scene-0167", "scene-0168", "scene-0170",
            "scene-0171", "scene-0172", "scene-0173", "scene-0174",
            "scene-0175", "scene-0176", "scene-0419"
            ]
        
        if my_scene_name in command_miss_scenes:
            driving_prompt = ""
        else:  
            try:
                driving_prompt = self.command_dict[my_scene_name][command_idx]
            except:
                # import pdb; pdb.set_trace()
                driving_prompt = ""
                pass
        driving_prompt = driving_prompt + ". " + caption
        
        # import pdb; pdb.set_trace()
        # print(driving_prompt)

        # frames = torch.Tensor(np.stack([image2arr(path) for path in frame_paths]))
        frames = np.stack([image2arr(path) for path in seek_path])

        selected_num_frames = frames.shape[0]

        assert (selected_num_frames - 1) % 4 == 0

        # Training transforms
        
        # no aug
        # frames = (frames - 127.5) / 127.5
        # frames = frames.permute(0, 3, 1, 2) # [F, C, H, W]
        # frames = resize(frames,size=[self.height, self.width],interpolation=InterpolationMode.BICUBIC)
        # NOTE aug, pil -> tensor
        state = torch.get_rng_state()
        frames = torch.stack([self.augmentation(v, self.transform, state) for v in frames], dim=0)

        return driving_prompt, frames

    def encode(self, prompt, video):
        prompt = self.encode_prompt(prompt).to("cpu")
        v1,v2 = self.encode_video(video)
        video = (v1,v2.sample().to("cpu"))
        return prompt, video

    def preload(self):
        self.instance_prompts = []
        self.instance_videos =[]

        progress_dataset_bar = tqdm(
            range(0, len(self.scenes)),
            desc="Loading prompts and videos",
        )

        for index in range(len(self.scenes)):
            progress_dataset_bar.update(1)

            prompt, video = self.load_sample(index)
            prompt, video = self.encode(prompt, video)

            self.instance_prompts.append(prompt)
            self.instance_videos.append(video)
        
    def __getitem__(self, index):
        try:
            if self.preload_all_data:
                return {
                    "instance_prompt": self.instance_prompts[index],
                    "instance_video": self.instance_videos[index],
                }
            else:
                prompt, video = self.load_sample(index)
                prompt, video = self.encode(prompt, video)
                return {
                    "instance_prompt": prompt,
                    "instance_video": video,
                }
        except Exception as e:
            print('Bad idx %s skipped because of %s' % (index, e))
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))

def merge_dicts(dict1, dict2):
    merged = dict1.copy()
    for key, value in dict2.items():
        if key in merged:
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged

class OpenDVFPS2OForCogvidx(Dataset):
    def __init__(
        self,
        data_root: str = "/data/wuzhirong/datasets/OpenDV-mini/full_images",
        height: int = 480,
        width: int = 720,
        max_num_frames: int = 13, # 8 * N + 1 (where N ≤ 6)
        split: str = "train",
        encode_prompt = None,
        encode_video = None,
        max_samples: int = 700,
        preload_all_data: bool = False,
        preprocessed_data_path = ""
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.height = height
        self.width = width
        self.max_num_frames = max_num_frames
        self.split = split
        self.encode_prompt=encode_prompt
        self.encode_video=encode_video
        self.preload_all_data = preload_all_data
        self.preprocessed_data_path = preprocessed_data_path

        if preload_all_data and preprocessed_data_path is not None:
            pass
        else:

            # read metadata from jsonl
            self.meta_data = {}
            if split == "train":
                jsonl_file_paths = [
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split0_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split1_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split2_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split3_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split4_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split5_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split6_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split7_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split8_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split9_id.json",
                ]
            else:
                pass

            for file in jsonl_file_paths:
                with open(file, 'r', encoding='utf-8') as f:
                    tmp_dict = json.load(f)
                    self.meta_data = merge_dicts(self.meta_data, tmp_dict)
            
            self.annotation = {}
            jsonl_file_paths = [
                f"/home/user/wangxd/LLaVA-NeXT/OpenDV_{split}_16_ov-7b_0_5_id.json",
                f"/home/user/wangxd/LLaVA-NeXT/OpenDV_{split}_16_ov-7b_5_10_id.json",
                f"/home/user/wangxd/LLaVA-NeXT/OpenDV_{split}_16_ov-7b_10_15_id.json",
                f"/home/user/wangxd/LLaVA-NeXT/OpenDV_{split}_16_ov-7b_15_22_id.json",
            ]
            for file in jsonl_file_paths:
                with open(file, 'r', encoding='utf-8') as f:
                    tmp_dict = json.load(f)
                    self.annotation = merge_dicts(self.annotation, tmp_dict)
            
            # take the joint ids
            ids = self.meta_data.keys()
            anno_ids = self.annotation.keys()

            joint_ids = list(set(ids) - (set(ids) - set(anno_ids)))

            # only use joint ids
            ids = joint_ids

            self.key_pairs = []
            for id in ids:
                for k in self.meta_data[id].keys():
                    self.key_pairs.append((id, k))
            
            # according OpenDV cmd
            print('Total samples: %d' % len(self.key_pairs)) # 521977
            
            # image transform
            self.transform = TT.Compose([
                    TT.ToPILImage('RGB'),
                    TT.RandomResizedCrop((height, width), scale=(0.9, 1.), ratio=(1., 1.75)),
                    # transforms.RandomHorizontalFlip(p=0.1),
                    TT.ColorJitter(brightness=0.05, contrast=0.15, saturation=0.15),
                    TT.ToTensor(),
                    TT.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True),
                ])
            
            if preload_all_data:
                self.preload()
            
            self.factor = 100

    
    def __len__(self):
        return len(self.key_pairs) // self.factor

    def augmentation(self, frame, transform, state):
        torch.set_rng_state(state)
        return transform(frame)
    
    def load_sample(self, index):
        id, start_f10 = self.key_pairs[index]
        item = self.meta_data[id][start_f10]
        cmd = item["cmd"]
        first_f10, end_f10 = item["first"], item["last"]
        len_of_frames = first_f10 - end_f10 # 40 frames (fps10) -> 80 frames (fps20)

        first_f20 = first_f10 * 2 # 20 -> 40

        seek_idxs = range(first_f20, first_f20+self.max_num_frames) # 25 frames
        seek_paths = [str(x).zfill(9)+'.jpg' for x in seek_idxs]

        first_f20 = str(first_f20 // 40 * 40)
        video_root = self.annotation[id][first_f20]["video_root"] # sometimes will missing annotations
        seek_paths = [os.path.join(video_root, p) for p in seek_paths]
        
        # annotations
        
        annotation = self.annotation[id][first_f20] # start from 40x, str
        caption = ""
        selected_attr = ["Weather", "Time", "Road environment", "Critical objects"]
        for attr in selected_attr:
            anno = annotation.get(attr, "")
            if anno == "":
                continue
            if anno[-1] == ".":
                anno = anno[:-1] 
            caption = caption + anno + ". "
        
        # driving command
        driving_prompt = map_category_to_caption(cmd)
        driving_prompt = driving_prompt + " " + caption

        frames = np.stack([image2arr(path) for path in seek_paths])

        state = torch.get_rng_state()
        frames = torch.stack([self.augmentation(v, self.transform, state) for v in frames], dim=0)

        return driving_prompt, frames

    def encode(self, prompt, video):
        prompt = self.encode_prompt(prompt).to("cpu")
        v1, v2, v3 = self.encode_video(video)
        video = (v1.sample().to("cpu"), v2.sample().to("cpu"), v3.sample().to("cpu"))
        return prompt, video
    
    def __getitem__(self, index):
        try:
            start_idx = index * self.factor
            end_idx = min(start_idx + self.factor, len(self.key_pairs))
            sampled_idx = random.randint(start_idx, end_idx - 1)

            prompt, video = self.load_sample(sampled_idx)
            prompt, video = self.encode(prompt, video)
            return {
                "instance_prompt": prompt,
                "instance_video": video,
            }
        except Exception as e:
            print('Bad idx %s skipped because of %s' % (index, e))
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))


class OpenDVFPS1fbfForCogvidx(Dataset):
    def __init__(
        self,
        data_root: str = "/data/wuzhirong/datasets/OpenDV-mini/full_images",
        height: int = 480,
        width: int = 720,
        max_num_frames: int = 13, # 8 * N + 1 (where N ≤ 6)
        split: str = "train",
        encode_prompt = None,
        encode_video = None,
        max_samples: int = 700,
        preload_all_data: bool = False,
        preprocessed_data_path = ""
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.height = height
        self.width = width
        self.max_num_frames = max_num_frames
        self.split = split
        self.encode_prompt=encode_prompt
        self.encode_video=encode_video
        self.preload_all_data = preload_all_data
        self.preprocessed_data_path = preprocessed_data_path

        if preload_all_data and preprocessed_data_path is not None:
            pass
        else:

            # read metadata from jsonl
            self.meta_data = {}
            if split == "train":
                jsonl_file_paths = [
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split0_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split1_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split2_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split3_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split4_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split5_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split6_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split7_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split8_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split9_id.json",
                ]
            else:
                pass

            for file in jsonl_file_paths:
                with open(file, 'r', encoding='utf-8') as f:
                    tmp_dict = json.load(f)
                    self.meta_data = merge_dicts(self.meta_data, tmp_dict)
            
            self.annotation = {}
            jsonl_file_paths = [
                f"/home/user/wangxd/LLaVA-NeXT/OpenDV_{split}_16_ov-7b_0_5_id.json",
                f"/home/user/wangxd/LLaVA-NeXT/OpenDV_{split}_16_ov-7b_5_10_id.json",
                f"/home/user/wangxd/LLaVA-NeXT/OpenDV_{split}_16_ov-7b_10_15_id.json",
                f"/home/user/wangxd/LLaVA-NeXT/OpenDV_{split}_16_ov-7b_15_22_id.json",
            ]
            for file in jsonl_file_paths:
                with open(file, 'r', encoding='utf-8') as f:
                    tmp_dict = json.load(f)
                    self.annotation = merge_dicts(self.annotation, tmp_dict)
            
            # take the joint ids
            ids = self.meta_data.keys()
            anno_ids = self.annotation.keys()

            joint_ids = list(set(ids) - (set(ids) - set(anno_ids)))

            # only use joint ids
            ids = joint_ids

            self.key_pairs = []
            for id in ids:
                for k in self.meta_data[id].keys():
                    self.key_pairs.append((id, k))
            
            # according OpenDV cmd
            print('Total samples: %d' % len(self.key_pairs)) # 521977
            
            # image transform
            self.transform = TT.Compose([
                    TT.ToPILImage('RGB'),
                    TT.RandomResizedCrop((height, width), scale=(0.9, 1.), ratio=(1., 1.75)),
                    # transforms.RandomHorizontalFlip(p=0.1),
                    TT.ColorJitter(brightness=0.05, contrast=0.15, saturation=0.15),
                    TT.ToTensor(),
                    TT.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True),
                ])
            
            if preload_all_data:
                self.preload()
            
            self.factor = 200

    def __len__(self):
        return len(self.key_pairs) // self.factor

    def augmentation(self, frame, transform, state):
        torch.set_rng_state(state)
        return transform(frame)
    
    def load_sample(self, index):
        id, start_f10 = self.key_pairs[index]

        segments = [int(start_f10), int(start_f10)+40, int(start_f10)+80, int(start_f10)+120, int(start_f10)+160] # 5 * 4 = 20 frames
        segments = [str(x) for x in segments]

        total_cmds = []
        caption = ""
        total_seek_paths = []

        for sidx, seg_start in enumerate(segments):
            # 0-40 -> 
            item = self.meta_data[id][seg_start]

            cmd = item["cmd"]
            first_f10, end_f10 = item["first"], item["last"]
            len_of_frames = first_f10 - end_f10 # 40 frames (fps10) -> 80 frames (fps20)

            first_f20 = first_f10 * 2 # 20 -> 40

            seek_idxs = range(first_f20, first_f20 + 80) # 40 frames -> 80 frames

            seek_idxs = seek_idxs[::20] # 20 fps -> 1 fps, 4 frame

            seek_paths = [str(x).zfill(9)+'.jpg' for x in seek_idxs]

            first_f20 = str(first_f20 // 40 * 40)
            video_root = self.annotation[id][first_f20]["video_root"] # sometimes will missing annotations
            seek_paths = [os.path.join(video_root, p) for p in seek_paths]
            total_seek_paths.extend(seek_paths)
        
            # annotations
            # first and last
            if sidx==0 or sidx==4:
                annotation = self.annotation[id][first_f20] # start from 40x, str
                if sidx==4:
                    caption = caption + "The scene then changes: "
                selected_attr = ["Weather", "Time", "Road environment", "Critical objects"]
                for attr in selected_attr:
                    anno = annotation.get(attr, "")
                    if anno == "":
                        continue
                    if anno[-1] == ".":
                        anno = anno[:-1] 
                    caption = caption + anno + ". "
            
            # collect cmd
            if not total_cmds or cmd!=total_cmds[-1]:
                total_cmds.append(cmd)

        driving_prompt = None
        for cd in total_cmds:
            # driving command
            tmp_prompt = map_category_to_caption(cd)
            driving_prompt = tmp_prompt if driving_prompt is None else driving_prompt + "Then " + tmp_prompt + ". "

        global_prompt = driving_prompt + " " + caption

        frames = np.stack([image2arr(path) for path in total_seek_paths])

        state = torch.get_rng_state()
        frames = torch.stack([self.augmentation(v, self.transform, state) for v in frames], dim=0)

        return global_prompt, frames

    def encode(self, prompt, video):
        prompt = self.encode_prompt(prompt).to("cpu")
        v1, v2 = self.encode_video(video)
        video = (v1, v2)
        return prompt, video
    
    def __getitem__(self, index):
        try:
            start_idx = index * self.factor
            end_idx = min(start_idx + self.factor, len(self.key_pairs))
            sampled_idx = random.randint(start_idx, end_idx - 1)

            prompt, video = self.load_sample(sampled_idx)
            prompt, video = self.encode(prompt, video)
            return {
                "instance_prompt": prompt,
                "instance_video": video,
            }
        except Exception as e:
            print('Bad idx %s skipped because of %s' % (index, e))
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))




class OpenDVFPS5fbfForCogvidx(Dataset):
    def __init__(
        self,
        data_root: str = "/data/wuzhirong/datasets/OpenDV-mini/full_images",
        height: int = 480,
        width: int = 720,
        max_num_frames: int = 13, # 8 * N + 1 (where N ≤ 6)
        split: str = "train",
        encode_prompt = None,
        encode_video = None,
        max_samples: int = 700,
        preload_all_data: bool = False,
        preprocessed_data_path = ""
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.height = height
        self.width = width
        self.max_num_frames = max_num_frames
        self.split = split
        self.encode_prompt=encode_prompt
        self.encode_video=encode_video
        self.preload_all_data = preload_all_data
        self.preprocessed_data_path = preprocessed_data_path

        if preload_all_data and preprocessed_data_path is not None:
            pass
        else:

            # read metadata from jsonl
            self.meta_data = {}
            if split == "train":
                jsonl_file_paths = [
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split0_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split1_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split2_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split3_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split4_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split5_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split6_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split7_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split8_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split9_id.json",
                ]
            else:
                pass

            for file in jsonl_file_paths:
                with open(file, 'r', encoding='utf-8') as f:
                    tmp_dict = json.load(f)
                    self.meta_data = merge_dicts(self.meta_data, tmp_dict)
            
            self.annotation = {}
            jsonl_file_paths = [
                f"/home/user/wangxd/LLaVA-NeXT/OpenDV_{split}_16_ov-7b_0_5_id.json",
                f"/home/user/wangxd/LLaVA-NeXT/OpenDV_{split}_16_ov-7b_5_10_id.json",
                f"/home/user/wangxd/LLaVA-NeXT/OpenDV_{split}_16_ov-7b_10_15_id.json",
                f"/home/user/wangxd/LLaVA-NeXT/OpenDV_{split}_16_ov-7b_15_22_id.json",
            ]
            for file in jsonl_file_paths:
                with open(file, 'r', encoding='utf-8') as f:
                    tmp_dict = json.load(f)
                    self.annotation = merge_dicts(self.annotation, tmp_dict)
            
            # take the joint ids
            ids = self.meta_data.keys()
            anno_ids = self.annotation.keys()

            joint_ids = list(set(ids) - (set(ids) - set(anno_ids)))

            # only use joint ids
            ids = joint_ids

            self.key_pairs = []
            for id in ids:
                for k in self.meta_data[id].keys():
                    self.key_pairs.append((id, k))
            
            # according OpenDV cmd
            print('Total samples: %d' % len(self.key_pairs)) # 521977
            
            # image transform
            self.transform = TT.Compose([
                    TT.ToPILImage('RGB'),
                    TT.Resize((height, width)),
                    # TT.RandomResizedCrop((height, width), scale=(0.9, 1.), ratio=(1., 1.75)),
                    # transforms.RandomHorizontalFlip(p=0.1),
                    # TT.ColorJitter(brightness=0.05, contrast=0.15, saturation=0.15),
                    TT.ToTensor(),
                    TT.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True),
                ])
            
            if preload_all_data:
                self.preload()
            
            self.factor = 200

    def __len__(self):
        return len(self.key_pairs) // self.factor

    def augmentation(self, frame, transform, state):
        torch.set_rng_state(state)
        return transform(frame)
    
    def load_sample(self, index):
        id, start_f10 = self.key_pairs[index]

        segments = [int(start_f10)] # 1 * 20 = 20 frames
        segments = [str(x) for x in segments]

        total_cmds = []
        caption = ""
        total_seek_paths = []

        for sidx, seg_start in enumerate(segments):
            # 0-40 -> 
            item = self.meta_data[id][seg_start]

            cmd = item["cmd"]
            first_f10, end_f10 = item["first"], item["last"]
            len_of_frames = first_f10 - end_f10 # 40 frames (fps10) -> 80 frames (fps20)

            first_f20 = first_f10 * 2 # 20 -> 40

            seek_idxs = range(first_f20, first_f20 + 80) # 40 frames -> 80 frames

            # seek_idxs = seek_idxs[::4] # 20 fps -> 5 fps, 20 frame
            seek_idxs = seek_idxs[::2] # 20 fps -> 10 fps, 40 frame
            seek_idxs = seek_idxs[:self.max_num_frames]

            seek_paths = [str(x).zfill(9)+'.jpg' for x in seek_idxs]

            first_f20 = str(first_f20 // 40 * 40)
            video_root = self.annotation[id][first_f20]["video_root"] # sometimes will missing annotations
            seek_paths = [os.path.join(video_root, p) for p in seek_paths]
            total_seek_paths.extend(seek_paths)
        
            # annotations
            # first and last
            if sidx==0 or sidx==4:
                annotation = self.annotation[id][first_f20] # start from 40x, str
                if sidx==4:
                    caption = caption + "The scene then changes: "
                selected_attr = ["Weather", "Time", "Road environment", "Critical objects"]
                for attr in selected_attr:
                    anno = annotation.get(attr, "")
                    if anno == "":
                        continue
                    if anno[-1] == ".":
                        anno = anno[:-1] 
                    caption = caption + anno + ". "
            
            # collect cmd
            if not total_cmds or cmd!=total_cmds[-1]:
                total_cmds.append(cmd)

        driving_prompt = None
        for cd in total_cmds:
            # driving command
            tmp_prompt = map_category_to_caption(cd)
            driving_prompt = tmp_prompt if driving_prompt is None else driving_prompt + "Then " + tmp_prompt + ". "

        global_prompt = driving_prompt + " " + caption

        frames = np.stack([image2arr(path) for path in total_seek_paths])

        state = torch.get_rng_state()
        frames = torch.stack([self.augmentation(v, self.transform, state) for v in frames], dim=0)

        return global_prompt, frames

    def encode(self, prompt, video):
        prompt = self.encode_prompt(prompt).to("cpu")
        v1, v2 = self.encode_video(video)
        video = (v1,v2.sample().to("cpu"))
        return prompt, video
    
    def __getitem__(self, index):
        try:
            start_idx = index * self.factor
            end_idx = min(start_idx + self.factor, len(self.key_pairs))
            sampled_idx = random.randint(start_idx, end_idx - 1)

            prompt, video = self.load_sample(sampled_idx)
            prompt, video = self.encode(prompt, video)
            return {
                "instance_prompt": prompt,
                "instance_video": video,
            }
        except Exception as e:
            print('Bad idx %s skipped because of %s' % (index, e))
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))



class OpenDVFPS10fbfForDistillCogvidx(Dataset):
    def __init__(
        self,
        data_root: str = "/data/wuzhirong/datasets/OpenDV-mini/full_images",
        height: int = 480,
        width: int = 720,
        max_num_frames: int = 229, # 8 * N + 1 (where N ≤ 6)
        split: str = "train",
        encode_prompt = None,
        encode_video = None,
        max_samples: int = 700,
        preload_all_data: bool = False,
        preprocessed_data_path = ""
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.height = height
        self.width = width
        self.max_num_frames = max_num_frames
        self.split = split
        self.encode_prompt=encode_prompt
        self.encode_video=encode_video
        self.preload_all_data = preload_all_data
        self.preprocessed_data_path = preprocessed_data_path

        if preload_all_data and preprocessed_data_path is not None:
            pass
        else:

            # read metadata from jsonl
            self.meta_data = {}
            if split == "train":
                jsonl_file_paths = [
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split0_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split1_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split2_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split3_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split4_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split5_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split6_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split7_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split8_id.json",
                    f"/home/user/wangxd/diffusers/mini-id-anno-{split}-split9_id.json",
                ]
            else:
                pass

            for file in jsonl_file_paths:
                with open(file, 'r', encoding='utf-8') as f:
                    tmp_dict = json.load(f)
                    self.meta_data = merge_dicts(self.meta_data, tmp_dict)
            
            self.annotation = {}
            jsonl_file_paths = [
                f"/home/user/wangxd/LLaVA-NeXT/OpenDV_{split}_16_ov-7b_0_5_id.json",
                f"/home/user/wangxd/LLaVA-NeXT/OpenDV_{split}_16_ov-7b_5_10_id.json",
                f"/home/user/wangxd/LLaVA-NeXT/OpenDV_{split}_16_ov-7b_10_15_id.json",
                f"/home/user/wangxd/LLaVA-NeXT/OpenDV_{split}_16_ov-7b_15_22_id.json",
            ]
            for file in jsonl_file_paths:
                with open(file, 'r', encoding='utf-8') as f:
                    tmp_dict = json.load(f)
                    self.annotation = merge_dicts(self.annotation, tmp_dict)
            
            # take the joint ids
            ids = self.meta_data.keys()
            anno_ids = self.annotation.keys()

            joint_ids = list(set(ids) - (set(ids) - set(anno_ids)))

            # only use joint ids
            ids = joint_ids

            self.key_pairs = []
            for id in ids:
                for k in self.meta_data[id].keys():
                    self.key_pairs.append((id, k))
            
            # according OpenDV cmd
            print('Total samples: %d' % len(self.key_pairs)) # 521977
            
            # image transform
            self.transform = TT.Compose([
                    TT.ToPILImage('RGB'),
                    TT.Resize((height, width)),
                    # TT.RandomResizedCrop((height, width), scale=(0.9, 1.), ratio=(1., 1.75)),
                    # transforms.RandomHorizontalFlip(p=0.1),
                    # TT.ColorJitter(brightness=0.05, contrast=0.15, saturation=0.15),
                    TT.ToTensor(),
                    TT.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True),
                ])
            
            if preload_all_data:
                self.preload()
            
            self.factor = 20000

    def __len__(self):
        return len(self.key_pairs) // self.factor

    def augmentation(self, frame, transform, state):
        torch.set_rng_state(state)
        return transform(frame)
    
    def load_sample(self, index):
        id, start_f10 = self.key_pairs[index]

        segments = [int(start_f10)] # 1 * 20 = 20 frames

        segments = [int(start_f10), int(start_f10)+40, int(start_f10)+80, int(start_f10)+120, int(start_f10)+160, int(start_f10)+200, int(start_f10)+240]

        # import pdb; pdb.set_trace()

        segments = [str(x) for x in segments]

        total_cmds = []
        caption = ""
        total_seek_paths = []

        for sidx, seg_start in enumerate(segments):
            # 0-40 -> 
            item = self.meta_data[id][seg_start]

            # import pdb; pdb.set_trace()

            cmd = item["cmd"]
            first_f10, end_f10 = item["first"], item["last"]
            len_of_frames = first_f10 - end_f10 # 40 frames (fps10) -> 80 frames (fps20)

            first_f20 = first_f10 * 2 # 20 -> 40

            seek_idxs = range(first_f20, first_f20 + 80) # 40 frames -> 80 frames

            # seek_idxs = seek_idxs[::4] # 20 fps -> 5 fps, 20 frame
            seek_idxs = seek_idxs[::2] # 20 fps -> 10 fps, 40 frame
            # seek_idxs = seek_idxs[:self.max_num_frames]

            seek_paths = [str(x).zfill(9)+'.jpg' for x in seek_idxs]

            first_f20 = str(first_f20 // 40 * 40)
            video_root = self.annotation[id][first_f20]["video_root"] # sometimes will missing annotations
            seek_paths = [os.path.join(video_root, p) for p in seek_paths]
            total_seek_paths.extend(seek_paths)
        
            # annotations
            # first and last
            if sidx==0 or sidx==4:
                annotation = self.annotation[id][first_f20] # start from 40x, str
                if sidx==4:
                    caption = caption + "The scene then changes: "
                selected_attr = ["Weather", "Time", "Road environment", "Critical objects"]
                for attr in selected_attr:
                    anno = annotation.get(attr, "")
                    if anno == "":
                        continue
                    if anno[-1] == ".":
                        anno = anno[:-1] 
                    caption = caption + anno + ". "
            
            # collect cmd
            if not total_cmds or cmd!=total_cmds[-1]:
                total_cmds.append(cmd)

        driving_prompt = None
        for cd in total_cmds:
            # driving command
            tmp_prompt = map_category_to_caption(cd)
            driving_prompt = tmp_prompt if driving_prompt is None else driving_prompt + "Then " + tmp_prompt + ". "

        global_prompt = driving_prompt + " " + caption

        # NOTE add max_frames
        total_seek_paths = total_seek_paths[:self.max_num_frames]
        # import pdb; pdb.set_trace()

        frames = np.stack([image2arr(path) for path in total_seek_paths])

        state = torch.get_rng_state()
        frames = torch.stack([self.augmentation(v, self.transform, state) for v in frames], dim=0)

        return global_prompt, frames

    def encode(self, prompt, video):
        prompt = self.encode_prompt(prompt).to("cpu")
        v1, v2 = self.encode_video(video)
        video = (v1,v2.sample().to("cpu"))
        return prompt, video
    
    def __getitem__(self, index):
        try:
            start_idx = index * self.factor
            end_idx = min(start_idx + self.factor, len(self.key_pairs))
            sampled_idx = random.randint(start_idx, end_idx - 1)

            prompt, video = self.load_sample(sampled_idx)
            prompt, video = self.encode(prompt, video)
            return {
                "instance_prompt": prompt,
                "instance_video": video,
            }
        except Exception as e:
            print('Bad idx %s skipped because of %s' % (index, e))
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))




if __name__ == "__main__":
    train_dataset = OpenDVFPS10fbfForDistillCogvidx(split="train")
    for i in tqdm(range(len(train_dataset))):
        item = train_dataset[i]
        import pdb; pdb.set_trace()