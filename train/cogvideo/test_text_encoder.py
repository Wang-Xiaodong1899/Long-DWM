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

text_inputs = tokenizer(
    # "a model \n a text \n some cars", # 3,    9,  825,    3,    9, 1499,  128, 2948,    1
    # "a model / a text / some cars", # 3,    9,  825,    3,   87,    3,    9, 1499,    3,   87,  128, 2948, 1
    # "a model a text some cars", # 3,    9,  825,    3,    9, 1499,  128, 2948,    1
    " / / /",
    padding="max_length",
    max_length=226,
    truncation=True,
    add_special_tokens=True,
    return_tensors="pt",
)

print(text_inputs["input_ids"])