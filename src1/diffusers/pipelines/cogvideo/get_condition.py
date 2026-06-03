import math
from typing import Iterable

import torch
import torch.fft as fft  # differentiable
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.

    :return: an [N x dim] Tensor of positional embeddings.
    """

    if repeat_only:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    else:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
        if dim % 2:
            embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
    return embedding

class Timestep(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return timestep_embedding(t, self.dim)

class ConcatTimestepEmbedderND(nn.Module):
    """
    Embeds each dimension independently and concatenates them.
    """

    def __init__(self, outdim, num_features=None, add_sequence_dim=False):
        super().__init__()
        self.timestep = Timestep(outdim)
        self.outdim = outdim
        self.num_features = num_features
        self.add_sequence_dim = add_sequence_dim

    def forward(self, x):
        if x.ndim == 1:
            x = x[:, None]
        assert len(x.shape) == 2
        b, dims = x.shape[0], x.shape[1]
        assert dims == self.num_features or self.num_features is None
        x = rearrange(x, "b d -> (b d)")
        emb = self.timestep(x)
        emb = rearrange(emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        if self.add_sequence_dim:
            emb = emb[:, None]
        return emb
    

class Conditioner(nn.Module):
    def __init__(self, action_emb_dim = 128):
        super().__init__()
        self.action_emb_dim = action_emb_dim
        self.emb_models={
            "trajectory":ConcatTimestepEmbedderND(action_emb_dim, 8, True),
            # "command":ConcatTimestepEmbedderND(action_emb_dim, 1, True),
            "speed":ConcatTimestepEmbedderND(action_emb_dim, 4, True),
            "angle":ConcatTimestepEmbedderND(action_emb_dim, 4, True),
            "goal":ConcatTimestepEmbedderND(action_emb_dim, 2, True),
        }
    def forward(self, cond_dict):
        embs = []
        for cond_name in self.emb_models.keys():
            embbder = self.emb_models[cond_name]
            if cond_name in cond_dict.keys():
                emb = embbder(cond_dict[cond_name])
                # print(emb)
            else:
                emb = torch.zeros((1,1,embbder.outdim*embbder.num_features))
            embs.append(emb)
            # print(emb.shape)
        embs = torch.cat(embs, dim=2)
        # print(embs.shape)
        # embs = embs.repeat(1,226,1)
        embs = embs.reshape(1,18,self.action_emb_dim).repeat(1,1,32)
        # print(embs.shape)
        return embs
