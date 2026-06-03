import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data as data

from pytorch_i3d import InceptionI3d
import os

from sklearn.metrics.pairwise import polynomial_kernel

MAX_BATCH = 1
FVD_SAMPLE_SIZE = 2048
TARGET_RESOLUTION = (224, 224)

def preprocess(videos, target_resolution):
    # videos in {0, ..., 255} as np.uint8 array
    b, t, h, w, c = videos.shape
    all_frames = torch.FloatTensor(videos).flatten(end_dim=1) # (b * t, h, w, c)
    all_frames = all_frames.permute(0, 3, 1, 2).contiguous() # (b * t, c, h, w)
    resized_videos = F.interpolate(all_frames, size=target_resolution,
                                   mode='bilinear', align_corners=False)
    resized_videos = resized_videos.view(b, t, c, *target_resolution)
    output_videos = resized_videos.transpose(1, 2).contiguous() # (b, c, t, *)
    scaled_videos = 2. * output_videos / 255. - 1 # [-1, 1]
    return scaled_videos

def get_fvd_logits(videos, i3d, device):
    videos = preprocess(videos, TARGET_RESOLUTION)
    embeddings = get_logits(i3d, videos, device)
    return embeddings

def load_fvd_model(device):
    i3d = InceptionI3d(400, in_channels=3).to(device)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    i3d_path = os.path.join(current_dir, 'i3d_pretrained_400.pt')
    i3d.load_state_dict(torch.load(i3d_path, map_location=device))
    print(f'loaded i3d from {i3d_path}')
    i3d.eval()
    return i3d


# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L161
def _symmetric_matrix_square_root(mat, eps=1e-10):
    u, s, v = torch.svd(mat)
    si = torch.where(s < eps, s, torch.sqrt(s))
    return torch.matmul(torch.matmul(u, torch.diag(si)), v.t())

# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L400
def trace_sqrt_product(sigma, sigma_v):
    sqrt_sigma = _symmetric_matrix_square_root(sigma)
    sqrt_a_sigmav_a = torch.matmul(sqrt_sigma, torch.matmul(sigma_v, sqrt_sigma))
    return torch.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))

# https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
def cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()

    fact = 1.0 / (m.size(1) - 1) # unbiased estimate
    m_center = m - torch.mean(m, dim=1, keepdim=True)
    mt = m_center.t()  # if complex: mt = m.t().conj()
    return fact * m_center.matmul(mt).squeeze()


def frechet_distance(x1, x2):
    x1 = x1.flatten(start_dim=1)
    x2 = x2.flatten(start_dim=1)
    m, m_w = x1.mean(dim=0), x2.mean(dim=0)
    sigma, sigma_w = cov(x1, rowvar=False), cov(x2, rowvar=False)

    sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)
    trace = torch.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

    mean = torch.sum((m - m_w) ** 2)
    fd = trace + mean
    return fd

#NOTE new evulation
import scipy

def calculate_fid(
    sample_mean: np.ndarray, sample_cov: np.ndarray, real_mean: np.ndarray,
    real_cov: np.ndarray, eps: float = 1e-6
):
    cov_sqrt, _ = scipy.linalg.sqrtm(sample_cov @ real_cov, disp=False)
    if not np.isfinite(cov_sqrt).all():
        print('product of cov matrices is singular')
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = scipy.linalg.sqrtm(
            (sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))
            raise ValueError(f'Imaginary component {m}')

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff
    # print(f'cov_sqrt: {np.trace(cov_sqrt)}, mean_diff: {mean_diff}')
    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)
    fid = mean_norm + trace
    return float(fid), float(mean_norm), float(trace)


def get_mean_cov_from_feature_list(feature_list):
    # features = np.concatenate(feature_list)
    features = feature_list
    mean = np.mean(features, 0)
    cov = np.cov(features, rowvar=False)
    return mean, cov



def polynomial_mmd(X, Y):
    m = X.shape[0]
    n = Y.shape[0]
    # compute kernels
    K_XX = polynomial_kernel(X)
    K_YY = polynomial_kernel(Y)
    K_XY = polynomial_kernel(X, Y)
    # compute mmd distance
    K_XX_sum = (K_XX.sum() - np.diagonal(K_XX).sum()) / (m * (m - 1))
    K_YY_sum = (K_YY.sum() - np.diagonal(K_YY).sum()) / (n * (n - 1))
    K_XY_sum = K_XY.sum() / (m * n)
    mmd = K_XX_sum + K_YY_sum - 2 * K_XY_sum
    return mmd



def get_logits(i3d, videos, device):
    assert videos.shape[0] % MAX_BATCH == 0
    with torch.no_grad():
        logits = []
        for i in range(0, videos.shape[0], MAX_BATCH):
            batch = videos[i:i + MAX_BATCH].to(device)
            logits.append(i3d(batch))
        logits = torch.cat(logits, dim=0)
        return logits


def compute_fvd(real, samples, i3d, device=torch.device('cpu')):
    # real, samples are (N, T, H, W, C) numpy arrays in np.uint8
    real, samples = preprocess(real, (224, 224)), preprocess(samples, (224, 224))
    first_embed = get_logits(i3d, real, device)
    second_embed = get_logits(i3d, samples, device)

    return frechet_distance(first_embed, second_embed)

def compute_fvd_1(real, samples, i3d, device=torch.device('cpu')):
    real, samples = preprocess(real, (224, 224)), preprocess(samples, (224, 224))
    first_embed = get_logits(i3d, real, device)
    second_embed = get_logits(i3d, samples, device)

    real_mean, real_cov = get_mean_cov_from_feature_list(
        first_embed.cpu().numpy())
    fake_mean, fake_cov = get_mean_cov_from_feature_list(
        second_embed.cpu().numpy())

    fid, mean_norm, trace = calculate_fid(
        fake_mean, fake_cov, real_mean, real_cov)
    print("FID {}, mean {}, trace {}".format(
        fid, mean_norm, trace))