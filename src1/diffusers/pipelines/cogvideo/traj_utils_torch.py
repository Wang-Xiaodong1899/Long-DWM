import torch
from PIL import Image
import numpy as np
from scipy.interpolate import splprep, splev

def interpolate_trajectory(points, num):
    tck, u = splprep(points.transpose(), s=0)
    new_points = splev(np.linspace(0, 1, num), tck)
    return np.array(new_points).transpose()

def unproject_pixel_to_point(depth, pixels, K):
    """
    将像素点根据深度反投影到三维空间
    :param depth: 深度图 (H, W)
    :param pixels: 像素坐标 (N, 2)
    :param K: 相机内参矩阵 (3, 3)
    :return: 三维点 (N, 3)
    """
    u, v = pixels[:, 0], pixels[:, 1]
    z = depth[v, u]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return torch.stack([x, y, z], dim=1)

def compute_rotation_matrix(trajectory_point, prev_point):
    """
    计算相机平面的旋转矩阵，使相机平面始终与轨迹垂直
    :param trajectory_point: 当前轨迹点 (3,)
    :param prev_point: 前一个轨迹点 (3,)
    :return: 旋转矩阵 (3, 3)
    """
    direction = trajectory_point - prev_point
    direction = direction / torch.norm(direction)
    up = torch.tensor([0, 1, 0], dtype=trajectory_point.dtype, device=trajectory_point.device)  # 假设相机平面初始时与z轴垂直
    right = torch.linalg.cross(up, direction)
    right = right / torch.norm(right)
    up = torch.linalg.cross(direction, right)
    up = up / torch.norm(up)
    R = torch.stack([right, up, direction], dim=1)
    return R

def project_point_to_pixel(points, K, R, t):
    """
    将三维点投影到相机平面
    :param points: 三维点 (N, 3)
    :param K: 相机内参矩阵 (3, 3)
    :param R: 旋转矩阵 (3, 3)
    :param t: 平移向量 (3,)
    :return: 像素坐标 (N, 2)
    """
    points_h = torch.cat([points, torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)], dim=1)
    P = K @ torch.cat([torch.inverse(R), -t.view(3, 1)], dim=1)
    pixel_h = points_h @ P.T
    pixel = pixel_h[:, :2] / pixel_h[:, 2].unsqueeze(1)
    return pixel

def fill_masked_region_with_nearest(image, mask):
    """
    对于掩码区域的每个值，从未掩码区域中选取一个最近的值填入。

    :param image: 输入图像，大小为 [H, W, C] 的 torch 张量
    :param mask: 掩码，大小为 [H, W] 的 torch 张量，掩码区域为 True 或非零值
    :return: 填充后的图像，大小为 [H, W, C] 的 torch 张量
    """
    # 确保 mask 是布尔类型
    mask = mask.bool()

    # 获取未掩码区域的像素坐标和对应的像素值
    unmasked_coords = torch.nonzero(~mask)  # 未掩码区域的坐标 [N, 2]
    unmasked_values = image[~mask]  # 未掩码区域的像素值 [N, C]

    # 获取掩码区域的像素坐标
    masked_coords = torch.nonzero(mask)  # 掩码区域的坐标 [M, 2]

    # 构建未掩码区域的 KDTree
    tree = torch.cdist(masked_coords.float(), unmasked_coords.float())
    indices = torch.argmin(tree, dim=1)

    # 用最近邻未掩码像素的值填充掩码区域
    filled_image = image.clone()
    filled_image[mask] = unmasked_values[indices]

    return filled_image

import torch.nn.functional as F

def fill_masked_region(image: torch.Tensor, mask: torch.Tensor):
    r1, r2 = 1,1
    H, W, C = image.shape
    
    # 将图像和掩码扩展为4D张量 (batch_size, channels, height, width)
    image_4d = image.permute(2, 0, 1).unsqueeze(0)  # shape: (1, C, H, W)
    mask_4d = mask.unsqueeze(0).unsqueeze(0)        # shape: (1, 1, H, W)
    
    # 创建一个卷积核，用于计算邻域内的像素值
    kernel = torch.ones(C, 1, 2 * r1 + 1, 2 * r2 + 1, device=image.device)
    
    # 计算邻域内的像素值之和
    neighborhood_sum = F.conv2d(image_4d, kernel, padding=(r1, r2), groups=C)
    
    # 计算邻域内的有效像素数量
    neighborhood_count = F.conv2d(1-mask_4d.to(torch.float32), torch.ones(1, 1, 2 * r1 + 1, 2 * r2 + 1, device=image.device), padding=(r1, r2))
    neighborhood_count[neighborhood_count==0]=1
    
    # 计算邻域内的平均值
    neighborhood_mean = neighborhood_sum / neighborhood_count
    
    # 将结果应用到掩码区域
    filled_image = image.clone()
    filled_image[mask == 1] = neighborhood_mean.squeeze(0).permute(1, 2, 0)[mask == 1]
    
    return filled_image
       

def get_trajectory_latent(image : torch.Tensor, trajectory_points : torch.Tensor, is_interplote_mode=None):
    """
    latent沿着轨迹运动后的结果

    :param image: 输入图像latent，大小为 [H, W, C] 的 torch 张量
    :param trajectory_points: 轨迹点，大小为 [N, 3] 的 torch 张量
    :return: 运动结果图像，大小为 [N, H, W, C] 的 torch 张量， Mask,大小为[N,H,W]
    """
    image_dtype = image.dtype
    image = image.to(dtype=torch.float32)
    device,dtype = image.device, torch.float32
    H, W, _ = image.shape
    default_depth_value = 30.
    depth = torch.full((image.shape[0], image.shape[1]), default_depth_value, dtype=dtype,device=device)

    K = torch.tensor([[71*W/90, 0, W/2], [0, 63*H/60, H/2], [0, 0, 1]], dtype=dtype, device=device)  # 相机内参矩阵

    # 反投影像素点到三维空间
    pixels = torch.stack(torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device),indexing="ij"), dim=-1).view(-1, 2)
    valid_depth = depth[pixels[:, 1], pixels[:, 0]] > 0
    points_3d = unproject_pixel_to_point(depth, pixels[valid_depth], K)
    colors = image[pixels[valid_depth][:, 1], pixels[valid_depth][:, 0]]

    trajectory_images = [image]
    trajectory_masks = [torch.ones((H, W), dtype=torch.uint8, device=device)]
    # 重新投影三维点到相机平面
    for i, trajectory_point in enumerate(trajectory_points):
        if i == 0:
            prev_point = trajectory_point
            continue
        R = compute_rotation_matrix(trajectory_point, prev_point)
        t = trajectory_point
        new_image = torch.zeros_like(image)
        mask = torch.ones((H, W), dtype=torch.uint8, device=device)

        pixels_projected = project_point_to_pixel(points_3d, K, R, t)
        valid_pixels = (pixels_projected[:, 0] >= 0) & (pixels_projected[:, 0] < W) & \
                       (pixels_projected[:, 1] >= 0) & (pixels_projected[:, 1] < H)
        pixels_projected = pixels_projected[valid_pixels].long()
        new_image[pixels_projected[:, 1], pixels_projected[:, 0]] = colors[valid_pixels]
        mask[pixels_projected[:, 1], pixels_projected[:, 0]] = 0

        if is_interplote_mode == "nearest":
            # new_image = fill_masked_region_with_nearest(new_image, mask)
            new_image = fill_masked_region(new_image, mask)
        elif is_interplote_mode == "cv2":
            import cv2
            new_image = torch.Tensor(cv2.inpaint(new_image.cpu().numpy().astype(np.uint8),mask.cpu().numpy(),3,cv2.INPAINT_NS)).to(dtype=new_image.dtype,device=device)

        trajectory_images.append(new_image)
        trajectory_masks.append(1-mask)
        prev_point = trajectory_point

    new_images = torch.stack(trajectory_images)
    new_image_masks = torch.stack(trajectory_masks)

    image = image.to(dtype=image_dtype)
    new_images = new_images.to(dtype=image_dtype)
    return new_images, new_image_masks
    

def get_trajectory_image_pil(image : Image, trajectory_points : torch.Tensor, device="cpu", is_interplote_image="cv2"):
    image_tensor = torch.Tensor(np.array(image)).to(device)
    trajectory_images, _ = get_trajectory_latent(image_tensor, trajectory_points, is_interplote_image)
    return [Image.fromarray(trajectory_images[i].cpu().numpy().astype(np.uint8)) for i in range(trajectory_images.shape[0])]