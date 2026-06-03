import numpy as np
from PIL import Image
import cv2
import json

def image2pil(filename):
    return Image.open(filename).convert('RGB')

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

def mp4toarr(file_path, resize=False, convert=True):
    cap = cv2.VideoCapture(file_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_uint8 = frame.astype(np.uint8)
        if resize:
            frame_uint8 = cv2.resize(frame_uint8, (512, 288))
        if convert:
            frame_uint8 = cv2.cvtColor(frame_uint8, cv2.COLOR_BGR2RGB)
        frames.append(frame_uint8)
    cap.release()
    return frames
    # return np.array(frames)

def preprocess_image(pil_img, target_width=448, target_height=256):
    # 448, 256
    # 384, 256
    image = pil_img
    ori_w, ori_h = image.size
    if ori_w / ori_h > target_width / target_height:
        tmp_w = int(target_width / target_height * ori_h)
        left = (ori_w - tmp_w) // 2
        right = (ori_w + tmp_w) // 2
        image = image.crop((left, 0, right, ori_h))
    elif ori_w / ori_h < target_width / target_height:
        tmp_h = int(target_height / target_width * ori_w)
        top = (ori_h - tmp_h) // 2
        bottom = (ori_h + tmp_h) // 2
        image = image.crop((0, top, ori_w, bottom))
    image = image.resize((target_width, target_height), resample=Image.LANCZOS)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    return image

def json2data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def data2json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f)