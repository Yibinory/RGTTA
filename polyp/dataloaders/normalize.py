import numpy as np
import torch

def normalize_image(img_npy, eps=1e-7):
    """
    :param img_npy: b, c, h, w
    """
    for b in range(img_npy.shape[0]):
        for c in range(img_npy.shape[1]):
            # img_npy[b, c] = (img_npy[b, c] - img_npy[b, c].mean()) / img_npy[b, c].std() 
            img_npy[b, c] = (img_npy[b, c] - img_npy[b, c].mean()) / (img_npy[b, c].std() + eps)
    return img_npy


# def normalize_image_to_0_1(img):
#     return (img-img.min())/(img.max()-img.min())

def normalize_image_to_0_1(img):
    if len(img.shape) == 4:
        for b in range(img.shape[0]):
            img[b] = (img[b]-img[b].min())/(img[b].max()-img[b].min())
    else:
        img = (img-img.min())/(img.max()-img.min())
    return img



def normalize_image_to_m1_1(img):
    return -1 + 2 * (img-img.min())/(img.max()-img.min())


def imagenet_normalize(img_npy):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    # 如果输入是 [0, 255]，先转到 [0, 1]
    if img_npy.max() > 1.0:
        img_npy = img_npy / 255.0
    
    # 支持 (B, C, H, W) 和 (C, H, W)
    if len(img_npy.shape) == 4:  # (B, C, H, W)
        for b in range(img_npy.shape[0]):
            for c in range(3):
                img_npy[b, c] = (img_npy[b, c] - mean[c]) / std[c]
    else:  # (C, H, W)
        for c in range(3):
            img_npy[c] = (img_npy[c] - mean[c]) / std[c]
    
    return img_npy
