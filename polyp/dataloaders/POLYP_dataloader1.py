from torch.utils import data
from PIL import Image, ImageFilter
from batchgenerators.utilities.file_and_folder_operations import *
# import torchvision.transforms as transforms
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import os
import numpy as np
import random
import torch
# from dataloaders.transform import get_train_transform
from dataloaders.normalize import imagenet_normalize

class POLYP_dataset(data.Dataset):
    def __init__(self, root, img_list, label_list, target_size=352, domain_list=None, img_normalize=True, augment=False, scales=[0.75, 1.0, 1.25]):
        super().__init__()
        self.root = root
        self.img_list = img_list
        self.label_list = label_list
        self.domain_list = domain_list
        self.len = len(img_list)
        self.target_size = (target_size, target_size)
        self.img_normalize = img_normalize
        self.augment = augment
        self.scales = 0

        self.transform = T.Compose([
            T.Resize(self.target_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.target_transform = T.Compose([
            T.Resize(self.target_size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
            lambda x: (x > 0.5).float()
        ])

    def __len__(self):
        return self.len

    def set_epoch_scale(self, scale):
        """每个epoch开始时设置尺度"""
        self.scales = scale
    

    def __getitem__(self, item):
        img_file = os.path.join(self.root, self.img_list[item])
        label_file = os.path.join(self.root, self.label_list[item])
        img = Image.open(img_file)#.convert('RGB')
        label = Image.open(label_file).convert('L')

        if self.augment:
            if random.random() > 0.5:
                img = TF.hflip(img)
                label = TF.hflip(label)
            if random.random() > 0.5:
                img = TF.vflip(img)
                label = TF.vflip(label)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                fill = 0
                img = TF.rotate(img, angle, fill=fill)
                label = TF.rotate(label, angle, fill=fill)
            if random.random() > 0.5:
                img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
                img = TF.adjust_contrast(img, random.uniform(0.8, 1.2))
                img = TF.adjust_saturation(img, random.uniform(0.8, 1.2))
        if self.transform:
            train_form = T.Compose([
            T.Resize((int(self.target_size[0] + 32 * self.scales), int(self.target_size[1] + 32 * self.scales))),
        ])
            img = train_form(img)
            img_npy = np.array(img).transpose(2, 0, 1).astype(np.float32)
            if self.img_normalize:
                img_npy = imagenet_normalize(img_npy)
            

        if self.target_transform:
            label = T.Compose([
            T.Resize((int(self.target_size[0] + 32 * self.scales), int(self.target_size[1] + 32 * self.scales)), interpolation=T.InterpolationMode.NEAREST),
            # lambda x: (x > 0.5).float()
        ])(label)
            label_npy = np.array(label)
            mask = np.zeros_like(label_npy, dtype=np.uint8)
            mask[label_npy > 0] = 1

        return img_npy, mask[np.newaxis], img_file , self.domain_list[item] if self.domain_list else None



