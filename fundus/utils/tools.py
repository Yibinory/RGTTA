import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn.functional import softmax
import pywt
import numpy as np
import os
import logging
from dataloaders.normalize import normalize_image
from .fourier import *
from typing import Tuple, Optional
import pickle

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_logger(log_dir, log_name="pre_train.log"):

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)
    

    formatter = logging.Formatter(
        "%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    

    logger = logging.getLogger("pre_train_logger")
    logger.setLevel(logging.INFO)  
    

    if logger.hasHandlers():
        logger.handlers.clear()
    

    
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger



class Mix_statistics(nn.Module):
    def __init__(self, p=0.5, eps=1e-6, activated=True, is_train=True):
        super().__init__()  
        self.p = p
        self.eps = eps
        self.activated = activated
        self.is_train = is_train

    def forward(self, x):
        if not self.activated:
            return x

        if self.is_train and random.random() > self.p:
            return x
   
        batch_size = x.shape[0]
        index = torch.randperm(batch_size, device=x.device)
        x_reordered = x[index]
        miu_reordered = x_reordered.mean(dim=[2, 3], keepdim=True)
        var_reordered = x_reordered.var(dim=[2, 3], keepdim=True)
        sig_reordered = (var_reordered + self.eps).sqrt()
        miu_reordered, sig_reordered = miu_reordered.detach(), sig_reordered.detach()


        miu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()  
        miu, sig = miu.detach(), sig.detach()
        x_normed = (x - miu) / sig

        lunda = self.generate_lam_for_statistics(x)
        
        miu_mix = miu * lunda + (1 - lunda) * miu_reordered
        sig_mix = sig * lunda + (1 - lunda) * sig_reordered

        return x_normed * sig_mix + miu_mix
    
    def generate_lam_for_statistics(self, x, alpha=0.1):
        batch_size = x.shape[0]
        channels = x.shape[1]
        lam = torch.distributions.beta.Beta(alpha, alpha).sample((batch_size, channels))
        lam = lam.to(x.device)
        lam = lam.view(batch_size, channels, 1, 1)
        return lam


    

class EpochLR(torch.optim.lr_scheduler._LRScheduler):
    # lr_n = lr_0 * (1 - epoch / epoch_nums)^gamma
    def __init__(self, optimizer, epochs, gamma=0.9, last_epoch=-1):
        self.lr = optimizer.param_groups[0]['lr']
        self.epochs = epochs
        self.gamma = gamma
        super(EpochLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.lr * pow((1. - (self.last_epoch + 1) / self.epochs), self.gamma)]
    



