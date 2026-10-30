import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def dice_coeff(pred, label):
    smooth = 1.
    bs = pred.size(0)
    m1 = pred.contiguous().view(bs, -1)
    m2 = label.contiguous().view(bs, -1)
    intersection = (m1 * m2).sum()
    score = 1 - ((2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth))
    return score


def bce_loss(pred, label):
    score = torch.nn.BCELoss()(pred, label)
    return score


def L2_loss(input1, input2):
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)

def focal_loss(pred, label, alpha=0.25, gamma=2, smooth=1e-6):
    # alpha：前景权重，gamma：难分样本权重
    bce = F.binary_cross_entropy(pred, label, reduction='none')
    pt = torch.where(label == 1, pred, 1 - pred)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


class Seg_loss(nn.Module):
    def __init__(self):
        super(Seg_loss, self).__init__()

    def forward(self, logit_pred, label):
        pred = torch.nn.Sigmoid()(logit_pred)
        # score = dice_coeff(pred=pred, label=label) + bce_loss(pred=pred, label=label)
        score = dice_coeff(pred=pred, label=label) + focal_loss(pred=pred, label=label, alpha=0.25, gamma=2)
        return score
    

class Rec_loss(nn.Module):
    def __init__(self):
        super(Rec_loss, self).__init__()

    def forward(self, x, rec_x):
        rec_sorce = L2_loss(x, rec_x)
        return rec_sorce


class SelfEntropyLoss(nn.Module):
    def forward(self, logits):
        if logits.shape[1] == 1:  # 二分类，单通道
            probs = torch.sigmoid(logits)
            entropy = - (probs * torch.log(probs + 1e-10) + 
                        (1-probs) * torch.log(1 - probs + 1e-10))
        else:  # 多分类或多通道
            probs = F.softmax(logits, dim=1)
            log_probs = F.log_softmax(logits, dim=1)
            entropy = -torch.sum(probs * log_probs, dim=1)
        
        return entropy.mean()


