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


class Seg_loss(nn.Module):
    def __init__(self):
        super(Seg_loss, self).__init__()

    def forward(self, logit_pred, label):
        pred = torch.nn.Sigmoid()(logit_pred)
        score = dice_coeff(pred=pred, label=label) + bce_loss(pred=pred, label=label)
        return score
    

class Rec_loss(nn.Module):
    def __init__(self):
        super(Rec_loss, self).__init__()

    def forward(self, x, rec_x):
        rec_sorce = L2_loss(x, rec_x)
        return rec_sorce


