import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

from .metric import compute_acc, compute_dice, compute_iou
from torchvision.transforms.functional import crop


class BCEWithLogitsMaskLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.reduction = reduction

    def forward(self, input, target, mask=None):
        loss = self.bce(input, target.float())

        reduce_axis = list(range(2, len(input.shape)))

        if mask is not None:
            mask = mask.float()
            loss = torch.sum(loss * mask, dim=reduce_axis) / (torch.sum(mask, dim=reduce_axis) + 1e-8)
        else:
            loss = loss.mean(reduce_axis)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise NotImplementedError


class DiceLoss(nn.Module):
    def __init__(self, reduction="mean", include_background=False):
        super().__init__()

        self.reduction = reduction
        self.include_background = include_background

    def forward(self, input, target, mask=None):
        loss = 1 - compute_dice(
            input,
            target,
            mask=mask,
            include_background=self.include_background,
            do_threshold=False,
        )

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise NotImplementedError


class DiceCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, ce_weight=0.5, reduction="mean", include_background=False):
        super().__init__()

        self.dice = DiceLoss(reduction=reduction, include_background=include_background)
        self.ce = BCEWithLogitsMaskLoss(reduction=reduction)

        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, input, target, mask=None):
        return self.dice_weight * self.dice(input, target, mask) + self.ce_weight * self.ce(input, target, mask)


def multi_level_consistency_loss(logits_l):
    prob_l = [torch.sigmoid(x) for x in logits_l]
    prob_vec_l = [torch.concat([1 - prob, prob], dim=1) for prob in prob_l]

    loss_l = []

    for i, prob in enumerate(prob_vec_l):
        prob_comp_l = prob_vec_l[:i] + prob_vec_l[i + 1 :]

        loss = 0
        for prob_comp in prob_comp_l:
            loss -= (prob * prob_comp.detach()).sum(1).mean() / len(prob_comp_l)

        loss_l.append(loss)

    return torch.stack(loss_l)


def multi_level_propgation_consistency_loss(logits_l, logits_prop_l):
    B, _, H, W = logits_l[0].shape

    prob_l = [torch.sigmoid(x) for x in logits_l]
    prob_prop_l = [torch.sigmoid(F.interpolate(x, size=(H, W))) for x in logits_prop_l]
    prob_vec_l = [torch.concat([1 - prob, prob], dim=1) for prob in prob_l]
    prob_prop_vec_l = [torch.concat([1 - prob, prob], dim=1) for prob in prob_prop_l]

    loss_l = []

    for i, prob_prop in enumerate(prob_prop_vec_l):
        prob_comp_l = prob_vec_l[:i] + prob_vec_l[i + 1 :]

        loss = 0
        for prob_comp in prob_comp_l:
            loss -= (prob_prop * prob_comp.detach()).sum(1).mean() / len(prob_comp_l)

        loss_l.append(loss)

    return torch.stack(loss_l)
