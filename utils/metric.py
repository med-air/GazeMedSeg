import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.linear_model import LogisticRegression


def compute_iou(input, target, mask=None, include_background=False, do_threshold=False):
    input = torch.sigmoid(input)
    if do_threshold:
        input = (input > 0.5).float()
    target = target.float()

    if mask is None:
        mask = torch.ones_like(target)
    mask = mask.int()

    assert input.shape == target.shape

    reduce_axis = list(range(2, len(input.shape)))

    intersection = torch.sum(input * target * mask, dim=reduce_axis)
    input_o = torch.sum(input * mask, dim=reduce_axis)
    target_o = torch.sum(target * mask, dim=reduce_axis)

    union = input_o + target_o - intersection

    if include_background and input.shape[1] > 1:
        intersection = intersection[:, 1:]
        union = union[:, 1:]

    iou = torch.mean(intersection / (union + 1e-7), dim=1)

    return iou


def compute_acc(input, target, mask=None, include_background=False):
    input = (torch.sigmoid(input) > 0.5).float()
    target = target.float()

    if mask is None:
        mask = torch.ones_like(target)
    mask = mask.int()

    assert input.shape == target.shape

    reduce_axis = list(range(2, len(input.shape)))

    corrects = (input == target).float()
    corrects = torch.sum(corrects * mask, dim=reduce_axis)

    divison = torch.sum(mask, dim=reduce_axis)

    if include_background and input.shape[1] > 1:
        corrects = corrects[:, 1:]
        divison = divison[:, 1:]

    iou = torch.mean(corrects / (divison + 1e-7), dim=1)

    return iou


def compute_dice(input, target, mask=None, include_background=False, do_threshold=False):
    input = torch.sigmoid(input)
    if do_threshold:
        input = (input > 0.5).float()
    target = target.float()

    if mask is None:
        mask = torch.ones_like(target)
    mask = mask.int()

    assert input.shape == target.shape

    reduce_axis = list(range(2, len(input.shape)))

    intersection = torch.sum(input * target * mask, dim=reduce_axis)
    input_o = torch.sum(input * mask, dim=reduce_axis)
    target_o = torch.sum(target * mask, dim=reduce_axis)

    if include_background and input.shape[1] > 1:
        intersection = intersection[:, 1:]
        input_o = input_o[:, 1:]
        target_o = target_o[:, 1:]

    dice = torch.mean(2 * intersection / (input_o + target_o + 1e-7), dim=1)

    return dice


def compute_cluster_nmi(input, target, mask=None):
    B, C, H, W = input.shape

    input = input.permute(0, 2, 3, 1).contiguous().view(-1, C)
    target = target.view(-1)

    assert input.shape[0] == target.shape[0]

    if mask is not None:
        mask = mask.int().view(-1)

        input = input[mask != 0, :]
        target = target[mask != 0]

    input = input.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    pred = KMeans(n_clusters=2, n_init="auto", max_iter=1000).fit_predict(input)

    nmi = normalized_mutual_info_score(target, pred)

    return nmi


def compute_downstream_cls_score(input, target, mask=None):
    B, C, H, W = input.shape

    input = input.permute(0, 2, 3, 1).contiguous().view(-1, C)
    target = target.view(-1)

    assert input.shape[0] == target.shape[0]

    if mask is not None:
        mask = mask.int().view(-1)

        input = input[mask != 0, :]
        target = target[mask != 0]

    input = input.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    clf = LogisticRegression(max_iter=100).fit(input, target)
    acc = clf.score(input, target)

    return acc
