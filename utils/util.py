import numpy as np
import time
import os
import logging
import sys

import torch
import torch.nn.functional as F
import itertools

import math


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def get_timestamp():
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%Y%m%d")
    return timestampDate + "-" + timestampTime


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter("%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s", datefmt="%y-%m-%d %H:%M:%S")
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + "_{}.log".format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

    return lg


def adjust_learning_rate(optimizer, epoch, total_epoch, base_lr, min_lr, warmup_ite):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_ite:
        lr = base_lr * epoch / warmup_ite
    else:
        lr = min_lr + (base_lr - min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - warmup_ite) / (total_epoch - warmup_ite))
        )

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
