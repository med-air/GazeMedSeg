import os

from .losses import *
from .util import *

import torch
import torch.nn as nn


def get_criterion(args):
    if args.data in ["kvasir", "prostate"]:
        return BCEWithLogitsMaskLoss()
    else:
        raise NotImplementedError
