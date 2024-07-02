import torch

from .unet import *
from .multi_level import MultiLevelModel


def get_model_opt(args):
    if args.model == "unet":
        if args.method == "gaze_sup":
            model = MultiLevelModel(in_channels=args.in_channels, num_levels=args.num_levels, num_classes=1)
        else:
            model = UNet(in_channels=args.in_channels, out_channels=1, feat_dim=128)
    else:
        raise NotImplementedError

    if args.opt == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.opt == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=0.99,
            nesterov=True,
        )

    return model, optimizer
