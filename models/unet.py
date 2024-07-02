from typing import Optional, Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets import BasicUNet


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, feat_dim=128):
        super().__init__()

        self.feat_dim = feat_dim

        self.encoder = BasicUNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=feat_dim,
            features=(64, 128, 256, 512, 1024, 128),
            norm=("group", {"num_groups": 4}),
        )

        self.head = nn.Sequential(
            nn.Conv2d(in_channels=feat_dim, out_channels=feat_dim, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=feat_dim, out_channels=feat_dim, kernel_size=1),
        )

        self.classifier = nn.Conv2d(in_channels=feat_dim, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        feat = self.encoder(x)
        logits = self.classifier(feat)

        return {"logits": logits, "feature": feat}
