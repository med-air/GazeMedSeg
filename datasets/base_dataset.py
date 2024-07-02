import os, glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from monai.transforms import (
    MapLabelValue,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    Resized,
    RandFlipd,
    ToTensord,
    Compose,
)


class BaseImageDataset(Dataset):
    def __init__(
        self,
        root,
        split,
        spatial_size=(384, 384),
        do_augmentation=False,
        size_rate=1,
        resize_label=True,
    ):
        super().__init__()

        if isinstance(spatial_size, int):
            spatial_size = (spatial_size, spatial_size)
        self.spatial_size = spatial_size

        self.img_norm_cfg = dict(mean=np.array([123.675, 116.28, 103.53]), std=np.array([58.395, 57.12, 57.375]))

        self.root = root
        self.split = split
        self.do_augmentation = do_augmentation
        self.size_rate = size_rate
        self.resize_label = resize_label

        self.sample_list = []
        self.images = None
        self.labels = None

        self.transform = self.get_transform()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        data = self._fetch_data(idx)
        data = self.transform(data)
        data = self._transform_custom(data)

        return {
            "idx": idx,
            "subject_id": self.sample_list[idx],
            "path": os.path.basename(self.images[idx]),
        } | data

    def get_transform(self):
        resize_keys = ["image", "label"] if self.resize_label else ["image"]
        resize_mode = ["bilinear", "nearest"] if self.resize_label else ["bilinear"]

        if self.split == "train" and self.do_augmentation:
            return Compose(
                [
                    EnsureChannelFirstd(
                        keys=["image"],
                        channel_dim=2,
                    ),
                    EnsureChannelFirstd(
                        keys=["label"],
                        channel_dim="no_channel",
                    ),
                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=self.img_norm_cfg["mean"],
                        divisor=self.img_norm_cfg["std"],
                        channel_wise=True,
                    ),
                    Resized(
                        keys=resize_keys,
                        spatial_size=self.spatial_size,
                        mode=resize_mode,
                    ),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                    ToTensord(keys=["image", "label"]),
                ]
            )
        else:
            return Compose(
                [
                    EnsureChannelFirstd(
                        keys=["image"],
                        channel_dim=2,
                    ),
                    EnsureChannelFirstd(
                        keys=["label"],
                        channel_dim="no_channel",
                    ),
                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=self.img_norm_cfg["mean"],
                        divisor=self.img_norm_cfg["std"],
                        channel_wise=True,
                    ),
                    Resized(
                        keys=resize_keys,
                        spatial_size=self.spatial_size,
                        mode=resize_mode,
                    ),
                    ToTensord(keys=["image", "label"]),
                ]
            )

    def _transform_custom(self, data):
        pass

    def _fetch_data(self, idx):
        pass
