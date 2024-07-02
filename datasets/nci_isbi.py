import os, glob
import pandas as pd
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
from .base_dataset import BaseImageDataset
from .transform import get_additional_transform
from utils import mkdirs
from skimage import color
import pydicom as dicom

from torchvision import transforms
from monai.transforms import (
    MapLabelValue,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    Resized,
    RandFlipd,
    ToTensord,
    Compose,
)


class NCIISBIProstateDataset(BaseImageDataset):
    def __init__(self, *argv, **kargs):
        super().__init__(*argv, **kargs)

        split_file = pd.read_csv(os.path.join(self.root, f"{self.split}.txt"), sep=" ", header=None)
        self.sample_list = split_file.iloc[:, 0].tolist()

        self.images = np.array([os.path.join(self.root, "images", f"{file}.dcm") for file in self.sample_list])[
            : int(len(self.sample_list) * self.size_rate)
        ]
        self.labels = np.array([os.path.join(self.root, "masks", f"{file}.png") for file in self.sample_list])[
            : int(len(self.sample_list) * self.size_rate)
        ]

    def _fetch_data(self, idx):
        image = dicom.dcmread(self.images[idx]).pixel_array.astype(np.float32)
        label = np.array(Image.open(self.labels[idx]).convert("L"), dtype=np.int16)

        return {"image": image, "label": label}

    def _transform_custom(self, data):
        data["label"] = (data["label"].float() / 255.0).long()

        return data

    def get_transform(self):
        resize_keys = ["image", "label"] if self.resize_label else ["image"]
        resize_mode = ["bilinear", "nearest"] if self.resize_label else ["bilinear"]

        if self.split == "train" and self.do_augmentation:
            return Compose(
                [
                    EnsureChannelFirstd(
                        keys=["image", "label"],
                        channel_dim="no_channel",
                    ),
                    NormalizeIntensityd(
                        keys=["image"],
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
                        keys=["image", "label"],
                        channel_dim="no_channel",
                    ),
                    NormalizeIntensityd(
                        keys=["image"],
                    ),
                    Resized(
                        keys=resize_keys,
                        spatial_size=self.spatial_size,
                        mode=resize_mode,
                    ),
                    ToTensord(keys=["image", "label"]),
                ]
            )


class NCIISBIProstateGazeDataset(NCIISBIProstateDataset):
    def __init__(self, pseudo_mask_root, level_cfg, *argv, **kargs):
        self.num_levels = len(level_cfg)

        super().__init__(*argv, **kargs)

        self.pseudo_mask_root = pseudo_mask_root
        self.level_cfg = level_cfg

        self.pseudo_labels = [
            np.array(
                [
                    os.path.join(
                        pseudo_mask_root,
                        f"crf_compat{level_cfg[i]['compat']}",
                        f"{file}.png",
                    )
                    for file in self.sample_list
                ]
            )
            for i in range(self.num_levels)
        ]

    def _transform_custom(self, data):
        data = super()._transform_custom(data)

        for i in range(self.num_levels):
            data[f"pseudo_label_{i+1}"] = (data[f"pseudo_label_{i+1}"] >= self.level_cfg[i]["thres"]).int()

        return data

    def get_transform(self):
        pseudo_label_keys = [f"pseudo_label_{i+1}" for i in range(self.num_levels)]
        resize_keys = ["image", "label"] + pseudo_label_keys if self.resize_label else ["image"]
        resize_mode = ["bilinear", "nearest"] + ["bilinear"] * self.num_levels if self.resize_label else ["bilinear"]

        if self.split == "train" and self.do_augmentation:
            return Compose(
                [
                    EnsureChannelFirstd(
                        keys=["image", "label"] + pseudo_label_keys,
                        channel_dim="no_channel",
                    ),
                    NormalizeIntensityd(
                        keys=["image"],
                    ),
                    Resized(
                        keys=resize_keys,
                        spatial_size=self.spatial_size,
                        mode=resize_mode,
                    ),
                    RandFlipd(
                        keys=["image", "label"] + pseudo_label_keys,
                        prob=0.5,
                        spatial_axis=0,
                    ),
                    RandFlipd(
                        keys=["image", "label"] + pseudo_label_keys,
                        prob=0.5,
                        spatial_axis=1,
                    ),
                    ToTensord(keys=["image", "label"] + pseudo_label_keys),
                ]
            )
        else:
            return Compose(
                [
                    EnsureChannelFirstd(
                        keys=["image", "label"] + pseudo_label_keys,
                        channel_dim="no_channel",
                    ),
                    NormalizeIntensityd(
                        keys=["image"],
                    ),
                    Resized(
                        keys=resize_keys,
                        spatial_size=self.spatial_size,
                        mode=resize_mode,
                    ),
                    ToTensord(keys=["image", "label"] + pseudo_label_keys),
                ]
            )

    def _fetch_data(self, idx):
        data = super()._fetch_data(idx)

        for i in range(self.num_levels):
            pseudo_label = np.array(Image.open(self.pseudo_labels[i][idx]).convert("L")).astype(np.float32)
            data[f"pseudo_label_{i+1}"] = pseudo_label / 255

        return data