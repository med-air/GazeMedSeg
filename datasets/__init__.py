import os

from torch.utils.data import Dataset, DataLoader

from .kvasir_seg import (
    KvasirSegDataset,
    KvasirGazeDataset,
)
from .nci_isbi import (
    NCIISBIProstateDataset,
    NCIISBIProstateGazeDataset,
)

import configs.static as static


def get_dataloader(args, split):
    resize_label = True if split == "train" else False

    if args.method == "full_sup" or split != "train":
        if args.data == "kvasir":
            dataset = KvasirSegDataset(
                root=args.root,
                split=split,
                spatial_size=args.spatial_size,
                do_augmentation=True,
                resize_label=resize_label,
                size_rate=args.data_size_rate,
            )
        elif args.data == "prostate":
            dataset = NCIISBIProstateDataset(
                root=args.root,
                split=split,
                spatial_size=args.spatial_size,
                do_augmentation=True,
                resize_label=resize_label,
                size_rate=args.data_size_rate,
            )
        else:
            raise NotImplementedError

    elif args.method == "gaze_sup":
        level_cfg = getattr(static, f"{str.upper(args.data)}_LEVEL_CONFIGS")[args.num_levels]
        if args.data == "kvasir":
            dataset = KvasirGazeDataset(
                root=args.root,
                pseudo_mask_root=args.pseudo_root,
                level_cfg=level_cfg,
                split=split,
                spatial_size=args.spatial_size,
                do_augmentation=True,
                resize_label=resize_label,
                size_rate=args.data_size_rate,
            )
        elif args.data == "prostate":
            dataset = NCIISBIProstateGazeDataset(
                root=args.root,
                pseudo_mask_root=args.pseudo_root,
                level_cfg=level_cfg,
                split=split,
                spatial_size=args.spatial_size,
                do_augmentation=True,
                resize_label=resize_label,
                size_rate=args.data_size_rate,
            )

        else:
            raise NotImplementedError

    if split == "train":
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_worker,
        )
    else:
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_worker)
