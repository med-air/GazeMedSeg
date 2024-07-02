import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader

import copy
import numpy as np
import shutil

from .seg import SegTrainer
from utils.metric import compute_iou, compute_acc, compute_dice
from utils.losses import (
    multi_level_consistency_loss,
    multi_level_propgation_consistency_loss,
)


class GazeSupTrainer(SegTrainer):
    def __init__(self, *argv, **kargs):
        super().__init__(*argv, **kargs)
        self.main_metric = "mdice"

    def _update(self, minibatch):
        image = minibatch["image"].cuda()

        loss_dict = {}
        loss_dict["lr"] = self.optimizer.param_groups[0]["lr"]

        self.model.train()

        with autocast(enabled=self.args.fp16):
            pred_dict = self.model(image)

            loss = 0
            for i in range(self.args.num_levels):
                pseudo_label = minibatch[f"pseudo_label_{i+1}"].cuda()
                loss_cls = self.criterion(pred_dict[f"logits_{i+1}"], pseudo_label.float())

                loss_dict[f"loss_cls_{i+1}"] = loss_cls.item()

                loss += loss_cls

            if self.args.cons_weight > 0 and self.args.num_levels > 1:
                if self.args.cons_mode == "pure":
                    loss_consistency = multi_level_consistency_loss(
                        [pred_dict[f"logits_{i+1}"] for i in range(self.args.num_levels)]
                    )
                elif self.args.cons_mode == "prop":
                    loss_consistency = multi_level_propgation_consistency_loss(
                        [pred_dict[f"logits_{i+1}"] for i in range(self.args.num_levels)],
                        [pred_dict[f"logits_prop_{i+1}"] for i in range(self.args.num_levels)],
                    )

                loss_dict["loss_cons"] = loss_consistency.mean().item()

                loss += self.args.cons_weight * loss_consistency.sum()

            loss_dict["loss"] = loss.item()

        if self.args.fp16:
            for param in self.model.parameters():
                param.grad = None
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)

        else:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss_dict

    def validate(self, dataloader, model=None, save_pred=False, save_root=None):
        if model is None:
            model = self.model
        model.eval()

        iou_sub_l = [[] for _ in range(self.args.num_levels)]
        dice_sub_l = [[] for _ in range(self.args.num_levels)]

        iou_l = []
        dice_l = []

        with torch.no_grad():
            for minibatch in dataloader:
                image, label = minibatch["image"].cuda(), minibatch["label"].cuda()

                mask = ~minibatch["trimap"].cuda() if "trimap" in minibatch.keys() else None

                with autocast(enabled=self.args.fp16):
                    pred_dict = model(image)
                    pred_sub_l = [
                        F.interpolate(
                            pred_dict[f"logits_{i+1}"],
                            size=label.shape[2:],
                            mode="bilinear",
                        )
                        for i in range(self.args.num_levels)
                    ]

                    pred = torch.stack(pred_sub_l, dim=0).mean(dim=0)

                if save_pred and save_root is not None:
                    self.save_pred_batch(
                        pred.clone(),
                        save_root=save_root,
                        save_filenames=minibatch["subject_id"],
                    )

                    for i in range(self.args.num_levels):
                        save_root_i = os.path.join(save_root, f"pred_level_{i+1}")

                        self.save_pred_batch(
                            pred_sub_l[i].clone(),
                            save_root=save_root_i,
                            save_filenames=minibatch["subject_id"],
                        )

                for i in range(self.args.num_levels):
                    iou_sub_l[i].append(compute_iou(pred_sub_l[i], label, mask=mask, do_threshold=True).cpu().numpy())
                    dice_sub_l[i].append(
                        compute_dice(pred_sub_l[i], label, mask=mask, do_threshold=True).cpu().numpy()
                    )

                iou_l.append(compute_iou(pred, label, mask=mask, do_threshold=True).cpu().numpy())
                dice_l.append(compute_dice(pred, label, mask=mask, do_threshold=True).cpu().numpy())

        iou_l = np.concatenate(iou_l, axis=0)
        dice_l = np.concatenate(dice_l, axis=0)

        performance_dict = {}
        performance_dict["miou"] = np.mean(iou_l)
        performance_dict["miou_std"] = np.std(iou_l)

        performance_dict["mdice"] = np.mean(dice_l)
        performance_dict["mdice_std"] = np.std(dice_l)

        for i in range(self.args.num_levels):
            iou_sub = np.concatenate(iou_sub_l[i], axis=0)
            dice_sub = np.concatenate(dice_sub_l[i], axis=0)

            performance_dict[f"miou_{i+1}"] = np.mean(iou_sub)
            performance_dict[f"miou_std_{i+1}"] = np.std(iou_sub)

            performance_dict[f"mdice_{i+1}"] = np.mean(dice_sub)
            performance_dict[f"mdice_std_{i+1}"] = np.std(dice_sub)

        return performance_dict
