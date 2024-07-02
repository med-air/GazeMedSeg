import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.cuda.amp import autocast

import copy
import numpy as np
from PIL import Image

from .base import BaseTrainer
from utils.metric import compute_iou, compute_acc, compute_dice
from utils import mkdirs

from utils.util import adjust_learning_rate


class SegTrainer(BaseTrainer):
    def __init__(self, *argv, **kargs):
        super().__init__(*argv, **kargs)

        self.main_metric = "miou"

    def _update(self, minibatch):
        image, label = minibatch["image"].cuda(), minibatch["label"].cuda()

        mask = ~minibatch["trimap"].cuda() if "trimap" in minibatch.keys() else None

        loss_dict = {}
        loss_dict["lr"] = self.optimizer.param_groups[0]["lr"]

        self.model.train()

        for param in self.model.parameters():
            param.grad = None

        with autocast(enabled=self.args.fp16):
            pred = self.model(image)["logits"]
            loss = self.criterion(pred, label, mask=mask)

        loss_dict["loss"] = loss.item()

        if self.args.fp16:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return loss_dict

    def validate(self, dataloader, model=None, save_pred=False, save_root=None):
        if model is None:
            model = self.model
        model.eval()

        iou_l = []
        dice_l = []

        with torch.no_grad():
            for minibatch in dataloader:
                image, label = minibatch["image"].cuda(), minibatch["label"].cuda()

                mask = ~minibatch["trimap"].cuda() if "trimap" in minibatch.keys() else None

                with autocast(enabled=self.args.fp16):
                    pred = model(image)["logits"]
                    pred = F.interpolate(pred, size=label.shape[2:], mode="bilinear")

                if save_pred and save_root is not None:
                    self.save_pred_batch(
                        pred.clone(),
                        save_root=save_root,
                        save_filenames=minibatch["subject_id"],
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

        return performance_dict

    def save_pred_batch(self, pred, save_root, save_filenames):
        for i_b in range(len(save_filenames)):
            save_pred = pred[i_b, 0].astype(np.float32)
            save_path = os.path.join(save_root, f"{os.path.splitext(save_filenames[i_b])[0]}.npy")
            mkdirs(os.path.dirname(save_path))
            np.save(save_path, save_pred)

    def _epoch_begin_hook(self):
        if self.args.lr_scheduler is not None:
            if self.args.lr_scheduler == "cos":
                adjust_learning_rate(
                    self.optimizer, self.epoch, self.total_epoch, self.args.lr, self.args.lr_min, self.args.warmup_ite
                )
            else:
                raise NotImplementedError

        return super()._epoch_begin_hook()
