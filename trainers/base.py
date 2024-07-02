import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.cuda.amp import GradScaler, autocast

import copy
import numpy as np

import wandb


class BaseTrainer(object):
    def __init__(
        self,
        args,
        logger,
        model,
        optimizer,
        criterion,
        train_dataloader=None,
        test_dataloader=None,
    ):
        self.args = args
        self.logger = logger

        self.model = model
        if isinstance(self.model, nn.Module):
            self.model.cuda()
        self.optimizer = optimizer

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.criterion = criterion

        self.init_metrics()

        self.epoch = 1
        self.iteration = 1

        self.total_epoch = self.args.max_ite // len(self.train_dataloader) + 1

        if args.fp16:
            self.scaler = GradScaler()

        if args.resume:
            self.resume_configure(os.path.join(args.ckpt_path, "model_latest.pth"))

        if args.wandb:
            wandb.init(project="gaze_sup", reinit=True, name=args.run_id)
            for report_mode in ["train", "val", "test"]:
                wandb.define_metric(f"{report_mode}/iteration")
                wandb.define_metric(f"{report_mode}/*", step_metric=f"{report_mode}/iteration")

    def _update(self, minibatch):
        x, y = minibatch["image"].cuda(), minibatch["label"].cuda()

        loss_dict = {}

        self.model.train()

        for param in self.model.parameters():
            param.grad = None

        with autocast(enabled=self.args.fp16):
            logits = self.model(x)["logits"]
            loss = self.criterion(logits, y.float())

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
        self.model.eval()

    def run(self):
        dataloader_iter = iter(self.train_dataloader)
        self._epoch_begin_hook()
        while self.iteration <= self.args.max_ite:
            try:
                minibatch = next(dataloader_iter)
            except:
                self._epoch_end_hook()
                self._epoch_begin_hook()

                dataloader_iter = iter(self.train_dataloader)
                minibatch = next(dataloader_iter)
                self.epoch += 1

            loss = self._update(minibatch)

            if self.iteration % self.args.log_step == 0:
                self.report_progress(progress_dict=loss, mode="train", use_wandb=self.args.wandb)

            if self.iteration % self.args.save_step == 0:
                self.save(os.path.join(self.args.exp_path, f"model_ite{self.iteration}.pth"))

            if self.test_dataloader and self.iteration % self.args.val_step == 0:
                performance_dict = self.validate(
                    self.test_dataloader,
                    save_pred=self.args.save_pred,
                    save_root=os.path.join(self.args.exp_path, f"pred_ite{self.iteration}"),
                )
                self.report_progress(
                    progress_dict=performance_dict,
                    mode="test",
                    use_wandb=self.args.wandb,
                )
                self.save(os.path.join(self.args.exp_path, f"model_latest.pth"))

                if (self.main_metric is not None) and (performance_dict[self.main_metric] > self.best_performance):
                    self.best_iteration = self.iteration
                    self.best_performance = performance_dict[self.main_metric]
                    self.best_performance_dict = performance_dict

                    self.save(os.path.join(self.args.exp_path, f"model_best.pth"))

                if self.best_performance_dict is not None:
                    self.report_progress(
                        progress_dict=self.best_performance_dict,
                        mode="best",
                        use_wandb=False,
                    )

            self.iteration += 1

        if self.test_dataloader:
            performance_dict = self.validate(self.test_dataloader)
            self.report_progress(progress_dict=performance_dict, mode="latest", use_wandb=False)
            if self.best_performance_dict is not None:
                self.report_progress(
                    progress_dict=self.best_performance_dict,
                    mode="best",
                    use_wandb=False,
                )

    def report_progress(self, progress_dict, mode, use_wandb=False):
        self.logger.info(
            "[{} | Epoch {} Ite {}] {}".format(
                mode,
                self.epoch,
                self.iteration,
                ", ".join("{}: {}".format(k, v) for k, v in progress_dict.items()),
            )
        )

        if use_wandb:
            wandb_dict = {f"{mode}/iteration": self.iteration}

            for k, v in progress_dict.items():
                wandb_dict[f"{mode}/{k}"] = v

            wandb.log(wandb_dict)

    def init_metrics(self):
        self.best_performance = 0
        self.best_iteration = None
        self.best_performance_dict = None

        self.main_metric = None

    def resume_configure(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")

        self.model.cpu().load_state_dict(ckpt["state_dict"])
        self.model = self.model.cuda()
        self.optimizer.load_state_dict(ckpt["opt"])

        self.epoch = ckpt["epoch"]
        self.iteration = ckpt["iteration"]
        self.best_performance = ckpt["best_performance"]
        self.best_iteration = ckpt["best_iteration"]

        self.logger.info(f"Resume training from iteration {self.iteration}.")

    def load(self, ckpt_path, target_model=None):
        ckpt = torch.load(ckpt_path, map_location="cpu")

        if target_model is None:
            model = self.model

        model.cpu().load_state_dict(ckpt["state_dict"])
        model = model.cuda()

        self.logger.info(f"Loading pretrained checkpoint {ckpt_path}")

    def save(self, path, model=None):
        if model is None:
            net = self.model
        else:
            net = model

        if isinstance(net, nn.DataParallel):
            state_dict = net.module.state_dict()
        else:
            state_dict = net.state_dict()

        torch.save(
            {
                "state_dict": state_dict,
                "opt": self.optimizer.state_dict(),
                "epoch": self.epoch + 1,
                "iteration": self.iteration + 1,
                "best_performance": self.best_performance,
                "best_iteration": self.best_iteration,
            },
            path,
        )

    def _epoch_begin_hook(self):
        pass

    def _epoch_end_hook(self):
        pass
