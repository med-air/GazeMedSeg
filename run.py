import random
import sys, os

import argparse
import time
import numpy as np
import math
import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from numpy.random import shuffle

from models import get_model_opt
from datasets import get_dataloader
from trainers import get_trainer_class
from utils import setup_logger, get_timestamp, mkdirs, get_criterion
from parse_args import args_parser


def init_arguments(args):
    args.run_id = (
        f"{args.method}_{args.data}_{args.opt}_bs{args.batch_size}_datasize{args.data_size_rate}_seed{args.seed}"
    )
    if args.method == "gaze_sup":
        args.run_id += f"_level{args.num_levels}_cons_{args.cons_mode}_weight{args.cons_weight}"

    if args.fp16:
        args.run_id += "_fp16"
    if args.resume:
        args.run_id += "_resume"
    if args.test:
        args.run_id += "_test"

    args.run_id = f"{args.run_id}_{get_timestamp()}"

    args.exp_path = os.path.join(args.exp_path, str(args.seed), args.data, args.run_id)
    mkdirs(args.exp_path)

    if args.method == "gaze_sup":
        args.pseudo_root = os.path.join(args.root, "gaze")

    return args


def main():
    # torch.autograd.set_detect_anomaly(True)
    args = args_parser()
    args = init_arguments(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    logger = setup_logger("train", args.exp_path, args.run_id, screen=False, tofile=True)
    logger.info("Using following arguments for training.")
    logger.info(args)

    # set the random seed so that the random permutations can be reproduced again
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    train_loader = get_dataloader(args, split="train")
    test_loader = get_dataloader(args, split="test")

    model, optimizer = get_model_opt(args)
    criterion = get_criterion(args)

    trainer = get_trainer_class(args)(
        args=args,
        logger=logger,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
    )

    trainer.run()


if __name__ == "__main__":
    main()
