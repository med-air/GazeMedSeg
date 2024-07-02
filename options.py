import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--method",
        help="method used for training",
        type=str,
        choices=[
            "full_sup",
            "gaze_sup",
        ],
    )
    parser.add_argument("--model", help="model architecture", type=str, default="unet", choices=["unet"])
    parser.add_argument("--in_channels", help="image dimension", type=int, default=3)
    parser.add_argument(
        "--data",
        help="dataset",
        choices=["kvasir", "prostate"],
        type=str,
    )
    parser.add_argument(
        "--root",
        help="path of the dataset",
        type=str,
    )
    parser.add_argument(
        "--exp_path",
        help="path of the experiment result",
        type=str,
    )
    parser.add_argument(
        "--ckpt_path",
        help="path of pretrained checkpoint path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--spatial_size",
        help="spatial size of input images",
        type=int,
        default=384,
    )
    parser.add_argument(
        "--feat_dim",
        help="dimension of representations",
        type=int,
        default=128,
    )
    parser.add_argument("--opt", help="optimizer", type=str, default="a")
    parser.add_argument("--lr", help="learning rate", type=float, default=4e-4)
    parser.add_argument("--lr_min", help="minimum learning rate", type=float, default=1e-6)
    parser.add_argument("--lr_scheduler", help="learning rate scheduler", type=str, default=None)
    parser.add_argument("--weight_decay", help="weight decay of optimizer", type=float, default=0.0004)
    parser.add_argument("--data_size_rate", type=float, default=1)
    parser.add_argument("--max_ite", help="total training iterations", type=int, default=15000)
    parser.add_argument("--warmup_ite", help="warmup iterations", type=int, default=0)
    parser.add_argument("-bs", "--batch_size", help="batch size", type=int, default=8)
    parser.add_argument(
        "--log_step",
        help="interval in iterations for reporting loss",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--val_step",
        help="interval in iterations for validation",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--save_step",
        help="interval in iterations for saving checkpoint",
        type=int,
        default=5000,
    )
    parser.add_argument("--resume", help="resume training", action="store_true")
    parser.add_argument("--test", help="test the pretrained model", action="store_true")
    parser.add_argument("--wandb", help="use wandb to report progress", action="store_true")
    parser.add_argument("--fp16", help="use mixed precision training", action="store_true")
    parser.add_argument("--seed", help="seed used for training", type=int, default=0)
    parser.add_argument("--num_worker", help="number of data loading workers", type=int, default=4)
    parser.add_argument("--device", help="device of running experiments", type=str, default="0")
    parser.add_argument("--save_pred", action="store_true")
    parser.add_argument("--finalize", action="store_true")

    """Gaze sup arguments"""
    parser.add_argument("--num_levels", type=int, default=2)
    parser.add_argument("--cons_weight", type=float, default=3.0)
    parser.add_argument("--cons_mode", type=str, choices=["pure", "prop"], default="prop")

    args = parser.parse_args()

    return args
