from .base import BaseTrainer
from .seg import SegTrainer
from .gaze_sup import GazeSupTrainer

trainer_dict = {
    "full_sup": SegTrainer,
    "gaze_sup": GazeSupTrainer,
}


def get_trainer_class(args):
    return trainer_dict[args.method]
