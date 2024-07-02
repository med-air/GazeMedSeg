import numpy as np
from PIL import ImageFilter, ImageOps

from torchvision import transforms


class GaussianBlur(object):
    """Gaussian Blur version 2"""

    def __call__(self, x):
        sigma = np.random.uniform(0.1, 2.0)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_additional_transform(data):
    if data == "kvasir":
        transform = transforms.Compose(
            [
                # transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
            ]
        )

    return transform
