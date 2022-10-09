import torch_audiomentations
from torch import Tensor
from torch import distributions
from hw_asr.augmentations.base import AugmentationBase


class Noise(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self.noiser = distributions.Normal(0, 0.001)

    def __call__(self, data: Tensor):
        data = data + self.noiser.sample(data.size())
        return data
