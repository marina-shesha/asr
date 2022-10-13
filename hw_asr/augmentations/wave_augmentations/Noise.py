import torch_audiomentations
from torch import Tensor
from torch import distributions
from hw_asr.augmentations.base import AugmentationBase
import random

class Noise(AugmentationBase):
    def __init__(self, p, **kwargs):
        self.noiser = distributions.Normal(0, 0.001)
        self.p = p
    def __call__(self, data: Tensor):
        if random.random() < self.p:
          data = data + self.noiser.sample(data.size())
        return data
