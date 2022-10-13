import torchaudio
from torch import Tensor
from hw_asr.augmentations.base import AugmentationBase
import random


class Volume(AugmentationBase):
    def __init__(self, gain, gain_type, p, **kwargs):
        self.voler = torchaudio.transforms.Vol(gain, gain_type)
        self.p = p

    def __call__(self, data: Tensor):
        if random.random() < self.p:
          data = self.voler(data)
        return data
