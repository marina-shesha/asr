import torchaudio
from torch import Tensor
from hw_asr.augmentations.base import AugmentationBase
import random

class TimeStretching(AugmentationBase):
    def __init__(self, n_freq, fix_rate, p, **kwargs):
        self._aug = torchaudio.transforms.TimeStretch(n_freq=n_freq)
        self.fix_rate = fix_rate
        self.p = p

    def __call__(self, data: Tensor):
        if random.random() < self.p:
            data = self._aug(data, self.fix_rate)
        return data
