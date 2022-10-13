from torch import Tensor
import torchaudio
from hw_asr.augmentations.base import AugmentationBase
import random

class PitchShift(AugmentationBase):
    def __init__(self, sr, n_step, p, **kwargs):
        self._aug = torchaudio.transforms.PitchShift(sr, n_step)
        self.p = p

    def __call__(self, data: Tensor):
        if random.random() < self.p:
            data = self._aug(data)
        return data
