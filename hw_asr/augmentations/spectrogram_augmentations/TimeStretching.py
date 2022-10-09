import torchaudio
from torch import Tensor
from hw_asr.augmentations.base import AugmentationBase


class TimeStretching(AugmentationBase):
    def __init__(self, n_freq, fix_rate, **kwargs):
        self._aug = torchaudio.transforms.TimeStretch(n_freq=n_freq)
        self.fix_rate = fix_rate

    def __call__(self, data: Tensor):
        res = self._aug(data, self.fix_rate)
        return res
