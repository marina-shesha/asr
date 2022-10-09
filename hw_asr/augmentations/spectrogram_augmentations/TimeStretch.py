import torchaudio
from torch import Tensor
from hw_asr.augmentations.base import AugmentationBase


class Volume(AugmentationBase):
    def __init__(self, n_freq, **kwargs):
        self._aug = torchaudio.transforms.TimeStretch(n_freq=n_freq)

    def __call__(self, data: Tensor):
        res = self._aug(data)
        return res
