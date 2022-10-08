from torch import Tensor
import torchaudio
from hw_asr.augmentations.base import AugmentationBase


class PitchShift(AugmentationBase):
    def __init__(self, sr, n_step, **kwargs):
        self._aug = torchaudio.transforms.PitchShift(sr, n_step)

    def __call__(self, data: Tensor):
        augumented_wav = self._aug(data)
        return augumented_wav
