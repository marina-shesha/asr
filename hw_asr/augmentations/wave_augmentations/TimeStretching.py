import librosa
from torch import Tensor
import torch
from hw_asr.augmentations.base import AugmentationBase


class TimeStretching(AugmentationBase):
    def __init__(self, rate, **kwargs):
        self.rate = rate

    def __call__(self, data: Tensor):
        augumented_wav = librosa.effects.time_stretch(data.numpy(), self.rate)
        augumented_wav = torch.from_numpy(augumented_wav)
        return augumented_wav
