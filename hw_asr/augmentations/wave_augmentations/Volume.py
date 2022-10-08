import torchaudio
from torch import Tensor
from hw_asr.augmentations.base import AugmentationBase


class Volume(AugmentationBase):
    def __init__(self, gain, gain_type, **kwargs):
        self.voler = torchaudio.transforms.Vol(gain, gain_type)

    def __call__(self, data: Tensor):
        augumented_wav = self.voler(data)
        return augumented_wav
