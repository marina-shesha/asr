from typing import List

import torch
from torch import Tensor

from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
from hw_asr.metric.utils import calc_wer


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            else:
                pred_text = self.text_encoder.decode(log_prob_vec[:length])
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)


class BeamSearchWERMetric(BaseMetric):
    def __init__(self, text_encoder: CTCCharTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], beam_size: int, **kwargs):
        wers = []
        lengths = log_probs_length.detach().numpy()
        for log_prob, length, target_text in zip(log_probs, lengths, text):
            hypos = self.text_encoder.fast_ctc_beam_search_decoder(log_prob.detach().cpu().numpy(), length, beam_size)
            pred = hypos[0][0]
            target_text = BaseTextEncoder.normalize_text(target_text)
            wers.append(calc_wer(target_text, pred))
        return sum(wers) / len(wers)


class BeamSearchWERMetricWithLm(BaseMetric):
    def __init__(self, text_encoder: CTCCharTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], beam_size: int, **kwargs):
        wers = []
        lengths = log_probs_length.detach().numpy()
        for log_prob, length, target_text in zip(log_probs, lengths, text):
            hypos = self.text_encoder.fast_ctc_beam_search_decoder_with_lm(log_prob.detach().cpu().numpy(), length,
                                                                           beam_size)
            pred = hypos[0][0]
            target_text = BaseTextEncoder.normalize_text(target_text)
            wers.append(calc_wer(target_text, pred))
        return sum(wers) / len(wers)
