from collections import defaultdict
from typing import List, NamedTuple

import torch

from .char_text_encoder import CharTextEncoder
from pyctcdecode import build_ctcdecoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"
    EMPTY_IND = 0

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        last_char = self.EMPTY_IND
        ans = []
        for ind in inds:
            if ind == last_char:
                continue
            if ind != self.EMPTY_IND:
                ans.append(self.ind2char[ind])
            last_char = ind
        return ''.join(ans)

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []
        dp = {
            ('', self.EMPTY_TOK): 1.0
        }
        # hypos.append(Hypothesis('', 1.0))
        sz = probs.shape[0]
        i = 0
        for prob in probs:
            if i < sz - 1:
                dp = self.extend_and_merge(dp, prob)
                dp = self.cut_beams(dp, beam_size)
            else:
                dp = self.extend_and_merge(dp, prob)
                dp = self.final_merge(dp)
                dp = self.cut_beams(dp, beam_size)
            i += 1
        for (res, last_char), v in dp.items():
            hypos.append(Hypothesis(res, v))
        return sorted(hypos, key=lambda x: x.prob, reverse=True)

    def final_merge(self, dp):
        new_dp = defaultdict(float)
        for (res, last_char), v in dp.items():
            if last_char == self.EMPTY_TOK:
                new_dp[(res, last_char)] += v
            else:
                new_dp[(res + last_char, last_char)] += v
        return new_dp

    def extend_and_merge(self, dp, prob):
        new_dp = defaultdict(float)
        for (res, last_char), v in dp.items():
            for i in range(len(prob)):
                if self.ind2char[i] == last_char:
                    new_dp[(res, last_char)] += v * prob[i]
                elif last_char == self.EMPTY_TOK:
                    new_dp[(res, self.ind2char[i])] += v * prob[i]
                else:
                    new_dp[(res + last_char, self.ind2char[i])] += v * prob[i]
        return new_dp

    def cut_beams(self, dp, beam_size):
        return dict(list(sorted(dp.items(), key=lambda x: x[1]))[-beam_size:])

    def fast_ctc_beam_search_decoder(self, logits: torch.tensor, log_probs_length,
                        beam_size: int = 100):

        beam_search = build_ctcdecoder([''] + self.alphabet)
        hypos = beam_search.decode_beams(logits[:log_probs_length], beam_width=beam_size)

        return hypos

    def fast_ctc_beam_search_decoder_with_lm(self, logits: torch.tensor, log_probs_length,
                        beam_size: int = 100):

        beam_search = build_ctcdecoder(
            [''] + self.alphabet,
            kenlm_model_path="asr/kenlm_model.arpa",
            alpha=0.5,
            beta=0.05
        )
        hypos = beam_search.decode_beams(logits[:log_probs_length], beam_width=beam_size)

        return hypos
