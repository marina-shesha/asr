from collections import defaultdict
from typing import List, NamedTuple

import torch

from .char_text_encoder import CharTextEncoder
from pyctcdecode import build_ctcdecoder
import kenlm
import os
from math import exp

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
        self.init_decode = None
        self.init_decode_lm = None
        self.lm_path = "asr/kenlm_model.arpa"
        self.lower_lm_path = "asr/lower_kenlm_model.arpa"
        self.unigrams_path = "asr/librispeech-vocab.txt"
        self.a = 0.5
        self.b = 0.05

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

    def ctc_beam_search_lm(self, probs: torch.tensor, probs_length,
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
        lm = kenlm.Model(self.lm_path)
        for prob in probs:
            if i < sz - 1:
                dp = self.extend_and_merge(dp, prob)
                dp = self.cut_beams_lm(dp, beam_size)
            else:
                dp = self.extend_and_merge(dp, prob)
                dp = self.final_merge(dp)
                dp = self.cut_beams_lm(dp, beam_size)
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

    def cut_beams_lm(self, dp, beam_size):
        return dict(list(sorted(dp.items(), key=self.get_probs))[-beam_size:])

    def get_probs(self, item_dp):
        return item_dp[1] * exp(self.a * self.lm(item_dp[0][0], bos=True, eos=True) + self.b * len(item_dp[0][0].split()))

    def fast_ctc_beam_search_decoder(self, logits: torch.tensor, log_probs_length,
                        beam_size: int = 100):
        if not self.init_decode:
            self.init_decode = build_ctcdecoder([''] + self.alphabet)
        hypos = self.init_decode.decode_beams(logits[:log_probs_length], beam_width=beam_size)

        return hypos

    def fast_ctc_beam_search_decoder_with_lm(self, logits: torch.tensor, log_probs_length,
                        beam_size: int = 100):

        if not self.init_decode_lm:
            if not os.path.exists(self.lower_lm_path):
                with open(self.lm_path, 'r') as f_upper:
                    with open(self.lower_lm_path, 'w') as f_lower:
                        for line in f_upper:
                            f_lower.write(line.lower())

            with open(self.unigrams_path) as f:
                unigram = [t.lower() for t in f.read().strip().split("\n")]

            self.init_decode_lm = build_ctcdecoder(
                [''] + self.alphabet,
                kenlm_model_path=self.lower_lm_path,
                unigrams=unigram,
                alpha=0.5,
                beta=0.05
            )
        hypos = self.init_decode_lm.decode_beams(logits[:log_probs_length], beam_width=beam_size)

        return hypos
