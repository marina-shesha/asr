from collections import defaultdict
from typing import List, NamedTuple

import torch

from .char_text_encoder import CharTextEncoder


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
        for prob in probs:
            dp = self.extend_and_merge(dp, prob)
            dp = self.cut_beams(dp, beam_size)
        for (res, last_char), v in dp.items():
            if last_char != self.EMPTY_TOK:
                text = res+last_char
                prob = v
                hypos.append(Hypothesis(text, prob))
            else:
                text = res
                prob = v
                hypos.append(Hypothesis(text, prob))
        return sorted(hypos, key=lambda x: x.prob, reverse=True)

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
        return dict(list(sorted(dp.items, key=lambda x: x[1]))[-beam_size:])
