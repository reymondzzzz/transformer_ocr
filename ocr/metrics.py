import editdistance
import torch
from pytorch_lightning.metrics import Metric

from ocr.utils.tokenizer import tokenize_vocab

__all__ = ['PhonemeErrorRate']


def phoneme_error_rate(p_seq1, p_seq2):
    p_vocab = set(p_seq1 + p_seq2)
    p2c = dict(zip(p_vocab, range(len(p_vocab))))
    c_seq1 = [chr(p2c[p]) for p in p_seq1]
    c_seq2 = [chr(p2c[p]) for p in p_seq2]
    return editdistance.eval(''.join(c_seq1), ''.join(c_seq2)) / len(c_seq2)


class PhonemeErrorRate(Metric):
    def __init__(self, letters):
        super().__init__(compute_on_step=True)
        _, self.token_to_letter = tokenize_vocab(letters)
        self.add_state('dists', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_seq', default=torch.tensor(0), dist_reduce_fx="sum")

    def _to_text(self, seq):
        symbol_list = [self.token_to_letter[token] for token in seq]
        if 'eos' in symbol_list:
            symbol_list = symbol_list[:symbol_list.index('eos')]
        symbol_list = [x for x in symbol_list if x not in ['sos', 'pad']]
        return ''.join(symbol_list)

    def update(self, pred_seq, gt_seq) -> None:
        pred_seq, gt_seq = pred_seq.permute(1, 0).detach().cpu().numpy(), gt_seq.permute(1, 0).detach().cpu().numpy()
        self.total_seq += pred_seq.shape[0]
        for seq1, seq2 in zip(pred_seq, gt_seq):
            self.dists += phoneme_error_rate(self._to_text(seq1), self._to_text(seq2))

    def compute(self):
        return self.dists / self.total_seq
