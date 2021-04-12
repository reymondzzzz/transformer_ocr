from typing import Iterable, Optional, Union

import editdistance
import torch
from torchmetrics import Metric

from ocr.utils.tokenizer import tokenize_vocab

__all__ = ['AccuracyWithIgnoreClasses', 'PhonemeErrorRate']


class AccuracyWithIgnoreClasses(Metric):
    def __init__(self, ignore_classes: Optional[Union[int, Iterable]] = None, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        if isinstance(ignore_classes, Iterable):
            self.ignore_classes = list(ignore_classes)
        elif isinstance(ignore_classes, int):
            self.ignore_classes = [ignore_classes]
        elif ignore_classes is None:
            self.ignore_classes = []
        else:
            assert False, 'incorrect type'

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        mask = torch.ones_like(target).bool()
        for class_idx in self.ignore_classes:
            mask = mask & (target != class_idx)

        assert preds.shape == target.shape

        acc_mask = preds == target
        acc_mask[~mask] = False
        self.correct += torch.sum(acc_mask)
        self.total += mask.sum()

    def compute(self):
        return self.correct.float() / self.total


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
        if len(pred_seq.shape) == 1:
            pred_seq = pred_seq.unsqueeze(0).detach().cpu().numpy()
            gt_seq = gt_seq.unsqueeze(0).detach().cpu().numpy()
        elif len(pred_seq.shape) == 2:
            pred_seq = pred_seq.detach().cpu().numpy()
            gt_seq = gt_seq.detach().cpu().numpy()
        else:
            assert False, 'not correct shapes'

        self.total_seq += pred_seq.shape[0]
        for seq1, seq2 in zip(pred_seq, gt_seq):
            self.dists += phoneme_error_rate(self._to_text(seq1), self._to_text(seq2))

    def compute(self):
        return self.dists / self.total_seq


class SymbolRate(Metric):
    def __init__(self, token):
        super().__init__(compute_on_step=False)
        self.token = token
        self.add_state('correct', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred_seq, gt_seq) -> None:
        gt_mask = gt_seq == self.token
        pred_mask = pred_seq == self.token
        res_mask = torch.logical_and(gt_mask, pred_mask)
        self.correct += res_mask.sum()
        self.total += pred_mask.sum()

    def compute(self):
        return self.correct / self.total
