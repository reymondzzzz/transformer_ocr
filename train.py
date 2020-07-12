import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from pathlib import Path

import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import wandb
import transformer_ocr.config as config
from transformer_ocr.config import char2token
from transformer_ocr.data.empty_dataset import EmptyDataset
from transformer_ocr.data.fake_dataset import FakeDataset
from transformer_ocr.data.real_dataset import RealDataset
from transformer_ocr.model import make_model
import numpy as np

from transformer_ocr.utils import cd


wandb.init(project='Transformer_OCR')
wandb.config.epochs = config.epochs
wandb.config.batch_size = config.batch_size


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, imgs, trg_y, trg, pad=0):
        self.imgs = Variable(imgs.cuda(), requires_grad=False)
        self.src_mask = Variable(torch.from_numpy(np.ones([imgs.size(0), 1, 36], dtype=np.bool)).cuda())
        if trg is not None:
            self.trg = Variable(trg.cuda(), requires_grad=False)
            self.trg_y = Variable(trg_y.cuda(), requires_grad=False)
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return Variable(tgt_mask.cuda(), requires_grad=False)

# def subsequent_mask(size):
#     "Mask out subsequent positions."
#     attn_shape = (1, size, size)
#     subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
#     return torch.from_numpy(subsequent_mask) == 0
#
# def make_std_mask(tgt, pad):
#         "Create a mask to hide padding and future words."
#         tgt_mask = (tgt != pad).unsqueeze(-2)
#         tgt_mask = tgt_mask & Variable(
#             subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
#         return tgt_mask
#
# class Batch:
#     "Object for holding a batch of data with mask during training."
#     def __init__(self, imgs, trg=None, pad=0):
#         self.imgs = Variable(imgs.float().cuda(), requires_grad=False)
#         # self.src_mask = Variable(torch.from_numpy(np.ones([imgs.size(0), 1, 36], dtype=np.bool)).cuda())
#         if trg is not None:
#             self.trg = Variable(trg.long().cuda(), requires_grad=False)
#             # self.trg_y = Variable(trg_y.cuda(), requires_grad=False)
#             # self.trg_mask = \
#             #     self.make_std_mask(self.trg, pad)
#             self.ntokens = (self.trg != pad).data.sum()

    # @staticmethod
    # def make_std_mask(tgt, pad):
    #     "Create a mask to hide padding and future words."
    #     tgt_mask = (tgt != pad).unsqueeze(-2)
    #     tgt_mask = tgt_mask & Variable(
    #         subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    #     return Variable(tgt_mask.cuda(), requires_grad=False)

class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        wandb.log({'lr': rate})
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, criterion, opt=None):
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm, model):
        x = model.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm

        # xx = x.detach().cpu().transpose(0, 1).numpy()[0]
        xx = x.detach().cpu().numpy()[0]
        word = []
        for i in range(36):
            word.append(np.argmax(xx[i]))
        yy = y.detach().cpu().numpy()[0]

        if self.opt is not None:
            self.opt.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            self.opt.step()
        return loss.data * norm

def greedy_decode(src, model, max_len=36, start_symbol=1):
    src_mask = Variable(torch.from_numpy(np.ones([1, 1, 36], dtype=np.bool)).cuda())
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).long().cuda()
    for i in range(max_len-1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .long().cuda()))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).long().cuda().fill_(next_word)], dim=1)
        if config.token2char[next_word.item()] == '>':
            break
    ret = ys.cpu().numpy()[0]
    out = [config.token2char[i] for i in ret]
    out = "".join(out[1:-1])
    return ret


def run_epoch(dataloader, model, loss_compute, is_train=False):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    once = True
    for i, (imgs, labels_y, labels) in enumerate(dataloader):
        batch = Batch(imgs, labels_y, labels)
        out = model(batch.imgs, batch.trg, batch.src_mask, batch.trg_mask)
        # out = model(batch.imgs, batch.trg)
        if not is_train and once:
            wordx = greedy_decode(batch.imgs[0].unsqueeze(0).float().cuda(), model)
            # wordy = labels.detach().cpu().numpy()[0]
            print(wordx)
            print(batch.trg_y.detach().cpu().numpy()[0])
            once = False
        loss = loss_compute(out, batch.trg_y, batch.ntokens, model)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens



def train():
    train_dataset = RealDataset(root_path=Path(config.data_config['root_path']),
                                subset='train',
                                lines_allowed=config.data_config['lines_allowed'],
                                balance_dataset=False,
                                empty_dropout=config.empty_dropout,
                                augment_dropout=config.augment_dropout,
                                ram_cache=True)
    empty_capacity = int(len(train_dataset) * 0.1)
    fake_capacity = int(len(train_dataset) * 0.2)

    empty_dataset = EmptyDataset(root_path=Path(config.data_config['root_path']),
                                 capacity=empty_capacity,
                                 augment_dropout=config.augment_dropout,
                                 ram_cache=True)

    fake_dataset = FakeDataset(capacity=fake_capacity)

    train_dataset = torch.utils.data.ConcatDataset([train_dataset, empty_dataset, fake_dataset])

    val_dataset = RealDataset(root_path=Path(config.data_config['root_path']),
                              subset='val',
                              lines_allowed=config.data_config['lines_allowed'],
                              balance_dataset=False,
                              empty_dropout=config.empty_dropout,
                              augment_dropout=config.augment_dropout,
                              ram_cache=True)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size,
                                                   shuffle=True, num_workers=12)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                                                 num_workers=12)
    model = make_model(len(char2token))
    # model.load_state_dict(torch.load('your-pretrain-model-path'))
    model.cuda()
    criterion = LabelSmoothing(size=len(char2token), padding_idx=0, smoothing=0.1)
    criterion.cuda()
    model_opt = NoamOpt(model.tgt_embed[0].d_model, 100, 2000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(config.epochs):
        model.train()
        train_loss = run_epoch(train_dataloader, model, SimpleLossCompute(criterion, model_opt), is_train=True)
        model.eval()
        val_loss = run_epoch(val_dataloader, model,
                              SimpleLossCompute(criterion, None))
        wandb.log({'train_loss': train_loss, 'val_loss': val_loss})
        # print("val_loss", val_loss)
        with cd(config.ckpt_dir):
            checkpoint_filepath = f'{epoch:08d}_{val_loss}.pth'
            latest_checkpoint_filepath = 'latest.pth'
            if Path(latest_checkpoint_filepath).exists():
                os.unlink(latest_checkpoint_filepath)
            torch.save(model.state_dict(), checkpoint_filepath)
            os.symlink(checkpoint_filepath, latest_checkpoint_filepath)


if __name__ == '__main__':
    train()
