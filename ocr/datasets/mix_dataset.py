from torch.utils.data import Dataset


class MixDataset(Dataset):
    def __init__(self, *datasets):
        self._datasets = datasets

    def __len__(self):
        return sum([len(d) for d in self._datasets])

    def __getitem__(self, idx):
        real_idx = idx
        for d in self._datasets:
            if real_idx <= len(d) - 1: # last idx
                return d[real_idx]
            else:
                real_idx -= len(d)
