from abc import ABC, abstractmethod
from typing import Tuple, List

import numpy as np
from torch.utils.data import Dataset

import transformer_ocr.config as config
from transformer_ocr.data.data_preparation import Pipeline


class BaseDataset(Dataset, ABC):
    def __init__(self,
                 data_preparation_pipeline: Pipeline,
                 augmentation_pipeline: Pipeline,
                 ram_cache: bool = False):
        self._data_preparation_pipeline: Pipeline = data_preparation_pipeline
        self._augmentation_pipeline: Pipeline = augmentation_pipeline
        if ram_cache:
            self._ram_cache = {}

    @abstractmethod
    def _get_sample(self, index) -> dict:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def __getitem__(self, index) -> Tuple[np.array, List[int]]:
        if not hasattr(self, '_ram_cache'):
            sample = self._get_sample(index)
            data = self._data_preparation_pipeline(**sample)
        else:
            if index not in self._ram_cache:
                sample = self._get_sample(index)
                data = self._data_preparation_pipeline(**sample)
                self._ram_cache[index] = data
            data = self._ram_cache[index]

        data = self._augmentation_pipeline(**data)
        assert 'image' in data and 'label' in data and 'label_y' in data
        assert data['image'].shape[0] == 3 and data['image'].shape[1] == config.input_size[0] \
               and data['image'].shape[2] == config.input_size[1]
        return data['image'], data['label_y'], data['label']
