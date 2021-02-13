from abc import ABC, abstractmethod
from typing import List, Optional
import torch
import cv2
import numpy as np

import ocr.config as config
from ocr.utils import load_image_with_crop, resize_with_aspect_ratio
from ocr.data.augment import FakeAugmentator as _FakeAugmentator
from ocr.data.augment import KillerAugmentator as _KillerAugmentator


class BaseDataProcessCallable(ABC):
    @abstractmethod
    def __call__(self, **kwargs):
        pass


class Sometimes(BaseDataProcessCallable):
    def __init__(self, node: BaseDataProcessCallable, prob: float = 0.5):
        self._p = prob
        self._node = node

    def __call__(self, **kwargs):
        if np.random.uniform(0.0, 1.0) < self._p:
            return self._node(**kwargs)
        return kwargs


class OneOf(BaseDataProcessCallable):
    def __init__(self, nodes: List[BaseDataProcessCallable], probs: Optional[List[float]] = None):
        assert len(nodes) > 0
        self._nodes = nodes
        if probs is None:
            self._probs = [1.0 / len(nodes)] * len(nodes)
        else:
            assert len(probs) == len(self._nodes) and sum(probs) == 1.0
            self._probs = probs

    def __call__(self, **kwargs):
        tgt_node = np.random.choice(self._nodes, size=1, p=self._probs)[0]
        return tgt_node(**kwargs)


class LoadImageWithCrop(BaseDataProcessCallable):
    def __call__(self, **kwargs):
        assert 'image_filepath' in kwargs and 'bbox' in kwargs
        image = cv2.imread(str(kwargs['image_filepath']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        kwargs['image'], kwargs['bbox'] = load_image_with_crop(image, kwargs['bbox'])
        return kwargs


class TextToSequence(BaseDataProcessCallable):
    def __call__(self, **kwargs):
        assert 'text' in kwargs
        # sequence = [config.char2token[config.PAD]] * config.label_len
        # for idx, sym in enumerate([config.SOS] + list(kwargs['text']) + [config.EOS]):
        #     sequence[idx] = config.char2token[sym]
        # kwargs['sequence'] = np.array(sequence, dtype=np.uint8)
        label = np.zeros(config.label_len, dtype=int)
        for i, c in enumerate('<'+kwargs['text']):
            label[i] = config.char2token[c]
        kwargs['label'] = torch.from_numpy(label)

        label_y = np.zeros(config.label_len, dtype=int)
        for i, c in enumerate(kwargs['text']+'>'):
            label_y[i] = config.char2token[c]
        kwargs['label_y'] = torch.from_numpy(label_y)

        return kwargs


class FinalResizeWithAspectRatioAndTranspose(BaseDataProcessCallable):
    def __call__(self, **kwargs):
        assert 'image' in kwargs
        kwargs['image'] = np.transpose(resize_with_aspect_ratio(kwargs['image'], config.input_size), [2, 0, 1]) / float(256)
        kwargs['image'] = torch.from_numpy(kwargs['image']).float()
        return kwargs


class FakeAugmentator(BaseDataProcessCallable):
    def __init__(self):
        self._augmentator = _FakeAugmentator()

    def __call__(self, **kwargs):
        assert 'image' in kwargs
        kwargs['image'] = self._augmentator.augment(kwargs['image'])
        return kwargs


class KillerAugmentator(BaseDataProcessCallable):
    def __init__(self):
        self._killer_augmentator = {
            1: _KillerAugmentator(inner_size_range=(20, 30),
                                  gaussian_blur_sigma=(7, 10),
                                  median_blur_k=(17, 21),
                                  average_blur_k=(19, 27),
                                  compression_blur_sigma=(7, 11),
                                  compression_jpeg=(95, 100)),
            2: _KillerAugmentator(inner_size_range=(10, 15),
                                  gaussian_blur_sigma=(13, 17),
                                  median_blur_k=(27, 31),
                                  average_blur_k=(29, 33),
                                  compression_blur_sigma=(13, 17),
                                  compression_jpeg=(95, 100)),
        }

    def __call__(self, **kwargs):
        assert 'image' in kwargs and 'text' in kwargs and 'lines' in kwargs
        if kwargs['lines'] not in self._killer_augmentator:
            return kwargs
        kwargs['image'], kwargs['text'], kwargs['lines'] = self._killer_augmentator[kwargs['lines']].augment(
            kwargs['image']), '', 0
        return kwargs


class Pipeline:
    def __init__(self, nodes: List[BaseDataProcessCallable]):
        self._nodes = nodes

    def __call__(self, **kwargs):
        for node in self._nodes:
            kwargs = node(**kwargs)
        return kwargs
