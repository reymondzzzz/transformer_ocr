import numpy as np

import transformer_ocr.config as config
from transformer_ocr.data.data_preparation import Pipeline, TextToSequence, FinalResizeWithAspectRatioAndTranspose, \
    FakeAugmentator
from transformer_ocr.data.dataset import BaseDataset
from transformer_ocr.data.fake_generator import FakeGenerator


class FakeDataset(BaseDataset):
    def __init__(self, capacity: int):
        self._capacity = capacity
        self._generator = FakeGenerator(settings=config.fake_generator_config)
        super().__init__(data_preparation_pipeline=Pipeline([

        ]), augmentation_pipeline=Pipeline([
            FakeAugmentator(),
            TextToSequence(),
            FinalResizeWithAspectRatioAndTranspose(),
        ]))

    def _get_sample(self, index: int):
        return self._generator.generate_one_plate()

    def __len__(self):
        return self._capacity


if __name__ == '__main__':
    import cv2

    dataset = FakeDataset(100)
    for image, sequence in dataset:
        print(sequence)
        image = np.transpose(image, [1, 2, 0])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('plate', image)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            exit()
