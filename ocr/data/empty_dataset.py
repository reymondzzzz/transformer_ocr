import pickle
import random
import re
from typing import List, Dict, Tuple

import cv2
import numpy as np

from ocr.data.data_preparation import Pipeline, LoadImageWithCrop, TextToSequence, \
    FinalResizeWithAspectRatioAndTranspose, Sometimes, KillerAugmentator, OneOf
from ocr.data.dataset import BaseDataset
import json
from pathlib import Path
import ocr.config as config
from ocr.data.plate_extracter import PlateExtractorWithAugment, PlateExtractor


class EmptyDataset(BaseDataset):
    def __init__(self,
                 root_path: Path,
                 capacity: int,
                 augment_dropout: float = 0.0,
                 ram_cache=False
                 ):
        self._dataset = self._load(root_path, capacity)
        super().__init__(data_preparation_pipeline=Pipeline([
            LoadImageWithCrop(),
        ]), augmentation_pipeline=Pipeline([
            OneOf([PlateExtractor(config.inner_size), PlateExtractorWithAugment(config.inner_size)],
                  probs=[augment_dropout, 1.0 - augment_dropout]),
            FinalResizeWithAspectRatioAndTranspose(),
            TextToSequence(),
        ]), ram_cache=ram_cache)

    def _get_sample(self, index) -> dict:
        return self._dataset[index]

    def __len__(self) -> int:
        return len(self._dataset)

    def _extract_box(self,
                     box: List[List[float]],
                     size: Tuple[int, int]):
        # plate bbox
        x0 = min([x for x, y in box])
        x1 = max([x for x, y in box])
        y0 = min([y for x, y in box])
        y1 = max([y for x, y in box])
        h, w = y1 - y0, x1 - x0

        # borders bbox
        bx0 = max(0, x0 - w)
        bx1 = min(1, x1 + w)
        by0 = max(0, y0 - h)
        by1 = min(1, y1 + h)

        if np.random.choice([False, True]):
            ey0 = np.random.uniform(by0, by1 - h)
            ey1 = ey0 + h
            if np.random.choice([False, True]):
                ex0, ex1 = bx0, x0
            else:
                ex0, ex1 = x1, bx1
        else:
            ex0 = np.random.uniform(bx0, bx1 - w)
            ex1 = ex0 + w
            if np.random.choice([False, True]):
                ey0, ey1 = by0, y0
            else:
                ey0, ey1 = y1, by1

        ih, iw = size
        iex0, iex1 = int(ex0 * iw), int(ex1 * iw)
        iey0, iey1 = int(ey0 * ih), int(ey1 * ih)
        if iex1 - iex0 <= 0 or iey1 - iey0 <= 0:
            raise RuntimeError()
        ebox = [[ex0, ey0], [ex1, ey0], [ex1, ey1], [ex0, ey1]]

        return ebox

    def _load(self, root_path: Path, capacity: int):
        dataset = []
        for filename in root_path.glob('**/annotations.pickle'):
            prefix = filename.parent
            with open(str(filename), 'rb') as f:
                annotations = pickle.load(f)
            images = {d['id']: d['file_name'].replace('full/', '') for d in annotations['images']}
            for anno in annotations['annotations']:
                if anno['shape_type'] != 'PolygonPlateShape':
                    continue
                image_fn = prefix / images[anno["image_id"]]
                box = anno['plate_box']
                if not image_fn.exists():
                    continue
                dataset.append((image_fn, box))
        random.shuffle(dataset)

        annotations = []
        for fn, box in dataset:
            try:
                h, w = cv2.imread(str(fn)).shape[:2]
                box = self._extract_box(box, (h, w))
                annotations.append({
                    'image_filepath': fn,
                    'bbox': box,
                    'text': '',
                    'lines': 0
                })
            except:
                continue
            if len(annotations) == capacity:
                break
        return annotations

if __name__ == '__main__':
    import cv2

    dataset = EmptyDataset(Path(config.data_config['root_path']), 100)
    for image, sequence in dataset:
        print(sequence)
        image = np.transpose(image, [1, 2, 0])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('plate', image)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            exit()