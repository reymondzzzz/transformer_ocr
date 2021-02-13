import pickle
import re
from typing import List, Dict

import numpy as np

from ocr.data.data_preparation import Pipeline, LoadImageWithCrop, TextToSequence, \
    FinalResizeWithAspectRatioAndTranspose, Sometimes, KillerAugmentator, OneOf
from ocr.data.dataset import BaseDataset
import json
from pathlib import Path
import ocr.config as config
from ocr.data.plate_extracter import PlateExtractorWithAugment, PlateExtractor


class RealDataset(BaseDataset):
    def __init__(self,
                 root_path: Path,
                 subset: str,
                 lines_allowed: List[int],
                 balance_dataset: bool = False,
                 empty_dropout: float = 0.0,
                 augment_dropout: float = 0.0,
                 ram_cache=False):
        self._dataset = self._load(root_path, subset, balance_dataset, lines_allowed)
        super().__init__(data_preparation_pipeline=Pipeline([
            LoadImageWithCrop(),
        ]), augmentation_pipeline=Pipeline([
            OneOf([PlateExtractor(config.inner_size), PlateExtractorWithAugment(config.inner_size)],
                  probs=[augment_dropout, 1.0 - augment_dropout]),
            Sometimes(KillerAugmentator(), prob=empty_dropout),
            FinalResizeWithAspectRatioAndTranspose(),
            TextToSequence(),
        ]), ram_cache=ram_cache)

    def __len__(self):
        return len(self._dataset)

    def _get_sample(self, index) -> dict:
        return self._dataset[index]

    def _greedy_balance_dataset(self,
                                dataset: Dict[int, List]):
        samples = max([len(annotations) for _, annotations in dataset.items()])
        for lines, annotations in dataset.items():
            total = len(annotations)
            annotations = [annotations[np.random.choice(total)] for _ in range(samples)]
            dataset[lines] = annotations
        return dataset

    def _prepare_plate_text(self,
                            text: str) -> str:
        text = text.replace(' ', '').lower()
        text = re.sub(r" ?\[[^)]+\]", "", text)
        text = re.sub("[^{}]+".format(config.vocab.lower()), "", text)
        return text

    def _load(self, root_path, subset, balance_dataset, lines_allowed):
        with open(str(root_path / f'{subset}.json'), 'rb') as f:
            subset_files = json.load(f)['files']
        subset_files = [f.split('/')[-1] for f in subset_files]

        dataset = {lines: [] for lines in lines_allowed}
        for filename in root_path.glob('**/annotations.pickle'):
            prefix = filename.parent
            with open(str(filename), 'rb') as f:
                annotations = pickle.load(f)
            images = {d['id']: d['file_name'].replace('full/', '') for d in annotations['images']}
            for anno in annotations['annotations']:
                if anno['shape_type'] != 'PolygonPlateShape':
                    continue
                image_fn = prefix / images[anno["image_id"]]
                image_fn_subset = image_fn.name
                box = anno['plate_box']
                lines = int(anno['plate_lines'])
                text = self._prepare_plate_text(anno['plate_text'])
                if not text:
                    continue
                if lines not in lines_allowed:
                    continue
                if not image_fn.exists() or str(image_fn_subset) not in subset_files:
                    continue
                dataset[lines].append({
                    'image_filepath': image_fn,
                    'bbox': box,
                    'text': text,
                    'lines': lines
                })

        if balance_dataset:
            dataset = self._greedy_balance_dataset(dataset)

        lines_counter = {lines: len(annotations) for lines, annotations in dataset.items()}
        dataset = [s for lines, samples in dataset.items() for s in samples]

        lines_stat = ', '.join([f'{lines}: {lines_counter[lines]} ({100. * lines_counter[lines] / len(dataset)}%)'
                                for lines in lines_allowed])
        print(f'Loaded dataset of size {len(dataset)} with lines: {lines_stat}')
        return dataset


if __name__ == '__main__':
    import cv2

    dataset = RealDataset(root_path=Path(config.data_config['root_path']),
                          subset='val',
                          lines_allowed=config.data_config['lines_allowed'],
                          empty_dropout=.1,
                          augment_dropout=.1)
    for image, sequence in dataset:
        print(sequence)
        image = np.transpose(image, [1, 2, 0])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('plate', image)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            exit()
