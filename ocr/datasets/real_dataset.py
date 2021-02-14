import json
import pickle
import re
from pathlib import Path
from typing import List, Union

from torch.utils.data import Dataset

from ocr.datasets.struct import AnnotationItem
from ocr.datasets.utils import debug_sample, load_image_with_cropping


class RealDataset(Dataset):
    def __init__(self, dataset_path: Union[Path, str],
                 subset: str,
                 transforms,
                 lines_allowed: List[int],
                 vocab: List[str],
                 balance_dataset: bool = False,
                 debug: bool = False,
                 ram_cache: bool = False,
                 name: str = ''):
        self.vocab = vocab
        self.ram_cache = ram_cache
        self._dataset = self._load(Path(dataset_path), subset, balance_dataset, lines_allowed)
        self.name = name
        self._to_debug = debug
        self._transforms = transforms

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        item = self._dataset[idx]
        if self.ram_cache and item.image is None:
            img = load_image_with_cropping(item)
            item.image = img
        elif self.ram_cache:
            img = item.image
        else:
            img = load_image_with_cropping(item)
        res = self._transforms(image=img, text=item.text)
        if self._to_debug:
            debug_sample(res)
        return res['image'], res['text']

    def _prepare_plate_text(self,
                            text: str) -> str:
        text = text.replace(' ', '').lower()
        text = re.sub(r" ?\[[^)]+\]", "", text)
        text = re.sub("[^{}]+".format(self.vocab.lower()), "", text)
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
                dataset[lines].append(AnnotationItem(
                    image_filepath=image_fn,
                    bbox=box,
                    text=text,
                    lines=lines
                ))

        if balance_dataset:
            dataset = self._greedy_balance_dataset(dataset)

        lines_counter = {lines: len(annotations) for lines, annotations in dataset.items()}
        dataset = [s for lines, samples in dataset.items() for s in samples]

        lines_stat = ', '.join([f'{lines}: {lines_counter[lines]} ({100. * lines_counter[lines] / len(dataset)}%)'
                                for lines in lines_allowed])
        print(f'Loaded dataset of size {len(dataset)} with lines: {lines_stat}')
        return dataset
