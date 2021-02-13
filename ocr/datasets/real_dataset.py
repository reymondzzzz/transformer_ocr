import json
import pickle
import re
from pathlib import Path
from typing import List, Union
import numpy as np
import cv2
from torch.utils.data import Dataset

from ocr.datasets.struct import AnnotationItem


class RealDataset(Dataset):
    def __init__(self, dataset_path: Union[Path, str],
                 subset: str,
                 transforms,
                 lines_allowed: List[int],
                 vocab: List[str],
                 balance_dataset: bool = False,
                 debug: bool = False,
                 name: str = ''):
        self.vocab = vocab
        self._dataset = self._load(Path(dataset_path), subset, balance_dataset, lines_allowed)
        self.name = name
        self._to_debug = debug
        self._transfroms = transforms

    def __len__(self):
        return len(self._dataset)

    @staticmethod
    def _convert_bbox(bbox, h, w):
        def clamp(x):
            return min(max(x, 0), 1)
        min_x, max_x = min([clamp(point[0]) for point in bbox]), max([clamp(point[0]) for point in bbox])
        min_y, max_y = min([clamp(point[1]) for point in bbox]), max([clamp(point[1]) for point in bbox])
        return int(min_x * w), int(min_y * h), int(max_x * w), int(max_y * h)

    @staticmethod
    def _load_image_with_cropping(anno_item: AnnotationItem):
        img = cv2.imread(str(anno_item.image_filepath))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        min_x, min_y, max_x, max_y = RealDataset._convert_bbox(anno_item.bbox, h, w)
        return img[min_y:max_y, min_x: max_x]


    def __getitem__(self, idx):
        item = self._dataset[idx]
        img = self._load_image_with_cropping(item)
        res = self._transfroms(image=img, text=item.text)
        if self._to_debug:
            self._debug(res)
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

    def _debug(self, sample):
        import cv2

        def _unprocess_img(img):
            img = (img.cpu().numpy() * 255.).astype(np.uint8)
            return np.transpose(img, (1, 2, 0))[..., ::-1].copy()

        def _draw_kps(img, kps):
            for x, y in kps[:, :2]:
                x, y = int(x * img.shape[1]), int(y * img.shape[0])
                cv2.circle(img, (x, y), 1, (255, 255, 255), -1)

        for field, data in sample.items():
            if field.startswith('image'):
                img = _unprocess_img(data)
                cv2.namedWindow(field, cv2.WINDOW_KEEPRATIO)
                cv2.imshow(field, img)
                cv2.waitKey(1)
            elif field == '3dfa_keypoints':
                img = _unprocess_img(sample['image'])
                _draw_kps(img, data)
                cv2.namedWindow(field, cv2.WINDOW_KEEPRATIO)
                cv2.imshow(field, img)
                cv2.waitKey(1)
            elif field.startswith('kps_series_'):
                series_idx = field.split('_')[-1]
                img = _unprocess_img(sample[f'image_series_{series_idx}'])
                _draw_kps(img, data)
                cv2.namedWindow(field, cv2.WINDOW_KEEPRATIO)
                cv2.imshow(field, img)
                cv2.waitKey(1)
            else:
                print(f'{field}: {data}')

        cv2.waitKey(0)