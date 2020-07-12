from typing import Tuple

import cv2

from transformer_ocr.data.augment import augment_real_lpr4
from transformer_ocr.data.data_preparation import BaseDataProcessCallable
import numpy as np


class PlateExtractor(BaseDataProcessCallable):
    def __init__(self, inner_size: Tuple[int, int]):
        self._inner_size = inner_size

    def _random_inner_size(self,
                           lines: int):
        h, w = self._inner_size

        if lines == 0:
            return w, h

        if lines == 1:
            aspect = np.random.uniform(2.0, 8.0)
        else:
            aspect = np.random.uniform(1.0, 4.0)
        if aspect > 1.:
            h = int(h / aspect)
        else:
            w = int(w * aspect)

        return h, w

    def _extract_plate(self, image, bbox, lines):
        h, w = image.shape[:2]
        hto, wto = self._random_inner_size(lines)

        def is_normalized(points):
            return all([(type(coord) is float) or (type(coord) is np.float64) and
                        0. <= coord <= 1. for point in points for coord in point])

        def restore_points_coords(points, image_size):
            if not is_normalized(points):
                return points

            w, h = image_size
            return [[max(min(int(point[0] * w), w), 0),
                     max(min(int(point[1] * h), h), 0)] for point in points]

        polygon = restore_points_coords(bbox, (w, h))

        pts1 = np.float32([[x, y] for x, y in polygon])
        pts2 = np.float32([[0, 0], [wto, 0], [wto, hto], [0, hto]])

        mat = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(image, mat, (wto, hto), flags=cv2.INTER_AREA)

    def _complement_to_inner_size(self,
                                  image: np.array):
        ch, cw = self._inner_size
        h, w, c = image.shape
        image_complemented = np.ones(self._inner_size + (c,), dtype=np.uint8) * 127
        h_shift, w_shift = ch - h, cw - w
        h_shift, w_shift = np.random.randint(h_shift) if h_shift else 0, np.random.randint(w_shift) if w_shift else 0
        image_complemented[h_shift:h_shift + h, w_shift:w_shift + w] = image
        return image_complemented

    def __call__(self, **kwargs):
        assert all([x in kwargs for x in ['image', 'bbox', 'lines']])

        kwargs['image'] = self._extract_plate(kwargs['image'], kwargs['bbox'], kwargs['lines'])
        kwargs['image'] = self._complement_to_inner_size(kwargs['image'])
        return kwargs


class PlateExtractorWithAugment(PlateExtractor):
    def _extract_plate(self, image, bbox, lines):
        image = augment_real_lpr4(image, bbox, self._random_inner_size(lines))
        h, w = map(float, image.shape[:2])
        hi, wi = self._inner_size
        scale = min(hi / h, wi / w)
        return cv2.resize(image, (int(scale * w), int(scale * h)), interpolation=cv2.INTER_AREA)
