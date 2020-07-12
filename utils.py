import os
from typing import List, Tuple

import cv2
import numpy as np


def load_image_with_crop(image: np.array,
                         box: List[int],
                         minimal_size: float = 64.0,
                         border_percent: Tuple[float, float] = (1.0, 0.5)):
    h, w = image.shape[:2]
    box = [[int(x * w), int(y * h)] for x, y in box]

    py0, py1 = min([y for _, y in box]), max([y for _, y in box])
    px0, px1 = min([x for x, _ in box]), max([x for x, _ in box])
    ph, pw = py1 - py0, px1 - px0

    scale = 1.0
    if ph > minimal_size and pw > minimal_size:
        scale = min(minimal_size / ph, minimal_size / pw)

    y0, y1 = max(0, py0 - int(border_percent[0] * ph)), min(h, py1 + int(border_percent[0] * ph))
    x0, x1 = max(0, px0 - int(border_percent[1] * pw)), min(w, px1 + int(border_percent[1] * pw))
    box = [[float(x - x0) / (x1 - x0), float(y - y0) / (y1 - y0)] for x, y in box]

    image = image[y0:y1, x0:x1, :]
    h, w = image.shape[:2]
    image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    return image, box


def resize_with_aspect_ratio(img: np.array,
                             size: Tuple[int, int]):
    h, w, c = img.shape
    if w > h:
        nw, nh = size[1], int(h * size[0] / w)
        if nh < 10: nh = 10
        img = cv2.resize(img, (nw, nh))
        a1 = int((size[0] - nh) / 2)
        a2 = size[0] - nh - a1
        pad1 = np.zeros((a1, size[0], c), dtype=np.uint8)
        pad2 = np.zeros((a2, size[0], c), dtype=np.uint8)
        img = np.concatenate((pad1, img, pad2), axis=0)
    else:
        nw, nh = int(w * size[1] / h), size[0]
        if nw < 10: nw = 10
        img = cv2.resize(img, (nw, nh))
        a1 = int((size[1] - nw) / 2)
        a2 = size[1] - nw - a1
        pad1 = np.zeros((size[1], a1, c), dtype=np.uint8)
        pad2 = np.zeros((size[1], a2, c), dtype=np.uint8)
        img = np.concatenate((pad1, img, pad2), axis=1)
    return img

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)
