import numpy as np
import cv2

from imgaug import augmenters as iaa

from typing import List, Tuple


__all__ = ['map_pts', 'augment_real_lpr4', 'get_random_interpolation', 'FakeAugmentator', 'KillerAugmentator']


def get_random_interpolation():
    methods = [cv2.INTER_AREA,
               cv2.INTER_NEAREST,
               cv2.INTER_LINEAR,
               cv2.INTER_CUBIC,
               cv2.INTER_LANCZOS4]
    return np.random.choice(methods)

def map_point_common(pt, mat):
    tp = np.array([pt[0], pt[1], 1.0])
    return mat.dot(tp)[:2]


def map_pts(pts, mat, perspective=False):
    xcoords = []
    ycoords = []
    for pt in pts:
        if perspective:
            tmp = cv2.perspectiveTransform(np.float32([[pt]]), mat)[0][0]
        else:
            tmp = map_point_common(pt, mat)
        xx, yy = map(int, tmp)
        xcoords.append(xx)
        ycoords.append(yy)
    return xcoords, ycoords


def map_bbox(bbox, mat, perspective=False):
    x, y, w, h = bbox
    xcoords, ycoords = map_pts([(x, y), (x, y + h), (x + w, y + h), (x + w, y)], mat, perspective)
    x, y = np.min(xcoords), np.min(ycoords)
    return x, y, np.max(xcoords) - x, np.max(ycoords) - y


def rot_skew(center, gain=1.0):
    rot = np.eye(3)
    skew = np.eye(3)
    if np.random.choice([False, True], p=[0.5, 0.5]):
        a = np.random.uniform(-1.5 * gain, 1.5 * gain)
        rot[:2, :] = cv2.getRotationMatrix2D(center, a, 1)
        sk = np.random.uniform(-0.008 * gain, 0.008 * gain)
        skew[0][1] = sk
    else:
        a = np.random.uniform(-6.0 * gain, 6.0 * gain)
        rot[:2, :] = cv2.getRotationMatrix2D(center, a, 1)
    return skew.dot(rot)


def margin(im, bbox, mar=0.15):
    hh, ww = im.shape[:2]
    x, y, w, h = bbox
    rm = lambda length: int(np.random.uniform(0, mar) * length)
    x0, y0 = np.max([0, x - rm(w)]), np.max([0, y - rm(2 * h)])
    x1, y1 = np.min([ww, x + w + rm(w)]), np.min([hh, y + h + rm(2 * h)])
    return im[y0:y1, x0:x1]


def augment_real_lpr4(image: np.array,
                      box: List[int],
                      plate_size: Tuple[int, int]):
    border = 256
    ph, pw = plate_size
    ww, hh = 2 * border + pw, 2 * border + ph

    pts = np.float32([[x * image.shape[1], y * image.shape[0]] for x, y in box])
    pts1 = np.float32([
        [border, border],
        [border + pw, border],
        [border + pw, border + ph],
        [border, border + ph]
    ])

    mat = rot_skew((border + pw // 2, border + ph // 2), gain=3.0)
    x, y, w, h = map_bbox([border, border, pw, ph], mat)

    mat = mat.dot(cv2.getPerspectiveTransform(pts, pts1))
    image = cv2.warpPerspective(image, mat, (ww, hh), flags=cv2.INTER_AREA)
    image = margin(image, (x, y, w, h), 0.3)

    return image


class FakeAugmentator:
    def __init__(self):
        self._pipeline = iaa.Sequential([
                iaa.Sometimes(0.5, iaa.Grayscale(alpha=1.0)),
                iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 128.0))),
                iaa.Sometimes(0.5, iaa.Sharpen(alpha=(0, 1.0), lightness=(0.5, 4.0))),
                iaa.Sometimes(0.5, iaa.ChannelShuffle(p=0.1)),
                iaa.Sometimes(0.5, iaa.MotionBlur(k=5)),
            ], random_order=True)

    def augment(self,
                image: np.array):
        pipeline = self._pipeline.to_deterministic()
        image = pipeline.augment_image(image)
        return image


class KillerAugmentator:
    def __init__(self,
                 inner_size_range: Tuple[int, int],
                 gaussian_blur_sigma: Tuple[int, int],
                 median_blur_k: Tuple[int, int],
                 average_blur_k: Tuple[int, int],
                 compression_blur_sigma: Tuple[int, int],
                 compression_jpeg: Tuple[int, int]):
        self._image_size = 256, 256
        self._inner_size_range = inner_size_range
        self._blur_pipeline = iaa.OneOf([
            iaa.GaussianBlur(sigma=gaussian_blur_sigma),
            iaa.MedianBlur(k=median_blur_k),
            iaa.AverageBlur(k=average_blur_k),
        ])
        self._compression_pipeline = iaa.Sequential([
            iaa.GaussianBlur(sigma=compression_blur_sigma),
            iaa.JpegCompression(compression=compression_jpeg),
        ])

    def _resize(self,
               image: np.array):
        h, w = image.shape[:2]
        inner_size = np.random.randint(self._inner_size_range[0], self._inner_size_range[1])
        if h > w:
            hh = inner_size
            ww = int(w * inner_size / float(h))
        else:
            ww = inner_size
            hh = int(h * inner_size / float(w))
        image = cv2.resize(image, (ww, hh), interpolation=get_random_interpolation())
        image = cv2.resize(image, (w, h), interpolation=get_random_interpolation())
        return image

    def _compress(self,
                 image: np.array):
        compress = self._compression_pipeline.to_deterministic()
        image = compress.augment_image(image)
        return image

    def _blur(self,
             image: np.array):
        blur = self._blur_pipeline.to_deterministic()
        image = blur.augment_image(image)
        return image

    def augment(self,
                image: np.array):
        image = cv2.resize(image, (self._image_size[1], self._image_size[0]), interpolation=cv2.INTER_AREA)
        image = np.random.choice([self._resize, self._blur, self._compress])(image)
        return image


if __name__ == '__main__':
    image = cv2.imread('plate.jpg', cv2.IMREAD_COLOR)

    augmentator = KillerAugmentator(inner_size_range=(20, 30),
                                    gaussian_blur_sigma=(7, 10),
                                    median_blur_k=(17, 21),
                                    average_blur_k=(19, 27),
                                    compression_blur_sigma=(7, 11),
                                    compression_jpeg=(95, 100))
    augmentator = FakeAugmentator()
    while True:
        augmented = augmentator.augment(image)
        original = cv2.resize(image, (augmented.shape[1], augmented.shape[0]), interpolation=cv2.INTER_AREA)
        output = np.concatenate([original, augmented], axis=1)
        output = cv2.resize(output, (1000, 500), interpolation=cv2.INTER_AREA)
        cv2.imshow('image/augmented', output)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            break
