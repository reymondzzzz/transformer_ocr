import cv2
import math
import numpy as np
import random
import torch
from albumentations import ImageOnlyTransform, BasicTransform

__all__ = ['SeriesTransformation']

from ocr.utils.tokenizer import tokenize_vocab


class Tokenizer(BasicTransform):
    def __init__(self, vocab, seq_size):
        super().__init__(always_apply=True, p=1)
        self.letter_to_token, _ = tokenize_vocab(vocab)
        self.seq_size = seq_size

    def _tokenize(self, text, **params):
        tokens = np.array([self.letter_to_token[letter] for letter in ['sos'] + list(text) + ['eos']], dtype=np.int64)
        return torch.from_numpy(tokens)

    def apply_with_params(self, params, force_apply=False, **kwargs):
        if params is None:
            return kwargs
        params = self.update_params(params, **kwargs)
        res = {}

        for key, arg in kwargs.items():
            if key != 'text':
                res[key] = arg
                continue

            if arg is not None:
                target_function = self._get_target_function(key)
                target_dependencies = {k: kwargs[k] for k in self.target_dependence.get(key, [])}
                res[key] = target_function(arg, **dict(params, **target_dependencies))
            else:
                res[key] = arg
        return res

    @property
    def targets(self):
        return {
            'text': self._tokenize
        }


class NullifyText(BasicTransform):
    def __init__(self):
        super().__init__(always_apply=True, p=1)

    def _nullify_text(self, text, **params):
        return ''

    def apply_with_params(self, params, force_apply=False, **kwargs):
        if params is None:
            return kwargs
        params = self.update_params(params, **kwargs)
        res = {}

        for key, arg in kwargs.items():
            if key != 'text':
                res[key] = arg
                continue

            if arg is not None:
                target_function = self._get_target_function(key)
                target_dependencies = {k: kwargs[k] for k in self.target_dependence.get(key, [])}
                res[key] = target_function(arg, **dict(params, **target_dependencies))
            else:
                res[key] = arg
        return res

    @property
    def targets(self):
        return {
            'text': self._nullify_text
        }


class SeriesTransformation(ImageOnlyTransform):
    def __init__(self, series_size=0,
                 pitch_angle=0.05, roll_angle=0.05, yaw_angle=5,
                 dx=3, dy=3, scale=0.1,
                 output_img_prefix='image_series_', always_apply=False, p=0.5, **kwargs):
        super().__init__(always_apply, p)
        self.series_size = series_size
        self.output_img_prefix = output_img_prefix
        # TODO fix this angles and move those arguments to albu params list
        self.pitch_angle = self._to_tuple(pitch_angle)
        self.roll_angle = self._to_tuple(roll_angle)
        self.yaw_angle = self._to_tuple(yaw_angle)
        self.f = 1
        self.dx, self.dy = self._to_tuple(dx), self._to_tuple(dy)  # 5, 5
        self.scale = self._to_tuple(scale)
        self.background_color = (128, 128, 128)

    def _to_tuple(self, value):
        if isinstance(value, tuple) or isinstance(value, list):
            return tuple(value)
        else:
            return (-value, value)

    def _transform(self, img):
        pitch_angle = math.radians(random.uniform(self.pitch_angle[0], self.pitch_angle[1]))
        roll_angle = math.radians(random.uniform(self.roll_angle[0], self.roll_angle[1]))
        yaw_angle = math.radians(random.uniform(self.yaw_angle[0], self.yaw_angle[1]))
        dx, dy = random.uniform(self.dx[0], self.dx[1]), random.uniform(self.dy[0], self.dy[1])
        scale = 1 + random.uniform(self.scale[0], self.scale[1])

        h, w = img.shape[:2]

        cx, cy = w // 2, h // 2

        RZ = np.array([
            math.cos(yaw_angle), math.sin(yaw_angle), 0, 0,
            -math.sin(yaw_angle), math.cos(yaw_angle), 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        ])
        RZ = RZ.reshape((4, 4))

        RX = np.array([
            1, 0, 0, 0,
            0, math.cos(pitch_angle), math.sin(pitch_angle), 0,
            0, -math.sin(pitch_angle), math.cos(pitch_angle), 0,
            0, 0, 0, 1
        ])
        RX = RX.reshape((4, 4))

        RY = np.array([
            math.cos(roll_angle), 0, math.sin(roll_angle), 0,
            0, 1, 0, 0,
            -math.sin(roll_angle), 0, math.cos(roll_angle), 0,
            0, 0, 0, 1
        ])
        RY = RY.reshape((4, 4))

        a2 = np.array([
            self.f, 0, cx + dx, 0,
            0, self.f, cy + dy, 0,
            0, 0, 1, 0
        ])
        a2 = a2.reshape((3, 4))

        a1 = np.array([
            1 / self.f, 0, -cx / self.f,
            0, 1 / self.f, -cy / self.f,
            0, 0, 1,
            0, 0, 0
        ])
        a1 = a1.reshape((4, 3))

        scale_mat = np.array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1 / scale, 0,
            0, 0, 0, 1
        ])
        scale_mat = scale_mat.reshape((4, 4))

        T = np.matmul(RZ, RY)
        T = np.matmul(T, RX)
        T = np.matmul(T, scale_mat)
        H = np.matmul(np.matmul(a2, T), a1).astype(np.float)

        img = cv2.warpPerspective(img, H, (w, h), borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=self.background_color)
        return img

    def apply_with_params(self, params, force_apply=False, **kwargs):
        if params is None:
            return kwargs
        params = self.update_params(params, **kwargs)
        res = {}

        for key, arg in kwargs.items():
            if key != 'image':
                res[key] = arg
                continue

            if arg is not None:
                target_function = self._get_target_function(key)
                target_dependencies = {k: kwargs[k] for k in self.target_dependence.get(key, [])}

                if self.series_size > 0:
                    res[key] = arg
                    for idx in range(self.series_size):
                        res[f'{self.output_img_prefix}{idx}'] = target_function(
                            arg, **dict(params, **target_dependencies))
                else:
                    res[key] = target_function(arg, **dict(params, **target_dependencies))

            else:
                res[key] = None
        return res

    def apply(self, image, **params):
        return self._transform(image)

    def get_params(self):
        return {"series_size": self.series_size}

    def get_transform_init_args_names(self):
        return ("series_size",)
