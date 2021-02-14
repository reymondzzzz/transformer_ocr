import sys
from functools import lru_cache
from pathlib import Path
import json

from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QFont, QImage, QFontDatabase, QPainter, QPen, QColor, QPixmap
from PyQt5.QtSvg import QSvgRenderer
from PyQt5.QtWidgets import QApplication
import numpy as np
from dataclasses import dataclass
from ocr.datasets.struct import AnnotationItem

_app = QApplication(sys.argv)  # fake app for QFontDatabase


@dataclass
class Symbol:
    sym: str
    x: int
    y: int
    w: int
    h: int
    font_size: int


@lru_cache(maxsize=None)
def get_font(path: Path):
    id = QFontDatabase.addApplicationFont(str(path))
    if id != -1:
        fontstr = QFontDatabase.applicationFontFamilies(id)[0]
        return QFont(fontstr)


def qimage_to_nparr(image):
    '''  Converts a QImage into an opencv MAT format  '''

    image = image.convertToFormat(QImage.Format.Format_RGB888)

    width = image.width()
    height = image.height()

    ptr = image.bits()
    ptr.setsize(height * width * 3)
    arr = np.array(ptr, np.uint8).reshape(height, width, 3)  # Copies the data
    return arr


class FakeGenerator:
    def __init__(self, settings: dict):
        self._settings: dict = settings
        self._templates: dict = self._load_templates()
        self._max_text_length = max([len(d['symbols']) for _, d in self._templates.items()])

    @property
    def max_text_length(self):
        return self._max_text_length

    def _load_templates(self):
        templates = {}
        root_path = Path(self._settings['lpr_resources'])

        templates_path = root_path / 'templates'
        for templates_path in templates_path.glob('*.json'):
            template_name = templates_path.stem
            with open(str(templates_path), 'rb') as f:
                templates[template_name] = json.load(f)
        return templates

    def _get_template(self):
        all_template_names = list(self._templates.keys())
        most_popular_templates = self._settings[
            'most_popular_templates'] if 'most_popular_templates' in self._settings else {}
        sum_prob_popular_templates = sum([prob for _, prob in most_popular_templates.items()])
        assert sum_prob_popular_templates <= 1.0
        other_prob = (1.0 - sum_prob_popular_templates) / (len(all_template_names) - len(most_popular_templates.keys()))
        probs = [other_prob] * len(all_template_names)
        for name, prob in most_popular_templates.items():
            index = all_template_names.index(name)
            probs[index] = prob
        template_name = np.random.choice(a=all_template_names, size=1, p=probs)[0]
        return self._templates[template_name]

    def _draw_plate(self, background_path, font_path, target_template, symbols):

        if background_path.suffix == '.svg':
            renderer = QSvgRenderer(str(background_path))
            assert renderer.isValid()
            width = target_template['width']
            w, h = renderer.viewBox().width(), renderer.viewBox().height()
            aspect_ratio = w / h
            w, h = width, width / aspect_ratio
            image = QImage(w, h, QImage.Format_ARGB32)
            image.fill(Qt.transparent)  # workaround for clean image
            painter = QPainter(image)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setRenderHint(QPainter.TextAntialiasing, True)
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            renderer.render(painter)
        else:
            image = QPixmap(str(background_path))
            assert not image.isNull()
            width = target_template['width']
            w, h = image.width(), image.height()
            aspect_ratio = w / h
            w, h = width, width / aspect_ratio
            image.scaled(w, h, Qt.KeepAspectRatio)
            image = image.toImage()
            painter = QPainter(image)

        font = get_font(font_path)
        pen = QPen(QColor.fromRgb(int(target_template['font_color'], base=16)), 2, Qt.SolidLine)
        painter.setPen(pen)

        for _, sym in symbols.items():
            rect = QRect(sym.x, sym.y, sym.w, sym.h)
            font.setPointSizeF(sym.font_size)
            painter.setFont(font)
            painter.drawText(rect, Qt.AlignCenter, sym.sym)

        painter.end()

        return qimage_to_nparr(image)

    def generate_one_plate(self):
        target_template = self._get_template()
        root_path = Path(self._settings['lpr_resources'])

        background_path = root_path / 'backgrounds' / target_template['background_image']
        font_path = root_path / 'fonts' / target_template['font_name']

        all_symbols = {x['idx']: x for x in target_template['symbols']}
        symbols = {}
        for group in target_template['groups']:
            symbols_ids = [int(x) for x in group['symbols'].split(',')]
            values = group['values'].split(',')
            if len(values) <= 1:
                values = values[0]
                if len(values) != len(symbols_ids):
                    values = np.random.choice(list(values), len(symbols_ids))
            else:
                assert all([len(x) == len(symbols_ids) for x in values])
                values = np.random.choice(values, 1)[0]

            for id, sym in zip(symbols_ids, values):
                symbols[id] = (Symbol(sym=sym,
                                      x=all_symbols[id]['coordinates']['x'],
                                      y=all_symbols[id]['coordinates']['y'],
                                      w=all_symbols[id]['coordinates']['w'],
                                      h=all_symbols[id]['coordinates']['h'],
                                      font_size=all_symbols[id]['font_size']))

        image = self._draw_plate(background_path, font_path, target_template, symbols)
        text = ''.join([x.sym for _, x in sorted(symbols.items(), key=lambda x: x[0])]).lower()
        lines = target_template['lines_count']
        bbox = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]
        return AnnotationItem(
            image=image,
            text=text,
            bbox=bbox,
            lines=lines
        )

if __name__ == '__main__':
    import cv2

    generator = FakeGenerator({
        'lpr_resources': '/home/kstarkov/t1s/tech1lpr/lpr_resources',
        'most_popular_templates': {
            'ru_type5_subtype1_lines1': 0.033,
            'ru_type5_subtype2_lines1': 0.033,
            'ru_type5_subtype3_lines1': 0.033,
            'ru_type6_subtype1_lines1': 0.1,
            'ru_type7_subtype1_lines1': 0.8,
        }
    })
    while True:
        sample = generator.generate_one_plate()
        print(sample['text'], sample['lines'])
        image = cv2.cvtColor(sample['image'], cv2.COLOR_RGB2BGR)
        cv2.imshow('plate', image)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            exit()
