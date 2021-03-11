import cv2
import numpy as np

from ocr.datasets.struct import AnnotationItem


def convert_bbox(bbox, h, w):
    def clamp(x):
        return min(max(x, 0), 1)

    min_x, max_x = min([clamp(point[0]) for point in bbox]), max([clamp(point[0]) for point in bbox])
    min_y, max_y = min([clamp(point[1]) for point in bbox]), max([clamp(point[1]) for point in bbox])
    return int(min_x * w), int(min_y * h), int(max_x * w), int(max_y * h)


def load_image_with_cropping(anno_item: AnnotationItem):
    img = cv2.imread(str(anno_item.image_filepath))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    min_x, min_y, max_x, max_y = convert_bbox(anno_item.bbox, h, w)
    return img[min_y:max_y, min_x: max_x]


def debug_sample(sample):
    import cv2

    def _unprocess_img(img):
        img = (img.cpu().numpy() * 255.).astype(np.uint8)
        return np.transpose(img, (1, 2, 0))[..., ::-1].copy()

    for field, data in sample.items():
        if field.startswith('image'):
            img = _unprocess_img(data)
            cv2.namedWindow(field, cv2.WINDOW_KEEPRATIO)
            cv2.imshow(field, img)
            cv2.waitKey(1)
        else:
            print(f'{field}: {data}')

    cv2.waitKey(0)
