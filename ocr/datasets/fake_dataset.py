from torch.utils.data import Dataset

from .fake_generator import FakeGenerator
from .utils import debug_sample


class FakeDataset(Dataset):
    def __init__(self, capacity: int,
                 generator_config: dict,
                 transforms,
                 debug: bool = False,
                 name: str = ''):
        self._capacity = capacity
        self._generator = FakeGenerator(settings=generator_config)
        self._transforms = transforms
        self._to_debug = debug
        self.name = name

    def __len__(self):
        return self._capacity

    def __getitem__(self, item):
        item = self._generator.generate_one_plate()
        res = self._transforms(image=item.image, text=item.text)
        if self._to_debug:
            debug_sample(res)
        return res['image'], res['text']
