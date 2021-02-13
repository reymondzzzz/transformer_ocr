import torch
from torch import Tensor
from torch import nn
from typing import Union, Tuple


class FlipModelWrapper(nn.Module):
    def __init__(self,
                 feature_extractor: nn.Module,
                 reid_features_number: int):
        super(FlipModelWrapper, self).__init__()
        self.feature_extractor = feature_extractor
        self.reid_features_number = reid_features_number
        self.linear_shuffle = nn.Linear(self.reid_features_number, self.reid_features_number)
        self.dropout = nn.Dropout2d(p=0.6)
        self.flipped_reid_vectors = nn.Linear(self.reid_features_number, self.reid_features_number)

    def forward(self, x: Union[torch.tensor, Tuple[torch.tensor]]):
        if isinstance(x, Tuple):
            x_initial = x[0]
            x_flipped = x[1]
            with torch.no_grad():
                vecs_from_flipped_image = self.feature_extractor(x_flipped)
        else:
            x_initial = x

        vecs_from_initial_image = self.feature_extractor(x_initial)

        vecs_shuffled = self.linear_shuffle(vecs_from_initial_image)
        vecs_shuffled = self.dropout(vecs_shuffled)
        reid_vectors_flipped = self.flipped_reid_vectors(vecs_shuffled)

        if isinstance(x, Tensor):
            reid_vectors = torch.add(vecs_from_initial_image, reid_vectors_flipped) / 2
        else:
            reid_vectors = vecs_from_initial_image

        return reid_vectors if isinstance(x, Tensor) else \
            (reid_vectors, vecs_from_flipped_image, reid_vectors_flipped)
