import numpy as np
import torch
from sklearn.datasets import make_moons
from torch.utils.data import Dataset

class MoonsDataset(Dataset):
    def __init__(
        self,
        n_samples: int = 1000,
        shuffle: bool = True,
        noise: float | None = None,
        random_state: int | None = None,
    ) -> None:
        X_np, y_np = make_moons(
            n_samples=n_samples,
            shuffle=shuffle,
            noise=noise,
            random_state=random_state,
        )
        # TODO: Initialize the torch dataset
        pass

    def __len__(self) -> int:
        # TODO: Implement __len__
        pass

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: Implement __getitem__
        pass