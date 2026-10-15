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
        # X_np: (n_samples, 2)
        # y_np: (n_samples,)
        self.X = torch.from_numpy(X_np.astype(np.float32))
        self.y = torch.from_numpy(y_np.astype(np.float32)).unsqueeze(1) # (n_samples, 1)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx, :], self.y[idx]