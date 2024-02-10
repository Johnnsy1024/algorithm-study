from torch.utils.data import Dataset
from numpy import ndarray
import torch
from typing import Any


class DataSet(Dataset):
    def __init__(
        self,
        x: ndarray,
        y: ndarray,
    ) -> None:
        super().__init__()
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx) -> Any:
        return self.x[idx], self.y[idx]
