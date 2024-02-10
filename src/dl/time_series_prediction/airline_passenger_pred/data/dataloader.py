from sklearn.preprocessing import MinMaxScaler
from numpy import ndarray, dtype
from typing import Any
from torch.utils.data import DataLoader
from .dataset import DataSet
import pandas as pd
import numpy as np
import os, sys

os.chdir(sys.path[0])


class DataLoaderBuilder:
    def __init__(
        self,
        file_name: str,
        seq_length: int = 10,
        split: float = 0.8,
    ) -> None:
        self.scaler = MinMaxScaler()
        self.train_x, self.train_y, self.test_x, self.test_y = self._load_data(
            file_name, seq_length, split
        )

    def _load_data(
        self, file_name: str, seq_length: int = 10, split: float = 0.8
    ) -> tuple[
        ndarray[Any, dtype[Any]],
        ndarray[Any, dtype[Any]],
        ndarray[Any, dtype[Any]],
        ndarray[Any, dtype[Any]],
    ]:
        df = pd.read_csv(file_name, sep=",", usecols=[1])
        data = df.values.astype(float)
        # data = self.scaler.fit_transform(data)
        dataset = []
        for i in range(len(data) - seq_length - 1):
            dataset.append(data[i : i + seq_length + 1])
        reshaped_data = np.array(dataset).astype("float32")
        np.random.shuffle(reshaped_data)
        # 对x进行统一归一化，而y则不归一化
        x = reshaped_data[:, :-1]
        y = reshaped_data[:, -1]
        split_boundary = int(reshaped_data.shape[0] * split)
        train_x = self.scaler.fit_transform(
            x[:split_boundary].reshape((-1, 1))
        ).reshape((-1, 10, 1))
        test_x = self.scaler.fit_transform(x[split_boundary:].reshape((-1, 1))).reshape(
            (-1, 10, 1)
        )

        train_y = self.scaler.fit_transform(y[:split_boundary])
        test_y = self.scaler.fit_transform(y[split_boundary:])

        return train_x, train_y, test_x, test_y

    def get_train_dataloader(self, batch_size: int = 6):
        train_dataset = DataSet(self.train_x, self.train_y)
        train_dataloader = DataLoader(
            train_dataset, batch_size, shuffle=True, drop_last=True
        )
        return train_dataloader

    def get_test_dataloader(self, batch_size: int = 3):
        test_dataset = DataSet(self.train_x, self.train_y)
        test_dataloader = DataLoader(
            test_dataset, batch_size, shuffle=True, drop_last=True
        )
        return test_dataloader

    def get_inverse_data(
        self,
    ):  # -> tuple[ndarray[Any, Any], ndarray]:# -> tuple[ndarray[Any, Any], ndarray]:
        return self.scaler.inverse_transform(self.train_x.reshape((-1, 1))).reshape(
            (-1, 10, 1)
        ), self.scaler.inverse_transform(self.train_y)


if __name__ == "__main__":
    d = DataLoaderBuilder("../file/international-airline-passengers.csv")
    tmp = d.get_inverse_data()
