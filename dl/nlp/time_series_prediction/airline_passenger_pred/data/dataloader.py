"""
Author: FaizalFeng fzx401@gmail.com
Date: 2024-02-09 15:29:10
LastEditors: FaizalFeng fzx401@gmail.com
LastEditTime: 2024-02-11 10:06:48
Copyright (c) 2024 by FaizalFeng, All Rights Reserved.
"""

from sklearn.preprocessing import MinMaxScaler
from numpy import ndarray, dtype
from typing import Any
from torch.utils.data import DataLoader
from .dataset import DataSet
import pandas as pd
import numpy as np
import os
import sys

os.chdir(sys.path[0])


class DataLoaderBuilder:
    def __init__(
        self,
        file_name: str,
        seq_length: int = 10,
        split: float = 0.8,
    ) -> None:
        """
        Initialize the object with the given file name, sequence length, and split ratio.

        Parameters:
            file_name (str): The name of the file to load the data from.
            seq_length (int): The length of the sequence (default is 10).
            split (float): The ratio to split the data into training and
            testing sets (default is 0.8).

        Returns:
            None
        """
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
        """
        Load the data from the given file_name and process it for training and testing.

        Args:
            file_name (str): The name of the file to load the data from.
            seq_length (int): The length of the sequence for data processing (default 10).
            split (float): The split ratio for training and testing data (default 0.8).

        Returns:
            tuple: A tuple containing train_x, train_y, test_x, and test_y.
                train_x (ndarray): The training input data.
                train_y (ndarray): The training output data.
                test_x (ndarray): The testing input data.
                test_y (ndarray): The testing output data.
        """
        df = pd.read_csv(file_name, sep=",", usecols=[1])
        data = df.values.astype(float)
        # data = self.scaler.fit_transform(data)
        dataset = []
        for i in range(len(data) - seq_length - 1):
            dataset.append(data[i : i + seq_length + 1])
        reshaped_data = np.array(dataset, dtype=np.float32)
        # np.random.shuffle(reshaped_data)
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

    def get_train_dataloader(self, batch_size: int = 6) -> DataLoader[Any]:
        """
        Create and return a training data loader for the given training data and batch size.

        Args:
            batch_size (int): The batch size for the data loader.

        Returns:
            DataLoader: The training data loader.
        """
        train_dataset = DataSet(self.train_x, self.train_y)
        train_dataloader = DataLoader(
            train_dataset, batch_size, shuffle=False, drop_last=True
        )
        return train_dataloader

    def get_test_dataloader(self, batch_size: int = 3) -> DataLoader[Any]:
        """
        Create and return a test dataloader using the provided batch size.

        Args:
            batch_size (int): The batch size for the dataloader.

        Returns:
            DataLoader: The test dataloader.
        """

        test_dataset = DataSet(self.train_x, self.train_y)
        test_dataloader = DataLoader(
            test_dataset, batch_size, shuffle=False, drop_last=True
        )
        return test_dataloader

    def get_inverse_train_data(
        self,
    ) -> tuple[ndarray[Any, Any], ndarray]:
        """
        get_inverse_data function takes no parameters and returns a tuple of two ndarrays.
        """
        return self.scaler.inverse_transform(self.train_x.reshape((-1, 1))).reshape(
            (-1, 10, 1)
        ), self.scaler.inverse_transform(self.train_y)

    def get_inverse_test_data(self):
        """
        Return the inverse transformed test data using the scaler.
        """
        return self.scaler.inverse_transform(self.test_x.reshape((-1, 1))).reshape(
            (-1, 10, 1)
        ), self.scaler.inverse_transform(self.test_y)

    def get_train_data(self) -> tuple[ndarray[Any, Any], ndarray]:
        """
        Return the training data as a tuple of numpy arrays.
        """
        return self.train_x, self.train_y

    def get_test_data(self) -> tuple[ndarray[Any, Any], ndarray]:
        """
        Returns the test data as a tuple containing self.test_x and self.test_y.
        """
        return self.test_x, self.test_y


if __name__ == "__main__":
    d = DataLoaderBuilder("../file/international-airline-passengers.csv")
    tmp = d.get_inverse_train_data()
