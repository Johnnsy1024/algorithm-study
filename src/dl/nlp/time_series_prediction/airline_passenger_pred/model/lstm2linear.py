"""
Author: FaizalFeng fzx401@gmail.com
Date: 2024-02-09 15:13:23
LastEditors: FaizalFeng fzx401@gmail.com
LastEditTime: 2024-02-11 10:37:44
Copyright (c) 2024 by FaizalFeng, All Rights Reserved.
"""

import torch
import torch.nn as nn

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


class LSTM2LINEAR(nn.Module):
    def __init__(self) -> None:
        """
        Constructor for the class. Initializes LSTM and linear layers,
        as well as activation and dropout functions.
        """
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=1,
            hidden_size=10,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            device=device,
        )
        self.linear = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.lstm1(x)[1][1]  # (1, batch_size, 10)
        x = x.reshape((x.shape[1], -1))  # (batch_size, 10)
        x = self.linear(x)

        return x
