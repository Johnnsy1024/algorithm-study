"""
Author: FaizalFeng fzx401@gmail.com
Date: 2024-02-09 22:42:09
LastEditors: FaizalFeng fzx401@gmail.com
LastEditTime: 2024-02-10 22:52:27
Copyright (c) 2024 by FaizalFeng, All Rights Reserved.
"""

from torch.utils.tensorboard.writer import SummaryWriter
from sklearn.preprocessing import MinMaxScaler
from data.dataloader import DataLoaderBuilder
import torch


if __name__ == "__main__":
    model = torch.load("./pth/model.pth")
    writer = SummaryWriter("./viz/")
    dataloader = DataLoaderBuilder("./file/international-airline-passengers.csv")
    input_x = dataloader.get_inverse_data()[0]
    input_y = dataloader.get_inverse_data()[1]
    # input_x = torch.tensor(dataloader.get_inverse_data()[0])
    # input_y = torch.tensor(dataloader.get_inverse_data()[1])
    writer.add_graph(model, torch.tensor(input_x))
    writer.close()

    x_scaler = MinMaxScaler()
    x_scaler.fit(input_x.reshape((-1, 1)))

    y_scaler = MinMaxScaler()
    y_scaler.fit(input_y)
    y_pred = model(x_scaler.transform(input_x.reshape((-1, 1)).reshape((-1, 10, 1))))
