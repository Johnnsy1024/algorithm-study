from typing import Any
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time

import os, sys

os.chdir(sys.path[0])

# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps" if torch.backends.mps.is_available() else "cpu"
# )
device = "cpu"


def load_data(file_name, sequence_length=10, split=0.8):
    # 用前10个时间段的数据去预测第11个时间段的数据
    df = pd.read_csv(file_name, sep=",", usecols=[1])
    data_all = np.array(df).astype(float)
    scaler = MinMaxScaler()
    data_all = scaler.fit_transform(data_all)
    data = []
    for i in range(len(data_all) - sequence_length - 1):
        data.append(data_all[i : i + sequence_length + 1])
    reshaped_data = np.array(data).astype("float64")
    np.random.shuffle(reshaped_data)
    # 对x进行统一归一化，而y则不归一化
    x = reshaped_data[:, :-1]
    y = reshaped_data[:, -1]
    split_boundary = int(reshaped_data.shape[0] * split)
    train_x = x[:split_boundary]
    test_x = x[split_boundary:]

    train_y = y[:split_boundary]
    test_y = y[split_boundary:]

    return train_x, train_y, test_x, test_y, scaler


class DataSet(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        super().__init__()
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx) -> Any:
        return self.x[idx], self.y[idx]


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=1,
            hidden_size=10,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            device=device,
        )
        self.lstm2 = nn.LSTM(
            input_size=20,
            hidden_size=10,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            device=device,
        )
        self.linear1 = nn.Linear(200, 200, device=device)
        self.linear2 = nn.Linear(100, 10, device=device)
        self.linear3 = nn.Linear(10, 1, device=device)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.tensor):
        x = self.lstm1(x)[0]  # (106, 10, 20)
        self.relu(x)
        x = x.reshape((x.shape[0], -1))  # (106, 200)
        x = self.linear1(x)  # (106, 200)
        self.relu(x)
        x = x.reshape((x.shape[0], 10, 20))  # (106, 10, 20)
        x = self.lstm2(x)[0]  # (106, 10, 10)
        self.relu(x)
        x = x.reshape((x.shape[0], 100))  # (106, 100)
        x = self.linear2(x)  # (106, 10)
        x = self.linear3(x)  # (10, 1)

        return x


def build_model():
    model = Model()
    model = model.to(device)
    return model


def train_model(train_x, train_y, test_x, test_y):
    model = build_model()
    train_dataset = DataSet(train_x, train_y)
    test_dataset = DataSet(test_x, test_y)

    train_dataloader = DataLoader(
        train_dataset, batch_size=10, shuffle=True, drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=5, shuffle=True, drop_last=True
    )

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(100):  # 每10个epoch进行一次验证
        train_loss = 0
        for _, (x, y) in enumerate(train_dataloader):
            x = x.to(torch.float32).to(device)
            y = y.to(torch.float32).to(device)
            y_pred = model(x)
            loss = loss_func(y_pred, y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        logger.info(f"Epoch {epoch + 1} loss: {train_loss / len(train_dataloader)}")
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                eval_indicator = 0
                eval_loss = 0
                for _, (x, y) in enumerate(test_dataloader):
                    x = x.to(torch.float32).to(device)
                    y = y.to(torch.float32).to(device)
                    y_pred = model(x)
                    acc = mean_absolute_percentage_error(y.cpu(), y_pred.cpu())
                    loss = loss_func(y_pred, y)
                    eval_loss += loss.item()
                    eval_indicator += acc
                logger.debug(
                    f"Epoch {epoch + 1} loss: {eval_loss / len(test_dataloader)}, acc: {eval_indicator / len(test_dataloader)}"
                )
    # # try:
    #     model.fit(train_x, train_y, batch_size=512, epochs=300, validation_split=0.1)
    #     predict = model.predict(test_x)
    #     predict = np.reshape(predict, (predict.size,))
    # except KeyboardInterrupt:
    #     print(predict)
    #     print(test_y)
    # print(predict)
    # print(test_y)
    # try:
    #     fig = plt.figure(1)
    #     plt.plot(predict, "r:")
    #     plt.plot(test_y, "g-")
    #     plt.legend(["predict", "true"])
    # except Exception as e:
    #     print(e)
    # return predict, test_y


if __name__ == "__main__":
    train_x, train_y, test_x, test_y, scaler = load_data(
        "./international-airline-passengers.csv"
    )
    start_time = time.time()
    train_model(train_x, train_y, test_x, test_y)
    end_time = time.time()
    logger.info(f"{device} takes {round(end_time - start_time, 1)} seconds to train")
    # fig2 = plt.figure(2)
    # plt.plot(predict_y, "g:")
    # plt.plot(test_y, "r-")
    # plt.show()
