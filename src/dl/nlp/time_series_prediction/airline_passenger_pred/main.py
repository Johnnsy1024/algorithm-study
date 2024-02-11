"""
Author: FaizalFeng fzx401@gmail.com
Date: 2024-02-09 15:32:49
LastEditors: FaizalFeng fzx401@gmail.com
LastEditTime: 2024-02-10 22:42:49
Copyright (c) 2024 by FaizalFeng, All Rights Reserved.
"""

from sklearn.metrics import (
    mean_absolute_percentage_error as mape,
    mean_absolute_error as mae,
)
from data.dataloader import DataLoaderBuilder
from model.lstm2linear import LSTM2LINEAR

# from model.lightgbm import lgb_model
from torch.utils.data import DataLoader
from loguru import logger
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import lightgbm as lgb
import argparse


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    device: str = "cpu",
    epoch: int = 100,
):
    """
    Args:
        model (nn.Module): Custom Model
        train_dataloader (DataLoader): Train dataloader
        test_dataloader (DataLoader): Test dataloader
        device (str, optional): Device of training. Defaults to "cpu".
        epoch (int, optional): Set epoch number. Defaults to 100.
    """
    model.train()
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loss_hist = []
    test_loss_hist = []
    train_mape_hist = []
    test_mape_hist = []
    for i in range(epoch):
        loss_record = 0
        mape_record = 0
        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            mape_val = mae(y_pred.detach().cpu().numpy(), y.detach().cpu().numpy())
            mape_record += mape_val
            loss = loss_func(y_pred, y)
            loss_record += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        logger.info(
            f"Epoch {i + 1} loss: {loss_record / len(train_dataloader)}, \
            mape: {mape_record / len(train_dataloader)}"
        )
        train_loss_hist.append(loss_record / len(train_dataloader))
        train_mape_hist.append(mape_record / len(train_dataloader))
        if (i + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                mape_record = 0
                loss_record = 0
                for x, y in test_dataloader:
                    x = x.to(device)
                    y = y.to(device)
                    y_pred = model(x)
                    mape_val = mape(y_pred.cpu(), y.cpu())
                    mape_record += mape_val
                    loss = loss_func(y_pred, y)
                    loss_record += loss.item()
                logger.info(
                    f"Epoch {(i + 1) // 10} test loss: \
                        {loss_record / len(test_dataloader)},\
                        test mape: {mape_record / len(test_dataloader)}"
                )
                test_loss_hist.append(loss_record / len(test_dataloader))
                test_mape_hist.append(mape_record / len(test_dataloader))
    fig = plt.figure(dpi=400)
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    (l1,) = ax.plot(train_loss_hist, ls="-", color="red", label="train_loss")
    (l2,) = ax.plot(train_mape_hist, ls="--", color="blue", label="train_mape")
    (l3,) = ax2.plot(test_loss_hist, ls="-", color="red", label="test_loss")
    (l4,) = ax2.plot(test_mape_hist, ls="--", color="blue", label="test_mape")
    ax.legend(handles=[l1, l2])
    ax2.legend(handles=[l3, l4])
    plt.savefig("./fig/train.png")
    torch.save(model, "./pth/model.pth")


if __name__ == "__main__":
    # device = (
    #     "cuda"
    #     if torch.cuda.is_available()
    #     else "mps" if torch.backends.mps.is_available() else "cpu"
    # )
    device = "cpu"
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--model", "-m", type=str, default="lstm", help="Setting the model to train"
    )
    parse.add_argument(
        "--epoch",
        "-e",
        type=int,  # type: ignore
        default=1000,
        help="Setting the epoch number to train",
    )
    args = parse.parse_args()
    dataloader = DataLoaderBuilder("./file/international-airline-passengers.csv")
    if args.model == "lstm":
        train_dataloader = dataloader.get_train_dataloader()
        test_dataloader = dataloader.get_test_dataloader()
        model = LSTM2LINEAR().to(device)
        train(model, train_dataloader, test_dataloader, device, epoch=args.epoch)
    elif args.model == "lgb":
        train_x, train_y = dataloader.get_train_data()
        test_x, test_y = dataloader.get_test_data()
        lgb_model = lgb.LGBMRegressor()
        lgb_model = lgb_model.fit(train_x.squeeze(axis=-1), train_y.squeeze(axis=-1))
        print(lgb_model.evals_result_)
