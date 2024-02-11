"""
Author: FaizalFeng fzx401@gmail.com
Date: 2024-02-09 22:42:09
LastEditors: FaizalFeng fzx401@gmail.com
LastEditTime: 2024-02-11 14:17:55
Copyright (c) 2024 by FaizalFeng, All Rights Reserved.
"""

from sklearn.preprocessing import MinMaxScaler
from data.dataloader import DataLoaderBuilder
import torch
import matplotlib.pyplot as plt
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Model Evaluate")
    parser.add_argument(
        "pattern",
        type=str,
    )
    args = parser.parse_args()
    model = torch.load("./pth/model.pth")
    dataloader = DataLoaderBuilder("./file/international-airline-passengers.csv")

    # Test data
    input_x_test = dataloader.get_inverse_test_data()[0]
    input_y_test = dataloader.get_inverse_test_data()[1]
    # Train data
    input_x_train = dataloader.get_inverse_train_data()[0]
    input_y_train = dataloader.get_inverse_train_data()[1]
    if args.pattern == "test":
        x_scaler = MinMaxScaler()
        x_scaler.fit(input_x_test.reshape((-1, 1)))

        y_scaler = MinMaxScaler()
        y_scaler.fit(input_y_test)
        y_pred = (
            model(
                torch.tensor(
                    x_scaler.transform(input_x_test.reshape((-1, 1))).reshape(
                        (-1, 10, 1)
                    )
                )
            )
            .detach()
            .numpy()
        )
        # 对比预测结果和真实值
        # Convert y_pred back to original scale
        y_pred_rescaled = y_scaler.inverse_transform(y_pred)

        # Plotting the results
        plt.figure(figsize=(10, 5))
        plt.plot(input_y_test, label="True Values")
        plt.plot(y_pred_rescaled, label="Predictions")
        plt.title("Comparison of True Values and Predictions")
        plt.xlabel("Time Steps")
        plt.ylabel("Number of Passengers")
        plt.legend()
        plt.savefig("./fig/eval.png")
    elif args.pattern == "train":
        x_scaler = MinMaxScaler()
        x_scaler.fit(input_x_train.reshape((-1, 1)))
        y_scaler = MinMaxScaler()
        y_scaler.fit(input_y_train)
        y_pred = (
            model(
                torch.tensor(
                    x_scaler.transform(input_x_train.reshape((-1, 1))).reshape(
                        (-1, 10, 1)
                    )
                )
            )
            .detach()
            .numpy()
        )
        # 对比预测结果和真实值
        # Convert y_pred back to original scale
        y_pred_rescaled = y_scaler.inverse_transform(y_pred)

        # Plotting the results
        plt.figure(figsize=(10, 5))
        plt.plot(input_y_train, label="True Values")
        plt.plot(y_pred_rescaled, label="Predictions")
        plt.title("Comparison of True Values and Predictions")
        plt.xlabel("Time Steps")
        plt.ylabel("Number of Passengers")
        plt.legend()
        plt.savefig("./fig/train.png")
