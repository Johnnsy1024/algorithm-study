"""
Author: FaizalFeng fzx401@gmail.com
Date: 2024-02-11 14:26:44
LastEditors: FaizalFeng fzx401@gmail.com
LastEditTime: 2024-02-11 14:28:40
Copyright (c) 2024 by FaizalFeng, All Rights Reserved.
"""

from data.dataloader import DataLoaderBuilder
import lightgbm as lgb
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_loader_builder = DataLoaderBuilder(
        "./file/international-airline-passengers.csv"
    )
    train_x, train_y = data_loader_builder.get_train_data()
    test_x, test_y = data_loader_builder.get_test_data()
    lgb_model = lgb.LGBMRegressor()
    lgb_model = lgb_model.fit(train_x.squeeze(axis=-1), train_y.squeeze(axis=-1))
    res = lgb_model.predict(test_x.squeeze(axis=-1))

    fig = plt.figure(dpi=400)
    ax = fig.add_subplot(111)
    ax.plot(test_y.squeeze(axis=-1), label="True Value")
    ax.plot(res, label="Predict Value")  # type: ignore
    plt.legend()
    plt.savefig("./fig/eval_lgb.png")
