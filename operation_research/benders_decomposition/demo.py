import gurobipy as gp
import numpy as np
from typing import Any, Dict
from loguru import logger
import matplotlib.pyplot as plt


# 单例实现
class Singleton(type):
    _instance = {}

    def __call__(cls, *args: Any, **kwds: Any) -> Any:
        if cls not in cls._instance:
            cls._instance[cls] = super(Singleton, cls).__call__(*args, **kwds)
        return cls._instance[cls]


# 问题定义
coefficient = dict()
coefficient["y"] = np.r_[[-1.045], [0] * 10]
coefficient["x"] = -np.r_[1.01:1.10:0.01]
coefficient["b"] = np.r_[[1000], [100] * 10]


# 主问题定义
class MP(metaclass=Singleton):
    def __init__(self, coefficient: Dict[str, np.ndarray]) -> None:
        self.master_problem = gp.Model("Master Problem")
        self.master_problem.setParam("OutputFlag", 0)
        self.y = self.master_problem.addMVar(1, vtype=gp.GRB.INTEGER, name="y")
        self.y = np.array(self.y.tolist())
        self.y = np.concatenate(
            [
                self.y,
                self.master_problem.addMVar(
                    coefficient["x"].shape, vtype=gp.GRB.INTEGER, lb=0, ub=0, name="y_pad"
                ).tolist(),
            ]
        ).tolist()
        self.y = gp.MVar.fromlist(self.y)

        self.q = self.master_problem.addMVar(1, lb=-float("inf"), vtype=gp.GRB.CONTINUOUS, name="q")
        self.master_problem.setObjective((coefficient["y"] * self.y).sum() + 1 * self.q, sense=gp.GRB.MINIMIZE)

        self.rhs = coefficient["b"] - self.y

    def update_constr(self, extrem_pnt: np.ndarray = None, extrem_ray: np.ndarray = None) -> None:
        if extrem_pnt is not None and extrem_pnt.size > 0:
            # 最优性约束
            self.master_problem.addConstr((extrem_pnt * self.rhs).sum() - self.q <= 0, name="optimal_constr")
        if extrem_ray is not None and extrem_ray.size > 0:
            # 可行性约束
            self.master_problem.addConstr((extrem_ray * self.rhs).sum() <= 0, name="feasible_constr")

    def get_lb(self) -> float:
        self.master_problem.optimize()
        return self.master_problem.ObjVal

    def get_y_value(self) -> float | Any:
        return self.y.X


# 子问题定义
class SP(metaclass=Singleton):
    def __init__(self, coefficient: Dict[str, np.ndarray]) -> None:
        self.sub_problem = gp.Model("Sub Problem")
        self.sub_problem.setParam("OutputFlag", 0)
        self.x = self.sub_problem.addMVar(coefficient["x"].shape, vtype=gp.GRB.CONTINUOUS, name="x")
        self.sub_problem.setObjective((coefficient["x"] * self.x).sum(), gp.GRB.MINIMIZE)
        self.sub_problem.setParam("InfUnbdInfo", 1)

    def update_y_value(self, y_value: list) -> None:
        self.y_value = y_value
        cur_constraints = self.sub_problem.getConstrs()
        self.sub_problem.remove(cur_constraints)
        self.sub_problem.update()
        self.sub_problem.addConstr(self.x.sum() - 1000 + self.y_value[0] <= 0, name="sub_problem_constr")
        for i in range(coefficient["x"].shape[0]):
            self.sub_problem.addConstr(self.x[i] <= 100)

    def get_fea_status(self) -> int:
        self.sub_problem.optimize()
        return self.sub_problem.Status


if __name__ == "__main__":
    # 初始化上下界
    lb = -float("inf")
    ub = float("inf")
    epsilon = 1e-3
    cur_epoch = 0
    lb_list = []
    ub_list = []

    mp = MP(coefficient)
    sp = SP(coefficient)
    lb = mp.get_lb()
    logger.info(f"Initial lower bound: {lb}")
    y_value = mp.get_y_value()
    logger.info(f"Initial y: {y_value}")
    while ub - lb > epsilon:
        cur_epoch += 1
        # 求解子问题
        sp.update_y_value(y_value)
        if sp.get_fea_status() == 2:
            # 子问题可行，更新原问题上界
            pole_list = []
            for c in sp.sub_problem.getConstrs():
                pole_list.append(c.Pi)
            # logger.info(f"Epoch {cur_epoch} pole: {pole_list}")
            Q_y = sp.sub_problem.ObjVal
            ub = (Q_y + y_value * coefficient["y"])[0]
            ub_list.append(ub)
            logger.info(f"Epoch {cur_epoch} ub: {ub}")

            # 向主问题添加最优性约束
            mp.update_constr(extrem_pnt=np.array(pole_list))
            lb = mp.get_lb()
            lb_list.append(lb)
            logger.info(f"Epoch {cur_epoch} lb: {lb}")
            y_value = mp.get_y_value()

        else:
            # 子问题不可行，更新原问题上界
            farkasdual_list = []
            for c in sp.sub_problem.getConstrs():
                farkasdual_list.append(-c.FarkasDual)  # gurobi中极射线的取FarkasDual的相反数

            logger.info(f"Epoch {cur_epoch} ub: {ub}")
            ub_list.append(ub)
            # 向主问题添加可行性约束
            mp.update_constr(extrem_ray=np.array(farkasdual_list))
            lb = mp.get_lb()
            lb_list.append(lb)
            print(f"Epoch {cur_epoch} lb: {lb}")
            y_value = mp.get_y_value()
    plt.figure(dpi=400)
    lb_list = [i for i in lb_list if i > -1e30]
    ub_list = [i for i in ub_list if i < 1e30]
    plt.plot(range(len(lb_list)), lb_list, label="Lower Bound")
    plt.plot(range(len(ub_list)), ub_list, label="Upper Bound")
    plt.legend()
    plt.savefig("./benders_decomposition.png")
