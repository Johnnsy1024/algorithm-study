"""
Author: FaizalFeng fzx401@gmail.com
Date: 2024-03-11 17:33:37
LastEditors: FaizalFeng fzx401@gmail.com
LastEditTime: 2024-03-11 17:35:17
Copyright (c) 2024 by FaizalFeng, All Rights Reserved.
"""

import gurobipy as gp
import numpy as np
from typing import Dict

coefficient = dict()
coefficient["y"] = np.array(-1.045)
coefficient["x"] = -np.r_[1.01:1.10:0.01]
coefficient["b"] = np.array(1000)


class MP(metaclass=Singleton):
    def __init__(self, coefficient: Dict[str, np.ndarray]) -> None:
        self.master_problem = gp.Model("Master Problem")
        self.master_problem.setParam("OutputFlag", 0)
        self.y = self.master_problem.addMVar(1, vtype=gp.GRB.INTEGER, name="y")
        # self.y = np.array(self.master_problem.addMVar(1, vtype=gp.GRB.INTEGER, name="y").tolist())
        # self.y = np.concatenate([self.y, self.master_problem.addMVar(coefficient['x'].shape, vtype=gp.GRB.INTEGER, lb=0, ub=0, name='y_pad').tolist()]).tolist()
        # self.y = gp.MVar.fromlist(self.y)

        self.q = self.master_problem.addMVar(1, lb=-float("inf"), vtype=gp.GRB.CONTINUOUS, name="q")
        self.master_problem.setObjective(coefficient["y"] * self.y - 1 * self.q, sense=gp.GRB.MINIMIZE)
        self.b = coefficient["b"]
        self.rhs = coefficient["b"] - self.y

    def update_constr(self, extrem_pnt: np.ndarray = None, extrem_ray: np.ndarray = None):
        cur_constrs = self.master_problem.getConstrs()
        self.master_problem.remove(cur_constrs)
        if extrem_pnt:
            # 最优性约束
            self.master_problem.addConstr(extrem_pnt * self.rhs - self.q <= 0, name="optimal_constr")
        if extrem_ray:
            # 可行约束
            self.master_problem.addConstr(extrem_ray * self.rhs <= 0, name="feasible_constr")

    def get_lower_bound(self):
        self.master_problem.optimize()
        return self.master_problem.ObjVal

    def get_y_value(self):
        return self.y.X
