"""
Author: FaizalFeng fzx401@gmail.com
Date: 2024-03-11 17:33:37
LastEditors: FaizalFeng fzx401@gmail.com
LastEditTime: 2024-03-11 17:35:17
Copyright (c) 2024 by FaizalFeng, All Rights Reserved.
"""

import gurobipy as gp


master_problem = gp.Model("Master Problem")
y = master_problem.addVar(vtype=gp.GRB.INTEGER, name="y")
q = master_problem.addVar(lb=-float("inf"), vtype=gp.GRB.CONTINUOUS, name="q")
master_problem.setObjective(1.045 * y + q, sense=gp.GRB.MINIMIZE)
master_problem.optimize()

print(f"y的值为{y.X}")
print(f"q的值为{q.X}")

sub_problem = gp.Model("Sub Problem")
x = sub_problem.addVar()
