"""
Author: FaizalFeng fzx401@gmail.com
Date: 2024-03-11 17:33:37
LastEditors: FaizalFeng fzx401@gmail.com
LastEditTime: 2024-03-11 17:35:17
Copyright (c) 2024 by FaizalFeng, All Rights Reserved.
"""

# import gurobipy as gp

# master_problem = gp.Model("Master Problem")
# y = master_problem.addVar(vtype=gp.GRB.INTEGER, name="y")
# q = master_problem.addVar(lb=float("inf"), vtype=gp.GRB.CONTINUOUS, name="q")
# master_problem.setObjective(1.045 * y + q)
# master_problem.optimize()

from gurobipy import *

MP = Model("Benders decomposition-MP")

""" create decision variables """
y = MP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name="y")
# z = MP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name="z")
q = MP.addVar(lb=-float("inf"), vtype=GRB.CONTINUOUS, name="q")
# MP.addConstr(z - 1.045 * y + q)
MP.setObjective(1.045 * y + q, GRB.MAXIMIZE)

MP.addConstr(1000 - y >= 0, name="benders feasibility cut iter 1")

MP.optimize()
print("\n\n\n")
print("Obj:", MP.ObjVal)
print("y = %4.1f" % (y.x))
