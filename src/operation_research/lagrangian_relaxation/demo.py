import numpy as np
import gurobipy as gp
from datetime import datetime
from loguru import logger
import matplotlib.pyplot as plt


class LagrangianProblem:
    def __init__(
        self, obj_coeff: np.ndarray, constraints_coeff: np.ndarray, rhs: np.ndarray
    ):
        self.n_dim = len(obj_coeff)
        self.obj_coeff = obj_coeff
        self.constraints_coeff = constraints_coeff
        self.rhs = rhs
        self.model = gp.Model("Lagrange Relaxtion")
        self.x = self.model.addMVar(self.n_dim, name="x")

    def compute_obj(self, lamd: np.ndarray):
        obj1 = gp.QuadExpr()
        obj1.addTerms(self.obj_coeff, self.x.tolist(), self.x.tolist())
        obj2 = gp.LinExpr()
        for c in range(self.constraints_coeff.shape[0]):
            obj2.addTerms(lamd[c] * self.constraints_coeff[c], self.x.tolist())
            obj2.add(lamd[c] * self.rhs[c])
        self.model.setObjective(obj1 + obj2)

    def solve(self):
        self.model.setParam("OutputFlag", False)
        self.model.optimize()
        self.opt_solution = self.x.X


class DualProblem(LagrangianProblem):
    def __init__(
        self,
        n_constraints: int,
        obj_coeff: np.ndarray,
        # constraints_coeff: np.ndarray,
        rhs: np.ndarray,
        init_lamb: np.ndarray,
    ):
        self.rhs = rhs
        self.lower_bound_hist = []
        self.obj_coeff = obj_coeff
        self.lamd = init_lamb
        self.iteration_time = 0
        self.subgradients = np.zeros(n_constraints)
        datetime_now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        logger.add(f"./log/{datetime_now}lower_bound_trace.log")

    def compute_subgradients(self, lagrange_problem):
        for i in range(self.lamd.shape[0]):
            self.subgradients[i] = (
                np.dot(
                    lagrange_problem.opt_solution.T,
                    lagrange_problem.constraints_coeff[i, :],
                )
                - self.rhs[i]
            )

    def compute_stepsize(self, lagrange_problem):
        # self.step = (417 - self.compute_costfun(subproblem_qip)) / np.linalg.norm(
        #     self.subgradients
        # ) ** 2
        self.step = 0.001

    def update_lamd(self):
        self.lamd = np.maximum(self.lamd + self.step * self.subgradients, 0)
        self.iteration_time += 1

    def compute_lower_bound(self, lagrange_problem):
        costfun_orig = np.dot(
            (lagrange_problem.opt_solution * lagrange_problem.opt_solution).T,
            self.obj_coeff,
        )
        cost_constraints = 0
        for i in range(len(self.lamd)):
            cost_constraints += self.lamd[i] * (
                np.dot(
                    lagrange_problem.opt_solution.T,
                    lagrange_problem.constraints_coeff[i, :],
                )
                - self.rhs[i]
            )
        logger.info(
            f"第{self.iteration_time}轮迭代中lower bound为{costfun_orig + cost_constraints}"
        )
        self.lower_bound_hist.append(costfun_orig + cost_constraints)


if __name__ == "__main__":
    n_constraints = 2
    lamd_init = np.zeros(n_constraints)
    obj_coeff = np.array([0.5, 0.1, 0.5, 0.1, 0.5, 0.1])
    constraints_coeff = np.array([[-1, 0.2, -1, 0.2, -1, 0.2], [-5, 1, -5, 1, -5, 1]])
    rhs = np.array([-48, -250])

    lagrange_problem = LagrangianProblem(obj_coeff, constraints_coeff, rhs)
    dual_problem = DualProblem(n_constraints, obj_coeff, rhs, lamd_init)
    max_itertimes = 500

    for i in range(max_itertimes):
        lagrange_problem.compute_obj(dual_problem.lamd)
        lagrange_problem.solve()
        dual_problem.compute_subgradients(lagrange_problem)
        dual_problem.compute_stepsize(lagrange_problem)
        dual_problem.update_lamd()
        dual_problem.compute_lower_bound(lagrange_problem)

    plt.plot(dual_problem.lower_bound_hist)
    plt.show()
