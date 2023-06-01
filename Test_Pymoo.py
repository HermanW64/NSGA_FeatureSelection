"""
the trial on the example codes from pymoo website
"""

# 1. define the problem
import numpy as np
np.set_printoptions(suppress=True)
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV


class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_ieq_constr=2,
                         xl=np.array([-2,-2]),
                         xu=np.array([2,2]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 100 * (x[0]**2 + x[1]**2)
        f2 = (x[0]-1)**2 + x[1]**2

        g1 = 2*(x[0]-0.1) * (x[0]-0.9) / 0.18
        g2 = - 20*(x[0]-0.4) * (x[0]-0.6) / 4.8

        out["F"] = [f1, f2]
        out["G"] = [g1, g2]


problem = MyProblem()


# 2. initialize algorithm
algorithm = NSGA2(
    pop_size=40,
    n_offsprings=40,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)

# 3. set termination condition
# -- note: each generation produces only one solution
termination = get_termination("n_gen", 500)

# 4. optimization process
print("start optimization process ...")
res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=False)

print("optimization ends ...")
X = res.X
# res.F size is always equal to the population size
F = res.F

# 5. visualize the initial and final pareto fronts
# length of history list is equal to number of the generation
# get the objectives f1 and f2 of the first and last generation
initial_gen_F = res.history[0].pop.get("F")
initial_gen_best = res.history[0].pop.get("best_F")

last_gen_F = res.history[-1].pop.get("F")

# show pareto front in the end
plt.figure(figsize=(7, 5))
plt.scatter(initial_gen_F[:, 0], initial_gen_F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.title("Initial Pareto Front")
plt.xlabel("objective 1")
plt.ylabel("objective 2")
plt.savefig("./pareto_images/initial_pareto.png")
print("initial pareto front plotted!")

# show pareto front in the end
plt.figure(figsize=(7, 5))
plt.scatter(last_gen_F[:, 0], last_gen_F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.title("Final Pareto Front")
plt.xlabel("objective 1")
plt.ylabel("objective 2")
plt.savefig("./pareto_images/final_pareto.png")
print("last pareto front plotted!")

# 6. calculate HV
ref_point = np.array([0, 0])

# get all points of the best pareto front in the last generation
ind = HV(ref_point=ref_point)
print("HV: ", ind(initial_gen_F))

# 7. record the best solutions and minimum classification error
# -- find out the lowest F1 error from the result F
min_f1_index = np.argmin(F[:, 0])
min_f1 = np.round(F[min_f1_index, 0],3)

# -- find out the corresponding solution in X
min_solution = np.round(X[min_f1_index, :], 3)
print("min f1 value: ", min_f1)
print("corresponding solution: ", min_solution)
# print("res X: \n", np.round(X, 3))



