"""
Simple example showing how to use a non-StateErrorResidual initial condition in 2D.
"""

import aligator
import numpy as np
import matplotlib.pyplot as plt

from aligator import manifolds
from aligator import dynamics


space = manifolds.R2()
ndx = 2
nu = 2
A = np.eye(ndx)
A[0, 1] = 0.1
B = np.eye(ndx)


x0 = space.rand()
INIT_COND_IDX = 1
x0[1] = 0.1
init_cond = aligator.StateErrorResidual(space, nu, x0)
init_cond = init_cond[INIT_COND_IDX]
print(init_cond)

stages = []
xtarget = space.neutral()
xtarget[1] = 0.0
term_cost = aligator.QuadraticStateCost(space, nu, xtarget, np.eye(ndx))

problem = aligator.TrajOptProblem(init_cond, term_cost)

dm = dynamics.LinearDiscreteDynamics(A, B, c=np.zeros(2))
cost = aligator.QuadraticControlCost(space, nu, np.eye(nu) * 1e-3)
stage = aligator.StageModel(cost, dm)

nsteps = 100

for i in range(nsteps):
    problem.addStage(stage)

solver = aligator.SolverProxDDP(1e-3, 0.01, verbose=aligator.VERBOSE)
solver.setup(problem)
solver.force_initial_condition = False

us_i = [np.zeros(nu) for _ in range(nsteps)]
xs_i = aligator.rollout(dm, x0, us_i)


flag = solver.run(problem, xs_i, us_i)
print(flag)


res = solver.results
xs = np.stack(res.xs)
us = np.stack(res.us)
plt.plot(*xs.T)
lab = "desired $x_0$ "
sc = plt.scatter(*x0, zorder=2)
plt.scatter(*xtarget, label="$x_\\mathrm{tgt}$", zorder=2)
if INIT_COND_IDX == 0:
    lab += "(only horiz. component counts)"
    plt.vlines(x0[0], *plt.ylim(), colors="k", linestyles="--", zorder=1)
elif INIT_COND_IDX == 1:
    lab += "(only vert. component counts)"
    plt.hlines(x0[1], *plt.xlim(), colors="k", linestyles="--", zorder=1)
sc.set_label(lab)
plt.legend()
plt.show()
