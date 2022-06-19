import proxddp

from proxddp import dynamics, manifolds

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
nx = 3
nu = 3
vecspace = manifolds.VectorSpace(nx)
x0 = vecspace.neutral() + (0.2, 0.3, -0.1)
x0 = np.clip(x0, -10, 10)

A = np.eye(nx)
B = np.eye(nx)[:, :nu]
B[2, :] = 0.4
c = np.zeros(nx)

Qroot = np.random.randn(20, nx)
Q = Qroot.T @ Qroot / 20 * 1e-2
R = 1e-2 * np.eye(nu)

Qf = np.eye(nx)

rcost = proxddp.QuadraticCost(Q, R)
term_cost = proxddp.QuadraticCost(Qf, R)
dynmodel = dynamics.LinearDiscreteDynamics(A, B, c)
stage = proxddp.StageModel(vecspace, nu, rcost, dynmodel)

nsteps = 20
problem = proxddp.ShootingProblem(x0, nu, vecspace, term_cost)
for i in range(nsteps):
    problem.add_stage(stage)

res = proxddp.Results(problem)
ws = proxddp.Workspace(problem)
mu_init = 1e-2
verbose = proxddp.VerboseLevel.VERBOSE
solver = proxddp.ProxDDP(1e-6, mu_init, verbose=verbose)

u0 = np.zeros(nu)
us_i = [u0] * nsteps
xs_i = proxddp.rollout(dynmodel, x0, us_i)

solver.run(problem, ws, res, xs_i, us_i)

print(res)


plt.plot(res.xs, ls='--', lw=1.)
plt.xlabel("Time $i$")
plt.show()
