import aligator
import numpy as np
import pinocchio as pin

from aligator import manifolds, dynamics


nx = 2
space = manifolds.VectorSpace(nx)
pin.seed(0)

A = np.array([[1.0, -0.2], [10.0, 1.0]])
B = np.eye(nx)
nu = nx
c = np.zeros(nx)

dt = 0.001
ode = dynamics.LinearODE(A, B, c)
dyn_model = dynamics.IntegratorEuler(ode, dt)

w_x = 0.1 * np.eye(nx)
w_u = 1e-3 * np.eye(nu)
rcost = aligator.QuadraticCost(w_x * dt, w_u * dt)

nsteps = 20
Tf = nsteps * dt

stm = aligator.StageModel(rcost, dyn_model)
stages = [stm] * nsteps

term_cost = rcost.copy()
term_cost.w_x /= dt
term_cost.w_u /= dt

x0 = space.rand()
problem = aligator.TrajOptProblem(x0, stages, term_cost)

xs_init = [x0] * (nsteps + 1)
us_init = [np.zeros(nu)] * nsteps

mu_init = 0.001
rho_init = 0.0
tol = 1e-5
verbose = aligator.VerboseLevel.VERBOSE

solver = aligator.SolverProxDDP(tol, mu_init, rho_init, verbose=verbose)
# solver = aligator.SolverFDDP(tol, verbose=verbose)

solver.setup(problem)
solver.run(problem, xs_init, us_init)
results = solver.results

print(results)
