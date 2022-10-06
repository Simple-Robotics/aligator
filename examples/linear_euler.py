import proxddp
import numpy as np
import pinocchio as pin

from proxddp import manifolds, dynamics


nx = 2
space = manifolds.VectorSpace(nx)

A = np.array([[1.0, -0.2], [10.0, 1.0]])
B = np.eye(nx)
nu = nx
c = np.zeros(nx)

timestep = 0.03
ode = dynamics.LinearODE(A, B, c)
dyn_model = dynamics.IntegratorEuler(ode, timestep)

w_x = 0.1 * np.eye(nx)
w_u = 1e-3 * np.eye(nu)
rcost = proxddp.QuadraticCost(w_x * timestep, w_u * timestep)

nsteps = 20
Tf = nsteps * timestep

stm = proxddp.StageModel(space, nu, rcost, dyn_model)
stages = [stm] * nsteps

term_cost = rcost.copy()
term_cost.w_x /= timestep
term_cost.w_u /= timestep

pin.seed(0)
x0 = space.rand()
problem = proxddp.TrajOptProblem(x0, stages, term_cost)

xs_init = [x0] * (nsteps + 1)
us_init = [np.zeros(nu)] * nsteps

mu_init = 0.001
rho_init = 0.0
tol = 1e-5
verbose = proxddp.VerboseLevel.VERBOSE

solver = proxddp.SolverProxDDP(tol, mu_init, rho_init, verbose=verbose)
# solver = proxddp.SolverFDDP(tol, verbose=verbose)

solver.setup(problem)
solver.run(problem, xs_init, us_init)
results = solver.getResults()

print(results)
