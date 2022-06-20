import proxddp

from proxddp import dynamics, manifolds
from proxnlp import constraints

import numpy as np
import matplotlib.pyplot as plt

import tap


class Args(tap.Tap):
    use_term_cstr: bool = False


args = Args().parse_args()
print(args)

np.random.seed(42)
nx = 3
nu = 3
space = manifolds.VectorSpace(nx)
x0 = space.neutral() + (0.2, 0.3, -0.1)
x0 = np.clip(x0, -10, 10)

A = np.eye(nx)
B = np.eye(nx)[:, :nu]
B[2, :] = 0.4
c = np.zeros(nx)

Qroot = np.random.randn(20, nx)
Q = Qroot.T @ Qroot / 20 * 1e-2
R = 1e-2 * np.eye(nu)

Qf = np.eye(nx)


class ControlBoxFunction(proxddp.StageFunction):
    def __init__(self, nx, nu, u_min, u_max) -> None:
        super().__init__(nx, nu, nr=2 * nu)
        self.u_min = u_min
        self.u_max = u_max

    def evaluate(self, x, u, y, data):
        data.value[:] = np.concatenate([u - self.u_max, self.u_min - u])

    def computeJacobians(self, x, u, y, data):
        nu = self.nu
        data.jac_buffer_[:] = 0.
        data.Ju[:nu, :] = np.eye(nu)
        data.Ju[nu:, :] = -np.eye(nu)


rcost = proxddp.QuadraticCost(Q, R)
rcost = proxddp.CostStack(nx, nu, [rcost], [1.])
term_cost = proxddp.QuadraticCost(Qf, R)
dynmodel = dynamics.LinearDiscreteDynamics(A, B, c)
stage = proxddp.StageModel(space, nu, rcost, dynmodel)
u_min = -0.1 * np.ones(nu)
u_max = +0.1 * np.ones(nu)
# ctrl_box = ControlBoxFunction(nx, nu, u_min, u_max)
ctrl_box = proxddp.ControlBoxFunction(nx, u_min, u_max)

stage.add_constraint(proxddp.StageConstraint(ctrl_box, constraints.NegativeOrthant()))

use_term_cstr = args.use_term_cstr


nsteps = 20
problem = proxddp.ShootingProblem(x0, nu, space, term_cost)
for i in range(nsteps):
    if i == nsteps - 1 and use_term_cstr:
        xtar = np.ones(nx)
        term_fun = proxddp.LinearFunction(np.zeros((nx, nx)),
                                          np.zeros((nx, nu)),
                                          np.eye(nx),
                                          xtar)
        stage.add_constraint(
            proxddp.StageConstraint(term_fun, constraints.EqualityConstraintSet())
        )
    problem.addStage(stage)

res = proxddp.Results(problem)
workspace = proxddp.Workspace(problem)
mu_init = 1e-4
verbose = proxddp.VerboseLevel.VERBOSE
solver = proxddp.ProxDDP(1e-6, mu_init, verbose=verbose)

u0 = np.zeros(nu)
us_i = [u0] * nsteps
xs_i = proxddp.rollout(dynmodel, x0, us_i)

solver.run(problem, workspace, res, xs_i, us_i)

print(res)
print(res.us.tolist())
print(workspace)

plt.subplot(121)
plt.plot(res.xs, ls='--', lw=1.)
plt.xlabel("Time $i$")

plt.subplot(122)
plt.plot(res.us, ls='--', lw=1.)
plt.title("Controls $u(t)$")

plt.tight_layout()
plt.show()
