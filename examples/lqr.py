import proxddp

from proxddp import dynamics, manifolds
from proxnlp import constraints

import numpy as np
import matplotlib.pyplot as plt

import tap
import pprint


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


rcost = proxddp.QuadraticCost(Q, R)
rcost = proxddp.CostStack(nx, nu, [rcost], [1.0])
term_cost = proxddp.QuadraticCost(Qf, R)
dynmodel = dynamics.LinearDiscreteDynamics(A, B, c)
stage = proxddp.StageModel(space, nu, rcost, dynmodel)
u_min = -0.17 * np.ones(nu)
u_max = +0.17 * np.ones(nu)
ctrl_box = proxddp.ControlBoxFunction(nx, u_min, u_max)
stage.addConstraint(ctrl_box, constraints.NegativeOrthant())


nsteps = 10
problem = proxddp.TrajOptProblem(x0, nu, space, term_cost)

for i in range(nsteps):
    problem.addStage(stage)

if args.use_term_cstr:
    xtar = 0.1 * np.ones(nx)
    term_fun = proxddp.StateErrorResidual(space, nu, xtar)
    problem.setTerminalConstraint(
        proxddp.StageConstraint(term_fun, constraints.EqualityConstraintSet())
    )

mu_init = 1e-2
verbose = proxddp.VerboseLevel.VERBOSE
solver = proxddp.SolverProxDDP(1e-6, mu_init, verbose=verbose)
his_cb = proxddp.HistoryCallback()
solver.registerCallback(his_cb)
solver.max_iters = 20

u0 = np.zeros(nu)
us_i = [u0] * nsteps
xs_i = proxddp.rollout(dynmodel, x0, us_i)
prob_data = proxddp.TrajOptData(problem)
problem.evaluate(xs_i, us_i, prob_data)

solver.setup(problem)
solver.run(problem, xs_i, us_i)
res = solver.getResults()

print(res)
print("xs")
pprint.pprint(res.xs.tolist())
print("us")
pprint.pprint(res.us.tolist())

plt.subplot(121)
lstyle = {"lw": 0.9, "marker": ".", "markersize": 5}
trange = np.arange(nsteps + 1)
plt.plot(res.xs, ls="-", **lstyle)
if args.use_term_cstr:
    plt.hlines(
        xtar,
        *trange[[0, -1]],
        ls="-",
        lw=1.0,
        colors="k",
        alpha=0.4,
        label=r"$x_\mathrm{tar}$"
    )
plt.legend()
plt.xlabel("Time $i$")

plt.subplot(122)
plt.plot(res.us, **lstyle)
plt.hlines(
    np.concatenate([u_min, u_max]),
    *trange[[0, -1]],
    ls="-",
    colors="k",
    lw=1.5,
    alpha=0.2,
    label=r"$\bar{u}$"
)
plt.title("Controls $u(t)$")
plt.legend()
plt.tight_layout()


plt.figure()
prim_infeas = his_cb.storage.prim_infeas
dual_infeas = his_cb.storage.dual_infeas
plt.plot(prim_infeas)
plt.plot(dual_infeas)
ax = plt.gca()
ax.set_yscale("log")
ax.set_xlabel("Iter")
ax.set_ylabel("Residuals")
plt.tight_layout()

plt.show()
