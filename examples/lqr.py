import proxddp
import tap

import numpy as np
import matplotlib.pyplot as plt

from proxddp import dynamics, constraints, manifolds


class Args(tap.Tap):
    term_cstr: bool = False
    bounds: bool = False


args = Args().parse_args()
print(args)

np.random.seed(42)
nx = 3
nu = 3
space = manifolds.VectorSpace(nx)
x0 = space.neutral() + (0.2, 0.3, -0.1)

A = np.eye(nx)
A[0, 1] = -0.2
A[1, 0] = 0.2
B = np.eye(nx)[:, :nu]
B[2, :] = 0.4
c = np.zeros(nx)
c[:] = (0.0, 0.0, 0.1)

Q = 1e-2 * np.eye(nx)
R = 1e-2 * np.eye(nu)

Qf = np.eye(nx)
if args.term_cstr:
    Qf[:, :] = 0.0


rcost0 = proxddp.QuadraticCost(Q, R)
rcost = proxddp.CostStack(nx, nu, [rcost0], [1.0])
term_cost = proxddp.QuadraticCost(Qf, R)
dynmodel = dynamics.LinearDiscreteDynamics(A, B, c)
stage = proxddp.StageModel(rcost, dynmodel)
if args.bounds:
    u_min = -0.15 * np.ones(nu)
    u_max = +0.15 * np.ones(nu)
    ctrl_fn = proxddp.ControlErrorResidual(nx, np.zeros(nu))
    stage.addConstraint(ctrl_fn, constraints.BoxConstraint(u_min, u_max))


nsteps = 20
problem = proxddp.TrajOptProblem(x0, nu, space, term_cost)

for i in range(nsteps):
    problem.addStage(stage)

xtar = space.neutral()
xtar2 = 0.1 * np.ones(nx)
if args.term_cstr:
    term_fun = proxddp.StateErrorResidual(space, nu, xtar2)
    problem.addTerminalConstraint(
        proxddp.StageConstraint(term_fun, constraints.EqualityConstraintSet())
    )

mu_init = 1e-1
rho_init = 0.0
verbose = proxddp.VerboseLevel.VERBOSE
tol = 1e-6
solver = proxddp.SolverProxDDP(tol, mu_init, rho_init, verbose=verbose)
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

plt.subplot(121)
lstyle = {"lw": 0.9, "marker": ".", "markersize": 5}
trange = np.arange(nsteps + 1)
plt.plot(res.xs, ls="-", **lstyle)

plt.hlines(
    xtar,
    *trange[[0, -1]],
    ls="-",
    lw=1.0,
    colors="k",
    alpha=0.4,
    label=r"$x_\mathrm{tar}$"
)
if args.term_cstr:
    plt.hlines(
        xtar2,
        *trange[[0, -1]],
        ls="-",
        lw=1.0,
        colors="k",
        alpha=0.4,
        label=r"$x_\mathrm{tar}$"
    )
plt.xlabel("Time $i$")
plt.title("State trajectory $x(t)$")
plt.legend(frameon=False)

plt.subplot(122)
plt.plot(res.us, **lstyle)
if args.bounds:
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
plt.legend(frameon=False)
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
