import proxddp
import tap

import numpy as np
import matplotlib.pyplot as plt

from proxddp import dynamics, constraints, manifolds
from utils import plot_convergence


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
N = 1e-5 * np.eye(nx, nu)

Qf = np.eye(nx)
if args.term_cstr:
    Qf[:, :] = 0.0


rcost0 = proxddp.QuadraticCost(Q, R, N)
assert np.allclose(rcost0.w_x, Q)
assert np.allclose(rcost0.w_u, R)
assert np.allclose(rcost0.weights_cross, N)
assert rcost0.has_cross_term
term_cost = proxddp.QuadraticCost(Qf, R)
dynmodel = dynamics.LinearDiscreteDynamics(A, B, c)
stage = proxddp.StageModel(rcost0, dynmodel)
if args.bounds:
    u_min = -0.18 * np.ones(nu)
    u_max = +0.18 * np.ones(nu)
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

if args.bounds:
    mu_init = 1e-3
else:
    mu_init = 1e-6
rho_init = 0.0
verbose = proxddp.VerboseLevel.VERBOSE
tol = 1e-6
solver = proxddp.SolverProxDDP(tol, mu_init, rho_init, verbose=verbose)


class CustomCallback(proxddp.BaseCallback):
    def __init__(self):
        super().__init__()
        self.active_sets = []
        self.x_dirs = []
        self.u_dirs = []
        self.lams = []
        self.Qus = []
        self.kkts = []

    def call(self, workspace: proxddp.Workspace, results: proxddp.Results):
        import copy

        self.active_sets.append(workspace.active_constraints.tolist())
        self.x_dirs.append(copy.deepcopy(workspace.dxs.tolist()))
        self.u_dirs.append(copy.deepcopy(workspace.dus.tolist()))
        self.lams.append(copy.deepcopy(results.lams.tolist()))
        Qus = [qq.Qu.copy() for qq in workspace.q_params]
        self.Qus.append(Qus)
        kkts = workspace.kkt_mat
        self.kkts.append(copy.deepcopy(kkts))

        def infNorm(xs):
            return max([np.linalg.norm(x, np.inf) for x in xs])

        print("Lxs: ", end="")
        Lxs = workspace.Lxs.tolist()
        if solver.force_initial_condition:
            Lxs = Lxs[1:]
        print("norm = {}".format(infNorm(Lxs)))
        Lus = workspace.Lus.tolist()
        print("Lus: ", end="")
        print("norm = {}".format(infNorm(Lus)))


cus_cb = CustomCallback()
solver.registerCallback("cus", cus_cb)
his_cb = proxddp.HistoryCallback()
solver.registerCallback("his", his_cb)
solver.max_iters = 20

u0 = np.zeros(nu)
us_i = [u0] * nsteps
xs_i = proxddp.rollout(dynmodel, x0, us_i)
prob_data = proxddp.TrajOptData(problem)
problem.evaluate(xs_i, us_i, prob_data)

solver.setup(problem)
for i in range(nsteps):
    psc = solver.workspace.getConstraintScaler(i)
    if args.bounds:
        psc.set_weight(100, 1)
solver.run(problem, xs_i, us_i)
res = solver.results
ws = solver.workspace

print(res)

plt.subplot(121)
fig1: plt.Figure = plt.gcf()

lstyle = {"lw": 0.9, "marker": ".", "markersize": 5}
trange = np.arange(nsteps + 1)
plt.plot(res.xs, ls="-", **lstyle)

if args.term_cstr:
    plt.hlines(
        xtar2,
        *trange[[0, -1]],
        ls="-",
        lw=1.0,
        colors="k",
        alpha=0.4,
        label=r"$x_\mathrm{tar}$",
    )
plt.hlines(
    0.0,
    *trange[[0, -1]],
    ls=":",
    lw=0.6,
    colors="k",
    alpha=0.4,
    label=r"$x=0$",
)
plt.title("State trajectory $x(t)$")
plt.xlabel("Time $i$")
plt.legend(frameon=False)

plt.subplot(122)
plt.plot(res.us, **lstyle)
if args.bounds:
    plt.hlines(
        np.concatenate([u_min, u_max]),
        *trange[[0, -1]],
        ls="-",
        colors="tab:red",
        lw=1.8,
        alpha=0.4,
        label=r"$\bar{u}$",
    )
plt.xlabel("Time $i$")
plt.title("Controls $u(t)$")
plt.legend(frameon=False, loc="lower right")
plt.tight_layout()


fig2: plt.Figure = plt.figure()
ax: plt.Axes = fig2.add_subplot()
niter = res.num_iters
ax.hlines(
    tol,
    0,
    niter,
    colors="k",
    linestyles="-",
    linewidth=2.0,
    label="$\\epsilon_\\mathrm{tol}$",
)
plot_convergence(his_cb, ax, res)
ax.set_title("Convergence (constrained LQR)")
fig2.tight_layout()

plt.show()

fig_dict = {"traj": fig1, "conv": fig2}
TAG = "LQR"
if args.bounds:
    TAG += "_bounded"
if args.term_cstr:
    TAG += "_cstr"
for name, fig in fig_dict.items():
    fig.savefig(f"assets/{TAG}_{name}.png")
    fig.savefig(f"assets/{TAG}_{name}.pdf")
