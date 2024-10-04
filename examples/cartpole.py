"""
@Time    :   2022/06/29 15:58:26
@Author  :   quentinll
@License :   (C)Copyright 2021-2022, INRIA
"""

import pinocchio as pin
import numpy as np
import aligator
import matplotlib.pyplot as plt

from utils import create_cartpole, ArgsBase, plot_convergence, get_endpoint_traj
from pinocchio.visualize import MeshcatVisualizer
from aligator import constraints, manifolds


class Args(ArgsBase):
    bounds: bool = False
    term_cstr: bool = False
    num_replay: int = 2


args = Args().parse_args()


model, geom_model, data, geom_data, ddl = create_cartpole(1)
visual_model = geom_model.clone()
dt = 0.01
nu = 1
act_mat = np.zeros((2, nu))
act_mat[0, 0] = 1.0
space = manifolds.MultibodyPhaseSpace(model)
nx = space.nx
ndx = space.ndx
cont_dyn = aligator.dynamics.MultibodyFreeFwdDynamics(space, act_mat)
disc_dyn = aligator.dynamics.IntegratorSemiImplEuler(cont_dyn, dt)

nq = model.nq
nv = model.nv
x0 = space.neutral()
x0[1] = 0.5

target_pos = np.array([0.0, 0.0, 1.0])
frame_id = model.getFrameId("end_effector_frame")

# running cost regularizes the control input
rcost = aligator.CostStack(space, nu)
wu = np.ones(nu) * 1e-2
rcost.addCost(aligator.QuadraticControlCost(space, np.zeros(nu), np.diag(wu) * dt))
frame_place_target = pin.SE3.Identity()
frame_place_target.translation[:] = target_pos
frame_err = aligator.FramePlacementResidual(
    ndx,
    nu,
    model,
    frame_place_target,
    frame_id,
)
weights_frame_place = np.zeros(6)
weights_frame_place[:3] = 1.0
weights_frame_place = np.diag(weights_frame_place)
rcost.addCost(
    aligator.QuadraticResidualCost(space, frame_err, weights_frame_place * dt)
)
term_cost = aligator.CostStack(space, nu)

# box constraint on control
u_min = -6.0 * np.ones(nu)
u_max = +6.0 * np.ones(nu)


def get_box_cstr():
    ctrl_fn = aligator.ControlErrorResidual(ndx, nu)
    return aligator.StageConstraint(ctrl_fn, constraints.BoxConstraint(u_min, u_max))


nsteps = 500
Tf = nsteps * dt
problem = aligator.TrajOptProblem(x0, nu, space, term_cost)

for i in range(nsteps):
    stage = aligator.StageModel(rcost, disc_dyn)
    if args.bounds:
        box_cstr = get_box_cstr()
        stage.addConstraint(box_cstr)
    problem.addStage(stage)

term_fun = aligator.FrameTranslationResidual(ndx, nu, model, target_pos, frame_id)

if args.term_cstr:
    term_cstr = aligator.StageConstraint(term_fun, constraints.EqualityConstraintSet())
    problem.addTerminalConstraint(term_cstr)
else:
    term_cost.addCost(
        aligator.QuadraticResidualCost(space, frame_err, weights_frame_place)
    )


mu_init = 1e-2
verbose = aligator.VerboseLevel.VERBOSE
TOL = 1e-4
MAX_ITER = 300
solver = aligator.SolverProxDDP(TOL, mu_init, max_iters=MAX_ITER, verbose=verbose)
solver.rollout_type = aligator.ROLLOUT_LINEAR
callback = aligator.HistoryCallback()
solver.registerCallback("his", callback)

u0 = np.zeros(nu)
us_i = [u0] * nsteps
xs_i = aligator.rollout(disc_dyn, x0, us_i)

solver.setup(problem)
workspace = solver.workspace
for i in range(nsteps):
    psc = workspace.getConstraintScaler(i)
    if args.bounds:
        psc.set_weight(10.0, 0)
solver.run(problem, xs_i, us_i)
res = solver.results
print(res)

fig1 = plt.figure(figsize=(7.2, 5.4))

xs_opt = np.asarray(res.xs)
trange = np.linspace(0, Tf, nsteps + 1)
gs = plt.GridSpec(2, 1)
gs0 = gs[0].subgridspec(1, 2)

_pts = get_endpoint_traj(model, data, xs_opt, frame_id)
_pts = _pts[:, 1:]

ax1 = fig1.add_subplot(gs0[0])
ax2 = fig1.add_subplot(gs0[1])
lstyle = {"lw": 0.9}
ax1.plot(trange, xs_opt[:, 0], ls="-", **lstyle)
ax1.plot(trange, xs_opt[:, 2], ls="-", label="$\\dot{x}$", **lstyle)
ax1.set_ylabel("$q(t)$")
if args.term_cstr:
    pass
ax1.legend()
ax2.plot(trange, xs_opt[:, 1], ls="-", **lstyle)
ax2.plot(trange, xs_opt[:, 3], ls="-", label="$\\dot{\\theta}$", **lstyle)
ax2.set_ylabel("Angle $\\theta(t)$")
ax2.legend()

plt.xlabel("Time $t$")

gs1 = gs[1].subgridspec(1, 2, width_ratios=[1, 2])
ax3 = plt.subplot(gs1[0])
plt.plot(*_pts.T, ls=":")
plt.scatter(*target_pos[1:], c="r", marker="^", zorder=2, label="EE target")
plt.legend()
ax3.set_aspect("equal")
plt.title("Endpoint trajectory")

plt.subplot(gs1[1])
plt.plot(trange[:-1], res.us, label="$u(t)$", **lstyle)
if args.bounds:
    plt.hlines(
        np.concatenate([u_min, u_max]),
        *trange[[0, -1]],
        ls="-",
        colors="k",
        lw=2.5,
        alpha=0.4,
        label=r"$\bar{u}$",
    )
plt.title("Controls $u(t)$")
plt.legend()
fig1.tight_layout()

fig2 = plt.figure(figsize=(6.4, 4.8))
ax: plt.Axes = plt.subplot(111)
ax.hlines(TOL, 0, res.num_iters, lw=2.2, alpha=0.8, colors="k")
plot_convergence(callback, ax, res)
prim_tols = np.array(callback.prim_tols)
al_iters = np.array(callback.al_iters)

itrange = np.arange(len(al_iters))
legends_ = [
    "$\\epsilon_\\mathrm{tol}$",
    "Prim. err $p$",
    "Dual err $d$",
]
if len(itrange) > 0:
    ax.step(itrange, prim_tols, c="green", alpha=0.9, lw=1.1)
    al_change = al_iters[1:] - al_iters[:-1]
    al_change_idx = itrange[:-1][al_change > 0]
    legends_.extend(
        [
            "Prim tol $\\eta_k$",
            "AL iters",
        ]
    )

    ax.vlines(al_change_idx, *ax.get_ylim(), colors="gray", lw=4.0, alpha=0.5)
ax.legend(
    [
        "$\\epsilon_\\mathrm{tol}$",
        "Prim. err $p$",
        "Dual err $d$",
        "Prim tol $\\eta_k$",
        "AL iters",
    ]
)
fig2.tight_layout()

fig_dict = {"traj": fig1, "conv": fig2}

TAG = "cartpole"
if args.bounds:
    TAG += "_bounds"
if args.term_cstr:
    TAG += "_cstr"

for name, fig in fig_dict.items():
    fig.savefig(f"assets/{TAG}_{name}.png")
    fig.savefig(f"assets/{TAG}_{name}.pdf")

plt.show()

if args.display:
    import hppfcl

    numrep = args.num_replay
    cp = [2.0, 0.0, 0.8]
    cps_ = [cp.copy() for _ in range(numrep)]
    qs = [x[:nq] for x in res.xs.tolist()]
    vs = [x[nq:] for x in res.xs.tolist()]

    obj = pin.GeometryObject("objective", 0, hppfcl.Sphere(0.05), frame_place_target)
    color = [255, 20, 83, 255]
    obj.meshColor[:] = color
    obj.meshColor /= 255
    visual_model.addGeometryObject(obj)

    vizer = MeshcatVisualizer(
        model, geom_model, visual_model, data=data, collision_data=geom_data
    )
    vizer.initViewer(open=args.display, loadModel=True)
    vizer.setBackgroundColor()

    if args.record:
        fps = 1.0 / dt
        filename = "examples/ur5_reach_ctrlbox.mp4"
        ctx = vizer.create_video_ctx(filename, fps=fps)
        print(f"[video will be recorded @ {filename}]")
    else:
        from contextlib import nullcontext

        ctx = nullcontext()
        print("[no recording]")

    def callback(i: int):
        pin.forwardKinematics(model, vizer.data, qs[i], vs[i])
        vizer.drawFrameVelocities(frame_id, color=0xFF9F1A)

    input("[Press enter]")
    with ctx:
        for i in range(numrep):
            vizer.setCameraPosition(cps_[i])
            vizer.play(qs, dt, callback)
