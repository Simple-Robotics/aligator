"""
@Time    :   2022/06/29 15:58:26
@Author  :   quentinll
@License :   (C)Copyright 2021-2022, INRIA
"""
import pinocchio as pin
import numpy as np
import proxddp
import tap
import matplotlib.pyplot as plt

from utils import create_cartpole
from pinocchio.visualize import MeshcatVisualizer
from proxddp import constraints, manifolds


class Args(tap.Tap):
    display: bool = False
    use_term_cstr: bool = False
    record: bool = False
    num_replay: int = 2

    def process_args(self):
        if self.record:
            self.display = True


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
cont_dyn = proxddp.dynamics.MultibodyFreeFwdDynamics(space, act_mat)
disc_dyn = proxddp.dynamics.IntegratorSemiImplEuler(cont_dyn, dt)

nq = model.nq
nv = model.nv
x0 = space.neutral()
x0[1] = 0.5

target_pos = np.array([0.0, 0.0, 1.0])
frame_id = model.getFrameId("end_effector_frame")

# running cost regularizes the control input
rcost = proxddp.CostStack(ndx, nu)
wu = np.ones(nu) * 1e-2
rcost.addCost(
    proxddp.QuadraticResidualCost(
        proxddp.ControlErrorResidual(ndx, np.zeros(nu)), np.diag(wu) * dt
    )
)
frame_place_target = pin.SE3.Identity()
frame_place_target.translation[:] = target_pos
frame_err = proxddp.FramePlacementResidual(
    ndx,
    nu,
    model,
    frame_place_target,
    frame_id,
)
weights_frame_place = np.zeros(6)
weights_frame_place[:3] = 1.0
# frame_err = proxddp.FrameTranslationResidual(ndx, nu, model, target_pos, frame_id)
# weights_frame_place = np.zeros(3)
# weights_frame_place[2] = np.ones(1) * 1.0
# weights_frame_place[:3] = 1.0
weights_frame_place = np.diag(weights_frame_place)
rcost.addCost(proxddp.QuadraticResidualCost(frame_err, weights_frame_place * dt))
term_cost = proxddp.CostStack(ndx, nu)

wx_term = np.zeros(ndx)
wx_term[nv:] = 0.001
x_reg_cost = proxddp.QuadraticResidualCost(
    proxddp.StateErrorResidual(space, nu, space.neutral()), np.diag(wx_term)
)
term_cost.addCost(proxddp.QuadraticResidualCost(frame_err, weights_frame_place * dt))
term_cost.addCost(x_reg_cost)

# box constraint on control
u_min = -25.0 * np.ones(nu)
u_max = +25.0 * np.ones(nu)


def get_box_cstr():

    ctrl_lin_fun = proxddp.LinearFunction(ndx, nu, ndx, nu)
    ctrl_lin_fun.B[:] = np.eye(nu)
    return proxddp.StageConstraint(
        ctrl_lin_fun, constraints.BoxConstraint(u_min, u_max)
    )


box_cstr = get_box_cstr()

nsteps = 500
Tf = nsteps * dt
problem = proxddp.TrajOptProblem(x0, nu, space, term_cost)

for i in range(nsteps):
    stage = proxddp.StageModel(rcost, disc_dyn)
    stage.addConstraint(box_cstr)
    problem.addStage(stage)

term_fun = proxddp.FrameTranslationResidual(ndx, nu, model, target_pos, frame_id)
term_cstr = proxddp.StageConstraint(term_fun, constraints.EqualityConstraintSet())
problem.addTerminalConstraint(term_cstr)

mu_init = 0.1
rho_init = 0.0
verbose = proxddp.VerboseLevel.VERBOSE
TOL = 1e-4
MAX_ITER = 100
solver = proxddp.SolverProxDDP(
    TOL, mu_init, rho_init, max_iters=MAX_ITER, verbose=verbose
)
solver.reg_init = 1e-5
callback = proxddp.HistoryCallback()
solver.registerCallback(callback)

u0 = np.zeros(nu)
us_i = [u0] * nsteps
xs_i = proxddp.rollout(disc_dyn, x0, us_i)

solver.setup(problem)
solver.run(problem, xs_i, us_i)
res = solver.getResults()
xtar = space.neutral()

plt.figure(figsize=(9.6, 4.8))
plt.subplot(121)
lstyle = {"lw": 0.9}
trange = np.linspace(0, Tf, nsteps + 1)
plt.plot(trange, res.xs, ls="-", **lstyle)
plt.title("State $x(t)$")
if args.use_term_cstr:
    plt.hlines(
        xtar,
        *trange[[0, -1]],
        ls="-",
        lw=1.3,
        colors="k",
        alpha=0.8,
        label=r"$x_\mathrm{tar}$",
    )
plt.xlabel("Time $i$")

plt.subplot(122)
plt.plot(trange[:-1], res.us, **lstyle)
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

if True:
    from proxnlp.utils import plot_pd_errs

    plt.figure(figsize=(6.4, 4.8))
    prim_errs = callback.storage.prim_infeas
    dual_errs = callback.storage.dual_infeas
    prim_tols = np.array(callback.storage.prim_tols)
    al_iters = np.array(callback.storage.al_iters)

    ax: plt.Axes = plt.subplot(111)
    plot_pd_errs(ax, prim_errs, dual_errs)
    itrange = np.arange(len(al_iters))
    ax.step(itrange, prim_tols, c="green", alpha=0.9, lw=1.1)
    al_change = al_iters[1:] - al_iters[:-1]
    al_change_idx = itrange[:-1][al_change > 0]

    ax.vlines(al_change_idx, *ax.get_ylim(), colors="gray", lw=4.0, alpha=0.5)
    ax.legend(["Prim. err $p$", "Dual err $d$", "Prim tol $\\eta_k$", "AL iters"])


plt.tight_layout()
plt.show()

if args.display:
    import hppfcl

    numrep = args.num_replay
    cp = [2.0, 0.0, 0.8]
    cps_ = [cp.copy() for _ in range(numrep)]
    qs = [x[:nq] for x in res.xs.tolist()]
    vs = [x[nq:] for x in res.xs.tolist()]

    obj = pin.GeometryObject("objective", 0, frame_place_target, hppfcl.Sphere(0.05))
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
