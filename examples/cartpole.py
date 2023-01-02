"""
@Time    :   2022/06/29 15:58:26
@Author  :   quentinll
@License :   (C)Copyright 2021-2022, INRIA
"""

import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
import proxddp
import proxnlp
import tap
import matplotlib.pyplot as plt
import meshcat_utils as msu
from utils import create_cartpole


class Args(tap.Tap):
    display: bool = False
    use_term_cstr: bool = False
    record: bool = False

    def process_args(self):
        if self.record:
            self.display = True


args = Args().parse_args()


model, geom_model, data, geom_data, ddl = create_cartpole(1)
time_step = 0.01
nu = 1
act_mat = np.zeros((2, nu))
act_mat[0, 0] = 1.0
space = proxnlp.manifolds.MultibodyPhaseSpace(model)
nx = space.nx
ndx = space.ndx
cont_dyn = proxddp.dynamics.MultibodyFreeFwdDynamics(space, act_mat)
disc_dyn = proxddp.dynamics.IntegratorSemiImplEuler(cont_dyn, time_step)

nq = model.nq
nv = model.nv
x0 = space.neutral()
x0[1] = 0.5
# x0[1] = np.pi

target_pos = np.array([0.0, 0.0, 1.0])
frame_id = model.getFrameId("end_effector_frame")

# running cost regularizes the control input
rcost = proxddp.CostStack(ndx, nu)
wu = np.ones(nu) * 1e-4
rcost.addCost(
    proxddp.QuadraticResidualCost(
        proxddp.ControlErrorResidual(ndx, np.zeros(nu)), np.diag(wu) * time_step
    )
)
frame_place_target = pin.SE3.Identity()
frame_place_target.translation = target_pos
frame_err = proxddp.FramePlacementResidual(
    ndx,
    nu,
    model,
    frame_place_target,
    frame_id,
)
weights_frame_place = np.zeros(6)
weights_frame_place[:3] = np.ones(3) * 1.0
# frame_err = proxddp.FrameTranslationResidual(ndx, nu, model, target_pos, frame_id)
# weights_frame_place = np.zeros(3)
# weights_frame_place[2] = np.ones(1) * 1.0
# weights_frame_place[:3] = 1.0
rcost.addCost(
    proxddp.QuadraticResidualCost(frame_err, np.diag(weights_frame_place) * time_step)
)
term_cost = proxddp.CostStack(ndx, nu)
term_cost.addCost(
    proxddp.QuadraticResidualCost(frame_err, np.diag(weights_frame_place) * time_step)
)

# box constraint on control
u_min = -25.0 * np.ones(nu)
u_max = +25.0 * np.ones(nu)
ctrl_box = proxddp.ControlBoxFunction(ndx, u_min, u_max)

nsteps = 1000
Tf = nsteps * time_step
problem = proxddp.TrajOptProblem(x0, nu, space, term_cost)

for i in range(nsteps):
    stage = proxddp.StageModel(rcost, disc_dyn)
    stage.addConstraint(
        proxddp.StageConstraint(ctrl_box, proxnlp.constraints.NegativeOrthant())
    )
    problem.addStage(stage)

term_fun = proxddp.FrameTranslationResidual(ndx, nu, model, target_pos, frame_id)
term_cstr = proxddp.StageConstraint(
    term_fun, proxnlp.constraints.EqualityConstraintSet()
)
# problem.setTerminalConstraint(term_cstr)

mu_init = 4e-2
rho_init = 1e-2
verbose = proxddp.VerboseLevel.VERBOSE
TOL = 1e-4
MAX_ITER = 200
solver = proxddp.SolverProxDDP(
    TOL, mu_init, rho_init=rho_init, max_iters=MAX_ITER, verbose=verbose
)
callback = proxddp.HistoryCallback()
solver.registerCallback(callback)

u0 = np.zeros(nu)
us_i = [u0] * nsteps
xs_i = proxddp.rollout(disc_dyn, x0, us_i)

solver.setup(problem)
solver.run(problem, xs_i, us_i)
res = solver.getResults()
print(res)
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
        label=r"$x_\mathrm{tar}$"
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
    label=r"$\bar{u}$"
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
    print(prim_tols)
    ax.step(itrange, prim_tols, c="green", alpha=0.9, lw=1.1)
    al_change = al_iters[1:] - al_iters[:-1]
    al_change_idx = itrange[:-1][al_change > 0]

    ax.vlines(al_change_idx, *ax.get_ylim(), colors="gray", lw=4.0, alpha=0.5)
    ax.legend(["Prim. err $p$", "Dual err $d$", "Prim tol $\\eta_k$", "AL iters"])


plt.tight_layout()
plt.show()

if args.display:
    vizer = MeshcatVisualizer(model, geom_model, geom_model)
    vizer.initViewer(open=args.display, loadModel=True)
    viz_util = msu.VizUtil(vizer)

    numrep = 2
    cp = [2.0, 0.0, 0.8]
    cps_ = [cp.copy() for _ in range(numrep)]
    vidrecord = msu.VideoRecorder("examples/ur5_reach_ctrlbox.mp4", fps=1.0 / time_step)
    input("[Press enter]")

    for i in range(numrep):
        viz_util.set_cam_pos(cps_[i])
        viz_util.draw_objective(frame_place_target.translation)
        viz_util.play_trajectory(
            res.xs.tolist(),
            res.us.tolist(),
            frame_ids=[frame_id],
            timestep=time_step,
            record=args.record,
            recorder=vidrecord,
        )
