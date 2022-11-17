"""
@Time    :   2022/06/29 15:58:26
@Author  :   quentinll
@License :   (C)Copyright 2021-2022, INRIA
"""

import pinocchio as pin
import numpy as np
import proxddp
import proxnlp
import hppfcl as fcl
import tap
import matplotlib.pyplot as plt
import meshcat_utils as msu

from pinocchio.visualize import MeshcatVisualizer
from proxddp import constraints


class Args(tap.Tap):
    display: bool = False
    use_term_cstr: bool = True
    record: bool = False

    def process_args(self):
        if self.record:
            self.display = True


args = Args().parse_args()


def create_pendulum(N, sincos=False):
    # base is fixed
    model = pin.Model()
    geom_model = pin.GeometryModel()

    parent_id = 0
    base_radius = 0.2
    shape_base = fcl.Sphere(base_radius)
    geom_base = pin.GeometryObject("base", 0, pin.SE3.Identity(), shape_base)
    geom_base.meshColor = np.array([1.0, 0.1, 0.1, 1.0])
    geom_model.addGeometryObject(geom_base)
    joint_placement = pin.SE3.Identity()
    body_mass = 1.0
    body_radius = 0.1
    for k in range(N):
        joint_name = "joint_" + str(k + 1)
        if sincos:
            joint_id = model.addJoint(
                parent_id, pin.JointModelRUBX(), joint_placement, joint_name
            )
        else:
            joint_id = model.addJoint(
                parent_id, pin.JointModelRX(), joint_placement, joint_name
            )

        body_inertia = pin.Inertia.FromSphere(body_mass, body_radius)
        body_placement = joint_placement.copy()
        body_placement.translation[2] = 1.0
        model.appendBodyToJoint(joint_id, body_inertia, body_placement)

        geom1_name = "ball_" + str(k + 1)
        shape1 = fcl.Sphere(body_radius)
        geom1_obj = pin.GeometryObject(geom1_name, joint_id, body_placement, shape1)
        geom1_obj.meshColor = np.ones((4))
        geom_model.addGeometryObject(geom1_obj)

        geom2_name = "bar_" + str(k + 1)
        shape2 = fcl.Cylinder(body_radius / 4.0, body_placement.translation[2])
        shape2_placement = body_placement.copy()
        shape2_placement.translation[2] /= 2.0

        geom2_obj = pin.GeometryObject(geom2_name, joint_id, shape2_placement, shape2)
        geom2_obj.meshColor = np.array([0.0, 0.0, 0.0, 1.0])
        geom_model.addGeometryObject(geom2_obj)

        parent_id = joint_id
        joint_placement = body_placement.copy()
    end_frame = pin.Frame(
        "end_effector_frame",
        model.getJointId("joint_" + str(N)),
        0,
        body_placement,
        pin.FrameType(3),
    )
    model.addFrame(end_frame)
    geom_model.collision_pairs = []
    model.qinit = np.zeros(model.nq)
    if sincos:
        model.qinit[0] = -1
    else:
        model.qinit[0] = 0.0 * np.pi
    model.qref = pin.neutral(model)
    data = model.createData()
    geom_data = geom_model.createData()
    actuation = np.eye(N)
    return model, geom_model, data, geom_data, actuation


model, geom_model, data, geom_data, ddl = create_pendulum(1)
dt = 0.01
nu = model.nv
space = proxnlp.manifolds.MultibodyPhaseSpace(model)
nx = space.nx
ndx = space.ndx
cont_dyn = proxddp.dynamics.MultibodyFreeFwdDynamics(space)
dyn_model = proxddp.dynamics.IntegratorSemiImplEuler(cont_dyn, dt)

np.random.seed(1)
nq = model.nq
nv = model.nv
x0 = space.neutral()
x0[0] = np.pi

target_pos = np.array([0.0, 0.0, 1.0])
frame_id = model.getFrameId("end_effector_frame")

# running cost regularizes the control input
rcost = proxddp.CostStack(ndx, nu)
w_u = np.ones(nu) * 1e-3
rcost.addCost(
    proxddp.QuadraticResidualCost(
        proxddp.ControlErrorResidual(ndx, np.zeros(nu)), np.diag(w_u) * dt
    )
)
# wx = np.zeros(ndx)
# wx[1] = 5e-1
# rcost.addCost(
#     proxddp.QuadraticResidualCost(
#         proxddp.StateErrorResidual(space, nu, np.zeros(nx)), np.diag(wx)
#     )
# )
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
    proxddp.QuadraticResidualCost(frame_err, np.diag(weights_frame_place) * dt)
)
term_cost = proxddp.CostStack(ndx, nu)
term_cost.addCost(
    proxddp.QuadraticResidualCost(frame_err, np.diag(weights_frame_place) * dt)
)

# box constraint on control
umin = -20.0 * np.ones(nu)
umax = +20.0 * np.ones(nu)
ctrl_fn = proxddp.ControlErrorResidual(ndx, np.zeros(nu))
box_cstr = proxddp.StageConstraint(ctrl_fn, constraints.BoxConstraint(umin, umax))
# ctrl_fn = proxddp.ControlBoxFunction(ndx, umin, umax)
# box_cstr = proxddp.StageConstraint(ctrl_fn, constraints.NegativeOrthant())

nsteps = 200
Tf = nsteps * dt
problem = proxddp.TrajOptProblem(x0, nu, space, term_cost)

for i in range(nsteps):
    stage = proxddp.StageModel(rcost, dyn_model)
    stage.addConstraint(box_cstr)
    problem.addStage(stage)

term_fun = proxddp.FrameTranslationResidual(ndx, nu, model, target_pos, frame_id)
if args.use_term_cstr:
    term_cstr = proxddp.StageConstraint(term_fun, constraints.EqualityConstraintSet())
    problem.setTerminalConstraint(term_cstr)

mu_init = 0.5
rho_init = 1e-6
verbose = proxddp.VerboseLevel.VERBOSE
TOL = 1e-4
MAX_ITER = 200
solver = proxddp.SolverProxDDP(
    TOL, mu_init, rho_init=rho_init, max_iters=MAX_ITER, verbose=verbose
)
solver.rollout_type = proxddp.ROLLOUT_NONLINEAR
callback = proxddp.HistoryCallback()
solver.registerCallback(callback)

u0 = pin.rnea(model, data, x0[:1], x0[1:], np.zeros(nv))
us_i = [u0] * nsteps
xs_i = proxddp.rollout(dyn_model, x0, us_i)

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
    np.concatenate([umin, umax]),
    *trange[[0, -1]],
    ls="-",
    colors="k",
    lw=2.5,
    alpha=0.4,
    label=r"$\bar{u}$"
)
plt.title("Controls $u(t)$")
plt.legend()
plt.tight_layout()
plt.savefig("assets/pendulum_controls.png")

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
    plt.savefig("assets/pendulum_convergence")


plt.show()

if args.display:
    vizer = MeshcatVisualizer(model, geom_model, geom_model)
    vizer.initViewer(open=args.display, loadModel=True)
    viz_util = msu.VizUtil(vizer)
    viz_util.set_bg_color()

    numrep = 2
    cp = [2.0, 0.0, 0.8]
    cps_ = [cp.copy() for _ in range(numrep)]
    vidrecord = msu.VideoRecorder("examples/ur5_reach_ctrlbox.mp4", fps=1.0 / dt)
    input("[Press enter]")

    for i in range(numrep):
        viz_util.set_cam_pos(cps_[i])
        viz_util.draw_objective(frame_place_target.translation)
        viz_util.play_trajectory(
            res.xs.tolist(),
            res.us.tolist(),
            frame_ids=[frame_id],
            timestep=dt,
            record=args.record,
            recorder=vidrecord,
        )
