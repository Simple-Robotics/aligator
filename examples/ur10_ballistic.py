import example_robot_data as erd
import pinocchio as pin
import numpy as np
import aligator
import coal
import matplotlib.pyplot as plt
import contextlib

from pathlib import Path
from typing import Tuple
from aligator.utils.plotting import (
    plot_controls_traj,
    plot_convergence,
    plot_velocity_traj,
)
from utils import (
    add_namespace_prefix_to_models,
    ArgsBase,
    IMAGEIO_KWARGS,
    manage_lights,
)
from aligator import dynamics, manifolds, constraints


class Args(ArgsBase):
    pass


args = Args().parse_args()

robot = erd.load("ur10")
q0_ref_arm = np.array([0.0, np.deg2rad(-120), 2 * np.pi / 3, np.deg2rad(-45), 0.0, 0.0])
robot.q0[:] = q0_ref_arm
print(f"Velocity limit (before): {robot.model.velocityLimit}")


def load_projectile_model(free_flyer: bool = True):
    ball_urdf = Path(__file__).parent / "mug.urdf"
    packages_dirs = [str(Path(__file__).parent)]
    ball_scale = 1.0
    model, cmodel, vmodel = pin.buildModelsFromUrdf(
        str(ball_urdf),
        package_dirs=packages_dirs,
        root_joint=pin.JointModelFreeFlyer()
        if free_flyer
        else pin.JointModelTranslation(),
    )
    print("Projectile model:\n", model)
    for geom in cmodel.geometryObjects:
        geom.meshScale *= ball_scale
    for geom in vmodel.geometryObjects:
        geom.meshScale *= ball_scale
    return model, cmodel, vmodel


def append_ball_to_robot_model(
    robot: pin.RobotWrapper,
) -> Tuple[pin.Model, pin.GeometryModel, pin.GeometryModel]:
    base_model: pin.Model = robot.model
    base_visual: pin.GeometryModel = robot.visual_model
    base_coll: pin.GeometryModel = robot.collision_model
    ee_link_id = base_model.getFrameId("tool0")
    _ball_model, _ball_coll, _ball_visu = load_projectile_model()
    add_namespace_prefix_to_models(_ball_model, _ball_coll, _ball_visu, "ball")

    pin.forwardKinematics(base_model, robot.data, robot.q0)
    pin.updateFramePlacement(base_model, robot.data, ee_link_id)

    tool_frame_pl = robot.data.oMf[ee_link_id]
    rel_placement = tool_frame_pl.copy()
    rel_placement.translation[1] = 0.0
    rmodel, cmodel = pin.appendModel(
        base_model, _ball_model, base_coll, _ball_coll, 0, rel_placement
    )
    _, vmodel = pin.appendModel(
        base_model, _ball_model, base_visual, _ball_visu, 0, rel_placement
    )

    ref_q0 = pin.neutral(rmodel)
    ref_q0[:6] = robot.q0
    return rmodel, cmodel, vmodel, ref_q0


nq_o = robot.model.nq
nv_o = robot.model.nv
rmodel, cmodel, vmodel, ref_q0 = append_ball_to_robot_model(robot)
print(f"New model velocity lims: {rmodel.velocityLimit}")
space = manifolds.MultibodyPhaseSpace(rmodel)
rdata: pin.Data = rmodel.createData()
nq_b = rmodel.nq
nv_b = rmodel.nv
nu = nv_b - 6
ndx = space.ndx
x0 = space.neutral()
x0[:nq_b] = ref_q0
print("X0 = {}".format(x0))
MUG_VEL_IDX = slice(robot.nv, nv_b)


def create_rcm(contact_type=pin.ContactType.CONTACT_6D):
    # create rigid constraint between ball & tool0
    tool_fid = rmodel.getFrameId("tool0")
    frame: pin.Frame = rmodel.frames[tool_fid]
    joint1_id = frame.parent
    joint2_id = rmodel.getJointId("ball/root_joint")
    pin.framesForwardKinematics(rmodel, rdata, ref_q0)
    pl1 = rmodel.frames[tool_fid].placement
    pl2 = rdata.oMf[tool_fid]
    rcm = pin.RigidConstraintModel(
        contact_type,
        rmodel,
        joint1_id,
        pl1,
        joint2_id,
        pl2,
        pin.LOCAL_WORLD_ALIGNED,
    )
    Kp = 1e-3
    rcm.corrector.Kp[:] = Kp
    rcm.corrector.Kd[:] = 2 * Kp**0.5
    return rcm


def configure_viz(target_pos):
    from pinocchio.visualize import MeshcatVisualizer

    gobj = pin.GeometryObject(
        "objective", 0, pin.SE3(np.eye(3), target_pos), coal.Sphere(0.04)
    )
    gobj.meshColor[:] = np.array([200, 100, 100, 200]) / 255.0

    viz = MeshcatVisualizer(
        model=rmodel, collision_model=cmodel, visual_model=vmodel, data=rdata
    )
    viz.initViewer(loadModel=True, zmq_url=args.zmq_url)
    manage_lights(viz)
    viz.addGeometryObject(gobj)
    # viz.setBackgroundColor()
    viz.setCameraZoom(1.7)
    return viz


target_pos = np.array([2.4, -0.2, 0.0])

dt = 0.01
tf = 2.0  # seconds
nsteps = int(tf / dt)
actuation_matrix = np.eye(nv_b, nu)

prox_settings = pin.ProximalSettings(accuracy=1e-8, mu=1e-6, max_iter=20)
rcm = create_rcm()
ode1 = dynamics.MultibodyConstraintFwdDynamics(
    space, actuation_matrix, [rcm], prox_settings
)
ode2 = dynamics.MultibodyFreeFwdDynamics(space, actuation_matrix)
dyn_model1 = dynamics.IntegratorSemiImplEuler(ode1, dt)
dyn_model2 = dynamics.IntegratorSemiImplEuler(ode2, dt)

q0 = x0[:nq_b]
v0 = x0[nq_b:]
u0_free = pin.rnea(robot.model, robot.data, robot.q0, robot.v0, robot.v0)
u0, lam_c = aligator.underactuatedConstrainedInverseDynamics(
    rmodel, rdata, q0, v0, actuation_matrix, [rcm], [rcm.createData()]
)
assert u0.shape == (nu,)


def testu0(u0):
    pin.initConstraintDynamics(rmodel, rdata, [rcm])
    rcd = rcm.createData()
    tau = actuation_matrix @ u0
    acc = pin.constraintDynamics(rmodel, rdata, q0, v0, tau, [rcm], [rcd])
    print("plugging in u0, got acc={}".format(acc))


with np.printoptions(precision=4, linewidth=200):
    print("invdyn (free): {}".format(u0_free))
    print("invdyn torque : {}".format(u0))
    testu0(u0)

dms = [dyn_model1] * nsteps
us_i = [u0] * len(dms)
xs_i = aligator.rollout(dms, x0, us_i)
qs_i = [x[:nq_b] for x in xs_i]

if args.display:
    viz = configure_viz(target_pos=target_pos)
    viz.play(qs_i, dt=dt)
else:
    viz = None


def create_running_cost():
    costs = aligator.CostStack(space, nu)
    w_x = np.array([1e-3] * nv_b + [0.1] * nv_b)
    w_v = w_x[nv_b:]
    # no costs on mug
    w_x[MUG_VEL_IDX] = 0.0
    w_v[MUG_VEL_IDX] = 0.0
    assert space.isNormalized(x0)
    xreg = aligator.QuadraticStateCost(space, nu, x0, np.diag(w_x) * dt)
    w_u = np.ones(nu) * 1e-5
    ureg = aligator.QuadraticControlCost(space, u0, np.diag(w_u) * dt)
    costs.addCost(xreg)
    costs.addCost(ureg)
    return costs


def create_term_cost(has_frame_cost=False, w_ball=1.0):
    w_xf = np.zeros(ndx)
    w_xf[: robot.nv] = 1e-4
    w_xf[nv_b + 6 :] = 1e-6
    costs = aligator.CostStack(space, nu)
    xreg = aligator.QuadraticStateCost(space, nu, x0, np.diag(w_xf))
    costs.addCost(xreg)
    if has_frame_cost:
        ball_pos_fn = get_ball_fn(target_pos)
        w_ball = np.eye(ball_pos_fn.nr) * w_ball
        ball_cost = aligator.QuadraticResidualCost(space, ball_pos_fn, w_ball)
        costs.addCost(ball_cost)
    return costs


def get_ball_fn(target_pos):
    fid = rmodel.getFrameId("ball/root_joint")
    return aligator.FrameTranslationResidual(ndx, nu, rmodel, target_pos, fid)


def create_term_constraint(target_pos):
    term_fn = get_ball_fn(target_pos)
    return (term_fn, constraints.EqualityConstraintSet())


def get_position_limit_constraint():
    state_fn = aligator.StateErrorResidual(space, nu, space.neutral())
    pos_fn = state_fn[:7]
    box_cstr = constraints.BoxConstraint(
        robot.model.lowerPositionLimit, robot.model.upperPositionLimit
    )
    return (pos_fn, box_cstr)


JOINT_VEL_LIM_IDX = [0, 1, 3, 4, 5, 6]
print("Joint vel. limits enforced for:")
for i in JOINT_VEL_LIM_IDX:
    print(robot.model.names[i])


def get_velocity_limit_constraint():
    state_fn = aligator.StateErrorResidual(space, nu, space.neutral())
    vel_fn = state_fn[[nv_b + i for i in JOINT_VEL_LIM_IDX]]
    vlim = rmodel.velocityLimit[JOINT_VEL_LIM_IDX]
    assert vel_fn.nr == vlim.shape[0]
    box_cstr = constraints.BoxConstraint(-vlim, vlim)
    return (vel_fn, box_cstr)


def get_torque_limit_constraint():
    ctrlfn = aligator.ControlErrorResidual(ndx, np.zeros(nu))
    eff = robot.model.effortLimit
    box_cstr = constraints.BoxConstraint(-eff, eff)
    return (ctrlfn, box_cstr)


def create_stage(contact: bool):
    dm = dyn_model1 if contact else dyn_model2
    rc = create_running_cost()
    stm = aligator.StageModel(rc, dm)
    stm.addConstraint(*get_torque_limit_constraint())
    # stm.addConstraint(get_position_limit_constraint())
    stm.addConstraint(*get_velocity_limit_constraint())
    return stm


stages = []
t_contact = int(0.4 * nsteps)
for k in range(nsteps):
    stages.append(create_stage(k <= t_contact))

term_cost = create_term_cost()

problem = aligator.TrajOptProblem(x0, stages, term_cost)
problem.addTerminalConstraint(*create_term_constraint(target_pos))
problem.addTerminalConstraint(*get_velocity_limit_constraint())
tol = 1e-4
mu_init = 1e-2
solver = aligator.SolverProxDDP(tol, mu_init, max_iters=300, verbose=aligator.VERBOSE)
# solver.linear_solver_choice = aligator.LQ_SOLVER_PARALLEL
solver.rollout_type = aligator.ROLLOUT_LINEAR
solver.reg_init = 1e-4
his_cb = aligator.HistoryCallback(solver)
solver.setNumThreads(4)
solver.registerCallback("his", his_cb)
solver.setup(problem)
flag = solver.run(problem, xs_i, us_i)

print(solver.results)
ws: aligator.Workspace = solver.workspace
rs: aligator.Results = solver.results
dyn_slackn_slacks = [np.max(np.abs(s)) for s in ws.dyn_slacks]

xs = solver.results.xs
us = solver.results.us
qs = [x[:nq_b] for x in xs]
vs = [x[nq_b:] for x in xs]
vs = np.asarray(vs)
proj_frame_id = rmodel.getFrameId("ball/root_joint")


def get_frame_vel(k: int):
    pin.forwardKinematics(rmodel, rdata, qs[k], vs[k])
    return pin.getFrameVelocity(rmodel, rdata, proj_frame_id)


vf_before_launch = get_frame_vel(t_contact)
vf_launch_t = get_frame_vel(t_contact + 1)
print("Before launch  :", vf_before_launch.np)
print("Launch velocity:", vf_launch_t.np)

EXPERIMENT_NAME = "ur10_mug_throw"

if args.display:

    def viz_callback(i: int):
        pin.forwardKinematics(rmodel, rdata, qs[i], xs[i][nq_b:])
        viz.drawFrameVelocities(proj_frame_id, v_scale=0.06)
        fid = rmodel.getFrameId("ball/root_joint")
        ctar: pin.SE3 = rdata.oMf[fid]
        viz.setCameraTarget(ctar.translation)

    VID_FPS = 30
    vid_ctx = (
        viz.create_video_ctx(
            f"assets/{EXPERIMENT_NAME}.mp4", fps=VID_FPS, **IMAGEIO_KWARGS
        )
        if args.record
        else contextlib.nullcontext()
    )

    input("[press enter]")

    with vid_ctx:
        viz.play(qs, dt, callback=viz_callback)

if args.plot:
    times = np.linspace(0.0, tf, nsteps + 1)
    _joint_names = robot.model.names
    _efflims = robot.model.effortLimit
    _vlims = robot.model.velocityLimit
    for i in range(nv_o):
        if i not in JOINT_VEL_LIM_IDX:
            _vlims[i] = np.inf
    figsize = (6.4, 4.0)
    fig1, _ = plot_controls_traj(
        times, us, rmodel=rmodel, effort_limit=_efflims, figsize=figsize
    )
    fig1.suptitle("Controls (N/m)")
    fig2, _ = plot_velocity_traj(
        times, vs[:, :-6], rmodel=robot.model, vel_limit=_vlims, figsize=figsize
    )

    PLOTDIR = Path("assets")

    fig3 = plt.figure()
    ax: plt.Axes = fig3.add_subplot(111)
    ax.plot(dyn_slackn_slacks)
    ax.set_yscale("log")
    ax.set_title("Dynamic slack errors $\\|s\\|_\\infty$")

    fig4 = plt.figure(figsize=(6.4, 3.6))
    ax = fig4.add_subplot(111)
    plot_convergence(
        his_cb,
        ax,
        res=solver.results,
        show_al_iters=True,
        legend_kwargs=dict(fontsize=8),
    )
    ax.set_title("Convergence")
    fig4.tight_layout()

    _fig_dict = {"controls": fig1, "velocity": fig2, "conv": fig4}
    for name, fig in _fig_dict.items():
        for ext in [".png", ".pdf"]:
            figpath: Path = PLOTDIR / f"{EXPERIMENT_NAME}_{name}"
            fig.savefig(figpath.with_suffix(ext))

    plt.show()
