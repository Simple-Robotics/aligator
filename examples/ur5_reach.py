import aligator
import numpy as np

import pinocchio as pin
import example_robot_data as erd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import coal

from aligator import constraints, manifolds, dynamics  # noqa

from utils import ArgsBase, get_endpoint_traj


class Args(ArgsBase):
    plot: bool = True
    fddp: bool = False
    bounds: bool = False
    collisions: bool = False


args = Args().parse_args()

print(args)


robot = erd.load("ur5")
rmodel: pin.Model = robot.model
rdata: pin.Data = robot.data
visual_model: pin.GeometryModel = robot.visual_model
space = manifolds.MultibodyPhaseSpace(rmodel)

fr_name = "universe"
fr_id = rmodel.getFrameId(fr_name)
joint_id = rmodel.frames[fr_id].parentJoint
if args.collisions:
    obstacle_loc = pin.SE3.Identity()
    obstacle_loc.translation[0] = 0.3
    obstacle_loc.translation[1] = 0.5
    obstacle_loc.translation[2] = 0.3
    geom_object = pin.GeometryObject(
        "capsule", fr_id, joint_id, coal.Capsule(0.05, 0.4), obstacle_loc
    )

    fr_id2 = rmodel.getFrameId("wrist_3_joint")
    joint_id2 = rmodel.frames[fr_id2].parentJoint
    geom_object2 = pin.GeometryObject(
        "endeffector", fr_id2, joint_id2, coal.Sphere(0.1), pin.SE3.Identity()
    )

    geometry = pin.GeometryModel()
    ig_frame = geometry.addGeometryObject(geom_object)
    ig_frame2 = geometry.addGeometryObject(geom_object2)
    geometry.addCollisionPair(pin.CollisionPair(ig_frame, ig_frame2))

    geom_object.meshColor[:] = [1.0, 1.0, 0.5, 1.0]
    visual_model.addGeometryObject(geom_object)

x0 = space.neutral()

ndx = space.ndx
nq = rmodel.nq
nv = rmodel.nv
nu = nv
q0 = x0[:nq]

B_mat = np.eye(nu)

dt = 0.01
Tf = 100 * dt
nsteps = int(Tf / dt)

ode = dynamics.MultibodyFreeFwdDynamics(space, B_mat)
discrete_dynamics = dynamics.IntegratorSemiImplEuler(ode, dt)

wt_x = 1e-4 * np.ones(ndx)
wt_x[nv:] = 1e-2
wt_x = np.diag(wt_x)
wt_u = 1e-4 * np.eye(nu)


tool_name = "tool0"
tool_id = rmodel.getFrameId(tool_name)
target_pos = np.array([0.15, 0.65, 0.5])
target_place = pin.SE3.Identity()
target_place.translation = target_pos
target_object = pin.GeometryObject(
    "target", fr_id, joint_id, coal.Sphere(0.05), target_place
)
target_object.meshColor[:] = [0.5, 0.5, 1.0, 1.0]
visual_model.addGeometryObject(target_object)
visual_data = visual_model.createData()

if args.display:
    from candlewick import Visualizer, VisualizerConfig

    _config = VisualizerConfig()
    _config.width = 1920
    _config.height = 1080
    vizer = Visualizer(
        _config,
        rmodel,
        visual_model,
        data=rdata,
        visual_data=visual_data,
    )
    vizer.display(q0)

frame_fn = aligator.FrameTranslationResidual(ndx, nu, rmodel, target_pos, tool_id)
v_ref = pin.Motion()
v_ref.np[:] = 0.0
frame_vel_fn = aligator.FrameVelocityResidual(
    ndx, nu, rmodel, v_ref, tool_id, pin.LOCAL
)
wt_x_term = wt_x.copy()
wt_x_term[:] = 1e-4
wt_frame_pos = 100.0 * np.eye(frame_fn.nr)
wt_frame_vel = 100.0 * np.ones(frame_vel_fn.nr)
wt_frame_vel = np.diag(wt_frame_vel)

term_cost = aligator.CostStack(space, nu)
term_cost.addCost("reg", aligator.QuadraticCost(wt_x_term, wt_u * 0))
term_cost.addCost(
    "frame", aligator.QuadraticResidualCost(space, frame_fn, wt_frame_pos)
)
term_cost.addCost(
    "vel", aligator.QuadraticResidualCost(space, frame_vel_fn, wt_frame_vel)
)

u_max = rmodel.effortLimit
u_min = -u_max


def make_control_bounds():
    fun = aligator.ControlErrorResidual(ndx, nu)
    cstr_set = constraints.BoxConstraint(u_min, u_max)
    return (fun, cstr_set)


def computeQuasistatic(model: pin.Model, x0, a):
    data = model.createData()
    q0 = x0[:nq]
    v0 = x0[nq : nq + nv]

    return pin.rnea(model, data, q0, v0, a)


init_us = [computeQuasistatic(rmodel, x0, a=np.zeros(nv)) for _ in range(nsteps)]
init_xs = aligator.rollout(discrete_dynamics, x0, init_us)


stages = []
for i in range(nsteps):
    rcost = aligator.CostStack(space, nu)
    rcost.addCost("reg", aligator.QuadraticCost(wt_x * dt, wt_u * dt))

    stm = aligator.StageModel(rcost, discrete_dynamics)
    if args.collisions:
        # Distance to obstacle constrained between 0.1 and 100 m
        cstr_set = constraints.BoxConstraint(np.array([0.1]), np.array([100]))
        frame_col = aligator.FrameCollisionResidual(ndx, nu, rmodel, geometry, 0)
        stm.addConstraint(frame_col, cstr_set)
    if args.bounds:
        stm.addConstraint(*make_control_bounds())
    stages.append(stm)


problem = aligator.TrajOptProblem(x0, stages, term_cost=term_cost)
tol = 1e-7

mu_init = 1e-7
verbose = aligator.VerboseLevel.VERBOSE
max_iters = 500
solver = aligator.SolverProxDDP(tol, mu_init, max_iters=max_iters, verbose=verbose)
solver.rollout_type = aligator.ROLLOUT_NONLINEAR
solver.sa_strategy = aligator.SA_LINESEARCH_NONMONOTONE
if args.fddp:
    solver = aligator.SolverFDDP(tol, verbose, max_iters=max_iters)
cb = aligator.HistoryCallback(solver)
solver.registerCallback("his", cb)
solver.setup(problem)
solver.run(problem, init_xs, init_us)


results = solver.results
print(results)

xs_opt = results.xs.tolist()
us_opt = np.asarray(results.us.tolist())


times = np.linspace(0.0, Tf, nsteps + 1)

fig: plt.Figure = plt.figure(constrained_layout=True)
fig.set_size_inches(6.4, 6.4)

gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 2])
_u_ncol = 2
_u_nrow, rmdr = divmod(nu, _u_ncol)
if rmdr > 0:
    _u_nrow += 1
gs1 = gs[1, :].subgridspec(_u_nrow, _u_ncol)

plt.subplot(gs[0, 0])
plt.plot(times, xs_opt)
plt.title("States")

axarr = gs1.subplots(sharex=True)
handles_ = []
lss_ = []

for i in range(nu):
    ax: plt.Axes = axarr.flat[i]
    (ls,) = ax.plot(times[1:], us_opt[:, i])
    lss_.append(ls)
    if args.bounds:
        col = lss_[i].get_color()
        hl = ax.hlines(
            (u_min[i], u_max[i]), *times[[0, -1]], linestyles="--", colors=col
        )
        handles_.append(hl)
    fontsize = 7
    ax.set_ylabel("$u_{{%d}}$" % (i + 1), fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize)
    ax.tick_params(axis="y", rotation=90)
    if i + 1 == nu - 1:
        ax.set_xlabel("time", loc="left", fontsize=fontsize)

pts = get_endpoint_traj(rmodel, rdata, xs_opt, tool_id)

ax = plt.subplot(gs[0, 1], projection="3d")
ax.plot(*pts.T, lw=1.0)
ax.scatter(*target_pos, marker="^", c="r")

ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$z$")

plt.figure()

nrang = range(1, results.num_iters + 1)
ax: plt.Axes = plt.gca()
plt.plot(nrang, cb.prim_infeas, ls="--", marker=".", label="primal err")
plt.plot(nrang, cb.dual_infeas, ls="--", marker=".", label="dual err")
ax.set_xlabel("iter")
ax.set_yscale("log")
plt.legend()
plt.tight_layout()
plt.show()


if args.display:
    import time

    input("[Press enter]")
    num_repeat = 3
    cp = np.array([0.8, 0.8, 0.8])
    cps_ = [cp.copy() for _ in range(num_repeat)]
    cps_[1][1] = -0.4

    qs = [x[:nq] for x in xs_opt]
    vs = [x[nq:] for x in xs_opt]
    VID_FPS = 30
    VID_DT = 1.0 / VID_FPS

    def callback(i: int):
        pin.forwardKinematics(rmodel, vizer.data, qs[i], vs[i])
        vizer.drawFrameVelocities(tool_id)

    for i in range(num_repeat):
        vizer.setCameraPosition(cps_[i])
        for t in range(nsteps):
            vizer.display(qs[t])
            time.sleep(VID_DT)
        # vizer.play(np.stack(qs), dt=VID_DT)
