import proxddp
import numpy as np

import pinocchio as pin
import meshcat_utils as msu
import example_robot_data as erd
import matplotlib.pyplot as plt

from proxddp import constraints, manifolds, dynamics
from pinocchio.visualize import MeshcatVisualizer

from common import ArgsBase

plt.rcParams["lines.linewidth"] = 1.0


class Args(ArgsBase):
    plot: bool = True

    def process_args(self):
        if self.record:
            self.display = True


args = Args().parse_args()

print(args)


robot = erd.load("ur5")
rmodel: pin.Model = robot.model
rdata: pin.Data = robot.data
space = manifolds.MultibodyPhaseSpace(rmodel)

vizer = MeshcatVisualizer(rmodel, robot.collision_model, robot.visual_model, data=rdata)
vizer.initViewer(open=args.display, loadModel=True)
viz_util = msu.VizUtil(vizer)


x0 = space.neutral()

nq = rmodel.nq
nv = rmodel.nv
nu = nv
q0 = x0[:nq]

vizer.display(q0)

B_mat = np.eye(nu)

Tf = 1.2
dt = 0.01
nsteps = int(Tf / dt)

ode = dynamics.MultibodyFreeFwdDynamics(space, B_mat)
discrete_dynamics = dynamics.IntegratorSemiImplEuler(ode, dt)

wt_x = 1e-4 * np.ones(space.ndx)
wt_x[nv:] = 1e-2
wt_x = np.diag(wt_x)
wt_u = 1e-4 * np.eye(nu)


tool_name = "tool0"
tool_id = rmodel.getFrameId(tool_name)
target_pos = np.array([0.15, 0.65, 0.5])
print(target_pos)

frame_fn = proxddp.FrameTranslationResidual(space.ndx, nu, rmodel, target_pos, tool_id)
v_ref = pin.Motion()
v_ref.np[:] = 0.0
frame_vel_fn = proxddp.FrameVelocityResidual(
    space.ndx, nu, rmodel, v_ref, tool_id, pin.LOCAL
)
wt_x_term = wt_x.copy()
wt_x_term[:] = 1e-4
wt_frame_pos = 10.0 * np.eye(frame_fn.nr)
wt_frame_vel = 100.0 * np.ones(frame_vel_fn.nr)
wt_frame_vel = np.diag(wt_frame_vel)

term_cost = proxddp.CostStack(space.ndx, nu)
term_cost.addCost(proxddp.QuadraticCost(wt_x_term, wt_u * 0))
term_cost.addCost(proxddp.QuadraticResidualCost(frame_fn, wt_frame_pos))
term_cost.addCost(proxddp.QuadraticResidualCost(frame_vel_fn, wt_frame_vel))

u_max = rmodel.effortLimit
u_min = -u_max
ctrl_box = proxddp.ControlBoxFunction(space.ndx, u_min, u_max)


def computeQuasistatic(model: pin.Model, x0, a):
    data = model.createData()
    q0 = x0[:nq]
    v0 = x0[nq : nq + nv]

    return pin.rnea(model, data, q0, v0, a)


init_us = [computeQuasistatic(rmodel, x0, a=np.zeros(nv)) for _ in range(nsteps)]
init_xs = proxddp.rollout(discrete_dynamics, x0, init_us)


stages = []

for i in range(nsteps):
    rcost = proxddp.CostStack(space.ndx, nu)
    rcost.addCost(proxddp.QuadraticCost(wt_x * dt, wt_u * dt))

    stm = proxddp.StageModel(space, nu, rcost, discrete_dynamics)
    stm.addConstraint(ctrl_box, constraints.NegativeOrthant())
    stages.append(stm)


problem = proxddp.TrajOptProblem(x0, stages, term_cost=term_cost)
problem.setTerminalConstraint(
    proxddp.StageConstraint(frame_fn, constraints.EqualityConstraintSet())
)
tol = 1e-3

mu_init = 1e-4
rho_init = 1e-8

solver = proxddp.SolverProxDDP(
    tol, mu_init, rho_init, verbose=proxddp.VerboseLevel.VERBOSE, max_iters=200
)
solver.rol_type = proxddp.RolloutType.NONLINEAR
solver.setup(problem)
solver.run(problem, init_xs, init_us)


results = solver.getResults()
print(results)

xs_opt = results.xs.tolist()
us_opt = np.asarray(results.us.tolist())


times = np.linspace(0.0, Tf, nsteps + 1)

gs = plt.GridSpec(2, 2)

plt.subplot(gs[0, 0])
plt.plot(times, xs_opt)
plt.title("States")

plt.subplot(gs[1, 0])
ls = plt.plot(times[1:], us_opt)
ylim = plt.ylim()
for i in range(nu):
    plt.hlines(
        (u_min[i], u_max[i]), *times[[0, -1]], linestyles="--", colors=ls[i].get_color()
    )
plt.ylim(ylim)
plt.title("Controls")


def get_endpoint(q: np.ndarray):
    pin.framesForwardKinematics(rmodel, rdata, q)
    return rdata.oMf[tool_id].translation.copy()


def get_endpoint_traj(xs: list[np.ndarray]):
    pts = []
    for i in range(len(xs)):
        pts.append(get_endpoint(q=xs[i][: rmodel.nq]))
    return np.array(pts)


pts = get_endpoint_traj(xs_opt)

ax = plt.subplot(gs[:, 1], projection="3d")
ax.plot(*pts.T, lw=1.0)
ax.scatter(*target_pos, marker="^", c="r")

ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$z$")
plt.tight_layout()
plt.show()


if args.display:
    import time

    input("[Press enter]")
    num_repeat = 3
    cp = np.array([0.8, 0.8, 0.8])
    cps_ = [cp.copy() for _ in range(num_repeat)]
    cps_[1][1] = -0.4
    vidrecord = msu.VideoRecorder("examples/ur5_reach_ctrlbox.mp4", fps=1.0 / dt)

    for i in range(num_repeat):
        viz_util.set_cam_pos(cps_[i])
        viz_util.draw_objective(target_pos)
        viz_util.play_trajectory(
            xs_opt,
            us_opt,
            frame_ids=[tool_id],
            timestep=dt,
            record=args.record,
            recorder=vidrecord,
        )
        time.sleep(0.5)
