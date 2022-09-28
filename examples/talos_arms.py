import numpy as np
import proxddp
import pinocchio as pin
import meshcat
import meshcat_utils as msu
import example_robot_data as erd
import matplotlib.pyplot as plt

import time

from proxddp import manifolds, dynamics
from common import get_endpoint_traj, compute_quasistatic, ArgsBase


class Args(ArgsBase):
    tcp: str = None


def loadTalos():
    robot = erd.load("talos")
    qref = robot.model.referenceConfigurations["half_sitting"]
    locked_joints = list(range(1, 14))
    locked_joints += [23, 31]
    locked_joints += [32, 33]
    red_bot = robot.buildReducedRobot(locked_joints, qref)
    return red_bot


args = Args().parse_args()
robot = loadTalos()
rmodel: pin.Model = robot.model
rdata: pin.Data = robot.data
nv = rmodel.nv
nu = nv


space = manifolds.MultibodyPhaseSpace(rmodel)

x0 = space.neutral()

Tf = 1.0
dt = 0.01
nsteps = int(Tf / dt)

w_x = np.ones(space.ndx) * 0.1
w_x[[0, 1]] = 1.0
w_x[nv:] = 1e-2
w_x = np.diag(w_x)
w_u = np.eye(nu) * 1e-4
act_matrix = np.eye(nu)

ode = dynamics.MultibodyFreeFwdDynamics(space, act_matrix)
dyn_model = dynamics.IntegratorSemiImplEuler(ode, dt)

rcost = proxddp.CostStack(space.ndx, nu)
rcost.addCost(proxddp.QuadraticCost(w_x, w_u), dt)

stm = proxddp.StageModel(space, nu, rcost, dyn_model)
umax = rmodel.effortLimit
umin = -umax
# stm.addConstraint(proxddp.ControlBoxFunction(space.ndx, umin, umax), constraints.NegativeOrthant())

term_cost = proxddp.CostStack(space.ndx, nu)
term_cost.addCost(
    proxddp.QuadraticResidualCost(proxddp.StateErrorResidual(space, nu, x0), w_x)
)

tool_id_lh = rmodel.getFrameId("gripper_left_base_link")
target_ee_pos = np.array([0.4, 0.2, 1.4])

frame_fn_lh = proxddp.FrameTranslationResidual(
    space.ndx, nu, rmodel, target_ee_pos, tool_id_lh
)
w_x_ee = 10.0 * np.eye(3)
frame_fn_cost = proxddp.QuadraticResidualCost(frame_fn_lh, w_x_ee)
term_cost.addCost(frame_fn_cost)


stages = [stm] * nsteps
problem = proxddp.TrajOptProblem(x0, stages, term_cost)


TOL = 1e-4
mu_init = 1e-5
rho_init = 1e-6
max_iters = 100
verbose = proxddp.VerboseLevel.VERBOSE
solver = proxddp.SolverProxDDP(TOL, mu_init, rho_init, verbose=verbose)
solver.rollout_type = proxddp.RolloutType.NONLINEAR
# solver = proxddp.SolverFDDP(TOL, verbose=verbose)
solver.max_iters = max_iters
solver.setup(problem)

xs_init = [x0] * (nsteps + 1)
u0 = compute_quasistatic(rmodel, rdata, x0, acc=np.zeros(nv))
us_init = [u0] * nsteps

solver.run(
    problem,
    xs_init,
    us_init,
)
results = solver.getResults()
print(results)


pts = get_endpoint_traj(rmodel, rdata, results.xs, tool_id_lh)
ax = plt.subplot(121, projection="3d")
ax.plot(*pts.T, ls="--")
ax.scatter(*target_ee_pos, marker="^", c="r")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$z$")

plt.subplot(122)
times = np.linspace(0.0, Tf, nsteps + 1)
plt.plot(times[1:], results.us)
plt.title("Controls trajectory")


plt.tight_layout()
plt.show()

if args.display:

    vizer = pin.visualize.MeshcatVisualizer(
        rmodel, robot.collision_model, robot.visual_model, data=rdata
    )
    if args.tcp:
        viewer = meshcat.Visualizer(zmq_url=args.tcp)
        vizer.initViewer(viewer, loadModel=True)
    else:
        vizer.initViewer(open=True, loadModel=True)
    vizutil = msu.VizUtil(vizer)
    vizutil.set_cam_pos([1.2, 0.0, 1.2])
    vizutil.set_cam_target([0.0, 0.0, 1.0])

    q0 = x0[: rmodel.nq]
    vizer.display(q0)

    for _ in range(3):
        vizutil.play_trajectory(results.xs, results.us, timestep=dt)
        time.sleep(0.5)
