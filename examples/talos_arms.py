import numpy as np
import proxddp
import pinocchio as pin
import meshcat_utils as msu
import example_robot_data as erd
import matplotlib.pyplot as plt

import time

from proxddp import manifolds, dynamics, constraints
from common import get_endpoint_traj, compute_quasistatic, ArgsBase


class Args(ArgsBase):
    tcp: str = None
    bounds: bool = True


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
nq = rmodel.nq
nv = rmodel.nv
nu = nv
print("nq:", nq)
print("nv:", nv)


if args.display:
    vizer = pin.visualize.MeshcatVisualizer(
        rmodel, robot.collision_model, robot.visual_model, data=rdata
    )
    vizer.initViewer(open=True, loadModel=True)
    vizer.display(pin.neutral(rmodel))

    vizutil = msu.VizUtil(vizer)
    vizutil.set_bg_color()
else:
    vizutil = None

space = manifolds.MultibodyPhaseSpace(rmodel)

x0 = space.neutral()

Tf = 0.8
dt = 0.01
nsteps = int(Tf / dt)

w_x = np.ones(space.ndx) * 0.01
w_x[:3] = 1.0
w_x[nv:] = 1e-4
w_x = np.diag(w_x)
w_u = np.eye(nu) * 1e-4
act_matrix = np.eye(nu)

ode = dynamics.MultibodyFreeFwdDynamics(space, act_matrix)
dyn_model = dynamics.IntegratorSemiImplEuler(ode, dt)

tool_id_lh = rmodel.getFrameId("gripper_left_base_link")
target_ee_pos = np.array([0.6, 0.4, 1.4])
frame_fn_lh = proxddp.FrameTranslationResidual(
    space.ndx, nu, rmodel, target_ee_pos, tool_id_lh
)
w_x_ee = 10.0 * np.eye(3)
frame_fn_cost = proxddp.QuadraticResidualCost(frame_fn_lh, w_x_ee)

rcost = proxddp.CostStack(space.ndx, nu)
rcost.addCost(proxddp.QuadraticCost(w_x, w_u), dt)
rcost.addCost(frame_fn_cost.copy(), 0.01 * dt)

stm = proxddp.StageModel(rcost, dyn_model)
umax = rmodel.effortLimit
umin = -umax
if args.bounds:
    stm.addConstraint(
        proxddp.ControlBoxFunction(space.ndx, umin, umax), constraints.NegativeOrthant()
    )

term_cost = proxddp.CostStack(space.ndx, nu)
term_cost.addCost(
    proxddp.QuadraticResidualCost(proxddp.StateErrorResidual(space, nu, x0), w_x)
)
term_cost.addCost(frame_fn_cost)


stages = [stm] * nsteps
problem = proxddp.TrajOptProblem(x0, stages, term_cost)


TOL = 1e-5
mu_init = 1e-3
rho_init = 0.0
max_iters = 200
verbose = proxddp.VerboseLevel.VERBOSE
solver = proxddp.SolverProxDDP(TOL, mu_init, rho_init, verbose=verbose)
solver.rollout_type = proxddp.ROLLOUT_NONLINEAR
print("LDLT algo choice:", solver.ldlt_algo_choice)
# solver = proxddp.SolverFDDP(TOL, verbose=verbose)
solver.max_iters = max_iters
solver.setup(problem)

u0 = compute_quasistatic(rmodel, rdata, x0, acc=np.zeros(nv))
us_init = [u0] * nsteps
xs_init = proxddp.rollout(dyn_model, x0, us_init).tolist()

solver.run(
    problem,
    xs_init,
    us_init,
)
workspace = solver.getWorkspace()
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
us_opt = np.array(results.us)
ls = plt.plot(times[1:], results.us)
mask_where_ctrl_saturate = np.any((us_opt <= umin) | (us_opt >= umax), axis=0)
idx_hit = np.argwhere(mask_where_ctrl_saturate).flatten()
if len(idx_hit) > 0:
    ls[idx_hit[0]].set_label("u{}".format(idx_hit[0]))
    plt.hlines(umin[idx_hit], *times[[0, -1]], colors="r", linestyles="--")
    plt.hlines(umax[idx_hit], *times[[0, -1]], colors="b", linestyles="--")
plt.title("Controls trajectory")
plt.legend()
plt.tight_layout()


plt.figure()
value_grads = [v.Vx for v in workspace.value_params]
value_grads = np.stack(value_grads).T
plt.imshow(value_grads)
plt.xlabel("Time $t$")
plt.ylabel("Dimension $\\partial V_t/\\partial x_i$")

plt.tight_layout()
plt.show()

if args.display:
    vizutil.set_cam_pos([1.2, 0.0, 1.2])
    vizutil.set_cam_target([0.0, 0.0, 1.0])

    for _ in range(3):
        vizutil.play_trajectory(results.xs, results.us, timestep=dt)
        time.sleep(0.5)
