"""
Simple quadrotor dynamics example.

Inspired by: https://github.com/loco-3d/crocoddyl/blob/master/examples/quadrotor.py
"""
import pinocchio as pin
import hppfcl as fcl
import example_robot_data as erd

import numpy as np
import meshcat_utils as msu
import matplotlib.pyplot as plt

import proxddp
import tap

from proxddp import manifolds
from proxnlp import constraints


class Args(tap.Tap):
    display: bool = False
    record: bool = False

    def process_args(self):
        if self.record:
            self.display = True

    obstacles: bool = False


args = Args().parse_args()
print(args)

robot = erd.load("hector")
rmodel = robot.model
rdata = robot.data
nq = rmodel.nq
nv = rmodel.nv

if args.obstacles:  # we add the obstacles to the geometric model
    R = np.eye(3)
    cyl_radius = 0.2
    cylinder = fcl.Cylinder(cyl_radius, 10.0)
    center_column1 = np.array([-0.2, 0.8, 0.0])
    geom_cyl1 = pin.GeometryObject(
        "column1", 0, 0, pin.SE3(R, center_column1), cylinder
    )
    center_column2 = np.array([0.3, 2.1, 0.0])
    geom_cyl2 = pin.GeometryObject(
        "column2", 0, 0, pin.SE3(R, center_column2), cylinder
    )
    geom_cyl1.meshColor = np.array([2.0, 0.2, 1.0, 0.6])
    geom_cyl2.meshColor = np.array([2.0, 0.2, 1.0, 0.6])
    robot.collision_model.addGeometryObject(geom_cyl1)
    robot.visual_model.addGeometryObject(geom_cyl1)
    robot.collision_model.addGeometryObject(geom_cyl2)
    robot.visual_model.addGeometryObject(geom_cyl2)
    robot.collision_model.geometryObjects[0].geometry.computeLocalAABB()
    quad_radius = robot.collision_model.geometryObjects[0].geometry.aabb_radius

vizer = pin.visualize.MeshcatVisualizer(
    rmodel, robot.collision_model, robot.visual_model, data=rdata
)
vizer.initViewer(loadModel=True)
if args.display:
    vizer.viewer.open()

space = manifolds.MultibodyPhaseSpace(rmodel)


# The matrix below maps rotor controls to torques

d_cog, cf, cm, u_lim, l_lim = 0.1525, 6.6e-5, 1e-6, 5.0, 0.1
QUAD_ACT_MATRIX = np.array(
    [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.0, d_cog, 0.0, -d_cog],
        [-d_cog, 0.0, d_cog, 0.0],
        [-cm / cf, cm / cf, -cm / cf, cm / cf],
    ]
)
nu = QUAD_ACT_MATRIX.shape[1]  # = no. of nrotors


ode_dynamics = proxddp.dynamics.MultibodyFreeFwdDynamics(space, QUAD_ACT_MATRIX)

dt = 0.033
Tf = 2.0
nsteps = int(Tf / dt)
print("nsteps: {:d}".format(nsteps))

dynmodel = proxddp.dynamics.IntegratorEuler(ode_dynamics, dt)

x0 = np.concatenate([robot.q0, np.zeros(nv)])
x0[2] = 0.2

tau = pin.rnea(rmodel, rdata, robot.q0, np.zeros(nv), np.zeros(nv))
u0, _, _, _ = np.linalg.lstsq(QUAD_ACT_MATRIX, tau)
vizer.display(x0[:nq])
out = space.neutral()

data = dynmodel.createData()
dynmodel.forward(x0, u0, data)
np.set_printoptions(precision=2, linewidth=250)
Jx = np.zeros((space.ndx, space.ndx))
Ju = np.zeros((space.ndx, nu))
Jx_nd = Jx.copy()

x1 = space.rand()
dynmodel.dForward(x1, u0, data)

us_init = [u0] * nsteps
xs_init = [x0] * (nsteps + 1)

x_tar1 = space.neutral()
x_tar1[:3] = (0.9, 0.8, 1.0)
x_tar2 = space.neutral()
x_tar2[:3] = (1.4, -0.6, 1.0)
x_tar3 = space.neutral()
x_tar3[:3] = (-0.3, 2.5, 1.0)

u_max = 4.5 * np.ones(nu)
u_min = 0.0 * np.ones(nu)

times = np.linspace(0, Tf, nsteps + 1)
idx_switch = int(0.7 * nsteps)
t_switch = times[idx_switch]


def make_task():
    if args.obstacles:
        weights = np.zeros(space.ndx)
        weights[:3] = 4.0
        weights[3:6] = 1e-2
        weights[nv:] = 1e-3

        def weight_target_selector(i):
            return weights, x_tar3

    else:
        weights1 = np.zeros(space.ndx)
        weights1[:3] = 4.0
        weights1[3:6] = 1e-2
        weights1[nv:] = 1e-3
        weights2 = weights1.copy()
        weights2[:3] = 1.0

        def weight_target_selector(i):
            x_tar = x_tar1
            weights = weights1
            if i == idx_switch:
                weights[:] /= dt
            if i > idx_switch:
                x_tar = x_tar2
                weights = weights2
            return weights, x_tar

    return weight_target_selector


class HalfspaceZ(proxddp.StageFunction):
    def __init__(self, ndx, nu, offset: float = 0.0, neg: bool = False) -> None:
        super().__init__(ndx, nu, 1)
        self.ndx = ndx
        self.offset = offset
        self.sign = -1.0 if neg else 1.0

    def evaluate(self, x, u, y, data):
        res = self.sign * (x[2] - self.offset)
        data.value[:] = res

    def computeJacobians(self, x, u, y, data):
        data.Jx[2] = self.sign
        data.Ju[:] = 0.0


class Column(proxddp.StageFunction):
    def __init__(
        self, ndx, nu, center, radius: float = 0.2, margin: float = 0.0
    ) -> None:
        super().__init__(ndx, nu, 1)
        self.ndx = ndx
        self.center = center
        self.radius = radius
        self.margin = margin

    def evaluate(self, x, u, y, data):  # distance function
        err = x[:2] - self.center
        res = np.dot(err, err) - (self.radius + self.margin) ** 2
        data.value[:] = -res

    def computeJacobians(self, x, u, y, data):  # TODO check jacobian
        err = x[:2] - self.center
        data.Jx[:2] = -2 * err
        data.Ju[:] = 0.0


task_fun = make_task()


def setup():

    w_x_term = np.ones(space.ndx)
    w_x_term[:nv] = 4.0
    w_x_term[nv:] = 0.1

    w_u = np.eye(nu) * 1e-2

    stages = []

    for i in range(nsteps):

        rcost = proxddp.CostStack(space.ndx, nu)

        weights, x_tar = task_fun(i)

        state_err = proxddp.StateErrorResidual(space, nu, x_tar)
        xreg_cost = proxddp.QuadraticResidualCost(state_err, np.diag(weights) * dt)

        rcost.addCost(xreg_cost)

        u_err = proxddp.ControlErrorResidual(space.ndx, nu)
        ucost = proxddp.QuadraticResidualCost(u_err, w_u * dt)
        rcost.addCost(ucost)

        stage = proxddp.StageModel(space, nu, rcost, dynmodel)
        ctrl_box = proxddp.ControlBoxFunction(space.ndx, u_min, u_max)
        stage.addConstraint(ctrl_box, constraints.NegativeOrthant())
        if args.obstacles:  # add obstacles' constraints
            ceiling = HalfspaceZ(space.ndx, nu, 2.0)
            stage.addConstraint(ceiling, constraints.NegativeOrthant())
            floor = HalfspaceZ(space.ndx, nu, 0.0, True)
            stage.addConstraint(floor, constraints.NegativeOrthant())
            column1 = Column(space.ndx, nu, center_column1[:2], margin=quad_radius)
            stage.addConstraint(column1, constraints.NegativeOrthant())
            column2 = Column(space.ndx, nu, center_column2[:2], margin=quad_radius)
            stage.addConstraint(column2, constraints.NegativeOrthant())
        stages.append(stage)
        if i == nsteps - 1:  # terminal constraint
            stage.addConstraint(
                proxddp.StateErrorResidual(space, nu, x_tar),
                constraints.EqualityConstraintSet(),
            )

        sd = stage.createData()
        stage.evaluate(x0, u0, x1, sd)

    term_cost = proxddp.QuadraticResidualCost(
        proxddp.StateErrorResidual(space, nu, x_tar), np.diag(w_x_term)
    )
    prob = proxddp.TrajOptProblem(x0, stages, term_cost=term_cost)
    return prob


_, x_term = task_fun(nsteps)
problem = setup()
tol = 1e-3
mu_init = 0.01
verbose = proxddp.VerboseLevel.VERBOSE
rho_init = 0.003
history_cb = proxddp.HistoryCallback()
solver = proxddp.ProxDDP(tol, mu_init, rho_init, verbose=verbose, max_iters=300)
solver.registerCallback(history_cb)
solver.setup(problem)
solver.run(problem, xs_init, us_init)

results = solver.getResults()
print(results)
xs_opt = results.xs.tolist()
us_opt = results.us.tolist()

fig: plt.Figure = plt.figure(figsize=(9.6, 5.4))
ax0: plt.Axes = fig.add_subplot(131)
ax0.plot(times[:-1], us_opt)
ax0.hlines((u_min[0], u_max[0]), *times[[0, -1]], colors="k", alpha=0.3, lw=1.4)
ax0.set_title("Controls")
ax0.set_xlabel("Time")
ax1: plt.Axes = fig.add_subplot(132)
root_pt_opt = np.stack(xs_opt)[:, :3]
ax1.plot(times, root_pt_opt)
ax1.hlines(
    x_term[:3],
    t_switch - 3 * dt,
    t_switch + 3 * dt,
    colors=["C0", "C1", "C2"],
    linestyles="dotted",
)
ax1.hlines(
    x_term[:3], Tf - 3 * dt, Tf + 3 * dt, colors=["C0", "C1", "C2"], linestyles="dashed"
)
ax2: plt.Axes = fig.add_subplot(133)
n_iter = [i for i in range(len(history_cb.storage.prim_infeas.tolist()))]
ax2.semilogy(n_iter, history_cb.storage.prim_infeas.tolist(), label="Primal err.")
ax2.semilogy(n_iter, history_cb.storage.dual_infeas.tolist(), label="Dual err.")
ax2.set_xlabel("Iterations")
ax2.legend()

if args.obstacles:
    TAG = "quadrotor_obstacles"
else:
    TAG = "quadrotor"


if args.display:
    viz_util = msu.VizUtil(vizer)
    input("[enter to play]")
    dist_ = 2.0
    directions_ = [np.array([1.0, 1.0, 0.5])]
    directions_.append(np.array([1.0, -1.0, 0.8]))
    directions_.append(np.array([1.0, 0.1, 0.2]))
    for d in directions_:
        d /= np.linalg.norm(d)

    vid_uri = "examples/{}.mp4".format(TAG)
    vid_recorder = msu.VideoRecorder(vid_uri, fps=1.0 / dt)
    for i in range(3):

        def post_callback(t):
            n = len(root_pt_opt)
            n = min(t, n)
            rp = root_pt_opt[n]
            pos = rp + directions_[i] * dist_
            viz_util.set_cam_pos(pos)
            viz_util.set_cam_target(rp)

        if args.obstacles:
            viz_util.draw_objectives([x_tar3], prefix="obj")
        else:
            viz_util.draw_objectives([x_tar1, x_tar2], prefix="obj")
        viz_util.play_trajectory(
            xs_opt,
            us_opt,
            frame_ids=[rmodel.getFrameId("base_link")],
            record=args.record,
            timestep=dt,
            show_vel=True,
            frame_sphere_size=0.06,
            recorder=vid_recorder,
            post_callback=post_callback,
        )

for ext in ["png", "pdf"]:
    fig.savefig("examples/{}.{}".format(TAG, ext))
plt.show()
