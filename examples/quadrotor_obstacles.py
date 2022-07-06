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
        "column1", 0, 0, cylinder, pin.SE3(R, center_column1)
    )
    center_column2 = np.array([0.3, 2.1, 0.0])
    geom_cyl2 = pin.GeometryObject(
        "column2", 0, 0, cylinder, pin.SE3(R, center_column2)
    )
    geom_cyl1.meshColor = np.array([2.0, 0.2, 1.0, 0.6])
    geom_cyl2.meshColor = np.array([2.0, 0.2, 1.0, 0.6])
    robot.collision_model.addGeometryObject(geom_cyl1)
    robot.visual_model.addGeometryObject(geom_cyl1)
    robot.collision_model.addGeometryObject(geom_cyl2)
    robot.visual_model.addGeometryObject(geom_cyl2)

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
Tf = 2.5
nsteps = int(Tf / dt)
print("nsteps: {:d}".format(nsteps))

dynmodel = proxddp.dynamics.IntegratorEuler(ode_dynamics, dt)

x0 = np.concatenate([robot.q0, np.zeros(nv)])
x0[2] = 0.2

u0 = np.zeros(nu)
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

x_tar = space.neutral()
x_tar[:3] = (-0.3, 2.5, 1.0)

u_max = 4.5 * np.ones(nu)
u_min = -1.0 * np.ones(nu)

times = np.linspace(0, Tf, nsteps + 1)
idx_switch = int(0.7 * nsteps)
t_switch = times[idx_switch]


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
        Jx = np.zeros((1, self.ndx))
        Jx[:, 2] = self.sign
        Ju = np.zeros((1, self.nu))
        data.Jx[:] = Jx
        data.Ju[:] = Ju


class Column(proxddp.StageFunction):
    def __init__(self, ndx, nu, center, radius: float = 0.2) -> None:
        super().__init__(ndx, nu, 1)
        self.ndx = ndx
        self.center = center
        self.radius = radius

    def evaluate(self, x, u, y, data):  # distance function
        res = -(np.sum(np.square(x[:2] - self.center)) - self.radius**2)
        data.value[:] = res

    def computeJacobians(self, x, u, y, data):  # TODO check jacobian
        Jx = np.zeros((1, self.ndx))
        Jx[:, :2] = -2 * (x[:2] - self.center)
        Ju = np.zeros((1, self.nu))
        data.Jx[:] = Jx
        data.Ju[:] = Ju


def setup():
    weights = np.zeros(space.ndx)
    weights[:3] = 4.0
    weights[3:6] = 1e-2
    weights[nv:] = 1e-3

    w_x_term = np.ones(space.ndx)
    w_x_term[:nv] = 4.0
    w_x_term[nv:] = 0.1

    w_u = np.eye(nu) * 1e-2

    stages = []

    for i in range(nsteps):

        rcost = proxddp.CostStack(space.ndx, nu)

        state_err = proxddp.StateErrorResidual(space, nu, x_tar)
        xreg_cost = proxddp.QuadraticResidualCost(state_err, np.diag(weights) * dt)

        rcost.addCost(xreg_cost)

        utar = np.zeros(nu)
        u_err = proxddp.ControlErrorResidual(space.ndx, nu, utar)
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
            column1 = Column(space.ndx, nu, center_column1[:2])
            stage.addConstraint(column1, constraints.NegativeOrthant())
            column2 = Column(space.ndx, nu, center_column2[:2])
            stage.addConstraint(column2, constraints.NegativeOrthant())
        stages.append(stage)

        sd = stage.createData()
        stage.evaluate(x0, u0, x1, sd)

    term_cost = proxddp.QuadraticResidualCost(
        proxddp.StateErrorResidual(space, nu, x_tar), np.diag(w_x_term)
    )
    prob = proxddp.TrajOptProblem(x0, stages, term_cost=term_cost)
    return prob


problem = setup()
tol = 1e-3
mu_init = 0.01
rho = 0.001
verbose = proxddp.VerboseLevel.VERBOSE
cb = proxddp.HistoryCallback()
solver = proxddp.ProxDDP(tol, mu_init, rho, verbose=verbose, max_iters=300)
solver.register_callback(cb)
solver.setup(problem)
solver.run(problem, xs_init, us_init)

results = solver.getResults()
print(results)
xs_opt = results.xs.tolist()
us_opt = results.us.tolist()

fig: plt.Figure = plt.figure()
ax0: plt.Axes = fig.add_subplot(121)
ax0.plot(times[:-1], us_opt)
ax0.hlines((u_min[0], u_max[0]), *times[[0, -1]], colors="k", alpha=0.3, lw=1.4)
ax0.set_title("Controls")
ax0.set_xlabel("Time")
ax1: plt.Axes = fig.add_subplot(122)
root_pt_opt = np.stack(xs_opt)[:, :3]
ax1.plot(times, root_pt_opt)
ax1.hlines(
    x_tar[:3],
    t_switch - 3 * dt,
    t_switch + 3 * dt,
    colors=["C0", "C1", "C2"],
    linestyles="dotted",
)


if args.display:
    viz_util = msu.VizUtil(vizer)
    input("[enter to play]")
    dist_ = 2.0
    directions_ = [np.array([1.0, 1.0, 0.5])]
    directions_.append(np.array([1.0, -1.0, 0.8]))
    directions_.append(np.array([1.0, 0.1, 0.2]))
    for d in directions_:
        d /= np.linalg.norm(d)

    vid_uri = "examples/quadrotor_fly.mp4"
    vid_recorder = msu.VideoRecorder(vid_uri, fps=1.0 / dt)
    for i in range(3):

        def post_callback(t):
            n = len(root_pt_opt)
            pos = root_pt_opt[min(t, n)].copy()
            pos += directions_[i] * dist_
            viz_util.set_cam_pos(pos)

        viz_util.draw_objectives([x_tar], prefix="obj")
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
    fig.savefig("examples/quadrotor_controls.{}".format(ext))
plt.show()
