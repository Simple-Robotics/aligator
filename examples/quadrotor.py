"""
Simple quadrotor dynamics example.

Inspired by: https://github.com/loco-3d/crocoddyl/blob/master/examples/quadrotor.py
"""
import pinocchio as pin
import example_robot_data as erd

import numpy as np
import meshcat_utils

import proxddp
from proxddp import manifolds
from proxnlp import constraints

import tap

VIDEO_ARGS = {
    "codec": "libx264",
    "macro_block_size": 8,
    "output_params": ["-crf", "17"],
}

class Args(tap.Tap):
    display: bool = False
    record: bool = False

    def process_args(self):
        if self.record:
            self.display = True


args = Args().parse_args()
print(args)

robot = erd.load("hector")
rmodel = robot.model
rdata = robot.data
nq = rmodel.nq
nv = rmodel.nv

vizer = pin.visualize.MeshcatVisualizer(rmodel, robot.collision_model, robot.visual_model, data=rdata)
vizer.initViewer(loadModel=True)
if args.display:
    vizer.viewer.open()
augvizer = meshcat_utils.ForceDraw(vizer)

space = manifolds.MultibodyPhaseSpace(rmodel)


# The matrix below maps rotor controls to torques

d_cog, cf, cm, u_lim, l_lim = 0.1525, 6.6e-5, 1e-6, 5., 0.1
QUAD_ACT_MATRIX = np.array(
    [[0., 0., 0., 0.],
     [0., 0., 0., 0.],
     [1., 1., 1., 1.],
     [0., d_cog, 0., -d_cog],
     [-d_cog, 0., d_cog, 0.],
     [-cm / cf, cm / cf, -cm / cf, cm / cf]]
)
nu = QUAD_ACT_MATRIX.shape[1]  # = no. of nrotors


class EulerIntegratorDynamics(proxddp.dynamics.ExplicitDynamicsModel):
    def __init__(self, dt: float, B: np.ndarray):
        self.dt = dt
        self.model = rmodel
        self.data = self.model.createData()
        self.B = B
        super().__init__(space, nu)

    def forward(self, x, u, out):
        assert out.size == space.nx
        q = x[:self.model.nq]
        v = x[self.model.nq:]
        tau = self.B @ u
        acc = pin.aba(self.model, self.data, q, v, tau)
        qout = out[:self.model.nq]
        vout = out[self.model.nq:]
        vout[:] = v + self.dt * acc
        qout[:] = pin.integrate(self.model, q, self.dt * vout)

    def dForward(self, x, u, Jx, Ju):
        Jx[:, :] = 0.
        Ju[:, :] = 0.
        q = x[:self.model.nq]
        v = x[self.model.nq:]
        tau = self.B @ u
        acc = pin.aba(self.model, self.data, q, v, tau)
        [dacc_dq, dacc_dv, dacc_dtau] = pin.computeABADerivatives(self.model, self.data, q, v, tau)
        dx = np.concatenate([self.dt * (v + self.dt * acc), self.dt * acc])

        dacc_dx = np.hstack([dacc_dq, dacc_dv])
        dacc_du = dacc_dtau @ self.B

        # Jx <- ddx_dx
        Jx[nv:, :] = dacc_dx * self.dt
        Jx[:nv, :] = Jx[nv:, :] * self.dt
        Jx[:nv, nv:] += np.eye(nv) * self.dt
        Ju[nv:, :] = dacc_du * self.dt
        Ju[:nv, :] = Ju[nv:, :] * self.dt
        space.JintegrateTransport(x, dx, Jx, 1)
        space.JintegrateTransport(x, dx, Ju, 1)

        Jtemp0 = np.zeros((space.ndx, space.ndx))
        space.Jintegrate(x, dx, Jtemp0, 0)
        Jx[:, :] = Jtemp0 + Jx


dt = 0.03
Tf = 2.5
nsteps = int(Tf / dt)

dynmodel = EulerIntegratorDynamics(dt, QUAD_ACT_MATRIX)

x0 = np.concatenate([robot.q0, np.zeros(nv)])
x0[2] = 0.2

u0 = np.zeros(nu)
vizer.display(x0[:nq])
out = space.neutral()

dynmodel.forward(x0, u0, out)
np.set_printoptions(precision=2, linewidth=250)
Jx = np.zeros((space.ndx, space.ndx))
Ju = np.zeros((space.ndx, nu))
Jx_nd = Jx.copy()

x1 = space.rand()
dynmodel.dForward(x1, u0, Jx, Ju)

us_init = [u0] * nsteps
xs_init = [x0] * (nsteps + 1)

x_tar1 = space.neutral()
x_tar1[:3] = (0.9, 0.8, 1.0)
x_tar2 = x_tar1.copy()
x_tar2[:3] = (1.4, -0.6, 1.0)

u_max = 4. * np.ones(nu)
u_min = -1. * np.ones(nu)

times = np.linspace(0, Tf, nsteps + 1)
idx_switch = int(0.7 * nsteps)
t_switch = times[idx_switch]

def setup():
    weights1 = np.zeros(space.ndx)
    weights1[:3] = 4.
    weights1[3:6] = 1e-2
    weights1[nv:] = 1e-3
    weights2 = weights1.copy()
    weights2[:3] = 1.

    w_x_term = np.ones(space.ndx)
    w_x_term[:nv] = 4.
    w_x_term[nv:] = 0.1

    w_u = np.eye(nu) * 1e-2

    stages = []

    for i in range(nsteps):

        rcost = proxddp.CostStack(space.ndx, nu)

        x_tar = x_tar1
        weights = weights1
        if i == idx_switch:
            weights[:] /= dt
        if i > idx_switch:
            x_tar = x_tar2
            weights = weights2

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
        stages.append(stage)

    term_cost = proxddp.QuadraticResidualCost(
        proxddp.StateErrorResidual(space, nu, x_tar2),
        np.diag(w_x_term))
    prob = proxddp.TrajOptProblem(x0, stages, term_cost=term_cost)
    return prob


problem = setup()
tol = 1e-3
mu_init = 0.01
verbose = proxddp.VerboseLevel.VERBOSE
solver = proxddp.ProxDDP(tol, mu_init, verbose=verbose)
solver.run(problem, xs_init, us_init)

results = solver.getResults()
print(results)
xs_opt = results.xs.tolist()
us_opt = results.us.tolist()

import matplotlib.pyplot as plt

fig: plt.Figure = plt.figure()
ax0: plt.Axes = fig.add_subplot(121)
ax0.plot(times[:-1], us_opt)
ax0.hlines((u_min[0], u_max[0]), *times[[0, -1]], colors='k', alpha=0.3, lw=1.4)
ax0.set_title("Controls")
ax0.set_xlabel("Time")
ax1: plt.Axes = fig.add_subplot(122)
root_pt_opt = np.stack(xs_opt)[:, :3]
ax1.plot(times, root_pt_opt)
ax1.hlines(x_tar1[:3], t_switch - 3*dt, t_switch + 3*dt, colors=['C0', 'C1', 'C2'], linestyles='dotted')
ax1.hlines(x_tar2[:3], Tf - 3*dt, Tf + 3*dt, colors=['C0', 'C1', 'C2'], linestyles='dashed')


if args.display:
    import imageio
    frames_ = []
    input("[enter to play]")
    dist_ = 2.
    directions_ = [np.array([1., 1., .5])]
    directions_.append(np.array([1., -1., .8]))
    directions_.append(np.array([1., 0.1, 0.2]))
    for d in directions_: d /= np.linalg.norm(d)


    for i in range(3):
        def post_callback(t):
            n = len(root_pt_opt)
            pos = root_pt_opt[min(t, n)].copy()
            pos += directions_[i] * dist_
            augvizer.set_cam_pos(pos, False)

        augvizer.draw_objectives([x_tar1, x_tar2], prefix='obj')
        frames_ += meshcat_utils.display_trajectory(vizer, augvizer, xs_opt,
                                                    frame_ids=[rmodel.getFrameId("base_link")],
                                                    record=args.record, wait=dt, show_vel=True,
                                                    frame_sphere_size=0.06,
                                                    post_callback=post_callback)

    if args.record:
        vid_uri = "examples/quadrotor_fly.mp4"
        imageio.mimwrite(vid_uri, frames_, fps=1. / dt, **VIDEO_ARGS)

for ext in ['png', 'pdf']:
    fig.savefig("examples/quadrotor_controls.{}".format(ext))
plt.show()
