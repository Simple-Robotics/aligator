import aligator
import pinocchio as pin
from utils.solo import (
    robot,
    rmodel,
    rdata,
    q0,
    create_ground_contact_model,
    manage_lights,
    add_plane,
    FOOT_FRAME_IDS,
)

import numpy as np
import matplotlib.pyplot as plt

from utils import ArgsBase, get_endpoint_traj, IMAGEIO_KWARGS
from aligator import manifolds, dynamics, constraints
from aligator.utils.plotting import (
    plot_controls_traj,
    plot_velocity_traj,
    plot_convergence,
)
from pinocchio.visualize import MeshcatVisualizer


class Args(ArgsBase):
    bounds: bool = False


args = Args().parse_args()
pin.framesForwardKinematics(rmodel, rdata, q0)


nq = rmodel.nq
nv = rmodel.nv
nu = nv - 6
space = manifolds.MultibodyPhaseSpace(rmodel)
ndx = space.ndx
act_matrix = np.eye(nv, nu, -6)
effort_limit = rmodel.effortLimit[6:]
print(f"Effort limit: {effort_limit}")

constraint_models = create_ground_contact_model(rmodel, (0, 0, 100), 50)
prox_settings = pin.ProximalSettings(1e-9, 1e-10, 10)
ode1 = dynamics.MultibodyConstraintFwdDynamics(
    space, act_matrix, constraint_models, prox_settings
)
ode2 = dynamics.MultibodyConstraintFwdDynamics(space, act_matrix, [], prox_settings)


def test():
    x0 = space.neutral()
    u0 = np.random.randn(nu)
    d1 = ode1.createData()
    ode1.forward(x0, u0, d1)
    ode1.dForward(x0, u0, d1)
    d2 = ode2.createData()
    ode2.forward(x0, u0, d2)
    ode2.dForward(x0, u0, d2)


test()


dt = 20e-3  # 20 ms
tf = 1.2  # in seconds
nsteps = int(tf / dt)

switch_t0 = 0.4
switch_t1 = 0.9  # landing time
k0 = int(switch_t0 / dt)
k1 = int(switch_t1 / dt)

times = np.linspace(0, tf, nsteps + 1)
mask = (switch_t0 <= times) & (times < switch_t1)

x0_ref = np.concatenate((q0, np.zeros(nv)))
x_ref_flight = x0_ref.copy()
w_x = np.ones(space.ndx) * 1e-2
w_x[:6] = 0.0
w_x[nv : nv + 6] = 0.0
w_x = np.diag(w_x)
w_u = np.eye(nu) * 1e-4


def add_fly_high_cost(costs: aligator.CostStack, slope):
    fly_high_w = 1.0
    for fname, fid in FOOT_FRAME_IDS.items():
        fn = aligator.FlyHighResidual(space, fid, slope, nu)
        fl_cost = aligator.QuadraticResidualCost(space, fn, np.eye(2) * dt)
        costs.addCost(fl_cost, fly_high_w / len(FOOT_FRAME_IDS))


def create_land_fns():
    out = {}
    for fname, fid in FOOT_FRAME_IDS.items():
        p_ref = rdata.oMf[fid].translation
        fn = aligator.FrameTranslationResidual(space.ndx, nu, rmodel, p_ref, fid)
        out[fid] = fn[2]
    return out


def create_land_vel_fns():
    out = {}
    for fname, fid in FOOT_FRAME_IDS.items():
        v_ref = pin.Motion.Zero()
        fn = aligator.FrameVelocityResidual(
            space.ndx, nu, rmodel, v_ref, fid, pin.LOCAL_WORLD_ALIGNED
        )
        out[fid] = fn[:3]
    return out


def create_land_cost(costs, w):
    fns = create_land_fns()
    land_cost_w = np.eye(1)
    for fid, fn in fns.items():
        land_cost = aligator.QuadraticResidualCost(space, fn, land_cost_w)
        costs.addCost(land_cost, w / len(FOOT_FRAME_IDS))


x_ref_flight[2] = 1.2
stages = []
for k in range(nsteps):
    vf = ode1
    wxlocal_k = w_x * dt
    if mask[k]:
        vf = ode2
    xref = x0_ref

    xreg_cost = aligator.QuadraticStateCost(space, nu, xref, weights=wxlocal_k)
    ureg_cost = aligator.QuadraticControlCost(space, nu, weights=w_u * dt)
    cost = aligator.CostStack(space, nu)
    cost.addCost(xreg_cost)
    cost.addCost(ureg_cost)
    add_fly_high_cost(cost, slope=50)

    dyn_model = dynamics.IntegratorSemiImplEuler(vf, dt)
    stm = aligator.StageModel(cost, dyn_model)

    if args.bounds:
        stm.addConstraint(
            aligator.ControlErrorResidual(ndx, nu),
            constraints.BoxConstraint(-effort_limit, effort_limit),
        )
    if k == k1:
        fns = create_land_fns()
        for fid, fn in fns.items():
            stm.addConstraint(fn, constraints.EqualityConstraintSet())
        for fid, fn in create_land_vel_fns().items():
            stm.addConstraint(fn, constraints.EqualityConstraintSet())

    stages.append(stm)


w_xterm = w_x.copy()
term_cost = aligator.QuadraticStateCost(space, nu, x0_ref, weights=w_x)

problem = aligator.TrajOptProblem(x0_ref, stages, term_cost)
mu_init = 1e-4
tol = 1e-4
solver = aligator.SolverProxDDP(tol, mu_init, verbose=aligator.VERBOSE, max_iters=200)
solver.rollout_type = aligator.ROLLOUT_LINEAR

cb_ = aligator.HistoryCallback()
solver.registerCallback("his", cb_)
solver.setup(problem)


xs_init = [x0_ref] * (nsteps + 1)
us_init = [np.zeros(nu) for _ in range(nsteps)]


add_plane(robot)


def make_plots(res: aligator.Results):
    fig1, axes = plt.subplots(nu // 2, 2, figsize=(9.6, 8.4))
    plot_controls_traj(times, res.us, axes=axes, rmodel=rmodel)
    fig1.suptitle("Joint torques")
    fig1.tight_layout()

    vs = [x[nq:] for x in res.xs]
    fig2, axes = plt.subplots(nv // 3, 3, figsize=(11.2, 8.4))
    plot_velocity_traj(times, vs, rmodel, axes=axes)
    fig2.suptitle("Joint velocities")

    pts = {}
    for fname, fid in FOOT_FRAME_IDS.items():
        pts[fid] = get_endpoint_traj(rmodel, rdata, res.xs, fid)
    assert len(pts) == 4
    fig3, axes = plt.subplots(2, 2)
    axes = axes.flatten()
    for i, (fname, fid) in enumerate(FOOT_FRAME_IDS.items()):
        axes[i].plot(times, pts[fid], label=["x", "y", "z"])
        axes[i].set_xlabel("Time $t$")
        axes[i].set_ylabel(fname.lower())
        axes[i].legend()
    fig3.suptitle("Foot 3D trajectories")
    fig3.tight_layout()

    fig4 = plt.figure()
    ax = fig4.add_subplot(111)
    plot_convergence(cb_, ax, res=res)
    fig4.tight_layout()

    _fig_dict = {"controls": fig1, "velocities": fig2, "foot_traj": fig3, "conv": fig4}

    for name, fig in _fig_dict.items():
        for ext in ("pdf",):
            _fpath = f"assets/solo_jump_{name}.{ext}"
            fig.savefig(_fpath)

    plt.show()


if __name__ == "__main__":
    if args.display:
        vizer = MeshcatVisualizer(
            rmodel,
            collision_model=robot.collision_model,
            visual_model=robot.visual_model,
            data=rdata,
        )
        vizer.initViewer(loadModel=True, zmq_url=args.zmq_url)
        # custom_color = np.asarray((53, 144, 243)) / 255.0
        # vizer.setBackgroundColor(col_bot=list(custom_color), col_top=(1, 1, 1, 1))
        manage_lights(vizer)
        vizer.display(q0)
        cam_pos = np.array((0.9, -0.3, 0.4))
        cam_pos *= 0.9 / np.linalg.norm(cam_pos)
        cam_tar = (0.0, 0.0, 0.3)
        vizer.setCameraPosition(cam_pos)
        vizer.setCameraTarget(cam_tar)

    solver.run(problem, xs_init, us_init)
    res = solver.results
    print(res)
    if args.plot:
        make_plots(res)

    xs = np.stack(res.xs)
    qs = xs[:, :nq]
    vs = xs[:, nq:]

    FPS = min(30, 1.0 / dt)

    if args.display:
        import contextlib

        def callback(i: int):
            pin.forwardKinematics(rmodel, rdata, qs[i], vs[i])
            for fid in FOOT_FRAME_IDS.values():
                vizer.drawFrameVelocities(fid)

        input("[display]")
        ctx = (
            vizer.create_video_ctx("assets/solo_jump.mp4", fps=FPS, **IMAGEIO_KWARGS)
            if args.record
            else contextlib.nullcontext()
        )

        with ctx:
            while True:
                vizer.play(qs, dt, callback=callback)
                input("[replay]")
