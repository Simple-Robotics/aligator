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


class Args(ArgsBase):
    bounds: bool = False
    num_threads: int = 4


args = Args().parse_args()
q0[2] += 0.02
pin.framesForwardKinematics(rmodel, rdata, q0)


nq = rmodel.nq
nv = rmodel.nv
nu = nv - 6
space = manifolds.MultibodyPhaseSpace(rmodel)
ndx = space.ndx
act_matrix = np.eye(nv, nu, -6)
effort_limit = rmodel.effortLimit[6:]
print(f"Effort limit: {effort_limit}")

constraint_models = create_ground_contact_model(rmodel, (0, 0, 100), 60)
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


dt = 5e-3  # 20 ms
tf = 1.2  # in seconds
nsteps = int(tf / dt)
print("Num steps: {:d}".format(nsteps))

switch_t0 = 0.4
switch_t1 = 1.0  # landing time
k0 = int(switch_t0 / dt)
k1 = int(switch_t1 / dt)

times = np.linspace(0, tf, nsteps + 1)
mask = (switch_t0 <= times) & (times < switch_t1)

q1 = q0.copy()
# q1[3:7] = pin.exp3_quat(np.array([0.0, 0.0, np.pi / 3]))
v0 = np.zeros(nv)
x0_ref = np.concatenate((q0, v0))
w_x = np.ones(space.ndx) * 1e-2
w_x[:nv] = 1.0
w_x[3:6] = 0.1
w_x[nv : nv + 6] = 0.0
w_x = np.diag(w_x)
w_u = np.eye(nu) * 1e-1


def create_land_fns():
    out = {}
    for fname, fid in FOOT_FRAME_IDS.items():
        p_ref = rdata.oMf[fid].translation
        fn = aligator.FrameTranslationResidual(space.ndx, nu, rmodel, p_ref, fid)
        out[fid] = fn[2]  # [2]
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


stages = []
for k in range(nsteps):
    vf = ode1
    if mask[k]:
        vf = ode2
    xref = x0_ref.copy()

    xreg_cost = aligator.QuadraticStateCost(space, nu, xref, weights=w_x * dt)
    ureg_cost = aligator.QuadraticControlCost(space, nu, weights=w_u * dt)
    cost = aligator.CostStack(space, nu)
    cost.addCost(xreg_cost)
    cost.addCost(ureg_cost)

    dyn_model = dynamics.IntegratorSemiImplEuler(vf, dt)
    stm = aligator.StageModel(cost, dyn_model)

    if args.bounds:
        stm.addConstraint(
            aligator.ControlErrorResidual(ndx, nu),
            constraints.BoxConstraint(-effort_limit, effort_limit),
        )
    if k == k1:
        pin.framesForwardKinematics(rmodel, rdata, q1)
        for fid, fn in create_land_fns().items():
            stm.addConstraint(fn, constraints.EqualityConstraintSet())
        for fid, fn in create_land_vel_fns().items():
            stm.addConstraint(fn, constraints.EqualityConstraintSet())

    stages.append(stm)


w_xterm = w_x * 1
xterm = np.concatenate((q1, v0))
print("x0   :", x0_ref)
print("xterm:", xterm)
term_cost = aligator.QuadraticStateCost(space, nu, xterm, weights=w_xterm)

problem = aligator.TrajOptProblem(x0_ref, stages, term_cost)
mu_init = 1e-5
tol = 1e-5
solver = aligator.SolverProxDDP(tol, mu_init, verbose=aligator.VERBOSE, max_iters=300)
solver.rollout_type = aligator.ROLLOUT_LINEAR
solver.setNumThreads(args.num_threads)

cb_ = aligator.HistoryCallback(solver)
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
    ax.hlines(tol, 0, res.num_iters, lw=2.2, alpha=0.8, colors="k")
    plot_convergence(cb_, ax, res, show_al_iters=True)
    fig4.tight_layout()

    _fig_dict = {"controls": fig1, "velocities": fig2, "foot_traj": fig3, "conv": fig4}

    for name, fig in _fig_dict.items():
        for ext in ("pdf",):
            _fpath = f"assets/solo_jump_{name}.{ext}"
            fig.savefig(_fpath)

    plt.show()


if __name__ == "__main__":
    if args.display:
        from pinocchio.visualize import MeshcatVisualizer

        vizer = MeshcatVisualizer(
            rmodel,
            collision_model=robot.collision_model,
            visual_model=robot.visual_model,
            data=rdata,
        )
        vizer.initViewer(
            open=args.zmq_url is None, loadModel=True, zmq_url=args.zmq_url
        )
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

    # FPS = min(30, 1.0 / dt)
    FPS = 1.0 / dt

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
