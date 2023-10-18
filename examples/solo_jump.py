import proxddp
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
from proxddp import manifolds, dynamics, constraints
from proxddp.utils.plotting import (
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
w_x = np.ones(space.ndx) * 1e-3
w_x[:6] = 0.0
w_x[nv : nv + 6] = 0.0
w_x = np.diag(w_x)
w_u = np.eye(nu) * 1e-4


def add_fly_high_cost(costs: proxddp.CostStack, slope):
    fly_high_w = 1.0
    for fname, fid in FOOT_FRAME_IDS.items():
        fn = proxddp.FlyHighResidual(space, fid, slope, nu)
        fl_cost = proxddp.QuadraticResidualCost(space, fn, np.eye(2) * dt)
        costs.addCost(fl_cost, fly_high_w / len(FOOT_FRAME_IDS))


def create_land_fns():
    out = {}
    for fname, fid in FOOT_FRAME_IDS.items():
        p_ref = rdata.oMf[fid].translation
        fn = proxddp.FrameTranslationResidual(space.ndx, nu, rmodel, p_ref, fid)
        fn = fn[2]
        out[fid] = fn
    return out


def create_land_cost(costs, w):
    fns = create_land_fns()
    land_cost_w = np.eye(1)
    for fid, fn in fns.items():
        land_cost = proxddp.QuadraticResidualCost(space, fn, land_cost_w)
        costs.addCost(land_cost, w / len(FOOT_FRAME_IDS))


x_ref_flight[2] = 1.2
stages = []
for k in range(nsteps):
    vf = ode1
    wxlocal_k = w_x * dt
    if mask[k]:
        vf = ode2
    xref = x0_ref

    xreg_cost = proxddp.QuadraticStateCost(space, nu, xref, weights=wxlocal_k)
    ureg_cost = proxddp.QuadraticControlCost(space, nu, weights=w_u * dt)
    cost = proxddp.CostStack(space, nu)
    cost.addCost(xreg_cost)
    cost.addCost(ureg_cost)
    add_fly_high_cost(cost, slope=50)

    dyn_model = dynamics.IntegratorSemiImplEuler(vf, dt)
    stm = proxddp.StageModel(cost, dyn_model)

    if args.bounds:
        stm.addConstraint(
            proxddp.ControlErrorResidual(ndx, nu),
            constraints.BoxConstraint(-effort_limit, effort_limit),
        )
    if k == k1:
        fns = create_land_fns()
        for fid, fn in fns.items():
            stm.addConstraint(fn, constraints.EqualityConstraintSet())

    stages.append(stm)


w_xterm = w_x.copy()
term_cost = proxddp.QuadraticStateCost(space, nu, x0_ref, weights=w_x)

problem = proxddp.TrajOptProblem(x0_ref, stages, term_cost)
# mu_init = 0.1
mu_init = 1e-4
tol = 1e-4
solver = proxddp.SolverProxDDP(tol, mu_init, verbose=proxddp.VERBOSE, max_iters=200)
solver.rollout_type = proxddp.ROLLOUT_LINEAR

cb_ = proxddp.HistoryCallback()
solver.registerCallback("his", cb_)
solver.setup(problem)


xs_init = [x0_ref] * (nsteps + 1)
us_init = [np.zeros(nu) for _ in range(nsteps)]


add_plane(robot, (251, 127, 0, 240))
vizer = MeshcatVisualizer(
    rmodel,
    collision_model=robot.collision_model,
    visual_model=robot.visual_model,
    data=rdata,
)


def make_plots(res: proxddp.Results):
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
    vizer.initViewer(loadModel=True, zmq_url=args.zmq_url)
    custom_color = np.asarray((53, 144, 243)) / 255.0
    vizer.setBackgroundColor(col_bot=list(custom_color), col_top=(1, 1, 1, 1))
    manage_lights(vizer)
    vizer.display(q0)

    solver.run(problem, xs_init, us_init)
    res = solver.results
    print(res)
    make_plots(res)

    qs = [x[:nq] for x in res.xs]
    vs = [x[nq:] for x in res.xs]

    FPS = 1.0 / dt

    def callback(i: int):
        pin.forwardKinematics(rmodel, rdata, qs[i], vs[i])
        for fid in FOOT_FRAME_IDS.values():
            vizer.drawFrameVelocities(fid)

    cam_pos = np.array([1.0, 0.7, 1.0])
    cam_pos *= 0.9 / np.linalg.norm(cam_pos)
    vizer.setCameraPosition(cam_pos)
    vizer.setCameraTarget((0.0, 0.0, 0.3))

    if args.display:
        input("[display]")
        if args.record:
            with vizer.create_video_ctx(
                "assets/solo_jump.mp4", fps=FPS, **IMAGEIO_KWARGS
            ):
                print("[Recording video]")
                vizer.play(qs, dt, callback=callback)

        while True:
            vizer.play(qs, dt, callback=callback)
            input("[replay]")
