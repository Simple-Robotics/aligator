import proxddp
import pinocchio as pin
from solo_utils import (
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
import tap

from proxddp import manifolds, dynamics
from pinocchio.visualize import MeshcatVisualizer


class Args(tap.Tap):
    record: bool = True


args = Args().parse_args()
pin.framesForwardKinematics(rmodel, rdata, q0)


nq = rmodel.nq
nv = rmodel.nv
nu = nv - 6
space = manifolds.MultibodyPhaseSpace(rmodel)
act_matrix = np.eye(nv, nu, -6)

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
tf = 1.0  # in seconds
nsteps = int(tf / dt)

switch_t0 = 0.3
switch_t1 = 0.8  # landing time
k0 = int(switch_t0 / dt)
k1 = int(switch_t1 / dt)

times = np.linspace(0, tf, nsteps + 1)
mask = (switch_t0 <= times) & (times < switch_t1)

x0_ref = np.concatenate((q0, np.zeros(nv)))
w_x = np.ones(space.ndx) * 1e-3
w_x[:6] = 0.0
w_x[nv : nv + 6] = 0.0
w_x = np.diag(w_x)
w_u = np.eye(nu) * 1e-4


def create_fly_high_cost(costs: proxddp.CostStack, slope=50):
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


stages = []
for k in range(nsteps):
    vf = ode1
    if mask[k]:
        vf = ode2

    wxlocal_k = w_x * dt
    xreg_cost = proxddp.QuadraticStateCost(space, nu, x0_ref, weights=wxlocal_k)
    ureg_cost = proxddp.QuadraticControlCost(space, nu, weights=w_u * dt)
    cost = proxddp.CostStack(space, nu)
    cost.addCost(xreg_cost)
    cost.addCost(ureg_cost)
    fly_high_cost = create_fly_high_cost(cost)

    dyn_model = dynamics.IntegratorSemiImplEuler(vf, dt)
    stm = proxddp.StageModel(cost, dyn_model)

    if k == k1:
        fns = create_land_fns()
        for fid, fn in fns.items():
            stm.addConstraint(fn, proxddp.constraints.EqualityConstraintSet())

    stages.append(stm)


term_cost = proxddp.QuadraticStateCost(space, nu, x0_ref, weights=w_x)

problem = proxddp.TrajOptProblem(x0_ref, stages, term_cost)
mu_init = 0.1
solver = proxddp.SolverProxDDP(1e-3, mu_init, verbose=proxddp.VERBOSE)
solver.setup(problem)


xs_init = [x0_ref] * (nsteps + 1)
us_init = [np.zeros(nu) for _ in range(nsteps)]


add_plane(robot)
vizer = MeshcatVisualizer(
    rmodel,
    collision_model=robot.collision_model,
    visual_model=robot.visual_model,
    data=rdata,
)


if __name__ == "__main__":
    vizer.initViewer(loadModel=True, zmq_url=args.zmq_url)
    custom_color = np.asarray((53, 144, 243)) / 255.0
    vizer.setBackgroundColor(col_bot=list(custom_color), col_top=(1, 1, 1, 1))
    manage_lights(vizer)
    vizer.display(q0)

    solver.run(problem, xs_init, us_init)
    res = solver.results
    print(res)

    input("[display]")
    qs = [x[:nq] for x in res.xs]
    vs = [x[nq:] for x in res.xs]

    FPS = 1.0 / dt * 0.5

    def callback(i: int):
        pin.forwardKinematics(rmodel, rdata, qs[i], vs[i])
        for fid in FOOT_FRAME_IDS.values():
            vizer.drawFrameVelocities(fid)

    if args.record:
        with vizer.create_video_ctx("examples/solo_jump.mp4", fps=FPS):
            print("[Recording video]")
            vizer.play(qs, dt, callback=callback)

    while True:
        vizer.play(qs, dt, callback=callback)
        input("[replay]")
