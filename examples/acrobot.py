import example_robot_data as erd
import aligator
from aligator import manifolds, dynamics, constraints
from utils import IMAGEIO_KWARGS, manage_lights
import numpy as np
import pinocchio as pin

from utils import ArgsBase


class Args(ArgsBase):
    bounds: bool = False
    term_cstr: bool = False


args = Args().parse_args()
print(args)

robot = erd.load("double_pendulum_continuous")
rmodel: pin.Model = robot.model
rmodel.effortLimit[:] = 1e40
nq = rmodel.nq
nv = rmodel.nv

space = manifolds.MultibodyPhaseSpace(rmodel)
actuation_matrix = np.array([[0.0], [1.0]])
nu = actuation_matrix.shape[1]

timestep = 0.01
target = space.neutral()
x0 = space.neutral()
x0[0] = -1.0

dyn_model = dynamics.IntegratorRK2(
    dynamics.MultibodyFreeFwdDynamics(space, actuation_matrix), timestep
)
w_x = np.eye(space.ndx) * 1e-4
w_u = np.eye(nu) * 1e-3
cost = aligator.CostStack(space, nu)
cost.addCost(aligator.QuadraticStateCost(space, nu, target, w_x * timestep))
cost.addCost(aligator.QuadraticControlCost(space, np.zeros(nu), w_u * timestep))
term_cost = aligator.CostStack(space, nu)
if not args.term_cstr:
    term_cost.addCost(
        aligator.QuadraticStateCost(space, nu, target, np.eye(space.ndx) * 10)
    )

Tf = 1.0
nsteps = int(Tf / timestep)

ubound = 4.0
umin = -ubound * np.ones(nu)
umax = +ubound * np.ones(nu)
if args.bounds:
    rmodel.effortLimit[:] = umax

stages = []
box_set = constraints.BoxConstraint(umin, umax)
for i in range(nsteps):
    stm = aligator.StageModel(cost, dyn_model)
    if args.bounds:
        stm.addConstraint(aligator.ControlErrorResidual(space.ndx, nu), box_set)
    stages.append(stm)

problem = aligator.TrajOptProblem(x0, stages, term_cost)
if args.term_cstr:
    problem.addTerminalConstraint(
        aligator.StateErrorResidual(space, nu, target),
        constraints.EqualityConstraintSet(),
    )

tol = 1e-3
mu_init = 1e-2
solver = aligator.SolverProxDDP(tol, mu_init=mu_init, verbose=aligator.VERBOSE)
solver.max_iters = 200
solver.rollout_type = aligator.ROLLOUT_NONLINEAR
solver.linear_solver_choice = aligator.LQ_SOLVER_STAGEDENSE
solver.setup(problem)


us_init = [np.zeros(nu) for _ in range(nsteps)]
xs_init = aligator.rollout(dyn_model, x0, us_init).tolist()
conv = solver.run(problem, xs_init, us_init)

res = solver.results
print(res)

tag = "acrobot"
if args.bounds:
    tag += "_bounds"
if args.term_cstr:
    tag += "_termcstr"

if args.plot:
    import matplotlib.pyplot as plt
    from aligator.utils.plotting import plot_controls_traj

    times = np.linspace(0, Tf, nsteps + 1)
    fig1, axes = plot_controls_traj(
        times, res.us, ncols=1, rmodel=rmodel, figsize=(6.4, 3.2)
    )
    fig1.tight_layout()
    xs = np.stack(res.xs)
    vs = xs[:, nq:]

    theta_s = np.zeros((nsteps + 1, 2))
    theta_s0 = space.difference(space.neutral(), x0)[:2]
    theta_s = theta_s0 + np.cumsum(vs * timestep, axis=0)
    fig2 = plt.figure(figsize=(6.4, 6.4))
    plt.subplot(211)
    plt.plot(times, theta_s, label=("$\\theta_0$", "$\\theta_1$"))
    plt.title("Joint angles")
    plt.legend()
    plt.subplot(212)
    plt.plot(times, xs[:, nq:], label=("$\\dot\\theta_0$", "$\\dot\\theta_1$"))
    plt.legend()
    plt.title("Joint velocities")
    fig2.tight_layout()

    _fig_dict = {"controls": fig1, "velocities": fig2}

    for name, fig in _fig_dict.items():
        fig.savefig(f"assets/{tag}_{name}.pdf")

    plt.show()


if args.display:
    from pinocchio.visualize import MeshcatVisualizer

    vizer = MeshcatVisualizer(
        rmodel, robot.collision_model, robot.visual_model, data=robot.data
    )
    vizer.initViewer(open=True, loadModel=True)
    manage_lights(vizer)
    vizer.display(x0[:nq])

    vizer.setCameraPreset("acrobot")
    qs = [x[:nq] for x in res.xs]

    FPS = 1.0 / timestep
    if args.record:
        ctx = vizer.create_video_ctx(f"assets/{tag}.mp4", fps=FPS, **IMAGEIO_KWARGS)
    else:
        from contextlib import nullcontext

        ctx = nullcontext()

    input("[press enter]")
    with ctx:
        vizer.play(qs, timestep)
    vizer.play(qs, timestep)
