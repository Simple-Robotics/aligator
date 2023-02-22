import example_robot_data as erd
import proxddp
from proxddp import manifolds, dynamics, constraints
import numpy as np
import pinocchio as pin

from common import ArgsBase


class Args(ArgsBase):
    bounds: bool = False


args = Args().parse_args()
print(args)

robot = erd.load("double_pendulum_continuous")
rmodel: pin.Model = robot.model
nq = rmodel.nq
nv = rmodel.nv

space = manifolds.MultibodyPhaseSpace(rmodel)
actuation_matrix = np.array([[0.0], [1.0]])
nu = actuation_matrix.shape[1]

vf = dynamics.MultibodyFreeFwdDynamics(space, actuation_matrix)
timestep = 0.01
target = space.neutral()
x0 = target.copy()
x0[:2] *= -1
dyn_model = dynamics.IntegratorRK2(vf, timestep)
w_x = np.eye(space.ndx) * 1e-4
w_u = np.eye(nu) * 1e-2
cost = proxddp.CostStack(space.ndx, nu)
cost.addCost(
    proxddp.QuadraticResidualCost(
        proxddp.StateErrorResidual(space, nu, target), w_x * timestep
    )
)
cost.addCost(
    proxddp.QuadraticResidualCost(
        proxddp.ControlErrorResidual(space.ndx, nu), w_u * timestep
    )
)
term_cost = proxddp.CostStack(space.ndx, nu)

Tf = 1.0
nsteps = int(Tf / timestep)

ubound = 6.0
umin = -ubound * np.ones(nu)
umax = +ubound * np.ones(nu)

stages = []
for i in range(nsteps):
    stm = proxddp.StageModel(cost, dyn_model)
    if args.bounds:
        stm.addConstraint(
            func=proxddp.ControlErrorResidual(space.ndx, nu),
            cstr_set=constraints.BoxConstraint(umin, umax),
        )
    stages.append(stm)

problem = proxddp.TrajOptProblem(x0, stages, term_cost)
term_cstr = proxddp.StageConstraint(
    proxddp.StateErrorResidual(space, nu, target), constraints.EqualityConstraintSet()
)
problem.addTerminalConstraint(term_cstr)

tol = 1e-3
mu_init = 1e-1
solver = proxddp.SolverProxDDP(tol, mu_init=mu_init, verbose=proxddp.VERBOSE)
solver.max_iters = 200
solver.setup(problem)

us_init = [np.zeros(nu) for _ in range(nsteps)]
xs_init = proxddp.rollout(dyn_model, x0, us_init).tolist()
conv = solver.run(problem, xs_init, us_init)

res = solver.getResults()
print(res)

if args.plot:
    import matplotlib.pyplot as plt

    times = np.linspace(0, Tf, nsteps)
    plt.plot(times, res.us)
    if args.bounds:
        plt.axhline(-ubound, *plt.xlim(), ls="--", c="k")
        plt.axhline(+ubound, *plt.xlim(), ls="--", c="k")
        plt.title("Controls")
    plt.show()


if args.display:
    from pinocchio.visualize import MeshcatVisualizer

    vizer = MeshcatVisualizer(
        rmodel, robot.collision_model, robot.visual_model, data=robot.data
    )
    vizer.initViewer(open=True, loadModel=True)

    vizer.setCameraPreset("acrobot")
    vizer.setBackgroundColor()
    qs = [x[:nq] for x in res.xs]

    print("[press enter]")
    for i in range(4):
        vizer.play(qs, timestep)
