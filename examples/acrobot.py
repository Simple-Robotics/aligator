import example_robot_data as erd
import proxddp
from proxddp import manifolds, dynamics, constraints
import numpy as np

from common import ArgsBase


class Args(ArgsBase):
    plot: bool = False


args = Args().parse_args()
print(args)

robot = erd.load("double_pendulum_continuous")
rmodel = robot.model
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

stages = []
for i in range(nsteps):
    stages.append(proxddp.StageModel(space, nu, cost, dyn_model))

problem = proxddp.TrajOptProblem(x0, stages, term_cost)
term_cstr = proxddp.StageConstraint(
    proxddp.StateErrorResidual(space, nu, target), constraints.EqualityConstraintSet()
)
problem.setTerminalConstraint(term_cstr)

tol = 1e-3
mu_init = 1e-6
rho_init = 1e-8
solver = proxddp.SolverProxDDP(
    tol, mu_init=mu_init, rho_init=rho_init, verbose=proxddp.VerboseLevel.VERBOSE
)
cb = proxddp.HistoryCallback()
solver.registerCallback(cb)
solver.setup(problem)

us_init = [np.zeros(nu) for _ in range(nsteps)]
xs_init = proxddp.rollout(dyn_model, x0, us_init).tolist()
conv = solver.run(problem, xs_init, us_init)
assert conv

result = solver.getResults()
print(result)


if args.plot:
    import matplotlib.pyplot as plt

    ax: plt.Axes = plt.axes()
    plt.plot(cb.storage.prim_infeas.tolist())
    plt.plot(cb.storage.dual_infeas.tolist())
    ax.set_yscale("log")
    plt.show()


if args.display:
    from pinocchio.visualize import MeshcatVisualizer
    import meshcat_utils as msu

    vizer = MeshcatVisualizer(
        rmodel, robot.collision_model, robot.visual_model, data=robot.data
    )
    vizer.initViewer(open=True, loadModel=True)

    viz_util = msu.VizUtil(vizer)
    viz_util.set_cam_angle_preset("acrobot")

    print("[press enter]")
    for i in range(4):
        viz_util.play_trajectory(result.xs, result.us, timestep=timestep)
