import example_robot_data as erd
import proxddp
from proxddp import manifolds, dynamics, constraints
import numpy as np


robot = erd.load("double_pendulum")
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
x0[0] = np.pi
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

Tf = 1.0
nsteps = int(Tf / timestep)

stages = []
for i in range(nsteps):
    stages.append(proxddp.StageModel(space, nu, cost, dyn_model))

# problem = proxddp.TrajOptProblem(x0, stages, term_cost)
