import numpy as np
import proxddp
import pinocchio as pin
import meshcat_utils as msu
import example_robot_data as erd

from proxddp import manifolds, dynamics, constraints


def loadTalos():
    robot = erd.load("talos")
    qref = robot.model.referenceConfigurations["half_sitting"]
    locked_joints = list(range(1, 14))
    locked_joints += [23, 31]
    locked_joints += [32, 33]
    red_bot = robot.buildReducedRobot(locked_joints, qref)
    return red_bot


robot = loadTalos()
rmodel = robot.model
rdata = robot.data
nv = rmodel.nv
nu = nv


space = manifolds.MultibodyPhaseSpace(rmodel)

x0 = space.neutral()
x_hsit = rmodel.referenceConfigurations["half_sitting"]
print(x_hsit)

Tf = 1.2
dt = 0.02
nsteps = int(Tf / dt)

w_x = np.eye(space.ndx) * 0.1
w_u = np.eye(nu) * 1e-3
act_matrix = np.eye(nu)

ode = dynamics.MultibodyFreeFwdDynamics(space, act_matrix)
dyn_model = dynamics.IntegratorSemiImplEuler(ode, dt)

rcost = proxddp.CostStack(space.ndx, nu)
rcost.addCost(proxddp.QuadraticCost(w_x, w_u), dt)

stm = proxddp.StageModel(space, nu, rcost, dyn_model)

term_cost = proxddp.QuadraticResidualCost(
    proxddp.StateErrorResidual(space, nu, x_hsit), w_x
)

stages = [stm] * nsteps
problem = proxddp.TrajOptProblem(x0, stages, term_cost)


TOL = 1e-3
MU_INIT = 1e-2
solver = proxddp.SolverProxDDP(TOL, MU_INIT, 1e-8)
solver.setup(problem)

xs_init = [x0] * (nsteps + 1)
u0 = np.zeros(nu)
us_init = [u0] * nsteps

solver.run(
    problem,
    xs_init,
    us_init,
)
results = solver.getResults()
print(results)


vizer = pin.visualize.MeshcatVisualizer(
    rmodel, robot.collision_model, robot.visual_model, data=rdata
)
vizer.initViewer(loadModel=True)

vizutil = msu.VizUtil(vizer)

import time

for _ in range(3):
    vizutil.play_trajectory(results.xs, results.us, timestep=dt / 0.5)
    time.sleep(0.5)
