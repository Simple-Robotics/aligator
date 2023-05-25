"""
Original script:
https://github.com/loco-3d/crocoddyl/blob/master/examples/arm_manipulation.py

In this script, we demonstrate use the Python Crocoddyl API, by defining
a manipulation problem using Crocoddyl and converting it to a proxddp problem.
"""

import tap
import time

import crocoddyl
import pinocchio
import numpy as np
import example_robot_data
from pinocchio.visualize import MeshcatVisualizer

from proxddp.croc import convertCrocoddylProblem


class Args(tap.Tap):
    display: bool = False
    plot: bool = False


parser = Args("croc_arm_manipulation")
args = parser.parse_args()

WITHDISPLAY = args.display
WITHPLOT = args.plot

# In this example test, we will solve the reaching-goal task with the Talos arm.
# For that, we use the forward dynamics (with its analytical derivatives)
# developed inside crocoddyl; it describes inside DifferentialActionModelFullyActuated class.
# Finally, we use an Euler sympletic integration scheme.

# First, let's load the Pinocchio model for the Talos arm.
talos_arm = example_robot_data.load("talos_arm")
robot_model = talos_arm.model

# Create a cost model per the running and terminal action model.
state = crocoddyl.StateMultibody(robot_model)
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)

# Note that we need to include a cost model (i.e. set of cost functions) in
# order to fully define the action model for our optimal control problem.
# For this particular example, we formulate three running-cost functions:
# goal-tracking cost, state and control regularization; and one terminal-cost:
# goal cost. First, let's create the common cost functions.
framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
    state,
    robot_model.getFrameId("gripper_left_joint"),
    pinocchio.SE3(np.eye(3), np.array([0.0, 0.0, 0.4])),
)
uResidual = crocoddyl.ResidualModelControl(state)
xResidual = crocoddyl.ResidualModelState(state)
goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)

# Then let's added the running and terminal cost functions
runningCostModel.addCost("gripperPose", goalTrackingCost, 1)
runningCostModel.addCost("xReg", xRegCost, 1e-4)
runningCostModel.addCost("uReg", uRegCost, 1e-4)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 1)

# Next, we need to create an action model for running and terminal knots. The
# forward dynamics (computed using ABA) are implemented
# inside DifferentialActionModelFullyActuated.
actuationModel = crocoddyl.ActuationModelFull(state)
dt = 1e-3
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuationModel, runningCostModel
    ),
    dt,
)
runningModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0])
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuationModel, terminalCostModel
    ),
    0.0,
)
terminalModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0])

# For this optimal control problem, we define 250 knots (or running action
# models) plus a terminal knot
T = 250
q0 = np.array([0.173046, 1.0, -0.52366, 0.0, 0.0, 0.1, -0.005])
x0 = np.concatenate([q0, pinocchio.utils.zero(robot_model.nv)])
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

if WITHDISPLAY:
    vizer = MeshcatVisualizer(
        robot_model,
        talos_arm.collision_model,
        talos_arm.visual_model,
        data=talos_arm.data,
    )
    vizer.initViewer(loadModel=True, open=True)
    vizer.display(q0)
else:
    vizer = None
fid_display = robot_model.getFrameId("gripper_left_joint")

t_f = T * dt
times = np.linspace(0, t_f, T + 1)
TOLERANCE = 1e-4

xs_i = [x0] * (T + 1)
us_i = [np.zeros(actuationModel.nu) for _ in range(T)]

if True:
    # Creating the DDP solver for this OC problem, defining a logger
    solver = crocoddyl.SolverFDDP(problem)
    cameraTF = [2.0, 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]
    if WITHDISPLAY and WITHPLOT:
        display = crocoddyl.MeshcatDisplay(talos_arm, 4, 4)
        solver.setCallbacks(
            [
                crocoddyl.CallbackLogger(),
                crocoddyl.CallbackVerbose(),
                crocoddyl.CallbackDisplay(display),
            ]
        )
    elif WITHDISPLAY:
        display = crocoddyl.MeshcatDisplay(talos_arm, 4, 4)
        solver.setCallbacks(
            [crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)]
        )
    elif WITHPLOT:
        solver.setCallbacks(
            [
                crocoddyl.CallbackLogger(),
                crocoddyl.CallbackVerbose(),
            ]
        )
    else:
        solver.setCallbacks([crocoddyl.CallbackVerbose()])

    # Solving it with the DDP algorithm
    solver.th_stop = TOLERANCE**2 * T

    elapsed_time = time.time()
    solver.solve(xs_i, us_i)
    elapsed_time = time.time() - elapsed_time
    print(f"CROCODDYL TIME = {elapsed_time:.4g}")

    # Plotting the solution and the DDP convergence
    if WITHPLOT:
        log = solver.getCallbacks()[0]
        crocoddyl.plotOCSolution(log.xs, log.us, figIndex=1, show=False)
        crocoddyl.plotConvergence(
            log.costs,
            log.u_regs,
            log.x_regs,
            log.grads,
            log.stops,
            log.steps,
            figIndex=2,
        )

    # Visualizing the solution in gepetto-viewer
    if WITHDISPLAY:
        import pinocchio as pin

        def vis_callback(i: int):
            pin.forwardKinematics(robot_model, vizer.data, qs[i], vs[i])
            vizer.drawFrameVelocities(fid_display)

        # display = crocoddyl.MeshcatDisplay(talos_arm, 4, 4)
        input("[enter to play]")
        # display.displayFromSolver(solver)
        nq = robot_model.nq
        qs = [x[:nq] for x in solver.xs]
        vizer.play(qs, dt, vis_callback)

    croc_xs = solver.xs
    croc_us = solver.us

    croc_dual_err = np.max([np.linalg.norm(q, np.inf) for q in solver.Qu])
    print("Croc inf. err: {:.4e}".format(croc_dual_err))

if True:
    import proxddp

    print("running proxddp")
    prox_problem = convertCrocoddylProblem(problem)

    # mu_init = 1e-7
    # rho_init = 1e-10
    # proxsolver = proxddp.SolverProxDDP(croc_dual_err * 0.8, 0.001, max_iters=300)
    proxsolver = proxddp.SolverFDDP(tol=croc_dual_err)
    proxsolver.verbose = proxddp.VerboseLevel.VERBOSE
    proxsolver.setup(prox_problem)

    elapsed_time = time.time()
    proxsolver.run(prox_problem, xs_i, us_i)
    elapsed_time = time.time() - elapsed_time
    print(f"PROXDDP TIME = {elapsed_time:.4g}")

    results = proxsolver.getResults()
    assert results.num_iters <= 24
    print("Results {}".format(results))
    prox_xs = results.xs
    prox_us = results.us
    prox_xs = np.stack(prox_xs)
    prox_us = np.stack(prox_us)

    print("prox_cost - croc_cost:", results.traj_cost - solver.cost)

    if WITHPLOT:
        import matplotlib.pyplot as plt

        plt.subplot(121)
        plt.plot(times, prox_xs)
        plt.title("States")
        plt.subplot(122)
        plt.plot(times[:-1], prox_us)
        plt.title("Controls")
        plt.show()

    if WITHDISPLAY:
        qs = [x[:nq] for x in prox_xs]
        vs = [x[nq:] for x in prox_xs]
        input("[press enter to play]")
        for i in range(3):
            vizer.play(qs, dt, vis_callback)


croc_xs = np.stack(croc_xs.tolist())
croc_us = np.stack(croc_us.tolist())
nq = robot_model.nq
if WITHPLOT:
    for i in range(robot_model.nq):
        plt.subplot(2, nq // 2 + 1, i + 1)
        plt.plot(times, croc_xs[:, i], ls="--")
        plt.plot(times, prox_xs[:, i], ls="--")
    fig = plt.gcf()
    fig.legend(["croco", "prox"])
    plt.tight_layout()
    plt.show()

dist_x = [np.linalg.norm(croc_xs[i] - prox_xs[i]) for i in range(T + 1)]
dist_u = [np.linalg.norm(croc_us[i] - prox_us[i]) for i in range(T)]

dist_x = np.max(dist_x)
dist_u = np.max(dist_u)

print("Errs:")
print(dist_x)
print(dist_u)
