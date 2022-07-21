"""
Original script:
https://github.com/loco-3d/crocoddyl/blob/master/examples/arm_manipulation.py

In this script, we demonstrate use the Python Crocoddyl API, by defining
a manipulation problem using Crocoddyl and converting it to a proxddp problem.
"""

import os
import sys

import crocoddyl
import pinocchio
import numpy as np
import example_robot_data
import meshcat_utils as msu
from pinocchio.visualize import MeshcatVisualizer


from proxddp.croc import convertCrocoddylProblem

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ

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
xResidual = crocoddyl.ResidualModelControl(state)
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

vizer = MeshcatVisualizer(
    robot_model, talos_arm.collision_model, talos_arm.visual_model, data=talos_arm.data
)
vizer.initViewer(loadModel=True, open=True)
viz_util = msu.VizUtil(vizer)
vizer.display(q0)
fid_display = robot_model.getFrameId("gripper_left_joint")

if True:
    # Creating the DDP solver for this OC problem, defining a logger
    solver = crocoddyl.SolverDDP(problem)
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
    solver.solve()

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
        # display = crocoddyl.MeshcatDisplay(talos_arm, 4, 4)
        input("[enter to play]")
        # display.displayFromSolver(solver)
        viz_util.play_trajectory(
            solver.xs.tolist(), solver.us.tolist(), timestep=dt, frame_ids=[fid_display]
        )

    croc_xs = solver.xs
    croc_us = solver.us

if True:
    import proxddp

    print("running proxddp")
    prox_problem = convertCrocoddylProblem(problem)

    tol = 1e-4
    mu_init = 1e-2
    rho_init = 1e-5
    solver = proxddp.ProxDDP(tol, mu_init, rho_init=rho_init)
    solver.verbose = proxddp.VerboseLevel.VERBOSE
    solver.max_iters = 300
    solver.bcl_params.rho_factor = 0.1
    solver.setup(prox_problem)
    xs_i = [x0] * (T + 1)
    us_i = [np.zeros(actuationModel.nu) for _ in range(T)]
    solver.run(prox_problem, xs_i, us_i)

    results = solver.getResults()
    print("Results {}".format(results))
    xs_opt = results.xs
    us_opt = results.us

    if WITHPLOT:
        import matplotlib.pyplot as plt

        xs_opt_flat = np.stack(xs_opt)
        us_opt_flat = np.stack(us_opt)
        t_f = T * dt
        times = np.linspace(0, t_f, T + 1)

        plt.subplot(121)
        plt.plot(times, xs_opt_flat)
        plt.title("States")
        plt.subplot(122)
        plt.plot(times[:-1], us_opt_flat)
        plt.title("Controls")
        plt.show()

    if WITHDISPLAY:

        input("[press enter to play]")
        for i in range(3):
            viz_util.play_trajectory(
                xs_opt, us_opt, timestep=dt, frame_ids=[fid_display], show_vel=True
            )


dist_x = [np.linalg.norm(croc_xs[i] - xs_opt[i]) for i in range(T + 1)]
dist_u = [np.linalg.norm(croc_us[i] - us_opt[i]) for i in range(T)]

print("Errs:")
print(dist_x)
print(dist_u)
