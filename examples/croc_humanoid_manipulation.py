"""
Original script:
https://github.com/loco-3d/crocoddyl/blob/master/examples/humanoid_manipulation.py

In this script, we demonstrate use the Python Crocoddyl API, by defining
a manipulation problem using Crocoddyl and converting it to a proxddp problem.
"""
import os
import sys

import crocoddyl
from crocoddyl.utils.biped import plotSolution
import numpy as np
import example_robot_data
import pinocchio

import proxddp
import time

import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(add_help=True)
parser.add_argument(
    "--solver",
    help="Choice of the second solver",
    choices=["fddp", "proxddp"],
    required=True,
)


args = parser.parse_args()

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ

# Load robot
robot = example_robot_data.load("talos")
rmodel = robot.model

# Create data structures
rdata = rmodel.createData()
state = crocoddyl.StateMultibody(rmodel)
actuation = crocoddyl.ActuationModelFloatingBase(state)

# Set integration time
DT = 5e-2
nsteps = 10
target = np.array([0.4, 0, 1.2])

# Initialize reference state, target and reference CoM
rightFoot = "right_sole_link"
leftFoot = "left_sole_link"
endEffector = "gripper_left_joint"
endEffectorId = rmodel.getFrameId(endEffector)
rightFootId = rmodel.getFrameId(rightFoot)
leftFootId = rmodel.getFrameId(leftFoot)
q0 = rmodel.referenceConfigurations["half_sitting"]
x0 = np.concatenate([q0, np.zeros(rmodel.nv)])
pinocchio.forwardKinematics(rmodel, rdata, q0)
pinocchio.updateFramePlacements(rmodel, rdata)
rfPos0 = rdata.oMf[rightFootId].translation
lfPos0 = rdata.oMf[leftFootId].translation
jointId = rmodel.getFrameId("gripper_left_joint")
refGripper = rdata.oMf[jointId].translation
comRef = (rfPos0 + lfPos0) / 2
comRef[2] = pinocchio.centerOfMass(rmodel, rdata, q0)[2].item()

# Initialize Gepetto viewer
if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(robot, frameNames=[rightFoot, leftFoot])
    display.robot.viewer.gui.addSphere(
        "world/point", 0.05, [1.0, 0.0, 0.0, 1.0]
    )  # radius = .1, RGBA=1001
    display.robot.viewer.gui.applyConfiguration(
        "world/point", target.tolist() + [0.0, 0.0, 0.0, 1.0]
    )  # xyz+quaternion

# Add contact to the model
contactModel = crocoddyl.ContactModelMultiple(state, actuation.nu)
supportContactModelLeft = crocoddyl.ContactModel6D(
    state, leftFootId, pinocchio.SE3.Identity(), actuation.nu, np.array([0, 0])
)
contactModel.addContact(leftFoot + "_contact", supportContactModelLeft)
supportContactModelRight = crocoddyl.ContactModel6D(
    state, rightFootId, pinocchio.SE3.Identity(), actuation.nu, np.array([0, 0])
)
contactModel.addContact(rightFoot + "_contact", supportContactModelRight)
contactData = contactModel.createData(rdata)

# Cost for self-collision
maxfloat = sys.float_info.max
xlb = np.concatenate(
    [
        -maxfloat * np.ones(6),  # dimension of the SE(3) manifold
        rmodel.lowerPositionLimit[7:],
        -maxfloat * np.ones(state.nv),
    ]
)
xub = np.concatenate(
    [
        maxfloat * np.ones(6),  # dimension of the SE(3) manifold
        rmodel.upperPositionLimit[7:],
        maxfloat * np.ones(state.nv),
    ]
)
bounds = crocoddyl.ActivationBounds(xlb, xub, 1.0)
xLimitResidual = crocoddyl.ResidualModelState(state, x0, actuation.nu)
xLimitActivation = crocoddyl.ActivationModelQuadraticBarrier(bounds)
limitCost = crocoddyl.CostModelResidual(state, xLimitActivation, xLimitResidual)

# Cost for state and control
xResidual = crocoddyl.ResidualModelState(state, x0, actuation.nu)
xActivation = crocoddyl.ActivationModelWeightedQuad(
    np.array([0] * 3 + [10.0] * 3 + [0.01] * (state.nv - 6) + [10] * state.nv) ** 2
)
uResidual = crocoddyl.ResidualModelControl(state, actuation.nu)
xTActivation = crocoddyl.ActivationModelWeightedQuad(
    np.array([0] * 3 + [10.0] * 3 + [0.01] * (state.nv - 6) + [100] * state.nv) ** 2
)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
xRegTermCost = crocoddyl.CostModelResidual(state, xTActivation, xResidual)

# Cost for target reaching
framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
    state, endEffectorId, pinocchio.SE3(np.eye(3), target), actuation.nu
)
framePlacementActivation = crocoddyl.ActivationModelWeightedQuad(
    np.array([1] * 3 + [0.0001] * 3) ** 2
)
goalTrackingCost = crocoddyl.CostModelResidual(
    state, framePlacementActivation, framePlacementResidual
)

# Cost for CoM reference
comResidual = crocoddyl.ResidualModelCoMPosition(state, comRef, actuation.nu)
comTrack = crocoddyl.CostModelResidual(state, comResidual)

# Create cost model per each action model
runningCostModel = crocoddyl.CostModelSum(state, actuation.nu)
terminalCostModel = crocoddyl.CostModelSum(state, actuation.nu)

# Then let's added the running and terminal cost functions
runningCostModel.addCost("gripperPose", goalTrackingCost, 1e2)
runningCostModel.addCost("stateReg", xRegCost, 1e-3)
runningCostModel.addCost("ctrlReg", uRegCost, 1e-4)
runningCostModel.addCost("limitCost", limitCost, 1e3)

terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e2)
terminalCostModel.addCost("stateReg", xRegTermCost, 1e-3)
terminalCostModel.addCost("limitCost", limitCost, 1e3)

# Create the action model
dmodelRunning = crocoddyl.DifferentialActionModelContactFwdDynamics(
    state, actuation, contactModel, runningCostModel
)
dmodelTerminal = crocoddyl.DifferentialActionModelContactFwdDynamics(
    state, actuation, contactModel, terminalCostModel
)
runningModel = crocoddyl.IntegratedActionModelEuler(dmodelRunning, DT)
terminalModel = crocoddyl.IntegratedActionModelEuler(dmodelTerminal, 0)

# Problem definition
x0 = np.concatenate([q0, pinocchio.utils.zero(state.nv)])
croc_problem = crocoddyl.ShootingProblem(x0, [runningModel] * nsteps, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
max_iters = 30
solver = crocoddyl.SolverFDDP(croc_problem)
cbs = [crocoddyl.CallbackVerbose()]
if WITHDISPLAY and WITHPLOT:
    cbs.extend(
        [
            crocoddyl.CallbackLogger(),
            crocoddyl.CallbackDisplay(
                crocoddyl.GepettoDisplay(robot, 4, 4, frameNames=[rightFoot, leftFoot])
            ),
        ]
    )
elif WITHDISPLAY:
    cbs.extend(
        [
            crocoddyl.CallbackDisplay(
                crocoddyl.GepettoDisplay(robot, 4, 4, frameNames=[rightFoot, leftFoot])
            )
        ]
    )
elif WITHPLOT:
    cbs.extend([crocoddyl.CallbackLogger()])
else:
    cbs.extend([])
solver.setCallbacks(cbs)

# Solving it with the FDDP algorithm
xs_init = [x0] * (solver.problem.T + 1)
us_init = solver.problem.quasiStatic([x0] * solver.problem.T)
reg_init = 1e-9
tol = 1e-5
solver.th_stop = tol**2

elapsed_time_croc = time.time()
solver.solve(xs_init, us_init, max_iters, False, reg_init)
elapsed_time_croc = time.time() - elapsed_time_croc

croc_inf_norm = max([np.linalg.norm(q, np.inf) for q in solver.Qu])
print("Croc final norm: {}".format(croc_inf_norm))

# Visualizing the solution in gepetto-viewer
if WITHDISPLAY:
    display.displayFromSolver(solver)

# Get final state and end effector position
xT = solver.xs[-1]
pinocchio.forwardKinematics(rmodel, rdata, xT[: state.nq])
pinocchio.updateFramePlacements(rmodel, rdata)
com = pinocchio.centerOfMass(rmodel, rdata, xT[: state.nq])
finalPosEff = np.array(
    rdata.oMf[rmodel.getFrameId("gripper_left_joint")].translation.T.flat
)

print("Final cost = ", solver.cost)
print("Finally reached = ", finalPosEff)
print("Distance between hand and target = ", np.linalg.norm(finalPosEff - target))
print("Distance to default state = ", np.linalg.norm(x0 - np.array(xT.flat)))
print("XY distance to CoM reference = ", np.linalg.norm(com[:2] - comRef[:2]))

# Plotting the entire motion
if WITHPLOT:
    log = solver.getCallbacks()[0]
    plotSolution(solver, bounds=False, figIndex=1, show=False)

    crocoddyl.plotConvergence(
        log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps, figIndex=3
    )


prox_problem = proxddp.croc.convertCrocoddylProblem(croc_problem)
verbose = proxddp.VerboseLevel.VERBOSE
if args.solver == "proxddp":
    mu_init = 1e-4
    solver2 = proxddp.SolverProxDDP(croc_inf_norm, mu_init, verbose=verbose)
elif args.solver == "fddp":
    solver2 = proxddp.SolverFDDP(croc_inf_norm, verbose=verbose)
else:
    raise ValueError("unknown choice of second solver ({})".format(args.solver))
solver2.verbose = verbose
solver2.setup(prox_problem)
solver2.max_iters = max_iters
solver2.reg_init = reg_init

elapsed_time_prox = time.time()
solver2.run(prox_problem, xs_init, us_init)
elapsed_time_prox = time.time() - elapsed_time_prox

rs = solver2.getResults()
ws = solver2.getWorkspace()
print(rs)

print("CROC TIME = {:.4e}".format(elapsed_time_croc))
print("PROX TIME = {:.4e}".format(elapsed_time_prox))

x_err_v = [np.max(abs(solver.xs[t] - rs.xs[t])) for t in range(nsteps + 1)]
print("xerr = {}".format(max(x_err_v)))
u_err_v = [np.max(abs(solver.us[t] - rs.us[t])) for t in range(nsteps)]
print("uerr = {}".format(max(u_err_v)))

print("cost diff = {}".format(croc_problem.calc(solver.xs, solver.us) - rs.traj_cost))


nu = actuation.nu
fig: plt.Figure = plt.figure(constrained_layout=True, figsize=(9.6, 5.4))
gs = fig.add_gridspec(4, nu // 4)
axes = gs.subplots(sharex=True)
axes = axes.flatten()
us_ = rs.us
us_ = np.asarray(us_)

times = np.arange(0, DT * nsteps, DT)
for i in range(nu):
    axes[i].plot(times, us_[:, i])
    axes[i].tick_params(axis="both", labelsize=8)
plt.xlabel("Time [s]")
plt.suptitle("Controls $u_i(t)$")
plt.show()
