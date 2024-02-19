import aligator
import pinocchio as pin
from utils.solo import robot, rmodel, rdata, q0
import time

import numpy as np
import matplotlib.pyplot as plt

from utils import ArgsBase
from aligator import manifolds, dynamics, constraints
import copy


class Args(ArgsBase):
    bounds: bool = False


args = Args().parse_args()

if args.display:
    vizer = pin.visualize.MeshcatVisualizer(
        rmodel, robot.collision_model, robot.visual_model, data=rdata
    )
    vizer.initViewer(open=True, loadModel=True)
    vizer.display(pin.neutral(rmodel))
    vizer.setBackgroundColor()

pin.forwardKinematics(rmodel, rdata, q0)
pin.updateFramePlacements(rmodel, rdata)

nq = rmodel.nq
nv = rmodel.nv
nk = 4
force_size = 3
nu = nv - 6 + nk * force_size
space = manifolds.MultibodyPhaseSpace(rmodel)
ndx = space.ndx

x0 = np.concatenate((q0, np.zeros(nv)))
u0 = np.zeros(nu)
com0 = pin.centerOfMass(rmodel, rdata, x0[:nq])

dt = 20e-3  # Timestep
gravity = np.array([0, 0, -9.81])
mu = 0.8  # Friction coefficient
mass = pin.computeTotalMass(rmodel)
f_ref = np.array([0, 0, -mass * gravity[2] / 4.0])  # Initial contact force

FL_id = rmodel.getFrameId("FL_FOOT")
FR_id = rmodel.getFrameId("FR_FOOT")
HL_id = rmodel.getFrameId("HL_FOOT")
HR_id = rmodel.getFrameId("HR_FOOT")

feet_ids = [FL_id, FR_id, HL_id, HR_id]

FL_pose = rdata.oMf[FL_id].copy()
FR_pose = rdata.oMf[FR_id].copy()
HL_pose = rdata.oMf[HL_id].copy()
HR_pose = rdata.oMf[HR_id].copy()

# Cost weights

w_x = np.ones(space.ndx) * 1e-2
w_x[0:6] = 0.0
w_x = np.diag(w_x)
w_u = np.eye(nu) * 1e-6

w_trans = np.ones(3) * 100
w_trans = np.diag(w_trans)

w_cent_mom = np.ones(6) * 1e-3
w_cent_mom = np.diag(w_cent_mom)


def create_dynamics(cont_states):
    ode = dynamics.KinodynamicsFwdDynamics(
        space, rmodel, gravity, cont_states, feet_ids, force_size
    )
    dyn_model = dynamics.IntegratorEuler(ode, dt)
    return dyn_model


def createStage(cont_states, cont_pos):
    rcost = aligator.CostStack(space, nu)

    cent_mom = aligator.CentroidalMomentumDerivativeResidual(
        space.ndx, rmodel, gravity, cont_states, feet_ids, force_size
    )

    rcost.addCost(aligator.QuadraticStateCost(space, nu, x0, w_x))
    rcost.addCost(aligator.QuadraticControlCost(space, u0, w_u))
    rcost.addCost(aligator.QuadraticResidualCost(space, cent_mom, w_cent_mom))
    for i in range(len(cont_pos)):
        frame_res = aligator.FrameTranslationResidual(
            space.ndx, nu, rmodel, cont_pos[i], feet_ids[i]
        )
        rcost.addCost(aligator.QuadraticResidualCost(space, frame_res, w_trans))

    stm = aligator.StageModel(rcost, create_dynamics(cont_states))
    for i in range(len(cont_states)):
        if cont_states[i]:
            cone_cstr = aligator.FrictionConeResidual(space.ndx, nu, i, mu, 1e-5)
            stm.addConstraint(cone_cstr, constraints.NegativeOrthant())
            frame_res = aligator.FrameTranslationResidual(
                space.ndx, nu, rmodel, cont_pos[i], feet_ids[i]
            )
            stm.addConstraint(frame_res, constraints.EqualityConstraintSet())
    return stm


# Define contact points throughout horizon

n_qs = 5  # Full contact support
n_ds = 40  # Two-contact support

steps = 2
contact_poses = []
contact_states = []
now_trans = [
    FL_pose.translation.copy(),
    FR_pose.translation.copy(),
    HL_pose.translation.copy(),
    HR_pose.translation.copy(),
]
swing_apex = 0.05
x_forward = 0.2


def ztraj(swing_apex, t_ss, ts):
    return swing_apex * np.sin(ts / float(t_ss) * np.pi)


def xtraj(x_forward, t_ss, ts):
    return x_forward * ts / float(t_ss)


for i in range(steps):
    for j in range(n_qs):
        contact_states.append([True, True, True, True])
        contact_poses.append(copy.deepcopy(now_trans))
    for j in range(n_ds):
        contact_states.append([False, True, True, False])
        new_trans = copy.deepcopy(now_trans)
        new_trans[0][0] = xtraj(x_forward, n_ds, j) + now_trans[0][0]
        new_trans[0][2] = ztraj(swing_apex, n_ds, j) + now_trans[0][2]
        new_trans[3][0] = xtraj(x_forward, n_ds, j) + now_trans[3][0]
        new_trans[3][2] = ztraj(swing_apex, n_ds, j) + now_trans[3][2]
        contact_poses.append(copy.deepcopy(new_trans))
        if j == n_ds - 1:
            now_trans = copy.deepcopy(new_trans)
    for j in range(n_qs):
        contact_states.append([True, True, True, True])
        contact_poses.append(copy.deepcopy(now_trans))
    for j in range(n_ds):
        contact_states.append([True, False, False, True])
        new_trans = copy.deepcopy(now_trans)
        new_trans[1][0] = xtraj(x_forward, n_ds, j) + now_trans[1][0]
        new_trans[1][2] = ztraj(swing_apex, n_ds, j) + now_trans[1][2]
        new_trans[2][0] = xtraj(x_forward, n_ds, j) + now_trans[2][0]
        new_trans[2][2] = ztraj(swing_apex, n_ds, j) + now_trans[2][2]
        contact_poses.append(copy.deepcopy(new_trans))
        if j == n_ds - 1:
            now_trans = copy.deepcopy(new_trans)

nsteps = len(contact_states)
tf = nsteps * dt  # in seconds
times = np.linspace(0, tf, nsteps + 1)

w_xterm = w_x.copy()
term_cost = aligator.CostStack(space, nu)
term_cost.addCost(aligator.QuadraticStateCost(space, nu, x0, weights=10 * w_x))
com_pose = aligator.CenterOfMassTranslationResidual(space.ndx, nu, rmodel, com0)

stages = [createStage(contact_states[i], contact_poses[i]) for i in range(nsteps)]

cent_mom_cst_ter = aligator.CentroidalMomentumDerivativeResidual(
    space.ndx, rmodel, gravity, contact_states[-1], feet_ids, force_size
)
cent_mom_cst_init = aligator.CentroidalMomentumDerivativeResidual(
    space.ndx, rmodel, gravity, contact_states[0], feet_ids, force_size
)
stages[0].addConstraint(cent_mom_cst_init, constraints.EqualityConstraintSet())
stages[-1].addConstraint(cent_mom_cst_ter, constraints.EqualityConstraintSet())

problem = aligator.TrajOptProblem(x0, stages, term_cost)

TOL = 1e-5
mu_init = 1e-8
rho_init = 0.0
max_iters = 100
verbose = aligator.VerboseLevel.VERBOSE
solver = aligator.SolverProxDDP(TOL, mu_init, rho_init, verbose=verbose)
# solver = aligator.SolverFDDP(TOL, verbose=verbose)
solver.rollout_type = aligator.ROLLOUT_LINEAR
print("LDLT algo choice:", solver.ldlt_algo_choice)
solver.max_iters = max_iters
solver.sa_strategy = aligator.SA_FILTER  # FILTER or LINESEARCH
solver.force_initial_condition = True
solver.filter.beta = 1e-5
solver.setup(problem)

xs_init = [x0] * (nsteps + 1)
u_ref = np.concatenate((f_ref, f_ref, f_ref, f_ref, np.zeros(rmodel.nv - 6)))
us_init = [u_ref for _ in range(nsteps)]

solver.run(
    problem,
    xs_init,
    us_init,
)
workspace = solver.workspace
results = solver.results
print(results)

if args.display:
    vizer.setCameraPosition([1.2, 0.0, 1.2])
    vizer.setCameraTarget([0.0, 0.0, 1.0])
    qs = [x[:nq] for x in results.xs.tolist()]

    for _ in range(5):
        vizer.play(qs, dt)
        time.sleep(0.5)

""" Plots results """
com_traj = [[], [], []]
linear_momentum = [[], [], []]
angular_momentum = [[], [], []]
base_pose = [[], [], []]
forces_z = [[] for _ in range(nk)]
FL_poses = [[], [], []]
HL_poses = [[], [], []]
FR_poses = [[], [], []]
HR_poses = [[], [], []]
FL_desired = [[], [], []]
HL_desired = [[], [], []]
FR_desired = [[], [], []]
HR_desired = [[], [], []]
ttlin = np.linspace(0, nsteps * dt, nsteps)
for i in range(nsteps):
    pin.forwardKinematics(rmodel, rdata, results.xs[i][: rmodel.nq])
    pin.updateFramePlacements(rmodel, rdata)
    com = pin.centerOfMass(rmodel, rdata, results.xs[i][: rmodel.nq])
    pin.computeCentroidalMomentum(
        rmodel, rdata, results.xs[i][: rmodel.nq], results.xs[i][rmodel.nq :]
    )
    for j in range(3):
        com_traj[j].append(com[j])
        base_pose[j].append(results.xs[i][j])
        FL_poses[j].append(rdata.oMf[FL_id].translation[j])
        FR_poses[j].append(rdata.oMf[FR_id].translation[j])
        HL_poses[j].append(rdata.oMf[HL_id].translation[j])
        HR_poses[j].append(rdata.oMf[HR_id].translation[j])
        FL_desired[j].append(contact_poses[i][0][j])
        FR_desired[j].append(contact_poses[i][1][j])
        HL_desired[j].append(contact_poses[i][2][j])
        HR_desired[j].append(contact_poses[i][3][j])
        linear_momentum[j].append(rdata.hg.linear[j])
        angular_momentum[j].append(rdata.hg.angular[j])
    for j in range(nk):
        if contact_states[i][j]:
            forces_z[j].append(results.us[i][j * 3 + 2])
        else:
            forces_z[j].append(0)

fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(3.5, 2.5), layout="constrained")
axs[0].plot(ttlin, base_pose[0])
axs[0].set_title("Base X")
axs[0].grid(True)
axs[1].plot(ttlin, base_pose[1])
axs[1].grid(True)
axs[1].set_title("Base Y")
axs[2].plot(ttlin, base_pose[2])
axs[2].grid(True)
axs[2].set_title("Base Z")

fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(3.5, 2.5), layout="constrained")
axs[0].plot(ttlin, com_traj[0])
axs[0].set_title("CoM X")
axs[0].grid(True)
axs[1].plot(ttlin, com_traj[1])
axs[1].grid(True)
axs[1].set_title("CoM Y")
axs[2].plot(ttlin, com_traj[2])
axs[2].grid(True)
axs[2].set_title("CoM Z")

fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(3.5, 2.5), layout="constrained")
axs[0].plot(ttlin, linear_momentum[0])
axs[0].set_title("h X")
axs[0].grid(True)
axs[1].plot(ttlin, linear_momentum[1])
axs[1].grid(True)
axs[1].set_title("h Y")
axs[2].plot(ttlin, linear_momentum[2])
axs[2].grid(True)
axs[2].set_title("h Z")

fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(3.5, 2.5), layout="constrained")
axs[0].plot(ttlin, angular_momentum[0])
axs[0].set_title("L X")
axs[0].grid(True)
axs[1].plot(ttlin, angular_momentum[1])
axs[1].grid(True)
axs[1].set_title("L Y")
axs[2].plot(ttlin, angular_momentum[2])
axs[2].grid(True)
axs[2].set_title("L Z")

fig, axs = plt.subplots(ncols=1, nrows=4, figsize=(3.5, 2.5), layout="constrained")
axs[0].plot(ttlin, forces_z[0])
axs[0].set_title("f_z LF")
axs[0].grid(True)
axs[1].plot(ttlin, forces_z[1])
axs[1].grid(True)
axs[1].set_title("f_z RF")
axs[2].plot(ttlin, forces_z[2])
axs[2].grid(True)
axs[2].set_title("f_z LB")
axs[3].plot(ttlin, forces_z[3])
axs[3].grid(True)
axs[3].set_title("f_z RB")

fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(3.5, 2.5), layout="constrained")
axs[0].plot(ttlin, FL_poses[0])
axs[0].plot(ttlin, FL_desired[0], "r")
axs[0].set_title("FL x")
axs[0].grid(True)
axs[1].plot(ttlin, FL_poses[1])
axs[1].plot(ttlin, FL_desired[1], "r")
axs[1].grid(True)
axs[1].set_title("FL y")
axs[2].plot(ttlin, FL_poses[2])
axs[2].plot(ttlin, FL_desired[2], "r")
axs[2].grid(True)
axs[2].set_title("FL z")

fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(3.5, 2.5), layout="constrained")
axs[0].plot(ttlin, FR_poses[0])
axs[0].plot(ttlin, FR_desired[0], "r")
axs[0].set_title("FR x")
axs[0].grid(True)
axs[1].plot(ttlin, FR_poses[1])
axs[1].plot(ttlin, FR_desired[1], "r")
axs[1].grid(True)
axs[1].set_title("FR y")
axs[2].plot(ttlin, FR_poses[2])
axs[2].plot(ttlin, FR_desired[2], "r")
axs[2].grid(True)
axs[2].set_title("FR z")

fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(3.5, 2.5), layout="constrained")
axs[0].plot(ttlin, HL_poses[0])
axs[0].plot(ttlin, HL_desired[0], "r")
axs[0].set_title("HL x")
axs[0].grid(True)
axs[1].plot(ttlin, HL_poses[1])
axs[1].plot(ttlin, HL_desired[1], "r")
axs[1].grid(True)
axs[1].set_title("HL y")
axs[2].plot(ttlin, HL_poses[2])
axs[2].plot(ttlin, HL_desired[2], "r")
axs[2].grid(True)
axs[2].set_title("HL z")

fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(3.5, 2.5), layout="constrained")
axs[0].plot(ttlin, HR_poses[0])
axs[0].plot(ttlin, HR_desired[0], "r")
axs[0].set_title("HR x")
axs[0].grid(True)
axs[1].plot(ttlin, HR_poses[1])
axs[1].plot(ttlin, HR_desired[1], "r")
axs[1].grid(True)
axs[1].set_title("HR y")
axs[2].plot(ttlin, HR_poses[2])
axs[2].plot(ttlin, HR_desired[2], "r")
axs[2].grid(True)
axs[2].set_title("HR z")


plt.show()
