import aligator
import pinocchio as pin
from utils.solo import robot, rmodel, rdata, q0
import time

import numpy as np
import matplotlib.pyplot as plt

from utils import ArgsBase
from aligator import manifolds, dynamics, constraints


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
nu = nv + nk * 3
space_multibody = manifolds.MultibodyPhaseSpace(rmodel)
space_centroidal = manifolds.VectorSpace(6)
space = manifolds.CartesianProduct(space_centroidal, space_multibody)
ndx = space.ndx

effort_limit = rmodel.effortLimit[6:]
print(f"Effort limit: {effort_limit}")

x0_m = np.concatenate((q0, np.zeros(nv)))
x0 = np.concatenate((np.zeros(6), q0, np.zeros(nv)))
u0 = np.zeros(nu)
com0 = pin.centerOfMass(rmodel, rdata, x0_m[:nq])
com1 = com0.copy()
com2 = com0.copy()
com1[2] -= 0.1
com2[2] += 0.05
dt = 20e-3  # 20 ms
gravity = np.array([0, 0, -9.81])
mu = 0.8  # Friction coefficient
mass = pin.computeTotalMass(rmodel)
f_ref = np.array([0, 0, -mass * gravity[2] / 4.0])


""" dx = np.random.rand(nv)
x1 = space_multibody.integrate(x0_m, dx)
q1 = x1[:nq]
v1 = np.random.rand(nv)
a1 = np.random.rand(nv)
hdot = pin.computeCentroidalMomentumTimeVariation(rmodel, rdata, q1, v1, a1)
dh_dq, dhdot_dq, dhdot_dv, dhdot_da = pin.computeCentroidalDynamicsDerivatives(rmodel, rdata, q1, v1, a1)

exit() """
FL_id = rmodel.getFrameId("FL_FOOT")
FR_id = rmodel.getFrameId("FR_FOOT")
HL_id = rmodel.getFrameId("HL_FOOT")
HR_id = rmodel.getFrameId("HR_FOOT")

feet_ids = [FL_id, FR_id, HL_id, HR_id]

FL_pose = rdata.oMf[FL_id].copy()
FR_pose = rdata.oMf[FR_id].copy()
HL_pose = rdata.oMf[HL_id].copy()
HR_pose = rdata.oMf[HR_id].copy()

w_x = np.ones(space.ndx) * 1e-2
w_x[6:12] = 0.0
w_x = np.diag(w_x)
w_u = np.eye(nu) * 1e-5

w_trans = np.ones(3) * 100
w_trans = np.diag(w_trans)

w_cent_mom = np.ones(6) * 1e-3
w_cent_mom = np.diag(w_cent_mom)

w_com = np.ones(3) * 0
w_com = np.diag(w_com)

""" Define contact points throughout horizon"""
transoffset = np.array([0, 0, 0])
n_qs = 50
n_ds = 50
contact_points = (
    [
        [
            [True, True, True, True],
            [
                FL_pose.translation.copy(),
                FR_pose.translation.copy(),
                HL_pose.translation.copy(),
                HR_pose.translation.copy(),
            ],
        ]
        for _ in range(n_qs)
    ]
    + [
        [
            [False, True, True, False],
            [
                FL_pose.translation.copy(),
                FR_pose.translation.copy(),
                HL_pose.translation.copy(),
                HR_pose.translation.copy(),
            ],
        ]
        for _ in range(n_ds)
    ]
    + [
        [
            [True, True, True, True],
            [
                FL_pose.translation.copy() + transoffset,
                FR_pose.translation.copy(),
                HL_pose.translation.copy(),
                HR_pose.translation.copy() + transoffset,
            ],
        ]
        for _ in range(n_qs)
    ]
    + [
        [
            [True, False, False, True],
            [
                FL_pose.translation.copy(),
                FR_pose.translation.copy(),
                HL_pose.translation.copy(),
                HR_pose.translation.copy(),
            ],
        ]
        for _ in range(n_ds)
    ]
    + [
        [
            [True, True, True, True],
            [
                FL_pose.translation.copy() + transoffset,
                FR_pose.translation.copy(),
                HL_pose.translation.copy(),
                HR_pose.translation.copy() + transoffset,
            ],
        ]
        for _ in range(n_qs)
    ]
)

nsteps = len(contact_points)
tf = nsteps * dt  # in seconds
times = np.linspace(0, tf, nsteps + 1)


def create_dynamics(contact_map):
    ode = dynamics.KinodynamicsFwdDynamics(
        space, rmodel, gravity, contact_map, feet_ids
    )
    dyn_model = dynamics.IntegratorEuler(ode, dt)
    return dyn_model


def createStage(cp):
    contact_map = aligator.ContactMap(cp[0], cp[1])
    u0 = np.zeros(nu)
    rcost = aligator.CostStack(space, nu)

    cent_mom = aligator.CentroidalMomentumDerivativeResidual(
        rmodel, gravity, contact_map
    )
    """com_pose = aligator.CenterOfMassTranslationResidual(
        space_multibody.ndx, nu, rmodel, com0
    )"""
    # wrapped_com = aligator.KinodynamicsWrapperResidual(com_pose, nq, nv, nk)

    rcost.addCost(aligator.QuadraticStateCost(space, nu, x0, w_x))
    rcost.addCost(aligator.QuadraticControlCost(space, u0, w_u))
    rcost.addCost(aligator.QuadraticResidualCost(space, cent_mom, w_cent_mom))
    # rcost.addCost(aligator.QuadraticResidualCost(space, wrapped_com, w_com))
    for i in range(len(cp[0])):
        frame_res = aligator.FrameTranslationResidual(
            space_multibody.ndx, nu, rmodel, cp[1][i], feet_ids[i]
        )
        wrapped_frame_res = aligator.KinodynamicsWrapperResidual(frame_res, nq, nv, nk)
        rcost.addCost(aligator.QuadraticResidualCost(space, wrapped_frame_res, w_trans))

    stm = aligator.StageModel(rcost, create_dynamics(contact_map))
    for i in range(len(cp[0])):
        if cp[0][i]:
            cone_cstr = aligator.FrictionConeResidual(space.ndx, nu, i, mu, 1e-5)
            stm.addConstraint(cone_cstr, constraints.NegativeOrthant())
            frame_res = aligator.FrameTranslationResidual(
                space_multibody.ndx, nu, rmodel, cp[1][i], feet_ids[i]
            )
            wrapped_cstr = aligator.KinodynamicsWrapperResidual(frame_res, nq, nv, nk)
            stm.addConstraint(wrapped_cstr, constraints.EqualityConstraintSet())
    return stm


swing_apex = 0.1
x_forward = 0.1


def ztraj(swing_apex, t_ss, ts):
    return swing_apex * np.sin(ts / float(t_ss) * np.pi)


def xtraj(x_forward, t_ss, ts):
    return x_forward * ts / float(t_ss)


ts = [0, 0, 0, 0]
ref_poses = [FL_pose, FR_pose, HL_pose, HR_pose]
now_trans = [
    FL_pose.translation.copy(),
    FR_pose.translation.copy(),
    HL_pose.translation.copy(),
    HR_pose.translation.copy(),
]
for r in range(len(contact_points)):
    for i in range(4):
        if contact_points[r][0][i]:
            contact_points[r][1][i] = now_trans[i]
            ts[i] = 0
        else:
            contact_points[r][1][i][0] = (
                xtraj(x_forward, n_ds, ts[i]) + ref_poses[i].translation[0]
            )
            contact_points[r][1][i][2] = (
                ztraj(swing_apex, n_ds, ts[i]) + ref_poses[i].translation[2]
            )
            now_trans[i] = contact_points[r][1][i]
            ts[i] += 1

w_xterm = w_x.copy()
term_cost = aligator.CostStack(space, nu)
term_cost.addCost(aligator.QuadraticStateCost(space, nu, x0, weights=10 * w_x))
com_pose = aligator.CenterOfMassTranslationResidual(
    space_multibody.ndx, nu, rmodel, com0
)
wrapped_com = aligator.KinodynamicsWrapperResidual(com_pose, nq, nv, nk)
# term_cost.addCost(aligator.QuadraticResidualCost(space, wrapped_com, w_com * 10))

stages = [createStage(contact_points[i]) for i in range(nsteps)]

contact_map = aligator.ContactMap(contact_points[-1][0], contact_points[-1][1])
cent_mom_cst = aligator.CentroidalMomentumDerivativeResidual(
    rmodel, gravity, contact_map
)
stages[-1].addConstraint(cent_mom_cst, constraints.EqualityConstraintSet())

problem = aligator.TrajOptProblem(x0, stages, term_cost)
term_constraint_cent = aligator.StageConstraint(
    cent_mom_cst, constraints.EqualityConstraintSet()
)
# problem.addTerminalConstraint(term_constraint_cent)

TOL = 1e-5
mu_init = 1e-8
rho_init = 0.0
max_iters = 100
verbose = aligator.VerboseLevel.VERBOSE
solver = aligator.SolverProxDDP(TOL, mu_init, rho_init, verbose=verbose)
# solver = aligator.SolverFDDP(TOL, verbose=verbose)
solver.rollout_type = aligator.ROLLOUT_LINEAR
print("LDLT algo choice:", solver.ldlt_algo_choice)
# solver = aligator.SolverFDDP(TOL, verbose=verbose)
solver.max_iters = max_iters
solver.sa_strategy = aligator.SA_FILTER  # FILTER or LINESEARCH
solver.force_initial_condition = True
solver.filter.beta = 1e-5
solver.setup(problem)

xs_init = [x0] * (nsteps + 1)
u_ref = np.concatenate((f_ref, f_ref, f_ref, f_ref, np.zeros(rmodel.nv)))
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
    qs = [x[6 : 6 + nq] for x in results.xs.tolist()]

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
    pin.forwardKinematics(rmodel, rdata, results.xs[i][6 : 6 + rmodel.nq])
    pin.updateFramePlacements(rmodel, rdata)
    com = pin.centerOfMass(rmodel, rdata, results.xs[i][6 : 6 + rmodel.nq])
    for j in range(3):
        com_traj[j].append(com[j])
        base_pose[j].append(results.xs[i][6 + j])
        FL_poses[j].append(rdata.oMf[FL_id].translation[j])
        FR_poses[j].append(rdata.oMf[FR_id].translation[j])
        HL_poses[j].append(rdata.oMf[HL_id].translation[j])
        HR_poses[j].append(rdata.oMf[HR_id].translation[j])
        FL_desired[j].append(contact_points[i][1][0][j])
        FR_desired[j].append(contact_points[i][1][1][j])
        HL_desired[j].append(contact_points[i][1][2][j])
        HR_desired[j].append(contact_points[i][1][3][j])
    linear_momentum[0].append(results.xs[i][0])
    linear_momentum[1].append(results.xs[i][1])
    linear_momentum[2].append(results.xs[i][2])
    angular_momentum[0].append(results.xs[i][3])
    angular_momentum[1].append(results.xs[i][4])
    angular_momentum[2].append(results.xs[i][5])
    for j in range(nk):
        if contact_points[i][0][j]:
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
