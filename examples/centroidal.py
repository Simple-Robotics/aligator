import numpy as np
import aligator
import matplotlib.pyplot as plt

from aligator import (
    manifolds,
    dynamics,
    constraints,
)
from utils import ArgsBase


class Args(ArgsBase):
    tcp: str = None


args = Args().parse_args()
""" Define centroidal parameters """
nx = 9  # State size: [c, h, L]
nk = 4  # Number of contacts
force_size = 3
nu = nk * force_size
mass = 10.5
gravity = np.array([0, 0, -9.81])
mu = 0.8  # Friction coefficient

foot_length = 0.1
foot_width = 0.1
space = manifolds.VectorSpace(nx)
x0 = space.neutral()
u0 = np.zeros(nu)

""" Define initial and final desired CoM """
com_initial = np.array([0.1, 0.05, 0.15])
com_final = np.array([0.175, 0.05, 0.15])

x0[:3] = com_initial

""" Define gait and time parameters"""
T_ds = 10  # Double support time
T_ss = 40  # Singel support time
# Contacts state: [LF, RF, LB, RB]

""" Define contact points throughout horizon"""
cp1 = [
    ["LF", "RF", "LB", "RB"],
    [True, True, True, True],
    [
        np.array([0.2, 0.1, 0.0]),
        np.array([0.2, 0.0, 0.0]),
        np.array([0.0, 0.1, 0.0]),
        np.array([0.0, 0.0, 0]),
    ],
]
cp2 = [
    ["LF", "RF", "LB", "RB"],
    [False, True, True, False],
    [
        np.array([0.2, 0.1, 0.0]),
        np.array([0.2, 0.0, 0.0]),
        np.array([0.0, 0.1, 0.0]),
        np.array([0.0, 0.0, 0]),
    ],
]
cp3 = [
    ["LF", "RF", "LB", "RB"],
    [True, True, True, True],
    [
        np.array([0.25, 0.1, 0.0]),
        np.array([0.2, 0.0, 0.0]),
        np.array([0.0, 0.1, 0.0]),
        np.array([0.05, 0.0, 0]),
    ],
]
cp4 = [
    ["LF", "RF", "LB", "RB"],
    [True, False, False, True],
    [
        np.array([0.25, 0.1, 0.0]),
        np.array([0.2, 0.0, 0.0]),
        np.array([0.0, 0.1, 0.0]),
        np.array([0.05, 0.0, 0]),
    ],
]
cp5 = [
    ["LF", "RF", "LB", "RB"],
    [True, True, True, True],
    [
        np.array([0.25, 0.1, 0.0]),
        np.array([0.25, 0.0, 0.0]),
        np.array([0.05, 0.1, 0.0]),
        np.array([0.05, 0.0, 0]),
    ],
]
cp6 = [
    ["LF", "RF", "LB", "RB"],
    [False, True, True, False],
    [
        np.array([0.25, 0.1, 0.0]),
        np.array([0.25, 0.0, 0.0]),
        np.array([0.05, 0.1, 0.0]),
        np.array([0.05, 0.0, 0]),
    ],
]
cp7 = [
    ["LF", "RF", "LB", "RB"],
    [True, True, True, True],
    [
        np.array([0.3, 0.1, 0.0]),
        np.array([0.25, 0.0, 0.0]),
        np.array([0.05, 0.1, 0.0]),
        np.array([0.1, 0.0, 0]),
    ],
]
contact_points = (
    [cp1] * T_ds
    + [cp2] * T_ss
    + [cp3] * T_ds
    + [cp4] * T_ss
    + [cp5] * T_ds
    + [cp6] * T_ss
    + [cp7] * T_ds
)


T = len(contact_points)  # Size of the problem
dt = 0.01  # timestep

""" Create dynamics and costs """

w_angular_acc = 0.1 * np.eye(3)
w_linear_mom = 10 * np.eye(3)
w_linear_acc = 100 * np.eye(3)
w_control = np.eye(nu) * 1e-1

# Regularize linear momentum only
state_w = np.diag(np.array([0, 0, 0, 10, 10, 10, 0, 0, 0]))


def create_dynamics(contact_map):
    ode = dynamics.CentroidalFwdDynamics(space, mass, gravity, contact_map, force_size)
    dyn_model = dynamics.IntegratorEuler(ode, dt)
    return dyn_model


def createStage(cp):
    contact_map = aligator.ContactMap(cp[0], cp[1], cp[2])
    rcost = aligator.CostStack(space, nu)

    linear_acc = aligator.CentroidalAccelerationResidual(
        nx, nu, mass, gravity, contact_map, force_size
    )
    angular_acc = aligator.AngularAccelerationResidual(
        nx, nu, mass, gravity, contact_map, force_size
    )
    linear_mom = aligator.LinearMomentumResidual(nx, nu, np.zeros(3))

    rcost.addCost(aligator.QuadraticControlCost(space, u0, w_control))
    rcost.addCost(aligator.QuadraticResidualCost(space, linear_mom, w_linear_mom))
    rcost.addCost(aligator.QuadraticResidualCost(space, angular_acc, w_angular_acc))
    rcost.addCost(aligator.QuadraticResidualCost(space, linear_acc, w_linear_acc))
    stm = aligator.StageModel(rcost, create_dynamics(contact_map))
    for i in range(len(cp[0])):
        if cp[0][i]:
            cone_cstr = aligator.FrictionConeResidual(space.ndx, nu, i, mu, 0)
            if force_size == 6:
                cone_cstr = aligator.CentroidalWrenchConeResidual(
                    space.ndx, nu, i, mu, foot_length, foot_width
                )
            stm.addConstraint(cone_cstr, constraints.NegativeOrthant())

    return stm


term_cost = aligator.CostStack(space, nu)

""" Initial and final acceleration (linear + angular) must be null"""
stages = []
for i in range(T):
    stages.append(createStage(contact_points[i]))

contact_map_init = aligator.ContactMap(
    contact_points[0][0], contact_points[0][1], contact_points[0][2]
)
contact_map_ter = aligator.ContactMap(
    contact_points[-1][0], contact_points[-1][1], contact_points[-1][2]
)
init_linear_acc_cstr = aligator.CentroidalAccelerationResidual(
    nx, nu, mass, gravity, contact_map_init, force_size
)
ter_linear_acc_cstr = aligator.CentroidalAccelerationResidual(
    nx, nu, mass, gravity, contact_map_ter, force_size
)
ter_angular_acc_cstr = aligator.AngularAccelerationResidual(
    nx, nu, mass, gravity, contact_map_ter, force_size
)
init_angular_acc_cstr = aligator.AngularAccelerationResidual(
    nx, nu, mass, gravity, contact_map_init, force_size
)
init_linear_mom = aligator.LinearMomentumResidual(nx, nu, np.array([0, 0, 0]))
ter_angular_mom = aligator.AngularMomentumResidual(nx, nu, np.array([0, 0, 0]))
stages[0].addConstraint(init_linear_acc_cstr, constraints.EqualityConstraintSet())
# stages[0].addConstraint(init_angular_acc_cstr, constraints.EqualityConstraintSet())
stages[0].addConstraint(init_linear_mom, constraints.EqualityConstraintSet())

# stages[-1].addConstraint(init_linear_mom, constraints.EqualityConstraintSet())
# stages[-1].addConstraint(ter_linear_acc_cstr, constraints.EqualityConstraintSet())
stages[-1].addConstraint(ter_angular_mom, constraints.EqualityConstraintSet())
stages[-1].addConstraint(ter_angular_acc_cstr, constraints.EqualityConstraintSet())
problem = aligator.TrajOptProblem(x0, stages, term_cost)

""" Final CoM placement constraints """
com_cstr = aligator.CentroidalCoMResidual(space.ndx, nu, com_final)
linear_mom_cstr = aligator.LinearMomentumResidual(nx, nu, np.array([0, 0, 0]))
ang_mom_cstr = aligator.AngularMomentumResidual(nx, nu, np.array([0, 0, 0]))

term_constraint_com = aligator.StageConstraint(
    com_cstr, constraints.EqualityConstraintSet()
)
term_constraint_linmom = aligator.StageConstraint(
    linear_mom_cstr, constraints.EqualityConstraintSet()
)
term_constraint_angmom = aligator.StageConstraint(
    ang_mom_cstr, constraints.EqualityConstraintSet()
)
problem.addTerminalConstraint(term_constraint_com)
# problem.addTerminalConstraint(term_constraint_linmom)
# problem.addTerminalConstraint(term_constraint_angmom)

""" Solver initialization """
TOL = 1e-5
mu_init = 1e-8
rho_init = 0.0
max_iters = 100
verbose = aligator.VerboseLevel.VERBOSE
solver = aligator.SolverProxDDP(TOL, mu_init, rho_init, verbose=verbose)
solver.rollout_type = aligator.ROLLOUT_LINEAR
solver.max_iters = max_iters
solver.sa_strategy = aligator.SA_FILTER  # FILTER or LINESEARCH
solver.setup(problem)
solver.filter.beta = 1e-5

us_init = [np.zeros(nk * force_size)] * T
xs_init = [x0] * (T + 1)

solver.run(
    problem,
    xs_init,
    us_init,
)

workspace = solver.workspace
results = solver.results
print(results)

""" Compute linear and angular acceleration """
linear_acceleration = [[], [], []]
angular_acceleration = [[], [], []]
for i in range(T):
    linacc = gravity * mass
    angacc = np.zeros(3)
    ncontact = len(contact_points[i][2])
    for j in range(ncontact):
        fj = results.us[i][j * force_size : (j + 1) * force_size]
        ci = results.xs[i][0:3]
        linacc += fj[:3]
        angacc += np.cross(contact_points[i][2][j] - ci, fj[:3])
        if force_size == 6:
            angacc += fj[3:]
    for z in range(3):
        linear_acceleration[z].append(linacc[z])
        angular_acceleration[z].append(angacc[z])


""" Plots results """
com_traj = [[], [], []]
linear_momentum = [[], [], []]
angular_momentum = [[], [], []]
forces_z = [[] for _ in range(nk)]
ttlin = np.linspace(0, T * dt, T)
for i in range(T):
    com_traj[0].append(results.xs[i][0])
    com_traj[1].append(results.xs[i][1])
    com_traj[2].append(results.xs[i][2])
    linear_momentum[0].append(results.xs[i][3])
    linear_momentum[1].append(results.xs[i][4])
    linear_momentum[2].append(results.xs[i][5])
    angular_momentum[0].append(results.xs[i][6])
    angular_momentum[1].append(results.xs[i][7])
    angular_momentum[2].append(results.xs[i][8])
    for j in range(nk):
        if contact_points[i][0][j]:
            forces_z[j].append(results.us[i][j * force_size + 2])
        else:
            forces_z[j].append(0)


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

fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(3.5, 2.5), layout="constrained")
axs[0].plot(ttlin, linear_acceleration[0])
axs[0].set_title("h_dot X")
axs[0].grid(True)
axs[1].plot(ttlin, linear_acceleration[1])
axs[1].grid(True)
axs[1].set_title("h_dot Y")
axs[2].plot(ttlin, linear_acceleration[2])
axs[2].grid(True)
axs[2].set_title("h_dot Z")

fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(3.5, 2.5), layout="constrained")
axs[0].plot(ttlin, angular_acceleration[0])
axs[0].set_title("L_dot X")
axs[0].grid(True)
axs[1].plot(ttlin, angular_acceleration[1])
axs[1].grid(True)
axs[1].set_title("L_dot Y")
axs[2].plot(ttlin, angular_acceleration[2])
axs[2].grid(True)
axs[2].set_title("L_dot Z")

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

plt.show()
