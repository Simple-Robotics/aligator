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
# Define centroidal parameters
nk = 4  # Max number of contacts
nxc = 9
force_size = 3
nx = 9 + force_size * nk  # State size: [c, h, L, u]
mass = 10.5
gravity = np.array([0, 0, -9.81])
mu = 0.8  # Friction coefficient
nu = force_size * nk

foot_length = 0.1
foot_width = 0.1
space = manifolds.VectorSpace(nx)
wrapped_space = manifolds.VectorSpace(9)
x0 = space.neutral()

# Define initial and final desired CoM
com_initial = np.array([0.1, 0.05, 0.15])

x0[:3] = com_initial
for i in range(nk):
    x0[9 + force_size * i + 2] = -gravity[2] * mass / 4.0

# Define gait and time parameters
T_ds = 10  # Double support time
T_ss = 40  # Singel support time

# Define contact points throughout horizon
x_forward = 0.05
cp1 = [
    [True, True, True, True],
    [
        np.array([0.2, 0.1, 0.0]),
        np.array([0.2, 0.0, 0.0]),
        np.array([0.0, 0.1, 0.0]),
        np.array([0.0, 0.0, 0]),
    ],
]
cp2 = [
    [False, True, True, False],
    [
        np.array([0.2, 0.1, 0.0]),
        np.array([0.2, 0.0, 0.0]),
        np.array([0.0, 0.1, 0.0]),
        np.array([0.0, 0.0, 0]),
    ],
]
cp3 = [
    [True, True, True, True],
    [
        np.array([0.25, 0.1, 0.0]),
        np.array([0.2, 0.0, 0.0]),
        np.array([0.0, 0.1, 0.0]),
        np.array([0.05, 0.0, 0]),
    ],
]
cp4 = [
    [True, False, False, True],
    [
        np.array([0.25, 0.1, 0.0]),
        np.array([0.2, 0.0, 0.0]),
        np.array([0.0, 0.1, 0.0]),
        np.array([0.05, 0.0, 0]),
    ],
]
cp5 = [
    [True, True, True, True],
    [
        np.array([0.25, 0.1, 0.0]),
        np.array([0.25, 0.0, 0.0]),
        np.array([0.05, 0.1, 0.0]),
        np.array([0.05, 0.0, 0]),
    ],
]
cp6 = [
    [False, True, True, False],
    [
        np.array([0.25, 0.1, 0.0]),
        np.array([0.25, 0.0, 0.0]),
        np.array([0.05, 0.1, 0.0]),
        np.array([0.05, 0.0, 0]),
    ],
]
cp7 = [
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
    + [cp7] * (T_ds + 50)
)
com_final = cp7[1][0]
com_final += cp7[1][1]
com_final += cp7[1][2]
com_final += cp7[1][3]
com_final /= 4
com_final[2] = com_initial[2]

T = len(contact_points)  # Size of the problem
dt = 0.01  # timestep

# Create dynamics and costs

w_angular_acc = 0.1 * np.eye(3)
w_linear_mom = 10 * np.eye(3)
w_linear_acc = 0.1 * np.eye(3)
w_control = np.eye(nu) * 1e-4
w_ter_com = np.eye(3) * 1e7

w_centroidal_state = np.zeros(9)
w_forces = np.ones(force_size) * 1e-2
w_state = np.diag(
    np.concatenate((w_centroidal_state, w_forces, w_forces, w_forces, w_forces))
)


def create_dynamics(nspace, contact_map):
    ode = dynamics.ContinuousCentroidalFwdDynamics(
        nspace, mass, gravity, contact_map, force_size
    )
    dyn_model = dynamics.IntegratorEuler(ode, dt)
    return dyn_model


def createStage(cp, cp_previous):
    contact_map = aligator.ContactMap(cp[0], cp[1])
    rcost = aligator.CostStack(space, nu)

    linear_acc = aligator.CentroidalAccelerationResidual(
        nxc, nu, mass, gravity, contact_map, force_size
    )
    linear_mom = aligator.LinearMomentumResidual(nxc, nu, np.zeros(3))
    angular_acc = aligator.AngularAccelerationResidual(
        nxc, nu, mass, gravity, contact_map, force_size
    )
    wrapped_linear_acc = aligator.CentroidalWrapperResidual(linear_acc)
    wrapped_angular_acc = aligator.CentroidalWrapperResidual(angular_acc)
    wrapped_linear_mom = aligator.CentroidalWrapperResidual(linear_mom)

    w_state_cstr = w_state.copy()
    for i in range(len(cp[0])):
        if cp[0][i] and not (cp_previous[0][i]):
            w_state_cstr[9 + i * force_size + 2] *= 100
        elif not (cp[0][i]) and cp_previous[0][i]:
            w_state_cstr[9 + i * force_size + 2] *= 100

    rcost.addCost(aligator.QuadraticStateCost(space, nu, np.zeros(nx), w_state_cstr))

    rcost.addCost(aligator.QuadraticControlCost(space, np.zeros(nu), w_control))
    rcost.addCost(
        aligator.QuadraticResidualCost(space, wrapped_linear_mom, w_linear_mom)
    )
    rcost.addCost(
        aligator.QuadraticResidualCost(space, wrapped_angular_acc, w_angular_acc)
    )
    rcost.addCost(
        aligator.QuadraticResidualCost(space, wrapped_linear_acc, w_linear_acc)
    )
    stm = aligator.StageModel(rcost, create_dynamics(space, contact_map))
    for i in range(len(cp[0])):
        if cp[0][i]:
            cone_cstr = aligator.FrictionConeResidual(nxc, nu, i, mu, 1e-3)
            if force_size == 6:
                cone_cstr = aligator.WrenchConeResidual(
                    nxc, nu, i, mu, foot_length, foot_width
                )
            wrapped_cstr = aligator.CentroidalWrapperResidual(cone_cstr)
            stm.addConstraint(wrapped_cstr, constraints.NegativeOrthant())

    return stm


term_cost = aligator.CostStack(space, nu)
ter_com = aligator.CentroidalWrapperResidual(
    aligator.CentroidalCoMResidual(nxc, nu, com_final)
)
# term_cost.addCost(aligator.QuadraticResidualCost(space,ter_com, w_ter_com))

# Initial and final acceleration (linear + angular) must be null
stages = [createStage(contact_points[0], contact_points[0])]
for i in range(1, T):
    stages.append(createStage(contact_points[i], contact_points[i - 1]))

contact_map_init = aligator.ContactMap(contact_points[0][0], contact_points[0][1])
contact_map_ter = aligator.ContactMap(contact_points[-1][0], contact_points[-1][1])

init_linear_acc_cstr = aligator.CentroidalWrapperResidual(
    aligator.CentroidalAccelerationResidual(
        nxc, nu, mass, gravity, contact_map_init, force_size
    )
)
ter_linear_acc_cstr = aligator.CentroidalWrapperResidual(
    aligator.CentroidalAccelerationResidual(
        nxc, nu, mass, gravity, contact_map_ter, force_size
    )
)
angular_acc_cstr = aligator.CentroidalWrapperResidual(
    aligator.AngularAccelerationResidual(
        nx, nu, mass, gravity, contact_map_ter, force_size
    )
)

init_linear_mom = aligator.CentroidalWrapperResidual(
    aligator.LinearMomentumResidual(nxc, nu, np.array([0, 0, 0]))
)
ter_angular_mom = aligator.CentroidalWrapperResidual(
    aligator.AngularMomentumResidual(nxc, nu, np.array([0, 0, 0]))
)

init_force_derivative = aligator.ControlErrorResidual(nx, nu)
init_state = aligator.StateErrorResidual(space, nu, x0)
stages[0].addConstraint(init_state, constraints.EqualityConstraintSet())
stages[0].addConstraint(init_force_derivative, constraints.EqualityConstraintSet())
stages[0].addConstraint(init_linear_acc_cstr, constraints.EqualityConstraintSet())
stages[0].addConstraint(init_linear_mom, constraints.EqualityConstraintSet())
# stages[0].addConstraint(angular_acc_cstr, constraints.EqualityConstraintSet())

stages[-1].addConstraint(ter_linear_acc_cstr, constraints.EqualityConstraintSet())
stages[-1].addConstraint(init_linear_mom, constraints.EqualityConstraintSet())
stages[-1].addConstraint(ter_angular_mom, constraints.EqualityConstraintSet())
# stages[-1].addConstraint(angular_acc_cstr, constraints.EqualityConstraintSet())
problem = aligator.TrajOptProblem(x0, stages, term_cost)

# Final CoM placement constraints
com_cstr = aligator.CentroidalWrapperResidual(
    aligator.CentroidalCoMResidual(nxc, nu, com_final)
)

term_constraint_com = aligator.StageConstraint(
    com_cstr, constraints.EqualityConstraintSet()
)
problem.addTerminalConstraint(term_constraint_com)

# Solver initialization
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
solver.setup(problem)
solver.filter.beta = 1e-5

us_init = []
xs_init = []
for el in contact_points:
    us_init.append(np.zeros(nk * 3))
    xi = np.zeros(9 + nk * 3)
    xi[:9] = x0[:9].copy()
    active_contact = 0
    for i in range(nk):
        if el[0][i]:
            active_contact += 1
    weight_reg = -mass * gravity[2] / active_contact
    for i in range(nk):
        if el[0][i]:
            xi[9 + 3 * i + 2] = weight_reg
    xs_init.append(xi)

xs_init.append(x0)

solver.run(
    problem,
    xs_init,
    us_init,
)

workspace = solver.workspace
results = solver.results
print(results)

# Compute linear and angular acceleration
linear_acceleration = [[], [], []]
angular_acceleration = [[], [], []]
for i in range(T):
    linacc = gravity * mass
    angacc = np.zeros(3)
    for j in range(nk):
        fj = np.zeros(force_size)
        if contact_points[i][0][j]:
            fj = results.xs[i][9 + j * force_size : 9 + (j + 1) * force_size]
        ci = results.xs[i][0:3]
        linacc += fj[:3]
        angacc += np.cross(contact_points[i][1][j] - ci, fj[:3])
        if force_size == 6:
            angacc += fj[3:]
    for z in range(3):
        linear_acceleration[z].append(linacc[z])
        angular_acceleration[z].append(angacc[z])


# Plots results
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
            forces_z[j].append(results.xs[i][9 + j * force_size + 2])
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
