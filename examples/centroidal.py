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
nu = nk * 3  # Control size (unilateral contacts)
mass = 10.5
gravity = np.array([0, 0, -9.81])
mu = 0.8  # Friction coefficient

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
gaits = (
    [[True, True, True, True]] * T_ds
    + [[False, True, True, False]] * T_ss
    + [[True, True, True, True]] * T_ds
    + [[True, False, False, True]] * T_ss
    + [[True, True, True, True]] * T_ds
    + [[False, True, True, False]] * T_ss
    + [[True, True, True, True]] * T_ds
)

T = len(gaits)  # Size of the problem
dt = 0.01  # timestep

""" Define contact points throughout horizon"""
cp1 = [
    np.array([0.2, 0.1, 0.0]),
    np.array([0.2, 0.0, 0.0]),
    np.array([0.0, 0.1, 0.0]),
    np.array([0.0, 0.0, 0]),
]
cp2 = [
    np.array([0.25, 0.1, 0.0]),
    np.array([0.2, 0.0, 0.0]),
    np.array([0.0, 0.1, 0.0]),
    np.array([0.05, 0.0, 0]),
]
cp3 = [
    np.array([0.25, 0.1, 0.0]),
    np.array([0.25, 0.0, 0.0]),
    np.array([0.05, 0.1, 0.0]),
    np.array([0.05, 0.0, 0]),
]
cp4 = [
    np.array([0.3, 0.1, 0.0]),
    np.array([0.25, 0.0, 0.0]),
    np.array([0.05, 0.1, 0.0]),
    np.array([0.1, 0.0, 0]),
]
contact_points = (
    [cp1] * (T_ds + T_ss) + [cp2] * (T_ds + T_ss) + [cp3] * (T_ds + T_ss) + [cp4] * T_ds
)

""" Create dynamics and costs """

w_control = np.eye(nu) * 1e-3
w_angular_acc = 0.1 * np.eye(3)
w_linear_mom = 10 * np.eye(3)
w_linear_acc = 10 * np.eye(3)


def create_dynamics(gait, cp):
    ode = dynamics.CentroidalFwdDynamics(space, nk, mass, gravity)
    ode.contact_points = cp
    ode.active_contacts = gait
    dyn_model = dynamics.IntegratorEuler(ode, dt)
    return dyn_model


def createStage(gait, cp):
    rcost = aligator.CostStack(space, nu)

    linear_acc = aligator.CentroidalAccelerationResidual(nx, nu, mass, gravity)
    linear_acc.active_contacts = gait
    angular_acc = aligator.AngularAccelerationResidual(nx, nu, mass, gravity)
    angular_acc.contact_points = cp
    angular_acc.active_contacts = gait
    linear_mom = aligator.LinearMomentumResidual(nx, nu, np.array([0, 0, 0]))

    rcost.addCost(aligator.QuadraticControlCost(space, u0, w_control))
    rcost.addCost(aligator.QuadraticResidualCost(space, angular_acc, w_angular_acc))
    rcost.addCost(aligator.QuadraticResidualCost(space, linear_mom, w_linear_mom))
    rcost.addCost(aligator.QuadraticResidualCost(space, linear_acc, w_linear_acc))
    stm = aligator.StageModel(rcost, create_dynamics(gait, cp))
    for i in range(len(gait)):
        if gait[i]:
            cone_cstr = aligator.FrictionConeResidual(space.ndx, nu, i, mu)
            stm.addConstraint(cone_cstr, constraints.NegativeOrthant())

    return stm


term_cost = aligator.CostStack(space, nu)
ter_linear_mom = aligator.LinearMomentumResidual(nx, nu, np.array([0, 0, 0]))
ter_linear_acc = aligator.CentroidalAccelerationResidual(nx, nu, mass, gravity)
term_cost.addCost(
    aligator.QuadraticResidualCost(space, ter_linear_mom, 10 * w_linear_mom)
)
term_cost.addCost(
    aligator.QuadraticResidualCost(space, ter_linear_acc, 1000 * w_linear_acc)
)

stages = []
for i in range(T):
    stages.append(createStage(gaits[i], contact_points[i]))
linear_acc_cstr = aligator.CentroidalAccelerationResidual(nx, nu, mass, gravity)
angular_acc_cstr = aligator.AngularAccelerationResidual(nx, nu, mass, gravity)
angular_acc_cstr.contact_points = cp4
stages[-1].addConstraint(linear_acc_cstr, constraints.EqualityConstraintSet())
stages[-1].addConstraint(angular_acc_cstr, constraints.EqualityConstraintSet())
problem = aligator.TrajOptProblem(x0, stages, term_cost)

""" Final constraints """
com_cstr = aligator.CentroidalCoMResidual(space.ndx, nu, com_final)
linear_mom_cstr = aligator.LinearMomentumResidual(nx, nu, np.array([0, 0, 0]))

term_constraint_acc = aligator.StageConstraint(
    linear_acc_cstr, constraints.EqualityConstraintSet()
)
term_constraint_angacc = aligator.StageConstraint(
    angular_acc_cstr, constraints.EqualityConstraintSet()
)
term_constraint_com = aligator.StageConstraint(
    com_cstr, constraints.EqualityConstraintSet()
)
term_constraint_linmom = aligator.StageConstraint(
    linear_mom_cstr, constraints.EqualityConstraintSet()
)
problem.addTerminalConstraint(term_constraint_com)
# problem.addTerminalConstraint(term_constraint_linmom)
# problem.addTerminalConstraint(term_constraint_angacc)
# problem.addTerminalConstraint(term_constraint_acc)

""" Solver initialization """
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
solver.setup(problem)
solver.filter.beta = 1e-5

us_init = [u0] * T
xs_init = [x0] * (T + 1)

solver.run(
    problem,
    xs_init,
    us_init,
)

workspace = solver.workspace
results = solver.results
print(results)

""" Plots results """
com_traj = [[], [], []]
linear_momentum = [[], [], []]
angular_momentum = [[], [], []]
forces_z = [[] for _ in range(nk)]
ttlin = np.linspace(0, T * 0.01, T)
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
        forces_z[j].append(results.us[i][j * 3 + 2])

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

plt.show()
