import aligator
import pinocchio as pin
import numpy as np
import coal

from aligator import manifolds, dynamics
from pinocchio.visualize import MeshcatVisualizer
from pinocchio.visualize.meshcat_visualizer import COLOR_PRESETS
from utils.solo import rmodel, rdata, robot, q0, create_ground_contact_model

COLOR_PRESETS["white"] = ([1, 1, 1], [1, 1, 1])


vizer = MeshcatVisualizer(
    rmodel,
    collision_model=robot.collision_model,
    visual_model=robot.visual_model,
    data=rdata,
)

pin.framesForwardKinematics(rmodel, rdata, q0)


nq = rmodel.nq
nv = rmodel.nv
nu = nv - 6
act_matrix = np.eye(nv, nu, -6)
space = manifolds.MultibodyPhaseSpace(rmodel)


def define_dynamics():
    constraint_models = create_ground_contact_model(rmodel)
    prox_settings = pin.ProximalSettings(1e-9, 1e-10, 10)
    ode = dynamics.MultibodyConstraintFwdDynamics(
        space, act_matrix, constraint_models, prox_settings
    )
    return ode


ode = define_dynamics()
timestep = 0.01
Tf = 4.0
nsteps = int(Tf / timestep)

dyn_model = dynamics.IntegratorSemiImplEuler(ode, timestep)

u0 = np.zeros(nu)
v0 = np.zeros(nv)
x0 = np.concatenate([q0, v0])


def create_target(i: int):
    x_target = x0.copy()
    x_target[:2] = -0.05, 0.07

    ti = timestep * i
    freq = 3.0
    z0 = x0[2] * 0.7
    amp = x0[2] * 0.4
    x_target[2] = z0 + amp * np.sin(freq * ti) ** 2

    return x_target


X_TARGETS = [create_target(i) for i in range(nsteps + 1)]


def update_target(sphere, x):
    sphere.placement.translation[:] = x[:3]


# Define cost functions
base_weight = 2.0
w_xreg = np.diag([1e-3] * nv + [1e-3] * nv)
w_xreg[range(3), range(3)] = base_weight

w_ureg = np.eye(nu) * 1e-3
ureg_cost = aligator.QuadraticControlCost(space, u0, w_ureg * timestep)

stages = []
for i in range(nsteps):
    x_cost = aligator.QuadraticStateCost(space, nu, X_TARGETS[i], w_xreg * timestep)
    rcost = aligator.CostStack(space, nu)
    rcost.addCost(x_cost)
    rcost.addCost(ureg_cost)
    stm = aligator.StageModel(rcost, dyn_model)
    stages.append(stm)

w_xterm = np.diag([1e-3] * nv + [1e-3] * nv)
w_xterm[range(3), range(3)] = base_weight
xreg_term = aligator.QuadraticStateCost(space, nu, X_TARGETS[nsteps], w_xterm)
term_cost = xreg_term


def main():
    vizer.initViewer(loadModel=True, open=True)
    vizer.display(q0)
    vizer.setBackgroundColor("white")

    # display target as a transparent sphere
    sphere = pin.GeometryObject("target", 0, pin.SE3.Identity(), coal.Sphere(0.01))
    sphere.meshColor[:] = 217, 101, 38, 120
    sphere.meshColor /= 255.0
    vizer.addGeometryObject(sphere)

    xs_i = [x0] * (nsteps + 1)
    us_i = [u0] * nsteps

    problem = aligator.TrajOptProblem(x0, stages, term_cost)

    mu_init = 1e-2
    solver = aligator.SolverProxDDP(1e-3, mu_init, verbose=aligator.VERBOSE)
    solver.reg_init = 1e-8
    solver.setup(problem)
    flag = solver.run(problem, xs_i, us_i)
    print(flag)

    rs = solver.results
    qs_opt = [x[:nq] for x in rs.xs]
    input("[display?]")

    def callback(i: int):
        update_target(sphere, X_TARGETS[i])

    NR = 3
    for _ in range(NR):
        vizer.play(qs_opt, timestep, callback=callback)
        input()


if __name__ == "__main__":
    main()
