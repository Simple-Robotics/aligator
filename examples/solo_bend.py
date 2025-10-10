import aligator
import pinocchio as pin
import numpy as np
import coal

from aligator import manifolds, dynamics
from utils import ArgsBase
from utils.solo import rmodel, rdata, robot, q0, create_ground_contact_model


class Args(ArgsBase):
    pass


args = Args().parse_args()

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
timestep = 0.02
Tf = 4.0
nsteps = int(Tf / timestep)

dyn_model = dynamics.IntegratorSemiImplEuler(ode, timestep)

v0 = np.zeros(nv)
x0 = np.concatenate([q0, v0])
u0, _ = aligator.underactuatedConstrainedInverseDynamics(
    rmodel,
    rdata,
    q0,
    v0,
    act_matrix,
    ode.constraint_models,
    [cm.createData() for cm in ode.constraint_models],
)


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
print(nv)
w_xreg = [0.1] * 3 + [0.01] * (nv - 3) + [1e-4] * nv
w_xreg = np.diag(w_xreg)
print(w_xreg)

w_ureg = np.eye(nu) * 1e-4
u_cost = aligator.QuadraticControlCost(space, u0, w_ureg * timestep)

stages = []
for i in range(nsteps):
    x_cost = aligator.QuadraticStateCost(space, nu, X_TARGETS[i], w_xreg * timestep)
    rcost = aligator.CostStack(space, nu)
    rcost.addCost(x_cost)
    rcost.addCost(u_cost)
    stm = aligator.StageModel(rcost, dyn_model)
    stages.append(stm)

w_xterm = 1.0 * np.eye(space.ndx)
xreg_term = aligator.QuadraticStateCost(space, nu, X_TARGETS[nsteps], w_xterm)
term_cost = xreg_term


def main():
    if args.display:
        from pinocchio.visualize import MeshcatVisualizer
        from pinocchio.visualize.meshcat_visualizer import COLOR_PRESETS

        COLOR_PRESETS["white"] = ([1, 1, 1], [1, 1, 1])
        sphere = pin.GeometryObject("target", 0, pin.SE3.Identity(), coal.Sphere(0.01))
        sphere.meshColor[:] = 217, 101, 38, 120
        sphere.meshColor /= 255.0
        robot.visual_model.addGeometryObject(sphere)
        vizer = MeshcatVisualizer(
            rmodel,
            collision_model=robot.collision_model,
            visual_model=robot.visual_model,
            data=rdata,
        )
        vizer.initViewer(loadModel=True, open=True)
        vizer.display(q0)
        vizer.setBackgroundColor("white")

    us_i = [u0] * nsteps
    xs_i = [x0] * (nsteps + 1)

    problem = aligator.TrajOptProblem(x0, stages, term_cost)

    mu_init = 1e-2
    tol = 1e-4
    solver = aligator.SolverProxDDP(tol, mu_init, verbose=aligator.VERBOSE)
    # solver.sa_strategy = aligator.SA_FILTER
    solver.setup(problem)
    flag = solver.run(problem, xs_i, us_i)
    print(flag)

    rs = solver.results
    qs_opt = [x[:nq] for x in rs.xs]

    if args.display:
        input("[display?]")

        def callback(i: int):
            update_target(sphere, X_TARGETS[i])

        num_repeat = 3
        for _ in range(num_repeat):
            vizer.play(qs_opt, timestep, callback=callback)
            input()


if __name__ == "__main__":
    main()
