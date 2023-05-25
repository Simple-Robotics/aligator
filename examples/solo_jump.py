import proxddp
import pinocchio as pin
from solo_utils import robot, rmodel, rdata, q0, create_ground_contact_model

import numpy as np

from proxddp import manifolds, dynamics
from pinocchio.visualize import MeshcatVisualizer

pin.framesForwardKinematics(rmodel, rdata, q0)


nq = rmodel.nq
nv = rmodel.nv
nu = nv - 6
space = manifolds.MultibodyPhaseSpace(rmodel)
act_matrix = np.eye(nv, nu, -6)

constraint_models = create_ground_contact_model(rmodel)
prox_settings = pin.ProximalSettings(1e-9, 1e-10, 10)
ode1 = dynamics.MultibodyConstraintFwdDynamics(
    space, act_matrix, constraint_models, prox_settings
)
ode2 = dynamics.MultibodyConstraintFwdDynamics(space, act_matrix, [], prox_settings)


def test():
    x0 = space.neutral()
    u0 = np.random.randn(nu)
    d1 = ode1.createData()
    ode1.forward(x0, u0, d1)
    ode1.dForward(x0, u0, d1)
    d2 = ode2.createData()
    ode2.forward(x0, u0, d2)
    ode2.dForward(x0, u0, d2)


test()


dt = 40e-3  # 40 ms
tf = 1.0  # in seconds
nsteps = int(tf / dt)

switch_t0 = 0.4
switch_t1 = 0.8  # landing time

times = np.linspace(0, tf, nsteps + 1)
mask = (switch_t0 <= times) & (times < switch_t1)

x0_ref = np.concatenate((q0, np.zeros(nv)))
w_x = np.eye(space.ndx) * 1e-3
w_u = np.eye(nu) * 1e-4


stages = []
for k in range(nsteps):
    _vf = ode1
    if mask[k]:
        _vf = ode2

    xreg_cost = proxddp.QuadraticStateCost(space, nu, target=x0_ref, weights=w_x * dt)
    ureg_cost = proxddp.QuadraticControlCost(space, nu, weights=w_u * dt)
    cost = proxddp.CostStack(space, nu)
    cost.addCost(xreg_cost)
    cost.addCost(ureg_cost)

    dyn_model = dynamics.IntegratorRK2(_vf, timestep=dt)
    stm = proxddp.StageModel(cost, dyn_model)
    stages.append(stm)


term_cost = proxddp.QuadraticStateCost(space, nu, target=x0_ref, weights=w_x)

problem = proxddp.TrajOptProblem(x0_ref, stages, term_cost)
mu_init = 1e-2
solver = proxddp.SolverProxDDP(1e-3, mu_init)
solver.setup(problem)


xs_init = [x0_ref] * (nsteps + 1)
us_init = [np.zeros(nu) for _ in range(nsteps)]

solver.run(problem, xs_init, us_init)
workspace = solver.workspace
res = solver.results


def addPlane():
    import hppfcl as fcl

    plane = fcl.Plane(np.array([0.0, 0.0, 1.0]), 0.0)
    plane_obj = pin.GeometryObject("plane", 0, pin.SE3.Identity(), plane)
    plane_obj.meshColor[:] = [1.0, 1.0, 0.95, 1.0]
    plane_obj.meshScale[:] = 2.0
    robot.visual_model.addGeometryObject(plane_obj)
    robot.collision_model.addGeometryObject(plane_obj)


addPlane()
vizer = MeshcatVisualizer(
    rmodel,
    collision_model=robot.collision_model,
    visual_model=robot.visual_model,
    data=rdata,
)


def main():
    vizer.initViewer(loadModel=True, open=True)
    vizer.display(q0)
    vizer.setBackgroundColor()

    input()
    qs = [x[:nq] for x in res.xs]
    vizer.play(qs, dt)


main()
