import example_robot_data as erd
import pinocchio as pin
import numpy as np
import proxddp
import tap

from proxddp import dynamics, manifolds
from pinocchio.visualize import MeshcatVisualizer


class Args(tap.Tap):
    display: bool = False
    record: bool = False
    dt: float = 0.02
    zmq_url = "tcp://127.0.0.1:6000"


robot = erd.load("ur10_limited")
rmodel: pin.Model = robot.model
visual_model = robot.visual_model
rdata: pin.Data = robot.data

space = manifolds.MultibodyPhaseSpace(rmodel)

# 20 ms
args = Args().parse_args()

dt = args.dt
ode = dynamics.MultibodyFreeFwdDynamics(space)
dyn_model = dynamics.IntegratorMidpoint(ode, dt)

nq = rmodel.nq
nv = rmodel.nv
ndx = space.ndx
nu = nv
print(f"nq={nq}")
print(f"nv={nv}")

tf = 1.0
nsteps = int(tf / dt)
u0 = np.zeros(nu)
x0 = space.neutral()
us_i = [u0] * nsteps
xs_i = proxddp.rollout_implicit(dyn_model, x0, us_i)
qs_i = [x[:nq] for x in xs_i]
ee_pos_target = np.array([1.0, 0.0, 1.0]) * 0.707


def define_cost():
    w_x = 1e-3
    w_u = 1e-3
    costs = proxddp.CostStack(space, nu)
    xreg = proxddp.QuadraticStateCost(space, nu, space.neutral(), np.eye(ndx) * dt)
    ureg = proxddp.QuadraticControlCost(space, nu, np.eye(nu) * dt)
    costs.addCost(xreg, w_x)
    costs.addCost(ureg, w_u)

    frame_name = "ee_link"
    ee_id = rmodel.getFrameId(frame_name)
    frame_err = proxddp.FrameTranslationResidual(ndx, nu, rmodel, ee_pos_target, ee_id)

    w_frame = np.eye(3) * 4.0
    term_cost = proxddp.CostStack(space, nu)
    frame_cost = proxddp.QuadraticResidualCost(space, frame_err, w_frame)
    term_cost.addCost(frame_cost)

    return costs, term_cost


running_cost, term_cost = define_cost()
stm = proxddp.StageModel(running_cost, dyn_model)
stages = [stm] * nsteps
problem = proxddp.TrajOptProblem(x0, stages, term_cost)


if __name__ == "__main__":
    from meshcat import Visualizer
    import hppfcl

    obj_placement = pin.SE3.Identity()
    obj_placement.translation = ee_pos_target
    obj_geom = pin.GeometryObject("obj", 0, obj_placement, hppfcl.Sphere(0.05))
    obj_geom.meshColor[:] = [255, 20, 83, 200]
    obj_geom.meshColor /= 255.0
    visual_model.addGeometryObject(obj_geom)

    viewer_ = Visualizer(zmq_url=args.zmq_url)
    viz = MeshcatVisualizer(rmodel, robot.collision_model, visual_model, data=rdata)
    viz.initViewer(viewer_, loadModel=True)
    viz.setBackgroundColor()

    input("[press enter]")
    viz.play(qs_i, dt)

    solver = proxddp.SolverProxDDP(1e-3, 0.01, verbose=proxddp.VERBOSE)
    solver.rollout_max_iters = 4
    solver.setup(problem)
    solver.run(problem, xs_i, us_i)

    results: proxddp.Results = solver.results
    print(results)

    input("[press enter - optimized trajectory]")
    us_opt = results.us
    qs_opt = [x[:nq] for x in results.xs]
    viz.play(qs_opt, dt)