import example_robot_data as erd
import pinocchio as pin
import numpy as np
import aligator

from aligator import dynamics, manifolds
from utils import plot_controls_traj, ArgsBase
from pinocchio.visualize import MeshcatVisualizer


class Args(ArgsBase):
    dt: float = 0.02


robot = erd.load("ur10_limited")
rmodel: pin.Model = robot.model
visual_model = robot.visual_model
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
x0 = space.neutral()
u0 = pin.rnea(rmodel, robot.data, q=x0[:nq], v=x0[nq:], a=np.zeros(nv))
us_i = [u0] * nsteps
xs_i = aligator.rollout_implicit(dyn_model, x0, us_i)
ee_pos_target = np.array([0.5, 0.7, 1.2]) * 0.707


def define_cost():
    w_x = np.ones(ndx) * 1e-6
    w_x[nv:] = 5e-2
    w_u = 1e-3
    costs = aligator.CostStack(space, nu)
    xreg = aligator.QuadraticStateCost(space, nu, space.neutral(), np.diag(w_x) * dt)
    ureg = aligator.QuadraticControlCost(space, u0, np.eye(nu) * dt)
    costs.addCost(xreg)
    costs.addCost(ureg, w_u)

    frame_name = "ee_link"
    ee_id = rmodel.getFrameId(frame_name)
    frame_err = aligator.FrameTranslationResidual(ndx, nu, rmodel, ee_pos_target, ee_id)

    w_frame = np.eye(3) * 6.0
    term_cost = aligator.CostStack(space, nu)
    frame_cost = aligator.QuadraticResidualCost(space, frame_err, w_frame)
    term_cost.addCost(frame_cost)

    return costs, term_cost


running_cost, term_cost = define_cost()
stm = aligator.StageModel(running_cost, dyn_model)
ctrl_fn = aligator.ControlErrorResidual(ndx, np.zeros(nu))
stm.addConstraint(
    ctrl_fn, aligator.constraints.BoxConstraint(-rmodel.effortLimit, rmodel.effortLimit)
)
stages = [stm] * nsteps
problem = aligator.TrajOptProblem(x0, stages, term_cost)


if __name__ == "__main__":
    import coal
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme("paper", "ticks")

    obj_placement = pin.SE3.Identity()
    obj_placement.translation = ee_pos_target
    obj_geom = pin.GeometryObject("obj", 0, obj_placement, coal.Sphere(0.05))
    obj_geom.meshColor[:] = [255, 20, 83, 200]
    obj_geom.meshColor /= 255.0
    visual_model.addGeometryObject(obj_geom)

    if args.display:
        viz = MeshcatVisualizer(
            rmodel, robot.collision_model, visual_model, data=robot.data
        )
        viz.initViewer(loadModel=True, zmq_url=args.zmq_url)
        viz.setBackgroundColor()
        viz.display(robot.q0)

    tol = 1e-6
    mu_init = 1e3
    solver = aligator.SolverProxDDP(tol, mu_init, verbose=aligator.VERBOSE)
    solver.rollout_max_iters = 10
    solver.max_iters = 100
    solver.linear_solver_choice = aligator.LQ_SOLVER_PARALLEL
    solver.rollout_type = aligator.ROLLOUT_LINEAR
    bcl: solver.AlmParams = solver.bcl_params
    solver.setNumThreads(4)
    solver.setup(problem)
    solver.run(problem, xs_i, us_i)

    results: aligator.Results = solver.results
    print(results)
    us_opt = results.us

    if args.display:
        input("[press enter - optimized trajectory]")
        qs_opt = [x[:nq] for x in results.xs]
        viz.play(qs_opt, dt)

    times = np.linspace(0.0, tf, nsteps + 1)
    plot_controls_traj(times, us_opt, rmodel=rmodel)
    plt.show()
