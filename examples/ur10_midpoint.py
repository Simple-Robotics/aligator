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
xs_i = proxddp.rollout_implicit(dyn_model, x0, us_i)
ee_pos_target = np.array([0.5, 0.7, 1.2]) * 0.707


def define_cost():
    w_x = np.ones(ndx) * 1e-6
    w_x[nv:] = 5e-2
    w_u = 1e-3
    costs = proxddp.CostStack(space, nu)
    xreg = proxddp.QuadraticStateCost(space, nu, space.neutral(), np.diag(w_x) * dt)
    ureg = proxddp.QuadraticControlCost(space, u0, np.eye(nu) * dt)
    costs.addCost(xreg)
    costs.addCost(ureg, w_u)

    frame_name = "ee_link"
    ee_id = rmodel.getFrameId(frame_name)
    frame_err = proxddp.FrameTranslationResidual(ndx, nu, rmodel, ee_pos_target, ee_id)

    w_frame = np.eye(3) * 6.0
    term_cost = proxddp.CostStack(space, nu)
    frame_cost = proxddp.QuadraticResidualCost(space, frame_err, w_frame)
    term_cost.addCost(frame_cost)

    return costs, term_cost


running_cost, term_cost = define_cost()
stm = proxddp.StageModel(running_cost, dyn_model)
ctrl_fn = proxddp.ControlErrorResidual(ndx, np.zeros(nu))
stm.addConstraint(
    ctrl_fn, proxddp.constraints.BoxConstraint(-rmodel.effortLimit, rmodel.effortLimit)
)
stages = [stm] * nsteps
problem = proxddp.TrajOptProblem(x0, stages, term_cost)


if __name__ == "__main__":
    from meshcat import Visualizer
    import hppfcl
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme("paper", "ticks")

    obj_placement = pin.SE3.Identity()
    obj_placement.translation = ee_pos_target
    obj_geom = pin.GeometryObject("obj", 0, obj_placement, hppfcl.Sphere(0.05))
    obj_geom.meshColor[:] = [255, 20, 83, 200]
    obj_geom.meshColor /= 255.0
    visual_model.addGeometryObject(obj_geom)

    if args.display:
        viewer_ = Visualizer(zmq_url=args.zmq_url)
        viz = MeshcatVisualizer(
            rmodel, robot.collision_model, visual_model, data=robot.data
        )
        viz.initViewer(viewer_, loadModel=True)
        viz.setBackgroundColor()
        viz.display(robot.q0)

    tol = 1e-3
    solver = proxddp.SolverProxDDP(tol, 0.01, verbose=proxddp.VERBOSE)
    solver.rollout_max_iters = 10
    solver.max_iters = 200
    solver.setup(problem)
    solver.run(problem, xs_i, us_i)

    results: proxddp.Results = solver.results
    print(results)
    us_opt = results.us

    if args.display:
        input("[press enter - optimized trajectory]")
        qs_opt = [x[:nq] for x in results.xs]
        viz.play(qs_opt, dt)

    times = np.linspace(0.0, tf, nsteps + 1)
    fig, axes = plt.subplots(3, 2, sharex="col", figsize=(6.4, 6.4))
    axes = axes.flatten()
    for i in range(nu):
        plt.sca(axes[i])
        plt.step(times[:-1], np.stack(us_opt)[:, i])
        ylim = plt.ylim()
        plt.hlines(-rmodel.effortLimit[i], 0.0, tf, colors="k", linestyles="--")
        plt.hlines(+rmodel.effortLimit[i], 0.0, tf, colors="k", linestyles="--")
        plt.ylim(*ylim)
        joint_name = rmodel.names[i]
        plt.ylabel(joint_name.lower())
    fig.supxlabel("Time $t$")
    plt.suptitle("Controls trajectory")
    plt.tight_layout()
    plt.show()
