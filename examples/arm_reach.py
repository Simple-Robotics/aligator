import proxddp
import numpy as np

import pinocchio as pin
import meshcat_utils as msu
import example_robot_data as erd

from proxddp import constraints, manifolds, dynamics
from pinocchio.visualize import MeshcatVisualizer

import tap


class Args(tap.Tap):
    display: bool = False
    record: bool = False

    def process_args(self):
        if self.record:
            self.display = True


args = Args().parse_args()

print(args)


robot = erd.load("ur5")
rmodel: pin.Model = robot.model
space = manifolds.MultibodyPhaseSpace(rmodel)

vizer = MeshcatVisualizer(rmodel, robot.collision_model, robot.visual_model)
vizer.initViewer(open=args.display, loadModel=True)
viz_util = msu.VizUtil(vizer)


x0 = space.neutral()

nq = rmodel.nq
nv = rmodel.nv
nu = nv
q0 = x0[:nq]

vizer.display(q0)

wt_x = 2e-4 * np.eye(space.ndx)
wt_u = 1e-5 * np.eye(nu)
wt_x_term = wt_x.copy()
wt_frame = 6.0 * np.eye(6)
wt_frame[3:] = 0.0
print(wt_frame)


idTool = rmodel.getFrameId("tool0")
target_frame: pin.SE3 = pin.SE3.Identity()
target_frame.translation[:] = (0.8, 0.0, 0.5)
print(target_frame)

frame_fun = proxddp.FramePlacementResidual(space.ndx, nu, rmodel, target_frame, idTool)
assert wt_frame.shape[0] == frame_fun.nr

B_mat = np.eye(nu)

Tf = 1.2
dt = 0.01
nsteps = int(Tf / dt)

rcost_ = proxddp.CostStack(space.ndx, nu)
rcost_.addCost(proxddp.QuadraticCost(wt_x, wt_u))

term_cost_ = proxddp.CostStack(space.ndx, nu)
term_cost_.addCost(proxddp.QuadraticCost(wt_x_term, wt_u))
term_cost_.addCost(proxddp.QuadraticResidualCost(frame_fun, wt_frame))

continuous_dynamics = dynamics.MultibodyFreeFwdDynamics(space, B_mat)
discrete_dynamics = dynamics.IntegratorSemiImplEuler(continuous_dynamics, dt)

u_max = rmodel.effortLimit
u_min = -u_max
ctrl_box = proxddp.ControlBoxFunction(space.ndx, u_min, u_max)


def computeQuasistatic(model: pin.Model, x0, a):
    data = model.createData()
    q0 = x0[:nq]
    v0 = x0[nq : nq + nv]

    return pin.rnea(model, data, q0, v0, a)


init_us = [computeQuasistatic(rmodel, x0, a=np.zeros(nv)) for _ in range(nsteps)]
init_xs = proxddp.rollout(discrete_dynamics, x0, init_us)


stages_ = []

for i in range(nsteps):
    stm = proxddp.StageModel(space, nu, rcost_, discrete_dynamics)
    stm.addConstraint(ctrl_box, constraints.NegativeOrthant())
    print("Stage {: 4d}: {}".format(i, stm))

    stages_.append(stm)


problem = proxddp.TrajOptProblem(x0, stages_, term_cost=term_cost_)
tol = 1e-3

mu_init = 0.001
rho_init = 1e-7

solver = proxddp.ProxDDP(
    tol, mu_init, rho_init, verbose=proxddp.VerboseLevel.VERBOSE, max_iters=300
)

solver.setup(problem)

solver.run(problem, init_xs, init_us)


results = solver.getResults()
print(results)

xs_opt = results.xs.tolist()
us_opt = results.us.tolist()


cp = [0.8, 0.8, 0.8]
viz_util.set_cam_pos(cp)
vidrecord = msu.VideoRecorder("examples/ur5_reach_ctrlbox.mp4", fps=1.0 / dt)
if args.display:
    input("[Press enter]")

    for i in range(3):
        viz_util.draw_objective(target_frame.translation)
        viz_util.play_trajectory(
            xs_opt,
            us_opt,
            frame_ids=[idTool],
            timestep=dt,
            record=args.record,
            recorder=vidrecord,
        )
