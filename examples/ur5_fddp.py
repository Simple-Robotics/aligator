"""
An example showcasing the use of our implementation of FDDP for a UR5 task.
"""
import proxddp
from proxddp import manifolds, dynamics, SolverFDDP

import numpy as np
import example_robot_data as erd
import pinocchio as pin

import tap


class Args(tap.Tap):
    display: bool = False
    record: bool = False

    def process_args(self):
        if self.record:
            self.display = True


args = Args().parse_args()
robot = erd.load("ur5")
rmodel = robot.model
rdata = robot.data

space = manifolds.MultibodyPhaseSpace(rmodel)
ndx = space.ndx
nv = rmodel.nv
nu = nv

x0 = space.neutral()

p_ref = np.array([0.7, 0.0, 0.6])

actuation_matrix = np.eye(nu)
ode = dynamics.MultibodyFreeFwdDynamics(space, actuation_matrix)
timestep = 0.033
discrete_dyn = dynamics.IntegratorRK2(ode, timestep)

w_x = timestep * np.ones(ndx) * 1e-2
w_x[nv:] = 0.02
w_x = np.diag(w_x)
w_u = timestep * np.eye(nu) * 1e-4
rcost = proxddp.QuadraticCost(w_x, w_u)

stages = []
Tf = 1.0
nsteps = int(Tf / timestep)
for i in range(nsteps):
    st = proxddp.StageModel(rcost, discrete_dyn)
    stages.append(st)

wx_term = np.eye(3) * 6.0
frame_id = rmodel.getFrameId("tool0")
term_cost = proxddp.CostStack(space, nu)
term_cost.addCost(
    proxddp.QuadraticResidualCost(
        proxddp.FrameTranslationResidual(ndx, nu, rmodel, p_ref, frame_id), wx_term
    )
)
problem = proxddp.TrajOptProblem(x0, stages, term_cost)

tol = 1e-4
verbose = proxddp.VerboseLevel.VERBOSE
solver = SolverFDDP(tol, verbose=verbose, reg_init=1e-9)
# solver = proxddp.SolverProxDDP(tol, 1e-6, verbose=verbose)
solver.registerCallback(proxddp.HistoryCallback(False, True, True))
solver.setup(problem)

us_init = [np.zeros(nu)] * nsteps
xs_init = [x0] * (nsteps + 1)

solver.run(problem, xs_init, us_init)

results = solver.getResults()
assert results.conv
print(results)

if args.display:
    vizer = pin.visualize.MeshcatVisualizer(
        rmodel, robot.collision_model, robot.visual_model, data=rdata
    )
    vizer.initViewer(open=args.display, loadModel=True)
    q0 = pin.neutral(rmodel)
    vizer.display(q0)
    vizer.setCameraPreset("preset1")

    nq = rmodel.nq
    qs = [x[:nq] for x in results.xs]

    def callback(i: int):
        v = results.xs[i][nq:]
        pin.forwardKinematics(rmodel, vizer.data, qs[i], v)
        vizer.drawFrameVelocities(frame_id)

    input("[press enter]")
    for _ in range(3):
        vizer.play(qs, timestep, callback)
