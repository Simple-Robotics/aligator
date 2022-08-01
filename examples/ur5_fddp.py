"""
An example showcasing the use of our implementation of FDDP for a UR5 task.
"""
import proxddp
from proxddp import manifolds, dynamics, SolverFDDP

import numpy as np
import example_robot_data as erd
import pinocchio as pin
import meshcat_utils as msu

import tap


class Args(tap.Tap):
    display: bool = False
    record: bool = False

    def process_args(self):
        if self.record:
            self.display = True


robot = erd.load("ur5")
rmodel = robot.model
rdata = robot.data

space = manifolds.MultibodyPhaseSpace(rmodel)
ndx = space.ndx
nv = rmodel.nv
nu = nv

x0 = space.neutral()

p_ref = np.array([0.9, 0.0, 0.9])

actuation_matrix = np.eye(nu)
ode = dynamics.MultibodyFreeFwdDynamics(space, actuation_matrix)
timestep = 0.033
rk2_dyn = dynamics.IntegratorRK2(ode, timestep)

w_x = timestep * np.eye(ndx) * 1e-2
w_u = timestep * np.eye(nu) * 1e-4
rcost = proxddp.QuadraticCost(w_x, w_u)

stages = []
nsteps = int(1.0 / timestep)
for i in range(nsteps):
    st = proxddp.StageModel(space, nu, rcost, rk2_dyn)
    stages.append(st)

wx_term = np.eye(3) * 2.0
rid = rmodel.getFrameId("tool0")
term_cost = proxddp.CostStack(space.ndx, nu)
term_cost.addCost(
    proxddp.QuadraticResidualCost(
        proxddp.FrameTranslationResidual(ndx, nu, rmodel, p_ref, rid), wx_term
    )
)
problem = proxddp.TrajOptProblem(x0, stages, term_cost)

tol = 1e-4
solver = SolverFDDP(tol, verbose=proxddp.VerboseLevel.VERBOSE)
solver.setup(problem)

us_init = [np.zeros(nu)] * nsteps
xs_init = proxddp.rollout(rk2_dyn, x0, us_init)

solver.run(problem, xs_init, us_init)

results = solver.getResults()
# ws = solver.getWorkspace()
print(results)

vizer = pin.visualize.MeshcatVisualizer(
    rmodel, robot.collision_model, robot.visual_model, data=rdata
)
vizer.initViewer(open=True, loadModel=True)
viz_util = msu.VizUtil(vizer)
q0 = pin.neutral(rmodel)
vizer.display(q0)
viz_util.draw_objective(p_ref)

input("[press enter]")
for _ in range(3):
    viz_util.play_trajectory(
        results.xs, results.us, frame_ids=[rid], timestep=timestep, show_vel=True
    )
