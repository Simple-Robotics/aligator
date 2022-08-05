import crocoddyl as croc
import proxddp

import numpy as np

import pinocchio as pin
import example_robot_data as erd


from pinocchio.visualize import MeshcatVisualizer

# --- Robot and viewer
robot = erd.load("ur5")
rmodel: pin.Model = robot.model

nq = rmodel.nq
nv = rmodel.nv
nu = nv
q0 = robot.q0.copy()
v0 = np.zeros(nv)
x0 = np.concatenate([q0, v0])

vizer = MeshcatVisualizer(rmodel, robot.collision_model, robot.visual_model)
vizer.initViewer(loadModel=True)
vizer.display(q0)

idTool = rmodel.getFrameId("tool0")
target_frame: pin.SE3 = pin.SE3.Identity()
target_frame.translation[:] = (-0.75, 0.1, 0.5)

# --- OCP hyperparams
Tf = 1.2
dt = 0.01
nsteps = int(Tf / dt)
tol = 1e-4
mu_init = 0.001
rho_init = 1e-7

wt_x = 1e-5 * np.ones(rmodel.nv * 2)
wt_x[nv:] = 2e-4
wt_u = 5e-6 * np.ones(nu)
wt_x_term = wt_x.copy()
wt_frame = 8.0 * np.ones(6)
wt_frame[3:] = 0.0

# --- Reference solution (computed by prox @ nmsd)
sol_ref = np.load("examples/urprox.npy", allow_pickle=True)[()]

# --- OCP
state = croc.StateMultibody(rmodel)

frame_fun = croc.ResidualModelFramePlacement(state, idTool, target_frame)
assert wt_frame.shape[0] == frame_fun.nr

rcost_ = croc.CostModelSum(state)
xRegCost = croc.CostModelResidual(
    state,
    croc.ActivationModelWeightedQuad(wt_x),
    croc.ResidualModelState(state, x0, nu),
)
uRegCost = croc.CostModelResidual(
    state, croc.ActivationModelWeightedQuad(wt_u), croc.ResidualModelControl(state, nu)
)
rcost_.addCost("xReg", xRegCost, 1 / dt)
rcost_.addCost("uReg", uRegCost, 1 / dt)

term_cost_ = croc.CostModelSum(state)
toolTrackingCost = croc.CostModelResidual(state, frame_fun)
xRegTermCost = croc.CostModelResidual(
    state,
    croc.ActivationModelWeightedQuad(wt_x_term),
    croc.ResidualModelState(state, x0, nu),
)
term_cost_.addCost("tool", toolTrackingCost, 0)
term_cost_.addCost("xReg", xRegTermCost, 1)

actuation = croc.ActuationModelFull(state)
continuous_dynamics = croc.DifferentialActionModelFreeFwdDynamics(
    state, actuation, rcost_
)
discrete_dynamics = croc.IntegratedActionModelEuler(continuous_dynamics, dt)

stages_ = [discrete_dynamics for i in range(nsteps)]

continuous_term_dynamics = croc.DifferentialActionModelFreeFwdDynamics(
    state, actuation, term_cost_
)
discrete_term_dynamics = croc.IntegratedActionModelEuler(continuous_term_dynamics)

problem = croc.ShootingProblem(x0, stages_, discrete_term_dynamics)

solver = croc.SolverFDDP(problem)
solver.th_grad = tol**2
solver.setCallbacks([croc.CallbackVerbose()])


init_us = [
    m.quasiStatic(d, x0) for m, d in zip(problem.runningModels, problem.runningDatas)
]
init_xs = problem.rollout(init_us)

# --- Solve
# croco solve
solver.solve(init_xs, init_us, 300)

# --- Results
print(
    f"""Results [
  converged  :  {solver.isFeasible and solver.stop<solver.th_stop},
  traj. cost :  {solver.cost},
  merit.value:  0,
  prim_infeas:  { sum([ sum(f**2) for f in solver.fs]) },
  dual_infeas:  { np.max(np.array([ np.max(np.abs(q)) for q in solver.Qu])) },
]"""
)

xs_opt = solver.xs.tolist()
us_opt = solver.us.tolist()
# np.save(open(f"urcroco.npy", "wb"),{'xs': xs_opt, 'us': us_opt})

pb_prox = proxddp.croc.convertCrocoddylProblem(problem)
fddp2 = proxddp.SolverFDDP(1e-6, verbose=proxddp.VerboseLevel.VERBOSE)
fddp2.setup(pb_prox)
conv = fddp2.run(pb_prox, init_xs, init_us)
rs = fddp2.getResults()
print("ourFDDP:", rs)
print("cost", rs.traj_cost)

print("cost_ours - cost_croc:", rs.traj_cost - solver.cost)
