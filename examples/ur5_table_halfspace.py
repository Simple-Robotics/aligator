import proxddp
import numpy as np

import pinocchio as pin

import meshcat_utils as msu
import example_robot_data as erd
import matplotlib.pyplot as plt

from proxddp import manifolds, constraints, dynamics
from common import ArgsBase, compute_quasistatic, get_endpoint_traj


class Args(ArgsBase):
    plot: bool = True

    def process_args(self):
        if self.record:
            self.display = True


args = Args().parse_args()
print(args)

robot = erd.load("ur5")
rmodel = robot.model
rdata = robot.data
nv = rmodel.nv

if args.display:
    vizer = pin.visualize.MeshcatVisualizer(
        rmodel, robot.collision_model, robot.visual_model, data=rdata
    )
    vizer.initViewer(open=True, loadModel=True)
    q0 = pin.neutral(rmodel)
    vizer.display(q0)
    vizutil = msu.VizUtil(vizer)
    vizutil.set_bg_color()
else:
    vizutil = None

space = manifolds.MultibodyPhaseSpace(rmodel)
ndx = space.ndx

x0 = space.neutral()


ode = dynamics.MultibodyFreeFwdDynamics(space)
print("Is underactuated:", ode.isUnderactuated)
print("Actuation rank:", ode.actuationMatrixRank)

dt = 0.01
Tf = 1.2
nsteps = int(Tf / dt)

nu = rmodel.nv
print(nu)
print(ode.nu)
assert nu == ode.nu
dyn_model = dynamics.IntegratorSemiImplEuler(ode, dt)
print(dyn_model)

frame_id = rmodel.getFrameId("tool0")
table_height = 0.65
table_side_y_l = -0.2
table_side_y_r = 0.6


def make_ee_residual():
    # creates a map -z_p(q) where p is the EE position
    A = np.array([[0.0, 0.0, 1.0]])
    b = np.array([-table_height])
    frame_fn = proxddp.FrameTranslationResidual(ndx, nu, rmodel, np.zeros(3), frame_id)
    frame_fn_neg_z = proxddp.LinearFunctionComposition(frame_fn, A, b)
    return frame_fn_neg_z


w_x = np.ones(ndx) * 0.01
w_x[:nv] = 1e-6
w_x = np.diag(w_x)
w_u = np.eye(nu) * 0.001

rcost = proxddp.CostStack(ndx, nu)
rcost.addCost(proxddp.QuadraticCost(w_x * dt, w_u * dt))

# define the terminal cost

weights_ee = 5.0 * np.eye(3)
weights_ee_term = 10.0 * np.eye(3)
p_ref_term = np.array([0.2, 0.3, table_height + 0.1])
frame_obj_fn = proxddp.FrameTranslationResidual(ndx, nu, rmodel, p_ref_term, frame_id)

rcost.addCost(proxddp.QuadraticResidualCost(frame_obj_fn, weights_ee * dt))

term_cost = proxddp.CostStack(ndx, nu)
term_cost.addCost(proxddp.QuadraticResidualCost(frame_obj_fn, weights_ee_term))

time_idx_require_above_table = int(0.4 * nsteps)
frame_fn_z = make_ee_residual()
frame_cstr = proxddp.StageConstraint(frame_fn_z, constraints.NegativeOrthant())
stages = []
for i in range(nsteps):
    stm = proxddp.StageModel(space, nu, rcost, dyn_model)
    if i >= time_idx_require_above_table:
        stm.addConstraint(frame_cstr)
    stages.append(stm)


problem = proxddp.TrajOptProblem(x0, stages, term_cost)
problem.setTerminalConstraint(frame_cstr)


tol = 1e-3
mu_init = 0.1
max_iters = 50
verbose = proxddp.VerboseLevel.VERBOSE
solver = proxddp.SolverProxDDP(tol, mu_init, max_iters=max_iters)
solver.verbose = verbose
solver.rollout_type = proxddp.ROLLOUT_NONLINEAR
# solver.dump_linesearch_plot = True

solver.setup(problem)

u0 = compute_quasistatic(rmodel, rdata, x0, acc=np.zeros(nv))
us_init = [u0] * nsteps
xs_init = proxddp.rollout(dyn_model, x0, us_init).tolist()

solver.run(problem, xs_init, us_init)

rs = solver.getResults()
print(rs)
xs_opt = np.array(rs.xs)
ws = solver.getWorkspace()

stage_datas: list[proxddp.StageData] = ws.problem_data.stage_data
ineq_cstr_datas = []
ineq_cstr_values = []
dyn_cstr_values = []
for i in range(nsteps):
    if len(stage_datas[i].constraint_data) > 1:
        icd: proxddp.FunctionData = stage_datas[i].constraint_data[1]
        ineq_cstr_datas.append(icd)
        ineq_cstr_values.append(icd.value.copy())
    dcd = stage_datas[i].constraint_data[0]
    dyn_cstr_values.append(dcd.value.copy())

times = np.linspace(0.0, Tf, nsteps + 1)
plt.subplot(131)
plt.plot(times[time_idx_require_above_table:-1], ineq_cstr_values)
plt.title("Inequality constraint values")
plt.xlabel("Time")
plt.subplot(132)
plt.plot(times[1:], np.array(dyn_cstr_values))
plt.xlabel("Time")
plt.subplot(133)
ee_traj = get_endpoint_traj(rmodel, rdata, xs_opt, frame_id)
plt.plot(times, np.array(ee_traj), label=["x", "y", "z"])
plt.hlines(table_height, *times[[0, -1]], colors="k")
plt.legend()
plt.show()

if args.display:
    import meshcat.geometry as mgeom
    import meshcat.transformations as mtransf

    video_fps = 0.5 / dt
    recorder = msu.VideoRecorder("assets/ur5_halfspace_under.mp4", fps=video_fps)

    def planehoz():
        p_height = table_height
        p_width = table_side_y_r - table_side_y_l
        p_center = (table_side_y_l + table_side_y_r) / 2.0
        plane_g = mgeom.Plane(height=p_width)
        _M = mtransf.translation_matrix([0.0, p_center, table_height])
        material = mgeom.MeshLambertMaterial(0x7FCB85, opacity=0.4)
        vizer.viewer["plane"].set_object(plane_g, material)
        vizer.viewer["plane"].set_transform(_M)
        plane_v = mgeom.Plane(height=p_height)
        _Mrot = mtransf.rotation_matrix(np.pi / 2, [1.0, 0.0, 0.0])
        _M2 = (
            mtransf.translation_matrix([0.0, table_side_y_l, 0.5 * table_height])
            @ _Mrot
        )
        vizer.viewer["plane_y1"].set_object(plane_v, material)
        vizer.viewer["plane_y1"].set_transform(_M2)
        _M3 = (
            mtransf.translation_matrix([0.0, table_side_y_r, 0.5 * table_height])
            @ _Mrot
        )
        vizer.viewer["plane_y2"].set_object(plane_v, material)
        vizer.viewer["plane_y2"].set_transform(_M3)

    planehoz()
    vizutil.draw_objective(target=p_ref_term)

    play_dt = 0.5 * dt
    play_dt = None
    for i in range(4):
        input("[enter to play]")
        vizutil.play_trajectory(
            rs.xs, rs.us, frame_ids=[frame_id], timestep=play_dt, recorder=recorder
        )
