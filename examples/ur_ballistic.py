import example_robot_data as erd
import pybullet_data
import pinocchio as pin
import meshcat
import numpy as np
import proxddp
import hppfcl
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Tuple
from pinocchio.visualize import MeshcatVisualizer
from utils import (
    add_namespace_prefix_to_models,
    plot_controls_traj,
)
from proxddp import dynamics, manifolds, constraints


PYBULLET_URDF_PATH = Path(pybullet_data.getDataPath())

robot = erd.load("ur10")


def load_projectile_model():
    ball_urdf = PYBULLET_URDF_PATH / "urdf/mug.urdf"
    ball_scale = 1.0
    model, cmodel, vmodel = pin.buildModelsFromUrdf(
        str(ball_urdf),
        package_dirs=str(ball_urdf.parent),
        root_joint=pin.JointModelFreeFlyer(),
    )
    for geom in cmodel.geometryObjects:
        geom.meshScale *= ball_scale
    for geom in vmodel.geometryObjects:
        geom.meshScale *= ball_scale
    return model, cmodel, vmodel


def append_ball_to_robot_model(
    robot: pin.RobotWrapper,
) -> Tuple[pin.Model, pin.GeometryModel, pin.GeometryModel]:
    locked_joints = []
    robot = robot.buildReducedRobot(locked_joints)
    base_model: pin.Model = robot.model
    base_visual: pin.GeometryModel = robot.visual_model
    base_coll: pin.GeometryModel = robot.collision_model
    ee_link_id = base_model.getFrameId("ee_link")
    _ball_model, _ball_coll, _ball_visu = load_projectile_model()
    add_namespace_prefix_to_models(_ball_model, _ball_coll, _ball_visu, "ball")

    pin.forwardKinematics(base_model, robot.data, robot.q0)
    pin.updateFramePlacement(base_model, robot.data, ee_link_id)

    tool_frame_pl = robot.data.oMf[ee_link_id]
    rel_placement = tool_frame_pl.copy()
    rel_placement.translation[1] = 0.0
    rmodel, cmodel = pin.appendModel(
        base_model, _ball_model, base_coll, _ball_coll, 0, rel_placement
    )
    _, vmodel = pin.appendModel(
        base_model, _ball_model, base_visual, _ball_visu, 0, rel_placement
    )

    return rmodel, cmodel, vmodel


rmodel, cmodel, vmodel = append_ball_to_robot_model(robot)
space = manifolds.MultibodyPhaseSpace(rmodel)
rdata: pin.Data = rmodel.createData()
nq = rmodel.nq
nv = rmodel.nv
nu = nv - 6
ndx = space.ndx

CONTACT_REF_FRAME = pin.LOCAL_WORLD_ALIGNED


def create_rcm():
    # create rigid constraint between ball & tool0
    tool_fid = rmodel.getFrameId("tool0")
    frame: pin.Frame = rmodel.frames[tool_fid]
    q0 = pin.neutral(rmodel)
    joint1_id = frame.parentJoint
    joint2_id = rmodel.getJointId("ball/root_joint")
    pin.framesForwardKinematics(rmodel, rdata, q0)
    pl1 = rmodel.frames[tool_fid].placement
    pl2 = rdata.oMf[tool_fid]
    return pin.RigidConstraintModel(
        pin.ContactType.CONTACT_6D,
        rmodel,
        joint1_id,
        pl1,
        joint2_id,
        pl2,
        CONTACT_REF_FRAME,
    )


target_pos = np.array([3.0, -0.4, 0.0])
gobj = pin.GeometryObject(
    "objective", 0, pin.SE3(np.eye(3), target_pos), hppfcl.Sphere(0.04)
)
gobj.meshColor[:] = np.array([200, 100, 100, 200]) / 255.0

viewer = meshcat.Visualizer("tcp://127.0.0.1:6000")
viz = MeshcatVisualizer(
    model=rmodel, collision_model=cmodel, visual_model=vmodel, data=rdata
)
viz.initViewer(viewer, loadModel=True)
viz.addGeometryObject(gobj)

q0 = pin.neutral(rmodel)
viz.display(q0)

input("[press enter]")

dt = 0.01
tf = 1.0  # seconds
nsteps = int(tf / dt)
actuation_matrix = np.eye(nv, nu, -nu)

prox_settings = pin.ProximalSettings(1e-8, 1e-9, 20)
rcm = create_rcm()
ode1 = dynamics.MultibodyConstraintFwdDynamics(
    space, actuation_matrix, [rcm], prox_settings
)
ode2 = dynamics.MultibodyConstraintFwdDynamics(
    space, actuation_matrix, [], prox_settings
)
dyn_model1 = dynamics.IntegratorSemiImplEuler(ode1, dt)
dyn_model2 = dynamics.IntegratorSemiImplEuler(ode2, dt)

x0 = space.neutral()


def constraint_quasistatic_torque(q, v, a, B):
    def _get_jacobian():
        pin.computeJointJacobians(rmodel, rdata, q)
        _tau = np.zeros(nv)
        # single 6D constraint
        joint_id = rcm.joint1_id
        J = pin.getJointJacobian(rmodel, rdata, joint_id, CONTACT_REF_FRAME)
        return J

    J = _get_jacobian()
    tau0 = pin.rnea(rmodel, rdata, q, v, a)
    matrix = np.hstack([B, -J.T])
    ret = np.linalg.lstsq(matrix, tau0, rcond=None)
    u, fc = np.split(ret[0], [nu])
    return u, fc


u0, lam_c = constraint_quasistatic_torque(
    x0[:nq], x0[nq:], np.zeros(nv), B=actuation_matrix
)
u0 = pin.rnea(robot.model, robot.data, robot.q0, robot.v0, robot.v0)

print(u0, u0.shape)
assert u0.shape == (nu,)

dms = [dyn_model1] * nsteps
us_i = [u0] * len(dms)
xs_i = proxddp.rollout(dms, x0, us_i)
qs_i = [x[:nq] for x in xs_i]
viz.play(qs_i, dt=dt)


def create_running_cost():
    costs = proxddp.CostStack(space, nu)
    w_x = np.array([1e-5] * nv + [1e-2] * nv)
    w_x[:6] = 0.0
    w_x[nv : nv + 6] = 0.0
    xreg = proxddp.QuadraticStateCost(space, nu, x0, np.diag(w_x) * dt)
    ureg = proxddp.QuadraticControlCost(space, u0, np.eye(nu) * dt)
    costs.addCost(xreg)
    costs.addCost(ureg, 1e-2)
    return costs


def create_term_cost():
    w_xf = np.ones(ndx)
    w_xf[:nv] = 1e-6
    w_xf[nv : nv + 6] = 1e-8
    return proxddp.QuadraticStateCost(space, nu, x0, np.diag(w_xf))


def get_ball_fn(target_pos):
    fid = rmodel.getFrameId("ball/root_joint")
    return proxddp.FrameTranslationResidual(ndx, nu, rmodel, target_pos, fid)


def create_term_constraint(target_pos):
    term_fn = get_ball_fn(target_pos)
    return proxddp.StageConstraint(term_fn, constraints.EqualityConstraintSet())


def create_stage(contact: bool):
    dm = dyn_model1 if contact else dyn_model2
    rc = create_running_cost()
    stm = proxddp.StageModel(rc, dm)
    return stm


stages = []
t_contact = int(0.4 * nsteps)
for k in range(nsteps):
    stages.append(create_stage(k <= t_contact))

term_cost = create_term_cost()
term_constraint = create_term_constraint(target_pos=target_pos)

problem = proxddp.TrajOptProblem(x0, stages, term_cost)
problem.addTerminalConstraint(term_constraint)
tol = 1e-3
solver = proxddp.SolverProxDDP(tol, 0.01, max_iters=200, verbose=proxddp.VERBOSE)
solver.setup(problem)
flag = solver.run(problem, xs_i, us_i)

print(flag)
print(solver.results)

qs = [x[:nq] for x in solver.results.xs]
viz.play(qs, dt)


times = np.linspace(0.0, tf, nsteps + 1)
_joint_names = rmodel.names[2:]
plot_controls_traj(times, solver.results.us, joint_names=_joint_names)
plt.show()
