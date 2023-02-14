import proxddp
import example_robot_data as erd
import pinocchio as pin
import numpy as np

from proxddp import manifolds, dynamics
from pinocchio.visualize import MeshcatVisualizer
from pinocchio.visualize.meshcat_visualizer import COLOR_PRESETS

COLOR_PRESETS["white"] = ([1, 1, 1], [1, 1, 1])


robot = erd.load("solo12")
rmodel: pin.Model = robot.model
rdata = robot.data
stance_name = "straight_standing"
q0 = rmodel.referenceConfigurations[stance_name]

vizer = MeshcatVisualizer(
    rmodel,
    collision_model=robot.collision_model,
    visual_model=robot.visual_model,
    data=rdata,
)

vizer.initViewer(loadModel=True, open=True)
vizer.display(q0)
vizer.setBackgroundColor("white")

pin.framesForwardKinematics(rmodel, rdata, q0)

foot_frame_ids = dict(
    fr_foot_id=rmodel.getFrameId("FR_FOOT"),
    fl_foot_id=rmodel.getFrameId("FL_FOOT"),
    hl_foot_id=rmodel.getFrameId("HL_FOOT"),
    hr_foot_id=rmodel.getFrameId("HR_FOOT"),
)
foot_joint_ids = {
    fname: rmodel.frames[fid].parentJoint for fname, fid in foot_frame_ids.items()
}


def create_ground_contact_model(rmodel: pin.Model):
    constraint_models = []
    for fname, fid in foot_frame_ids.items():
        print(fid)
        joint_id = foot_joint_ids[fname]
        pl1 = rmodel.frames[fid].placement
        pl2 = rdata.oMf[fid]
        print(pl2)
        cm = pin.RigidConstraintModel(
            pin.ContactType.CONTACT_3D, rmodel, joint_id, pl1, 0, pl2
        )
        cm.corrector.Kp = 20.0
        cm.corrector.Kd = 2 * cm.corrector.Kp**0.5
        constraint_models.append(cm)
    return constraint_models


nq = rmodel.nq
nv = rmodel.nv
nu = nv - 6
act_matrix = np.zeros((nv, nu))
act_matrix[6:] = np.eye(nu)
print(act_matrix)

space = manifolds.MultibodyPhaseSpace(rmodel)
constraint_models = create_ground_contact_model(rmodel)
print(constraint_models)
prox_settings = pin.ProximalSettings(1e-9, 1e-10, 4)
ode = dynamics.MultibodyConstraintFwdDynamics(
    space, act_matrix, constraint_models, prox_settings
)
timestep = 0.01
dyn_model = dynamics.IntegratorSemiImplEuler(ode, timestep)

u0 = np.zeros(nu)
v0 = np.zeros(nv)
x0 = np.concatenate([q0, v0])


def forward_sim(nsteps):
    us = [u0] * nsteps
    xs = proxddp.rollout(dyn_model, x0, us)

    print(xs.tolist())

    qs = [x[:nq] for x in xs]
    vizer.play(qs, timestep)


def main():
    input()
    forward_sim(100)


if __name__ == "__main__":
    main()
