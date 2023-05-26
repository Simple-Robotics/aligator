import example_robot_data as erd
import pinocchio as pin

robot = erd.load("solo12")
rmodel: pin.Model = robot.model
rdata: pin.Data = robot.data
stance_name = "straight_standing"
q0 = rmodel.referenceConfigurations[stance_name]

FOOT_FRAME_IDS = {
    fname: rmodel.getFrameId(fname)
    for fname in ["FR_FOOT", "FL_FOOT", "HL_FOOT", "HR_FOOT"]
}

FOOT_JOINT_IDS = {
    fname: rmodel.frames[fid].parentJoint for fname, fid in FOOT_FRAME_IDS.items()
}


def create_ground_contact_model(rmodel: pin.Model, Kp=0.0, Kd=100.0):
    """Create a list of pinocchio.RigidConstraintModel objects corresponding to ground contact."""
    constraint_models = []
    for fname, fid in FOOT_FRAME_IDS.items():
        joint_id = FOOT_JOINT_IDS[fname]
        pl1 = rmodel.frames[fid].placement
        pl2 = rdata.oMf[fid]
        cm = pin.RigidConstraintModel(
            pin.ContactType.CONTACT_3D, rmodel, joint_id, pl1, 0, pl2
        )
        cm.corrector.Kp = Kp
        cm.corrector.Kd = Kd
        constraint_models.append(cm)
    return constraint_models


def add_plane(robot):
    import hppfcl as fcl
    import numpy as np

    plane = fcl.Plane(np.array([0.0, 0.0, 1.0]), 0.0)
    plane_obj = pin.GeometryObject("plane", 0, pin.SE3.Identity(), plane)

    meshColor = np.array((239, 173, 143, 200))
    plane_obj.meshColor[:] = meshColor / 255.0
    plane_obj.meshScale[:] = 0.1
    robot.visual_model.addGeometryObject(plane_obj)
    robot.collision_model.addGeometryObject(plane_obj)


__all__ = ["robot", "rmodel", "rdata", "q0", "FOOT_FRAME_IDS", "FOOT_JOINT_IDS"]
