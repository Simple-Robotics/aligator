import example_robot_data as erd
import pinocchio as pin
import numpy as np

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
    fname: rmodel.frames[fid].parent for fname, fid in FOOT_FRAME_IDS.items()
}


def create_ground_contact_model(rmodel: pin.Model, Kp=(0.0, 0.0, 1000.0), Kd=None):
    """Create a list of pinocchio.RigidConstraintModel objects corresponding to ground contact."""
    constraint_models = []
    for fname, fid in FOOT_FRAME_IDS.items():
        joint_id = FOOT_JOINT_IDS[fname]
        pl1 = rmodel.frames[fid].placement
        pl2 = rdata.oMf[fid]
        cm = pin.RigidConstraintModel(
            pin.ContactType.CONTACT_3D,
            rmodel,
            joint_id,
            pl1,
            0,
            pl2,
            pin.LOCAL_WORLD_ALIGNED,
        )
        cm.corrector.Kp[:] = Kp
        cm.corrector.Kd[:] = Kd if Kd is not None else 2 * np.sqrt(Kp)
        constraint_models.append(cm)
    return constraint_models


def add_plane(robot, meshColor=(238, 66, 102, 240)):
    import hppfcl as fcl

    plane = fcl.Plane(np.array([0.0, 0.0, 1.0]), 0.0)
    plane_obj = pin.GeometryObject("plane", 0, pin.SE3.Identity(), plane)

    plane_obj.meshColor = np.asarray(meshColor) / 255.0
    plane_obj.meshScale[:] = 1.0
    robot.visual_model.addGeometryObject(plane_obj)
    robot.collision_model.addGeometryObject(plane_obj)


def manage_lights(vizer: pin.visualize.MeshcatVisualizer):
    import meshcat

    viewer: meshcat.Visualizer = vizer.viewer
    SPOTLIGHT_PATH = "/Lights/SpotLight/<object>"
    viewer["/Lights/SpotLight"].set_property("visible", True)
    props = [("intensity", 0.2), ("castShadow", True)]
    for name, value in props:
        viewer[SPOTLIGHT_PATH].set_property(name, value)

    AMBIENT_PATH = "/Lights/AmbientLight/<object>"
    props = [
        ("intensity", 0.5),
    ]
    for name, value in props:
        viewer[AMBIENT_PATH].set_property(name, value)


__all__ = ["robot", "rmodel", "rdata", "q0", "FOOT_FRAME_IDS", "FOOT_JOINT_IDS"]
