"""
Common utilities for examples.
"""

import numpy as np
import pinocchio as pin
import pinocchio.visualize
import tap

import matplotlib.pyplot as plt

from pathlib import Path
from aligator.utils.plotting import *  # noqa
from typing import Literal, List, Optional


plt.rcParams["lines.linewidth"] = 1.0
plt.rcParams["lines.markersize"] = 5

ASSET_DIR = Path("assets/")
ASSET_DIR.mkdir(exist_ok=True)

_integrator_choices = Literal["euler", "semieuler", "midpoint", "rk2"]
MESHCAT_ZMQ_DEFAULT = "tcp://127.0.0.1:6000"
IMAGEIO_KWARGS = {"macro_block_size": 8, "quality": 9}


class ArgsBase(tap.Tap):
    display: bool = False  # Display the trajectory using meshcat
    record: bool = False  # record video
    plot: bool = False
    integrator: _integrator_choices = "semieuler"
    """Numerical integrator to use"""
    zmq_url: Optional[str] = MESHCAT_ZMQ_DEFAULT

    def process_args(self):
        if self.record:
            self.display = True
        if self.zmq_url is not None and self.zmq_url.lower() == "none":
            self.zmq_url = None


def get_endpoint(rmodel, rdata, q: np.ndarray, tool_id: int):
    pin.framesForwardKinematics(rmodel, rdata, q)
    return rdata.oMf[tool_id].translation.copy()


def get_endpoint_traj(rmodel, rdata, xs: List[np.ndarray], tool_id: int):
    pts = []
    for i in range(len(xs)):
        pts.append(get_endpoint(rmodel, rdata, xs[i][: rmodel.nq], tool_id))
    return np.array(pts)


def compute_quasistatic(model: pin.Model, data: pin.Data, x0, acc):
    nq = model.nq
    q0 = x0[:nq]
    v0 = x0[nq:]
    return pin.rnea(model, data, q0, v0, acc)


def create_cartpole(N):
    import hppfcl as fcl

    model = pin.Model()
    geom_model = pin.GeometryModel()

    parent_id = 0

    cart_radius = 0.1
    cart_length = 5 * cart_radius
    cart_mass = 1.0
    joint_name = "joint_cart"

    geometry_placement = pin.SE3.Identity()
    geometry_placement.rotation = pin.Quaternion(
        np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])
    ).toRotationMatrix()

    joint_id = model.addJoint(
        parent_id, pin.JointModelPY(), pin.SE3.Identity(), joint_name
    )

    body_inertia = pin.Inertia.FromCylinder(cart_mass, cart_radius, cart_length)
    body_placement = geometry_placement
    model.appendBodyToJoint(
        joint_id, body_inertia, body_placement
    )  # We need to rotate the inertia as it is expressed in the LOCAL frame of the geometry

    shape_cart = fcl.Cylinder(cart_radius, cart_length)

    geom_cart = pin.GeometryObject(
        "shape_cart", joint_id, shape_cart, geometry_placement
    )
    geom_cart.meshColor = np.array([1.0, 0.1, 0.1, 1.0])
    geom_model.addGeometryObject(geom_cart)

    parent_id = joint_id
    joint_placement = pin.SE3.Identity()
    body_mass = 0.1
    body_radius = 0.1
    for k in range(N):
        joint_name = "joint_" + str(k + 1)
        joint_id = model.addJoint(
            parent_id, pin.JointModelRX(), joint_placement, joint_name
        )

        body_inertia = pin.Inertia.FromSphere(body_mass, body_radius)
        body_placement = joint_placement.copy()
        body_placement.translation[2] = 1.0
        model.appendBodyToJoint(joint_id, body_inertia, body_placement)

        geom1_name = "ball_" + str(k + 1)
        shape1 = fcl.Sphere(body_radius)
        geom1_obj = pin.GeometryObject(geom1_name, joint_id, shape1, body_placement)
        geom1_obj.meshColor = np.ones((4))
        geom_model.addGeometryObject(geom1_obj)

        geom2_name = "bar_" + str(k + 1)
        shape2 = fcl.Cylinder(body_radius / 4.0, body_placement.translation[2])
        shape2_placement = body_placement.copy()
        shape2_placement.translation[2] /= 2.0

        geom2_obj = pin.GeometryObject(geom2_name, joint_id, shape2, shape2_placement)
        geom2_obj.meshColor = np.array([0.0, 0.0, 0.0, 1.0])
        geom_model.addGeometryObject(geom2_obj)

        parent_id = joint_id
        joint_placement = body_placement.copy()
    end_frame = pin.Frame(
        "end_effector_frame",
        model.getJointId("joint_" + str(N)),
        0,
        body_placement,
        pin.FrameType(3),
    )
    model.addFrame(end_frame)
    geom_model.collision_pairs = []
    model.qinit = np.zeros(model.nq)
    model.qinit[1] = 0.0 * np.pi
    model.qref = pin.neutral(model)
    data = model.createData()
    geom_data = geom_model.createData()
    ddl = np.array([0])
    return model, geom_model, data, geom_data, ddl


def make_npendulum(N, ub=True, lengths=None):
    import hppfcl as fcl

    model = pin.Model()
    geom_model = pin.GeometryModel()

    parent_id = 0

    base_radius = 0.08
    shape_base = fcl.Sphere(base_radius)
    geom_base = pin.GeometryObject("base", 0, shape_base, pin.SE3.Identity())
    geom_base.meshColor = np.array([1.0, 0.1, 0.1, 1.0])
    geom_model.addGeometryObject(geom_base)

    joint_placement = pin.SE3.Identity()
    body_mass = 1.0
    body_radius = 0.06
    if lengths is None:
        lengths = [1.0 for _ in range(N)]

    for k in range(N):
        joint_name = "joint_" + str(k + 1)
        if ub:
            jmodel = pin.JointModelRUBX()
        else:
            jmodel = pin.JointModelRX()
        joint_id = model.addJoint(parent_id, jmodel, joint_placement, joint_name)

        body_inertia = pin.Inertia.FromSphere(body_mass, body_radius)
        body_placement = joint_placement.copy()
        body_placement.translation[2] = lengths[k]
        model.appendBodyToJoint(joint_id, body_inertia, body_placement)

        geom1_name = "ball_" + str(k + 1)
        shape1 = fcl.Sphere(body_radius)
        geom1_obj = pin.GeometryObject(geom1_name, joint_id, shape1, body_placement)
        geom1_obj.meshColor = np.ones((4))
        geom_model.addGeometryObject(geom1_obj)

        geom2_name = "bar_" + str(k + 1)
        shape2 = fcl.Cylinder(body_radius / 4, body_placement.translation[2])
        shape2_placement = body_placement.copy()
        shape2_placement.translation[2] /= 2.0

        geom2_obj = pin.GeometryObject(geom2_name, joint_id, shape2, shape2_placement)
        geom2_obj.meshColor = np.array([0.0, 0.0, 0.0, 1.0])
        geom_model.addGeometryObject(geom2_obj)

        parent_id = joint_id
        joint_placement = body_placement.copy()

    return model, geom_model, geom_model


def load_talos_upper_body():
    import example_robot_data as erd

    robot = erd.load("talos")
    qref = robot.model.referenceConfigurations["half_sitting"]
    locked_joints = list(range(1, 14))
    locked_joints += [23, 31]
    locked_joints += [32, 33]
    red_bot = robot.buildReducedRobot(locked_joints, qref)
    return red_bot


def load_talos_no_wristhead():
    import example_robot_data as erd

    robot = erd.load("talos")
    qref = robot.model.referenceConfigurations["half_sitting"]
    locked_joints = [20, 21, 22, 23, 28, 29, 30, 31, 32, 33]
    red_bot = robot.buildReducedRobot(locked_joints, qref)
    return robot, red_bot


def add_namespace_prefix_to_models(model, collision_model, visual_model, namespace):
    """
    Lifted from this GitHub discussion:
    https://github.com/stack-of-tasks/pinocchio/discussions/1841
    """
    # Rename geometry objects in collision model:
    for geom in collision_model.geometryObjects:
        geom.name = f"{namespace}/{geom.name}"

    # Rename geometry objects in visual model:
    for geom in visual_model.geometryObjects:
        geom.name = f"{namespace}/{geom.name}"

    # Rename frames in model:
    for f in model.frames:
        f.name = f"{namespace}/{f.name}"

    # Rename joints in model:
    for k in range(len(model.names)):
        model.names[k] = f"{namespace}/{model.names[k]}"


def manage_lights(vizer: pin.visualize.MeshcatVisualizer):
    import meshcat

    def apply_props(obj, props):
        for name, value in props.items():
            obj.set_property(name, value)

    viewer: meshcat.Visualizer = vizer.viewer
    apply_props(viewer["/Lights/SpotLight"], props={"visible": True})
    spotlight = viewer["/Lights/SpotLight/<object>"]
    apply_props(
        spotlight,
        props={
            "intensity": 1.0,
            "penumbra": 1.0,
            "decay": 1.0,
            # default: False
            "castShadow": False,
            # default: pi / 3
            "angle": np.pi / 3,
            "position": [4, -4, 4],
        },
    )

    ambient_light = viewer["/Lights/AmbientLight/<object>"]
    apply_props(
        ambient_light,
        props={
            # default: 0.6
            "intensity": 0.2
        },
    )

    fill_light = viewer["/Lights/FillLight/<object>"]
    apply_props(fill_light, props={"intensity": 3.0, "castShadow": False})

    plpx = viewer["/Lights/PointLightPositiveX"]
    apply_props(plpx, props={"visible": False})
    plnx = viewer["/Lights/PointLightNegativeX"]
    apply_props(plnx, props={"visible": False})
    plpx.delete()
    plnx.delete()
