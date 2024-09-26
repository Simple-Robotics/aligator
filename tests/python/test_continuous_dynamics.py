"""
Create continuous dynamics from Python.
"""

import pytest
import numpy as np
import aligator
from aligator import dynamics, manifolds
from eigenpy import Quaternion
from utils import finite_diff, infNorm, create_multibody_ode

epsilon = 1e-6
aligator.seed(42)
np.random.seed(42)
HAS_PINOCCHIO = aligator.has_pinocchio_features()


class MyODE(dynamics.ODEAbstract):
    def __init__(self, nu=0):
        super().__init__(manifolds.R3() * manifolds.SO3(), nu)

    def forward(self, x, u, data: dynamics.ODEData):
        assert isinstance(data, MyODEData)
        _, R_quat = x[:3], x[3:]  # R is quat
        R_quat = Quaternion(R_quat)
        R = R_quat.toRotationMatrix()
        data.xdot[:3] = R @ data.v0
        data.xdot[3:] = 0

    def dForward(self, x, u, data: dynamics.ODEData):
        Jp = data.Jx[:, :3]
        Jw = data.Jx[:, 3:]
        Jp[::3] = 0.0

        Jp[:, :3] = 0.0
        Jw[:, 3:] = 0.0

    def createData(self):
        return MyODEData(self)


class MyODEData(dynamics.ODEData):
    def __init__(self, obj: MyODE):
        super().__init__(obj.ndx, obj.nu)
        self.v0 = np.random.randn(3)


def test_abstract():
    """Test have the right types, etc."""
    space = manifolds.SO3()
    nu = 1
    dae = dynamics.ContinuousDynamicsAbstract(space, nu)
    dae_data = dae.createData()
    assert isinstance(dae_data, dynamics.ContinuousDynamicsData)
    assert hasattr(dae_data, "Jx")
    assert hasattr(dae_data, "Ju")
    assert hasattr(dae_data, "Jxdot")

    ode = dynamics.ODEAbstract(space, nu)
    ode_data = ode.createData()
    assert isinstance(ode_data, dynamics.ODEData)
    assert hasattr(ode_data, "xdot")


def test_custom_ode():
    ode = MyODE()
    itg = dynamics.IntegratorEuler(ode, 0.01)
    x0 = ode.space.rand()
    us = [np.zeros(0) for _ in range(10)]
    xs = aligator.rollout(itg, x0, us)
    print(xs.tolist())
    assert len(xs) == 11


@pytest.mark.skipif(
    not HAS_PINOCCHIO, reason="Aligator was compiled without Pinocchio."
)
def test_multibody_free():
    ode = create_multibody_ode(True)
    if ode is None:
        pass
    space = ode.space
    nu = ode.nu
    data = ode.createData()
    assert isinstance(data, dynamics.MultibodyFreeFwdData)
    assert hasattr(data, "tau")

    for i in range(100):
        aligator.seed(i)
        np.random.seed(i)
        x0 = space.rand()
        x0[:3] = 0.0
        u0 = np.random.randn(nu)

        ode.forward(x0, u0, data)
        ode.dForward(x0, u0, data)

        Jxdiff, Judiff = finite_diff(ode, space, x0, u0, epsilon)
        atol = epsilon
        assert np.allclose(Jxdiff, data.Jx, atol, atol), "Jxerr={}".format(
            infNorm(Jxdiff - data.Jx)
        )
        assert np.allclose(Judiff, data.Ju, atol, atol), "Juerr={}".format(
            infNorm(Judiff - data.Ju)
        )


def test_centroidal():
    space = manifolds.VectorSpace(9)
    nk = 3
    force_size = 3
    nu = force_size * nk
    mass = 10.5
    gravity = np.array([0, 0, -9.81])
    contact_names = ["foot1", "foot2", "foot3"]
    contact_states = [True, True, False]
    contact_poses = [
        np.array([0, 0.1, 0]),
        np.array([0.1, -0.1, 0]),
        np.array([0.1, 0.2, 0]),
    ]
    contact_map = aligator.ContactMap(contact_names, contact_states, contact_poses)
    ode = dynamics.CentroidalFwdDynamics(space, mass, gravity, contact_map, force_size)
    data = ode.createData()

    assert isinstance(data, dynamics.CentroidalFwdData)

    x0 = space.neutral()
    x0[2] = 0.5
    u0 = np.zeros(nu)
    force_x = 10
    force_y = 10
    force_z = -gravity[2] * mass / nk + 10
    # Build contact forces so that CoM moves up left forward
    for k in range(nk):
        u0[k * 3] = force_x
        u0[k * 3 + 1] = force_y
        u0[k * 3 + 2] = force_z

    # Integrate over several timesteps
    dt = 0.01
    N = 10
    for _ in range(N):
        ode.forward(x0, u0, data)
        x0 += dt * data.xdot

    # CoM has shifted up left forward
    assert x0[0] > 0
    assert x0[1] > 0
    assert x0[2] > 0


def test_centroidal_diff():
    # Test for 3D contact
    nx = 9
    space = manifolds.VectorSpace(nx)
    force_size = 3
    nk = 3
    nu = force_size * nk
    mass = 10.5
    gravity = np.array([0, 0, -9.81])
    contact_names = ["foot1", "foot2", "foot3"]
    contact_states = [True, True, False]
    contact_poses = [
        np.array([0, 0.1, 0]),
        np.array([0.1, -0.1, 0]),
        np.array([0.1, 0.2, 0]),
    ]
    contact_map = aligator.ContactMap(contact_names, contact_states, contact_poses)
    ode = dynamics.CentroidalFwdDynamics(space, mass, gravity, contact_map, force_size)
    data = ode.createData()

    x0 = np.random.randn(nx)
    u0 = np.random.randn(nu)

    ode.forward(x0, u0, data)
    ode.dForward(x0, u0, data)

    Jx0 = data.Jx.copy()
    Ju0 = data.Ju.copy()
    Jxdiff, Judiff = finite_diff(ode, space, x0, u0, epsilon)

    assert np.allclose(Jxdiff, Jx0, epsilon), "err={}".format(infNorm(Jxdiff - Jx0))
    assert np.allclose(Judiff, Ju0, epsilon), "err={}".format(infNorm(Judiff - Ju0))

    # Test for 6D contact
    force_size = 6
    nu = force_size * nk
    u0 = np.random.randn(nu)

    ode = dynamics.CentroidalFwdDynamics(space, mass, gravity, contact_map, force_size)
    data = ode.createData()

    ode.forward(x0, u0, data)
    ode.dForward(x0, u0, data)

    Jx0 = data.Jx.copy()
    Ju0 = data.Ju.copy()
    Jxdiff, Judiff = finite_diff(ode, space, x0, u0, epsilon)

    assert np.allclose(Jxdiff, Jx0, epsilon), "err={}".format(infNorm(Jxdiff - Jx0))
    assert np.allclose(Judiff, Ju0, epsilon), "err={}".format(infNorm(Judiff - Ju0))


def test_continuous_centroidal():
    nk = 3
    force_size = 3
    nu = force_size * nk
    space = manifolds.VectorSpace(9 + nu)
    mass = 10.5
    gravity = np.array([0, 0, -9.81])
    contact_names = ["foot1", "foot2", "foot3"]
    contact_states = [True, True, False]
    contact_poses = [
        np.array([0, 0.1, 0]),
        np.array([0.1, -0.1, 0]),
        np.array([0.1, 0.2, 0]),
    ]
    contact_map = aligator.ContactMap(contact_names, contact_states, contact_poses)
    ode = dynamics.ContinuousCentroidalFwdDynamics(
        space, mass, gravity, contact_map, force_size
    )
    data = ode.createData()

    assert isinstance(data, dynamics.ContinuousCentroidalFwdData)

    x0 = space.neutral()
    x0[2] = 0.5
    force_z = -gravity[2] * mass / nk
    for k in range(nk):
        x0[11 + k * nk] = force_z
    u0 = np.zeros(nu)
    # Build derivatives of contact forces so that CoM moves up left forward
    dforce_x = 10
    dforce_y = 10
    dforce_z = 10
    for k in range(nk):
        u0[k * 3] = dforce_x
        u0[k * 3 + 1] = dforce_y
        u0[k * 3 + 2] = dforce_z

    # Integrate over several timesteps
    dt = 0.01
    N = 10
    for _ in range(N):
        ode.forward(x0, u0, data)
        x0 += dt * data.xdot

    # CoM has shifted up left forward
    assert x0[0] > 0
    assert x0[1] > 0
    assert x0[2] > 0


def test_continuous_centroidal_diff():
    # Test for 3D contact
    nk = 3
    force_size = 3
    nu = force_size * nk
    nx = 9 + nu
    space = manifolds.VectorSpace(nx)
    mass = 10.5
    gravity = np.array([0, 0, -9.81])
    contact_names = ["foot1", "foot2", "foot3"]
    contact_states = [True, True, False]
    contact_poses = [
        np.array([0, 0.1, 0]),
        np.array([0.1, -0.1, 0]),
        np.array([0.1, 0.2, 0]),
    ]
    contact_map = aligator.ContactMap(contact_names, contact_states, contact_poses)
    ode = dynamics.ContinuousCentroidalFwdDynamics(
        space, mass, gravity, contact_map, force_size
    )
    data = ode.createData()

    x0 = np.random.randn(nx)
    u0 = np.random.randn(nu)

    ode.forward(x0, u0, data)
    ode.dForward(x0, u0, data)

    Jx0 = data.Jx.copy()
    Ju0 = data.Ju.copy()
    Jxdiff, Judiff = finite_diff(ode, space, x0, u0, epsilon)

    assert np.allclose(Jxdiff, Jx0, epsilon), "err={}".format(infNorm(Jxdiff - Jx0))
    assert np.allclose(Judiff, Ju0, epsilon), "err={}".format(infNorm(Judiff - Ju0))

    # Test for 6D contact
    force_size = 6
    nu = force_size * nk
    nx = 9 + nu
    space = manifolds.VectorSpace(nx)

    ode = dynamics.ContinuousCentroidalFwdDynamics(
        space, mass, gravity, contact_map, force_size
    )
    data = ode.createData()

    x0 = np.random.randn(nx)
    u0 = np.random.randn(nu)

    ode.forward(x0, u0, data)
    ode.dForward(x0, u0, data)

    Jx0 = data.Jx.copy()
    Ju0 = data.Ju.copy()
    Jxdiff, Judiff = finite_diff(ode, space, x0, u0, epsilon)

    assert np.allclose(Jxdiff, Jx0, epsilon), "err={}".format(infNorm(Jxdiff - Jx0))
    assert np.allclose(Judiff, Ju0, epsilon), "err={}".format(infNorm(Judiff - Ju0))


@pytest.mark.skipif(
    not HAS_PINOCCHIO, reason="Aligator was compiled without Pinocchio."
)
def test_kinodynamics():
    import pinocchio as pin

    model = pin.buildSampleModelHumanoid()
    contact_ids = [
        model.getFrameId("lleg_effector_body"),
        model.getFrameId("rleg_effector_body"),
        model.getFrameId("rarm_effector_body"),
    ]

    nk = 3
    nu = 3 * nk + model.nv - 6
    space = manifolds.MultibodyPhaseSpace(model)
    mass = 0
    for inertia in model.inertias:
        mass += inertia.mass
    gravity = np.array([0, 0, -9.81])
    contact_states = [True, True, False]

    ode = dynamics.KinodynamicsFwdDynamics(
        space, model, gravity, contact_states, contact_ids, 3
    )
    data = ode.createData()

    assert isinstance(data, dynamics.KinodynamicsFwdData)

    x0 = space.neutral()
    u0 = np.random.randn(nu)

    ode.forward(x0, u0, data)
    ode.dForward(x0, u0, data)


@pytest.mark.skipif(
    not HAS_PINOCCHIO, reason="Aligator was compiled without Pinocchio."
)
def test_kinodynamics_diff():
    import pinocchio as pin

    model = pin.buildSampleModelHumanoid()
    contact_ids = [
        model.getFrameId("lleg_effector_body"),
        model.getFrameId("rleg_effector_body"),
        model.getFrameId("rarm_effector_body"),
    ]
    # Test for 3D contact
    force_size = 3
    nk = 3
    nu = force_size * nk + model.nv - 6
    space = manifolds.MultibodyPhaseSpace(model)
    mass = 0
    for inertia in model.inertias:
        mass += inertia.mass
    gravity = np.array([0, 0, -9.81])
    contact_states = [True, True, False]

    ode = dynamics.KinodynamicsFwdDynamics(
        space, model, gravity, contact_states, contact_ids, force_size
    )
    data = ode.createData()

    x0 = space.neutral()
    ndx = space.ndx
    dx = np.random.randn(ndx)
    x0 = space.integrate(x0, dx)
    u0 = np.random.randn(nu)
    epsilon = 1e-6

    ode.forward(x0, u0, data)
    ode.dForward(x0, u0, data)

    Jx0 = data.Jx.copy()
    Ju0 = data.Ju.copy()
    Jxdiff, Judiff = finite_diff(ode, space, x0, u0, epsilon)

    assert np.linalg.norm(Jxdiff - Jx0) <= epsilon
    assert np.linalg.norm(Judiff - Ju0) <= epsilon

    # Test for 6D contact
    force_size = 6
    nu = force_size * nk + model.nv - 6

    ode = dynamics.KinodynamicsFwdDynamics(
        space, model, gravity, contact_states, contact_ids, force_size
    )
    data = ode.createData()
    u0 = np.random.randn(nu)

    ode.forward(x0, u0, data)
    ode.dForward(x0, u0, data)

    Jx0 = data.Jx.copy()
    Ju0 = data.Ju.copy()
    Jxdiff, Judiff = finite_diff(ode, space, x0, u0, epsilon)

    assert np.linalg.norm(Jxdiff - Jx0) <= epsilon
    assert np.linalg.norm(Judiff - Ju0) <= epsilon


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
