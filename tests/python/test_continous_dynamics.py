"""
Create continuous dynamics from Python.
"""
import pytest
import numpy as np
import aligator
from aligator import dynamics, manifolds
from pinocchio import Quaternion

space = manifolds.R3() * manifolds.SO3()
nu = 0


class MyODE(dynamics.ODEAbstract):
    def __init__(self):
        super().__init__(space, nu)

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
        return MyODEData()


class MyODEData(dynamics.ODEData):
    def __init__(self):
        super().__init__(space.ndx, nu)
        self.v0 = np.random.randn(3)


def test_abstract():
    """Test have the right types, etc."""
    space = manifolds.SO3()
    nu = 1
    dae = dynamics.ContinuousDynamicsBase(space, nu)
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
    x0 = space.rand()
    us = [np.zeros(0) for _ in range(10)]
    xs = aligator.rollout(itg, x0, us)
    print(xs.tolist())


def test_multibody_free():
    try:
        import pinocchio as pin

        model = pin.buildSampleModelHumanoid()
        space = manifolds.MultibodyPhaseSpace(model)
        nu = model.nv
        B = np.eye(nu)
        ode = dynamics.MultibodyFreeFwdDynamics(space, B)
        data = ode.createData()
        assert isinstance(data, dynamics.MultibodyFreeFwdData)
        assert hasattr(data, "tau")

        x0 = space.neutral()
        u0 = np.random.randn(nu)

        ode.forward(x0, u0, data)
        ode.dForward(x0, u0, data)
    except ImportError:
        pass


def test_centroidal():
    try:
        space = manifolds.VectorSpace(9)
        nk = 2
        nu = 3 * nk
        mass = 10.5
        gravity = np.array([0, 0, -9.81])
        contact_map = [(True, np.array([0, 0.1, 0])), (True, np.array([0.1, -0.1, 0]))]
        ode = dynamics.CentroidalFwdDynamics(space, nk, mass, gravity, contact_map)
        data = ode.createData()

        assert isinstance(data, dynamics.CentroidalFwdData)

        x0 = space.neutral()
        u0 = np.random.randn(nu)

        ode.forward(x0, u0, data)
        ode.dForward(x0, u0, data)
    except ImportError:
        pass


def test_centroidal_diff():
    nx = 9
    space = manifolds.VectorSpace(nx)
    nk = 2
    nu = 3 * nk
    mass = 10.5
    gravity = np.array([0, 0, -9.81])
    contact_map = [(True, np.array([0, 0.1, 0])), (True, np.array([0.1, -0.1, 0]))]
    ode = dynamics.CentroidalFwdDynamics(space, nk, mass, gravity, contact_map)
    data = ode.createData()

    x0 = np.random.randn(nx)
    u0 = np.random.randn(nu)
    epsilon = 1e-6

    ode.forward(x0, u0, data)
    ode.dForward(x0, u0, data)

    xdot0 = data.xdot.copy()
    Jx0 = data.Jx.copy()
    Ju0 = data.Ju.copy()
    Jxdiff = np.zeros((nx, nx))
    Judiff = np.zeros((nx, nu))

    for i in range(nx):
        evec = np.zeros(nx)
        evec[i] = epsilon
        xi = x0 + evec
        ode.forward(xi, u0, data)
        ode.dForward(xi, u0, data)
        Jxdiff[:, i] = (data.xdot - xdot0) / epsilon

    for i in range(nu):
        evec = np.zeros(nu)
        evec[i] = epsilon
        ui = u0 + evec
        ode.forward(x0, ui, data)
        ode.dForward(x0, ui, data)
        Judiff[:, i] = (data.xdot - xdot0) / epsilon

    assert np.linalg.norm(Jxdiff - Jx0) <= epsilon
    assert np.linalg.norm(Judiff - Ju0) <= epsilon


def test_continuous_centroidal():
    try:
        nk = 2
        nu = 3 * nk
        space = manifolds.VectorSpace(9 + nu)
        mass = 10.5
        gravity = np.array([0, 0, -9.81])
        contact_map = [(True, np.array([0, 0.1, 0])), (True, np.array([0.1, -0.1, 0]))]
        ode = dynamics.ContinuousCentroidalFwdDynamics(
            space, mass, gravity, contact_map
        )
        data = ode.createData()

        assert isinstance(data, dynamics.ContinuousCentroidalFwdData)

        x0 = space.neutral()
        u0 = np.random.randn(nu)

        ode.forward(x0, u0, data)
        ode.dForward(x0, u0, data)
    except ImportError:
        pass


def test_continuous_centroidal_diff():
    nk = 2
    nu = 3 * nk
    nx = 9 + nu
    space = manifolds.VectorSpace(nx)
    mass = 10.5
    gravity = np.array([0, 0, -9.81])
    contact_map = [(True, np.array([0, 0.1, 0])), (True, np.array([0.1, -0.1, 0]))]
    ode = dynamics.ContinuousCentroidalFwdDynamics(space, mass, gravity, contact_map)
    data = ode.createData()

    x0 = np.random.randn(nx)
    u0 = np.random.randn(nu)
    epsilon = 1e-6

    ode.forward(x0, u0, data)
    ode.dForward(x0, u0, data)

    xdot0 = data.xdot.copy()
    Jx0 = data.Jx.copy()
    Ju0 = data.Ju.copy()
    Jxdiff = np.zeros((nx, nx))
    Judiff = np.zeros((nx, nu))

    for i in range(nx):
        evec = np.zeros(nx)
        evec[i] = epsilon
        xi = x0 + evec
        ode.forward(xi, u0, data)
        ode.dForward(xi, u0, data)
        Jxdiff[:, i] = (data.xdot - xdot0) / epsilon

    for i in range(nu):
        evec = np.zeros(nu)
        evec[i] = epsilon
        ui = u0 + evec
        ode.forward(x0, ui, data)
        ode.dForward(x0, ui, data)
        Judiff[:, i] = (data.xdot - xdot0) / epsilon

    assert np.linalg.norm(Jxdiff - Jx0) <= epsilon
    assert np.linalg.norm(Judiff - Ju0) <= epsilon


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
