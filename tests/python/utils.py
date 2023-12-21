import numpy as np

from aligator import manifolds, dynamics


def create_multibody_ode(humanoid=True):
    try:
        import pinocchio as pin

        if humanoid:
            model = pin.buildSampleModelHumanoid()
        else:
            model = pin.buildSampleModelManipulator()
        space = manifolds.MultibodyPhaseSpace(model)
        nu = model.nv
        B = np.eye(nu)
        ode = dynamics.MultibodyFreeFwdDynamics(space, B)
        data = ode.createData()
        assert isinstance(data, dynamics.MultibodyFreeFwdData)
        return ode
    except ImportError:
        return None


def create_linear_ode(nx, nu):
    n = min(nx, nu)
    A = np.eye(nx)
    A[1, 0] = 0.1
    B = np.zeros((nx, nu))
    B[range(n), range(n)] = 1.0
    B[0, 1] = 0.5
    c = np.zeros(nx)
    ode = dynamics.LinearODE(A, B, c)
    cd = ode.createData()
    assert np.allclose(ode.A, cd.Jx)
    assert np.allclose(ode.B, cd.Ju)
    return ode
