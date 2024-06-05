import numpy as np

from aligator import dynamics, manifolds
from utils import configure_functions

import sys
import pytest


def test_abstract():
    nx = 3
    space = manifolds.VectorSpace(nx)
    nu = 2
    dm = dynamics.DynamicsModel(space, nu)
    common_containers_dm = configure_functions([dm])
    dd = dm.createData(common_containers_dm.datas)
    print(dd)

    em = dynamics.ExplicitDynamicsModel(space, nu)
    common_containers_em = configure_functions([dm])
    ed = em.createData(common_containers_em.datas)
    print(ed)


def _test_direct_sum(f, g):
    dm = dynamics.directSum(f, g)
    common_containers = configure_functions([dm])
    dd = dm.createData(common_containers.datas)
    print(dd)
    space = dm.space_next
    print(space)
    assert space.num_components == 2

    x0 = space.rand()
    u0 = np.random.randn(dm.nu)
    common_containers.evaluate(x0, u0)
    dm.forward(x0, u0, dd)
    print(dd.xnext)

    dd1 = f.createData(common_containers.datas)
    dd2 = g.createData(common_containers.datas)
    x01, x02 = space.split(x0).tolist()
    u01, u02 = u0[: f.nu], u0[f.nu :]
    f.forward(x01, u01, dd1)
    g.forward(x02, u02, dd2)
    assert np.allclose(dd1.xnext, dd.data1.xnext)
    assert np.allclose(dd2.xnext, dd.data2.xnext)

    common_containers.compute_gradients(x0, u0)
    dm.dForward(x0, u0, dd)
    f.dForward(x01, u01, dd1)
    g.dForward(x02, u02, dd2)
    print(dd.Jx)
    print(dd.data1.Jx)
    print(dd.data2.Jx)
    assert np.allclose(dd1.Jx, dd.data1.Jx)
    assert np.allclose(dd1.Ju, dd.data1.Ju)

    assert np.allclose(dd2.Jx, dd.data2.Jx)
    assert np.allclose(dd2.Ju, dd.data2.Ju)


def test_mb_direct_sum():
    from utils import create_multibody_ode

    ode = create_multibody_ode(humanoid=False)
    dm1 = dynamics.IntegratorEuler(ode, 0.01)

    nx2 = 2
    nu2 = 3
    A = np.random.randn(nx2, nx2)
    B = np.random.randn(nx2, nu2)
    dm2 = dynamics.LinearDiscreteDynamics(A, B, np.random.randn(nx2))

    _test_direct_sum(dm1, dm2)


class TestLinear:
    N = 2
    nu = 1
    A = np.random.randn(N, N)
    B = np.random.randn(N, nu)
    c = np.random.randn(N)
    ldd = dynamics.LinearDiscreteDynamics(A, B, c)

    def test_linear(self):
        space = self.ldd.space

        x0 = space.neutral()
        x1 = space.rand()
        u0 = np.random.randn(self.nu)
        common_containers = configure_functions([self.ldd])
        lddata = self.ldd.createData(common_containers.datas)
        print(lddata)
        common_containers.evaluate(x0, u0)
        self.ldd.forward(x0, u0, lddata)
        self.ldd.evaluate(x0, u0, x1, lddata)

        print("Data: {}".format(lddata.value))

    def test_direct_sum(self):
        _test_direct_sum(self.ldd, self.ldd)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
