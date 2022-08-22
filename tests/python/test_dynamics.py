import numpy as np
from proxddp import dynamics, manifolds

import sys
import pytest


def test_abstract():
    nx = 3
    space = manifolds.VectorSpace(nx)
    nu = 2
    dm = dynamics.DynamicsModel(space, nu)
    dd = dm.createData()
    print(dd)

    em = dynamics.ExplicitDynamicsModel(space, nu)
    ed = em.createData()
    print(ed)


def test_linear():
    N = 2
    nu = 1
    A = np.random.randn(N, N)
    B = np.random.randn(N, nu)
    c = np.random.randn(N)

    print(A)
    print(B)
    print(c)

    ldd = dynamics.LinearDiscreteDynamics(A, B, c)
    space = ldd.space

    x0 = space.neutral()
    x1 = space.rand()
    u0 = np.random.randn(nu)
    lddata = ldd.createData()
    print(lddata)
    ldd.evaluate(x0, u0, x1, lddata)

    print("Data: {}".format(lddata.value))


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
