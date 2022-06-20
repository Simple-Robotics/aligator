import numpy as np
import proxddp


def test_linear():
    N = 2
    nu = 1
    A = np.random.randn(N, N)
    B = np.random.randn(N, nu)
    c = np.random.randn(N)

    print(A)
    print(B)
    print(c)

    ldd = proxddp.dynamics.LinearDiscreteDynamics(A, B, c)
    space = ldd.space

    print("Space:", space)
    x0 = space.neutral()
    x1 = space.rand()
    u0 = np.random.randn(nu)
    lddata = ldd.createData()
    ldd.evaluate(x0, u0, x1, lddata)

    print("Data: {}".format(lddata.value))
