import numpy as np
import proxddp
from proxnlp import manifolds


N = 2
nu = 1
A = np.random.randn(N, N)
B = np.random.randn(N, nu)
c = np.random.randn(N)

print(A)
print(B)
print(c)
space = manifolds.VectorSpace(N)

ldd = proxddp.dynamics.LinearDiscreteDynamics(A, B, c)
ldd2 = proxddp.dynamics.LinearDiscreteDynamics(A, B, c, space)

print("Space:", space)
x0 = space.neutral()
x1 = space.rand()
u0 = np.random.randn(nu)
lddata = ldd.createData()
ldd.evaluate(x0, u0, x1, lddata)

print("Data 1: {}".format(lddata.value))

data2 = ldd2.createData()
ldd2.evaluate(x0, u0, x1, data2)
print("Data 2: {}".format(data2.value))

assert np.array_equal(lddata.value, data2.value)
