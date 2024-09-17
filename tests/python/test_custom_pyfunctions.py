import aligator
import numpy as np
import pytest


class CustomFunction(aligator.StageFunction):
    def __init__(self, space: aligator.manifolds.ManifoldAbstract, nu):
        self.space = space
        ndx = space.ndx
        super().__init__(ndx, nu, ndx, ndx)

    def __getinitargs__(self):
        return (self.space, self.nu)

    def evaluate(self, x, u, y, data: aligator.StageFunctionData):
        data.value[:] = self.space.difference(x, self.space.neutral())

    def computeJacobians(self, x, u, y, data: aligator.StageFunctionData):
        data.Jx[:] = self.space.Jdifference(x, self.space.neutral(), 0)


class TwistModelExplicit(aligator.dynamics.ExplicitDynamicsModel):
    def __init__(self, space, nu, dt: float):
        B = np.eye(nu)
        self.B = B
        self.dt = dt
        super().__init__(space, nu)

    def __getinitargs__(self):
        return (self.space, self.nu, self.dt)

    def forward(self, x, u, data: aligator.dynamics.ExplicitDynamicsData):
        self.space.integrate(x, self.dt * self.B @ u, data.xnext)

    def dForward(self, x, u, data: aligator.dynamics.ExplicitDynamicsData):
        Jx = data.Jx
        Ju = data.Ju
        v_ = self.dt * self.B @ u
        dv_du = self.dt * self.B

        self.space.Jintegrate(x, v_, Jx, 0)
        Jxnext_dv = self.space.Jintegrate(x, v_, 1)
        Ju[:, :] = Jxnext_dv @ dv_du


def test_abstract():
    space = aligator.manifolds.SE2()
    ndx = space.ndx
    nu = 3
    nr = 1
    fun = aligator.StageFunction(ndx, nu, nr)
    data = fun.createData()
    print(data)


def test_custom_controlbox():
    space = aligator.manifolds.SE2()
    ndx = space.ndx
    nu = 3

    fun = CustomFunction(space, nu)
    data1: aligator.StageFunctionData = fun.createData()

    lbd0 = np.zeros(fun.nr)
    x0 = space.rand()
    u0 = np.random.randn(nu)

    fun.evaluate(x0, u0, x0, data1)
    fun.computeJacobians(x0, u0, x0, data1)
    print(data1.value)
    print(data1.Ju)

    # expected behavior: initial value of vhp_buffer is 0
    assert np.allclose(data1.vhp_buffer, 0.0)

    rdm = np.random.randn(*data1.vhp_buffer.shape)
    data1.vhp_buffer[:, :] = rdm
    fun.computeVectorHessianProducts(x0, u0, x0, lbd0, data1)
    # expected behavior: unimplemented computeVectorHessianProducts does nothing.
    assert np.allclose(data1.vhp_buffer, rdm)

    cost = aligator.QuadraticStateCost(space, nu, space.neutral(), np.eye(ndx))
    dynamics = TwistModelExplicit(space, nu, 0.1)
    stage = aligator.StageModel(cost, dynamics)
    stage.addConstraint(fun, aligator.constraints.EqualityConstraintSet())
    data = stage.createData()
    stage.evaluate(x0, u0, x0, data)

    stages = [stage, stage, stage]
    prob = aligator.TrajOptProblem(x0, stages, cost)
    pd = aligator.TrajOptData(prob)
    print(pd)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
