import aligator
from aligator import manifolds

import numpy as np

space = manifolds.SE2()
nx = space.nx
ndx = space.ndx
nu = space.ndx


np.random.seed(42)  # of course
u0 = np.random.randn(nu)
x1 = space.neutral()


class TwistModelExplicit(aligator.dynamics.ExplicitDynamicsModel):
    def __init__(self, dt: float, B: np.ndarray = None):
        if B is None:
            B = np.eye(nu)
        self.B = B
        self.dt = dt
        super().__init__(space, nu)

    def __getinitargs__(self):
        return (self.dt, self.B)

    def forward(self, x, u, data: aligator.dynamics.ExplicitDynamicsData):
        assert data.good
        space.integrate(x, self.dt * self.B @ u, data.xnext)

    def dForward(self, x, u, data: aligator.dynamics.ExplicitDynamicsData):
        Jx = data.Jx
        Ju = data.Ju
        v_ = self.dt * self.B @ u
        dv_du = self.dt * self.B

        space.Jintegrate(x, v_, Jx, 0)
        Jxnext_dv = space.Jintegrate(x, v_, 1)
        Ju[:, :] = Jxnext_dv @ dv_du

    def createData(self):
        return TwistData(self)


class TwistData(aligator.dynamics.ExplicitDynamicsData):
    def __init__(self, model):
        super().__init__(model)
        self.good = True


class MyCostData(aligator.CostData):
    def __init__(self):
        super().__init__(space.ndx, nu)


def _call(space: manifolds.ManifoldAbstract, x_ref, x, W):
    err = space.difference(x_ref, x)
    return err.dot(W @ err)


def _jac(space: manifolds.ManifoldAbstract, x_ref, x, W):
    err = space.difference(x_ref, x)
    J = space.Jdifference(x_ref, x, 1)
    return J.T @ (W @ err)


def _hess(space: manifolds.ManifoldAbstract, x_ref, x, W):
    J = space.Jdifference(x_ref, x, 1)
    return J.T @ (W @ J)


class MyQuadCost(aligator.CostAbstract):
    def __init__(self, W: np.ndarray, x_ref: np.ndarray):
        self.x_ref = x_ref
        self.W = W
        super().__init__(space, nu)

    def __getinitargs__(self):
        return (self.W, self.x_ref)

    def __getstate__(self):
        return dict()

    def evaluate(self, x, u, data):
        assert isinstance(data, MyCostData)
        data.value = _call(space, self.x_ref, x, self.W)

    def computeGradients(self, x, u, data):
        assert isinstance(data, MyCostData)
        data.Lx[:] = _jac(space, self.x_ref, x, self.W)
        data.Lu[:] = 0.0

    def computeHessians(self, x, u, data):
        assert isinstance(data, MyCostData)
        data.hess[:, :] = 0.0
        data.Lxx[:, :] = _hess(space, self.x_ref, x, self.W)

    def createData(self):
        return MyCostData()


dt = 0.1
x0 = space.neutral()
dyn_model = TwistModelExplicit(dt)
cost = MyQuadCost(W=np.eye(space.ndx), x_ref=x1)
stage_model = aligator.StageModel(cost, dyn_model)
tol = 1e-5
mu_init = 1e-2
solver = aligator.SolverProxDDP(tol, mu_init)
