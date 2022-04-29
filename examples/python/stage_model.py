from proxnlp import manifolds
import proxddp
import numpy as np

space = manifolds.SE2()
nu = space.ndx


class TwistModel(proxddp.DynamicsModel):
    def __init__(self, B: np.ndarray = None):
        if B is None:
            B = np.eye(nu)
        proxddp.DynamicsModel.__init__(self, space.ndx, nu)
        self.B = B

    def evaluate(self, x, u, y, data: proxddp.FunctionData):
        data.value[:] = space.difference(
            space.integrate(x, self.B @ u),
            y)

    def computeJacobians(self, x, u, y, data: proxddp.FunctionData):
        v_ = self.B @ u

        xnext = space.integrate(x, v_)

        Jv_u = self.B
        Jxnext_x = space.Jintegrate(x, v_, 0)
        Jxnext_v = space.Jintegrate(x, v_, 1)

        # res = space.difference(xnext, y)

        Jres_xnext = np.eye(space.ndx)
        Jres_y = np.eye(space.ndx)
        Jres_xnext = space.Jdifference(xnext, y, 0)
        Jres_y = space.Jdifference(xnext, y, 1)

        data.Jx[:, :] = Jres_xnext @ Jxnext_x
        data.Ju[:, :] = Jres_xnext @ Jxnext_v @ Jv_u
        data.Jy[:, :] = Jres_y


dynmodel = TwistModel()
data = dynmodel.createData()

x0 = space.rand()
u0 = np.random.randn(nu)
x1 = space.rand()

dynmodel.evaluate(x0, u0, x1, data)
dynmodel.computeJacobians(x0, u0, x1, data)
