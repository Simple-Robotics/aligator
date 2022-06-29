import proxddp
import numpy as np
from proxddp import manifolds, dynamics


def test_aba_field():
    space = manifolds.SE3()
    nu = space.ndx

    class GravityDynamics(dynamics.ODEBase):
        def __init__(self, space, nu):
            dynamics.ContinuousDynamicsBase.__init__(space, nu)

        def forward(self, x, u, data: dynamics.ODEData):
            data.xdot

        def Jforward(self, x, u, data: dynamics.ODEData):
            data.Jx
            data.Ju
