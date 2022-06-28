"""
Some custom functions for the examples.
"""
import proxddp
import numpy as np


class ControlBoxFunction(proxddp.StageFunction):
    def __init__(self, ndx, nu, u_min, u_max) -> None:
        super().__init__(ndx, nu, 2 * nu)
        self.u_min = u_min
        self.u_max = u_max

    def evaluate(self, x, u, y, data):
        data.value[:] = np.concatenate([self.u_min - u, u - self.u_max])

    def computeJacobians(self, x, u, y, data):
        data.Ju[:, :] = np.block([[-np.eye(self.nu)], [np.eye(self.nu)]])
