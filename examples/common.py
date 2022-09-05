"""
Common definitions for examples.
"""
import tap
import numpy as np
import pinocchio as pin
from typing import Literal

integrator_choices = Literal["euler", "semieuler", "midpoint", "rk2"]


class ArgsBase(tap.Tap):
    display: bool = False  # Display the trajectory using meshcat
    record: bool = False  # record video
    integrator: integrator_choices = "semieuler"
    """Numerical integrator to use"""


def get_endpoint(rmodel, rdata, q: np.ndarray, tool_id: int):
    pin.framesForwardKinematics(rmodel, rdata, q)
    return rdata.oMf[tool_id].translation.copy()


def get_endpoint_traj(rmodel, rdata, xs: list[np.ndarray], tool_id: int):
    pts = []
    for i in range(len(xs)):
        pts.append(get_endpoint(rmodel, rdata, xs[i][: rmodel.nq], tool_id))
    return np.array(pts)
