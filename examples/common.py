"""
Common definitions for examples.
"""
import tap
from typing import Literal

integrator_choices = Literal["euler", "semieuler", "midpoint", "rk2"]


class ArgsBase(tap.Tap):
    display: bool = False  # Display the trajectory using meshcat
    record: bool = False  # record video
    integrator: integrator_choices = "semieuler"
    """Numerical integrator to use"""
