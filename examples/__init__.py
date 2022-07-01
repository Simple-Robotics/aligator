"""
Utils for examples.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms


def plot_se2_pose(q: np.ndarray, ax: plt.Axes, alpha=0.5, fc="tab:blue"):
    w = 1.0
    h = 0.4
    center = (q[0] - 0.5 * w, q[1] - 0.5 * h)
    rect = plt.Rectangle(center, w, h, fc=fc, alpha=alpha)
    theta = np.arctan2(q[3], q[2])
    transform_ = transforms.Affine2D().rotate_around(*q[:2], -theta) + ax.transData
    rect.set_transform(transform_)
    ax.add_patch(rect)
    return rect
