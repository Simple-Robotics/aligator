import matplotlib.pyplot as plt
import numpy as np

from aligator import HistoryCallback, Results


def plot_convergence(cb: HistoryCallback, ax: plt.Axes, res: Results = None):
    from proxsuite_nlp.utils import plot_pd_errs

    prim_infeas = cb.prim_infeas.tolist()
    dual_infeas = cb.dual_infeas.tolist()
    if res is not None:
        prim_infeas.append(res.primal_infeas)
        dual_infeas.append(res.dual_infeas)
    plot_pd_errs(ax, prim_infeas, dual_infeas)
    ax.grid(axis="y", which="major")
    return


def plot_se2_pose(
    q: np.ndarray, ax: plt.Axes, alpha=0.5, fc="tab:blue"
) -> plt.Rectangle:
    from matplotlib import transforms

    w = 1.0
    h = 0.4
    center = (q[0] - 0.5 * w, q[1] - 0.5 * h)
    rect = plt.Rectangle(center, w, h, fc=fc, alpha=alpha)
    theta = np.arctan2(q[3], q[2])
    transform_ = transforms.Affine2D().rotate_around(*q[:2], -theta) + ax.transData
    rect.set_transform(transform_)
    ax.add_patch(rect)
    return rect


def _axes_flatten_if_ndarray(axes) -> list[plt.Axes]:
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    elif not isinstance(axes, list):
        axes = [axes]
    return axes


def plot_controls_traj(
    times,
    us,
    ncols=2,
    axes=None,
    effort_limit=None,
    joint_names=None,
    rmodel=None,
    figsize=(6.4, 6.4),
) -> tuple[plt.Figure, list[plt.Axes]]:
    t0 = times[0]
    tf = times[-1]
    us = np.asarray(us)
    nu = us.shape[1]
    nrows, r = divmod(nu, ncols)
    nrows += int(r > 0)
    if axes is None:
        fig, axes = plt.subplots(nrows, ncols, sharex="col", figsize=figsize)
    else:
        fig = axes.flat[0].get_figure()
    axes = _axes_flatten_if_ndarray(axes)

    if rmodel is not None:
        effort_limit = rmodel.effortLimit
        joint_names = rmodel.names

    for i in range(nu):
        ax: plt.Axes = axes[i]
        ax.step(times[:-1], us[:, i])
        if effort_limit is not None:
            ylim = ax.get_ylim()
            ax.hlines(-effort_limit[i], t0, tf, colors="k", linestyles="--")
            ax.hlines(+effort_limit[i], t0, tf, colors="r", linestyles="dashdot")
            ax.set_ylim(*ylim)
        if joint_names is not None:
            joint_name = joint_names[i].lower()
            ax.set_ylabel(joint_name)
    fig.supxlabel("Time $t$")
    fig.suptitle("Control trajectories")
    fig.tight_layout()
    return fig, axes


def plot_velocity_traj(
    times,
    vs,
    rmodel,
    axes=None,
    ncols=2,
    vel_limit=None,
    figsize=(6.4, 6.4),
) -> tuple[plt.Figure, list[plt.Axes]]:
    vs = np.asarray(vs)
    nv = vs.shape[1]
    idx_to_joint_id_map = {}
    jid = 0
    for i in range(nv):
        if i in rmodel.idx_vs.tolist():
            jid += 1
        idx_to_joint_id_map[i] = jid
    nrows, r = divmod(nv, ncols)
    nrows += int(r > 0)

    t0 = times[0]
    tf = times[-1]

    if axes is None:
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        fig: plt.Figure
    else:
        fig = axes.flat[0].get_figure()
    axes = _axes_flatten_if_ndarray(axes)

    for i in range(nv):
        ax: plt.Axes = axes[i]
        ax.plot(times, vs[:, i])
        jid = idx_to_joint_id_map[i]
        joint_name = rmodel.names[jid].lower()
        if vel_limit is not None:
            ylim = ax.get_ylim()
            ax.hlines(-vel_limit[i], t0, tf, colors="k", linestyles="--")
            ax.hlines(+vel_limit[i], t0, tf, colors="r", linestyles="dashdot")
            ax.set_ylim(*ylim)
        ax.set_ylabel(joint_name)

    fig.supxlabel("Time $t$")
    fig.suptitle("Velocity trajectories")
    fig.tight_layout()
    return fig, axes
