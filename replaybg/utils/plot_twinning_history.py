import numpy as np
import matplotlib.pyplot as plt

from utils.plot_common import (
    KIND_DENSE,
    finalize,
    make_figure,
    series,
    set_panel_title,
)


def plot_twinning_history(
    history: dict,
    param_names: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    hover: bool = True,
) -> plt.Figure:
    """Plots the optimization history from a twinning run.

    The figure follows the same rationale as :func:`plot_replay` and
    :func:`plot_twinning`: a vertical stack of subplots sharing a common x axis,
    here the function evaluation rather than time.

    1. **Parameter trajectory**: each unknown parameter's value per evaluation.
    2. **Log-prior**, **log-likelihood**, **log-posterior**: the three terms the
       twinner is optimising, one per subplot.

    Parameters
    ----------
    history : dict
        Dict with keys ``theta``, ``log_prior``, ``log_likelihood`` and
        ``log_posterior``. Each value is a list of recorded values per function
        evaluation. ``theta`` entries are numpy arrays.
    param_names : list of str or None, optional, default : None
        Optional list of parameter names for the theta legend. Defaults to
        ``theta_0``, ``theta_1``, ...
    figsize : tuple of float or None, optional, default : None
        Optional figure size. Defaults to a height that grows with the number of
        subplots.
    hover : bool, optional, default : True
        Whether to attach the interactive readout: hovering any subplot drops a
        vertical cursor across the whole stack and reports the values at that
        evaluation. Ignored on non-interactive backends.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled figure.
    """
    thetas = np.array(history['theta'])          # (n_evals, n_params)
    log_prior = np.array(history['log_prior'])
    log_likelihood = np.array(history['log_likelihood'])
    log_posterior = np.array(history['log_posterior'])

    n_evals = len(log_posterior)
    steps = np.arange(n_evals)

    if thetas.ndim == 1:
        thetas = thetas[:, np.newaxis]

    n_params = thetas.shape[1]
    if param_names is None:
        param_names = [f'theta_{i}' for i in range(n_params)]

    fig, axes = make_figure(4, figsize or (10, 10))
    hover_series = {}

    # --- theta ---
    ax = axes[0]
    hover_series[ax] = []
    for i in range(n_params):
        mask = np.isfinite(thetas[:, i])
        ax.plot(steps[mask], thetas[mask, i], label=param_names[i], linewidth=0.8)
        hover_series[ax].append(series(param_names[i], steps[mask], thetas[mask, i], KIND_DENSE))
    ax.set_ylabel('Parameter value')
    set_panel_title(ax, 'Parameter trajectory')
    if n_params <= 10:
        ax.legend(fontsize='small', ncol=min(n_params, 4), framealpha=0.9)

    # --- the three optimisation terms ---
    terms = [
        ('Log-prior', log_prior, 'tab:blue'),
        ('Log-likelihood', log_likelihood, 'tab:orange'),
        ('Log-posterior', log_posterior, 'tab:green'),
    ]
    for ax, (label, values, color) in zip(axes[1:], terms):
        mask = np.isfinite(values)
        ax.plot(steps[mask], values[mask], color=color, linewidth=0.8)
        hover_series[ax] = [series(label, steps[mask], values[mask], KIND_DENSE)]
        ax.set_ylabel(label)
        set_panel_title(ax, label)

    finalize(fig, axes, 'Function evaluation', hover_series, hover, x_fmt='eval = {:.0f}')
    return fig
