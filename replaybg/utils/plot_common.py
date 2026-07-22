"""Shared building blocks for the ReplayBG plotting utilities.

:func:`~utils.plot_replay.plot_replay`, :func:`~utils.plot_twinning.plot_twinning`
and :func:`~utils.plot_twinning_history.plot_twinning_history` all render the same
kind of figure: a vertical stack of subplots sharing one x axis. This module holds
what they have in common — the channel colour map, the input-group layout, how a
channel is drawn, the visual style and the hover readout — so the three figures
look and behave alike and each behaviour is defined in exactly one place.

Every helper is domain-agnostic: nothing here assumes what an output, an input or
a parameter means.
"""

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# Fraction of steps a channel may be non-zero on and still read as impulses.
# Above it the channel is treated as continuous and drawn as a step line, since
# stemming a channel that fires nearly everywhere paints a solid block.
IMPULSE_MAX_DUTY = 0.2

# Series kinds understood by the hover readout. Dense series carry a value at
# every x, so the readout interpolates to the nearest sample unconditionally.
# Sparse series only carry a value where they fired, so the readout reports them
# only when the cursor is genuinely near one (see HOVER_SNAP_FRACTION).
KIND_DENSE = "dense"
KIND_SPARSE = "sparse"

# How close (as a fraction of the x range) the cursor must be to a sparse sample
# for the readout to report it.
HOVER_SNAP_FRACTION = 0.02

# Reference-level styling shared by the output and fit subplots.
_THRESHOLD_COLOR = "0.45"
_BAND_COLOR = "tab:green"

# Styling for the separators drawn between glued segments (see the interval
# plotters). Kept above the grid but below the data, so it delimits without
# competing with the traces.
_BOUNDARY_COLOR = "0.55"


def series(label: str, x, y, kind: str = KIND_DENSE) -> dict:
    """Returns a hover-readout descriptor for one plotted series.

    Collecting these alongside the drawing calls lets :func:`attach_hover` report
    values without having to introspect artists, which keeps the readout correct
    for stems (whose artists hold only the non-zero steps).

    Parameters
    ----------
    label : str
        The name shown in the readout.
    x : array_like
        The series x values.
    y : array_like
        The series y values, parallel to ``x``.
    kind : str, optional, default : KIND_DENSE
        ``KIND_DENSE`` when the series has a value at every x, ``KIND_SPARSE``
        when it only carries values where it fired.

    Returns
    -------
    dict
        The descriptor consumed by :func:`attach_hover`.
    """
    return {"label": label, "x": np.asarray(x), "y": np.asarray(y), "kind": kind}


def make_figure(n_rows: int, figsize: tuple[float, float] | None = None):
    """Returns a figure and its axes, styled and sized for ``n_rows`` stacked panels.

    Parameters
    ----------
    n_rows : int
        The number of vertically stacked subplots.
    figsize : tuple of float or None, optional, default : None
        Optional figure size. Defaults to a height that grows with ``n_rows``.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, numpy.ndarray)
        The figure and its axes as a 1-D array, one entry per row.
    """
    if figsize is None:
        figsize = (10, 2.2 * n_rows)
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes)
    for ax in axes:
        style_axes(ax)
    return fig, axes


def style_axes(ax) -> None:
    """Applies the shared panel style to one axes.

    Draws a light grid beneath the data and drops the top and right spines, so
    the data reads before the frame does.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to style.
    """
    ax.grid(True, which="major", linestyle="-", linewidth=0.5, color="0.9")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("0.6")
    ax.tick_params(labelsize="small", color="0.6")


def set_panel_title(ax, title: str) -> None:
    """Sets a left-aligned panel title in the shared style.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to title.
    title : str
        The title text.
    """
    ax.set_title(title, loc="left", fontsize="medium", pad=6)


def resolve_input_layout(data_to_input: dict,
                         input_groups: list[list[int]] | None,
                         mask_inputs: list[int] | None):
    """Returns the input subplot groups and a stable colour per channel.

    Colours are keyed by the channel's global index, so a channel reads the same
    wherever it appears and keeps its colour when other channels are masked out.

    Parameters
    ----------
    data_to_input : dict
        Mapping from channel index to channel name.
    input_groups : list of list of int or None
        Optional list of channel-index groups, each becoming one subplot with its
        channels overlaid. ``None`` puts every channel on its own subplot, in
        ``data_to_input`` order.
    mask_inputs : list of int or None
        Optional list of channel indices to hide. Masked channels are dropped
        from the groups, and a group left empty by masking gets no subplot.

    Returns
    -------
    tuple of (list of list of int, dict)
        The resolved groups and the index -> colour mapping.
    """
    all_idxs = sorted(data_to_input.keys())
    cmap = plt.get_cmap("tab20" if len(all_idxs) > 10 else "tab10")
    colors = {idx: mcolors.to_hex(cmap(i % cmap.N)) for i, idx in enumerate(all_idxs)}

    masked = set(mask_inputs or [])
    if input_groups is None:
        groups = [[idx] for idx in all_idxs if idx not in masked]
    else:
        groups = [[idx for idx in group if idx not in masked] for group in input_groups]
        groups = [group for group in groups if group]
    return groups, colors


def draw_input_group(ax, t, inputs, group: list[int],
                     data_to_input: dict, colors: dict) -> list[dict]:
    """Draws one group of input channels onto a subplot.

    Impulse channels (boluses, meals) fire on a few steps and read best as stems.
    Continuous channels (e.g. basal) are non-zero nearly everywhere, and stemming
    them paints a solid block, so they are drawn as a step line instead. The
    channel's own sparsity decides which, keeping the utility agnostic to what
    the channel means.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    t : numpy.ndarray
        The time axis, one entry per integration step.
    inputs : numpy.ndarray
        The input matrix, shape ``(tsteps, n_channels)``.
    group : list of int
        The channel indices to overlay on this subplot.
    data_to_input : dict
        Mapping from channel index to channel name.
    colors : dict
        Mapping from channel index to colour.

    Returns
    -------
    list of dict
        Hover descriptors for the channels drawn, as built by :func:`series`.
    """
    drawn = []
    for idx in group:
        name = data_to_input.get(idx, f"input_{idx}")
        color = colors[idx]
        channel = inputs[:, idx]
        nz = np.flatnonzero(channel)
        if not nz.size:
            # Keep the channel in the legend even when it never fires.
            ax.plot([], [], color=color, label=name)
        elif nz.size <= IMPULSE_MAX_DUTY * len(t):
            ax.stem(t[nz], channel[nz],
                    linefmt=color, markerfmt="none", basefmt=" ", label=name)
            drawn.append(series(name, t[nz], channel[nz], KIND_SPARSE))
        else:
            ax.step(t, channel, where="post", color=color, linewidth=1.2, label=name)
            drawn.append(series(name, t, channel, KIND_DENSE))

    ax.set_ylabel("Input")
    set_panel_title(ax, ", ".join(data_to_input.get(i, f"input_{i}") for i in group))
    if len(group) > 1:
        ax.legend(fontsize="small", loc="upper right", framealpha=0.9)
    return drawn


def draw_thresholds(ax, thresholds: list[float] | None) -> None:
    """Draws reference levels as dashed lines, shading the span between them.

    With two or more levels the region between the lowest and the highest is
    shaded, so the band a trace is meant to stay inside reads at a glance.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    thresholds : list of float or None
        The reference levels. ``None`` or empty draws nothing.
    """
    levels = sorted(thresholds or [])
    if len(levels) >= 2:
        ax.axhspan(levels[0], levels[-1], color=_BAND_COLOR, alpha=0.06, zorder=0)
    for level in levels:
        ax.axhline(level, linestyle="--", color=_THRESHOLD_COLOR, linewidth=0.8, zorder=1)


def segment_offsets(tsteps_list: list[int], ts_min: float = 1.0):
    """Returns the per-segment time offsets and the inner boundary times.

    Gluing several segments onto one time axis means laying them end to end: the
    second segment starts where the first ends, and so on. This computes, in the
    x units of the axis, where each segment begins and where the joins fall, so
    the interval plotters offset every segment the same way and draw the same
    separators.

    Parameters
    ----------
    tsteps_list : list of int
        The number of integration steps in each segment, in draw order.
    ts_min : float, optional, default : 1.0
        Minutes represented by one integration step, used to scale the offsets.

    Returns
    -------
    tuple of (list of float, list of float)
        The start time of each segment (one per segment, first is ``0``) and the
        times of the inner boundaries (the joins between consecutive segments;
        empty for a single segment).
    """
    starts = [0.0]
    for n in tsteps_list[:-1]:
        starts.append(starts[-1] + n * ts_min)
    boundaries = starts[1:]
    return starts, boundaries


def draw_segment_boundaries(ax, boundary_times: list[float] | None) -> None:
    """Draws vertical separators where glued segments meet.

    Each boundary is a thin dotted line spanning the panel, marking the join
    between two segments so the stack reads as glued-but-delimited without
    hiding the trace that crosses it.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    boundary_times : list of float or None
        The x positions of the joins, as returned by :func:`segment_offsets`.
        ``None`` or empty draws nothing.
    """
    for x in boundary_times or []:
        ax.axvline(x, linestyle=":", color=_BOUNDARY_COLOR, linewidth=0.9, zorder=2)


def finalize(fig, axes, xlabel: str, hover_series: dict | None = None,
             hover: bool = True, x_fmt: str = "{:g}") -> None:
    """Applies the shared finishing touches to a stacked figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to finish.
    axes : numpy.ndarray
        The figure's axes, one per row.
    xlabel : str
        The label for the shared x axis, set on the bottom panel.
    hover_series : dict or None, optional, default : None
        Mapping from axes to the list of :func:`series` descriptors drawn on it,
        used for the hover readout. ``None`` disables the readout.
    hover : bool, optional, default : True
        Whether to attach the hover readout.
    x_fmt : str, optional, default : "{:g}"
        Format string for the x value shown in the readout.
    """
    axes[-1].set_xlabel(xlabel)
    fig.tight_layout()
    if hover and hover_series:
        attach_hover(fig, axes, hover_series, x_fmt=x_fmt)


def attach_hover(fig, axes, hover_series: dict, x_fmt: str = "{:g}"):
    """Attaches a synchronised crosshair and value readout to a stacked figure.

    Moving the cursor over any panel drops a vertical line at that x on *every*
    panel, so values can be compared down the stack at one instant, and shows a
    readout of the series on the hovered panel at that x.

    The readout is a no-op on non-interactive backends (e.g. ``Agg``), where no
    mouse events are ever delivered, so figures still save unchanged.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to attach to.
    axes : numpy.ndarray
        The figure's axes, one per row.
    hover_series : dict
        Mapping from axes to the list of :func:`series` descriptors drawn on it.
    x_fmt : str, optional, default : "{:g}"
        Format string for the x value shown in the readout.

    Returns
    -------
    _Crosshair
        The live crosshair. It is also stored on the figure, which keeps it alive
        for as long as the figure is.
    """
    crosshair = _Crosshair(fig, axes, hover_series, x_fmt=x_fmt)
    # matplotlib holds only a weak reference to callbacks, so the crosshair has to
    # outlive this call by riding on the figure or the readout goes dead.
    fig._rbg_crosshair = crosshair
    return crosshair


class _Crosshair:
    """A synchronised vertical cursor with a per-panel value readout.

    ...
    Attributes
    ----------
    fig : matplotlib.figure.Figure
        The figure the crosshair is attached to.
    axes : list of matplotlib.axes.Axes
        The panels the cursor spans.
    """

    def __init__(self, fig, axes, hover_series: dict, x_fmt: str = "{:g}"):
        """Constructs all the necessary attributes for the _Crosshair object.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to attach to.
        axes : numpy.ndarray
            The figure's axes, one per row.
        hover_series : dict
            Mapping from axes to the list of :func:`series` descriptors drawn.
        x_fmt : str, optional, default : "{:g}"
            Format string for the x value shown in the readout.
        """
        self.fig = fig
        self.axes = list(axes)
        self._series = hover_series
        self._x_fmt = x_fmt
        self._last_pos = None

        # Adding artists must not disturb the limits the data established.
        xlim, ylims = self.axes[0].get_xlim(), [ax.get_ylim() for ax in self.axes]
        self._vlines = [
            ax.axvline(xlim[0], color="0.35", linewidth=0.8, alpha=0.9,
                       visible=False, zorder=5)
            for ax in self.axes
        ]
        self._annots = {
            ax: ax.annotate(
                "", xy=(0, 0), xycoords="data",
                xytext=(10, 10), textcoords="offset points",
                fontsize="x-small", visible=False, zorder=6,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="0.7", alpha=0.95),
            )
            for ax in self.axes
        }
        self.axes[0].set_xlim(xlim)
        for ax, ylim in zip(self.axes, ylims):
            ax.set_ylim(ylim)

        self._cids = [
            fig.canvas.mpl_connect("motion_notify_event", self._on_move),
            fig.canvas.mpl_connect("figure_leave_event", self._on_leave),
        ]

    def _on_move(self, event) -> None:
        """Redraws the cursor and readout for the pointer position in ``event``."""
        if event.inaxes not in self._series or event.xdata is None:
            self._hide()
            return

        x = event.xdata
        # Redrawing only on a real move keeps the readout cheap on dense traces.
        # The panel is part of the key: crossing into another one at the same x
        # still has to move the readout there.
        if self._last_pos == (x, event.inaxes):
            return
        self._last_pos = (x, event.inaxes)

        for vline in self._vlines:
            vline.set_xdata([x, x])
            vline.set_visible(True)
        for ax, annot in self._annots.items():
            annot.set_visible(False)

        text = self._readout(event.inaxes, x)
        if text:
            annot = self._annots[event.inaxes]
            annot.set_text(text)
            annot.xy = (x, event.ydata)
            # Flip the box back over the cursor near the right edge so it stays
            # inside the panel instead of being clipped away.
            x_lo, x_hi = event.inaxes.get_xlim()
            past_middle = x > (x_lo + x_hi) / 2
            annot.set_x(-10 if past_middle else 10)
            annot.set_horizontalalignment("right" if past_middle else "left")
            annot.set_visible(True)

        self.fig.canvas.draw_idle()

    def _readout(self, ax, x: float) -> str:
        """Returns the readout text for ``ax`` at ``x``, or ``""`` when there is none."""
        x_lo, x_hi = ax.get_xlim()
        tol = abs(x_hi - x_lo) * HOVER_SNAP_FRACTION

        lines = [self._x_fmt.format(x)]
        for s in self._series[ax]:
            if not s["x"].size:
                continue
            i = int(np.argmin(np.abs(s["x"] - x)))
            # A sparse series has no value between its samples, so report it only
            # when the cursor is actually near one rather than inventing a hold.
            if s["kind"] == KIND_SPARSE and abs(s["x"][i] - x) > tol:
                continue
            lines.append(f"{s['label']}: {s['y'][i]:.4g}")
        return "\n".join(lines) if len(lines) > 1 else ""

    def _on_leave(self, event) -> None:
        """Hides the cursor when the pointer leaves the figure."""
        self._hide()

    def _hide(self) -> None:
        """Hides the cursor and every readout, redrawing only if something changed."""
        if not any(v.get_visible() for v in self._vlines):
            return
        self._last_pos = None
        for vline in self._vlines:
            vline.set_visible(False)
        for annot in self._annots.values():
            annot.set_visible(False)
        self.fig.canvas.draw_idle()
