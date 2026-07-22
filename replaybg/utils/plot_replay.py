import numpy as np
import matplotlib.pyplot as plt

from utils.plot_common import (
    KIND_DENSE,
    KIND_SPARSE,
    draw_input_group,
    draw_segment_boundaries,
    draw_thresholds,
    finalize,
    make_figure,
    resolve_input_layout,
    segment_offsets,
    series,
    set_panel_title,
)


def plot_replay(
    replay_results: dict,
    thresholds: list[float] | None = None,
    input_groups: list[list[int]] | None = None,
    mask_inputs: list[int] | None = None,
    action_field: str | None = None,
    ts_min: float = 1.0,
    output_label: str = "Output",
    measurement_label: str = "Measurement",
    figsize: tuple[float, float] | None = None,
    hover: bool = True,
) -> plt.Figure:
    """Plot the outcome of a ``ReplayBG.replay()`` run.

    The figure is a vertical stack of subplots sharing a common time axis:

    1. **Output** (first subplot): the replayed ``output`` as a continuous line
       and, when a sensor was used, the replayed ``measurement`` as markers.
       Each value in ``thresholds`` is drawn as a dashed horizontal line
       spanning the whole width, acting as a visual reference level, and the span
       between the outermost levels is shaded.
    2. **Inputs** (one subplot per input *group*): the applied ``input``
       channels, labelled via ``data_to_input``. By default every channel gets
       its own subplot; pass ``input_groups`` to overlay channels together.
    3. **Callback actions** (one subplot per distinct callback): the values a
       callback logged, stemmed at the integration minute of each action.

    The utility is model-agnostic: nothing about the meaning of the output,
    inputs, or actions is assumed. Channel names come straight from
    ``data_to_input`` and callback labels from the action log.

    Parameters
    ----------
    replay_results : dict
        The dict returned by :meth:`ReplayBG.replay`. Must contain ``output``,
        ``input``, ``data_to_input`` and ``actions``; ``measurement`` and
        ``measurement_time`` are used when present.
    thresholds : list of float or None, optional, default : None
        Optional reference levels drawn as dashed horizontal lines on the output
        subplot. Two or more levels also shade the span between the outermost.
    input_groups : list of list of int or None, optional, default : None
        Optional list of channel-index groups. Each group becomes one subplot
        with its channels overlaid. ``None`` plots one channel per subplot, in
        ``data_to_input`` order.
    mask_inputs : list of int or None, optional, default : None
        Optional list of channel indices to hide. Masked channels are dropped
        from the (default or explicit) ``input_groups`` so they get no subplot;
        e.g. pass the ``t_hour`` index to skip it.
    action_field : str or None, optional, default : None
        Name of the scalar field to plot for each callback action. ``None``
        resolves it per callback to the first numeric field the callback logged
        (anything other than ``k``/``callback``), which by convention is the
        quantity the action delivered. An explicit name is used for every
        callback that logs it; callbacks that do not fall back to the automatic
        choice.
    ts_min : float, optional, default : 1.0
        Minutes represented by one integration step, used to scale the time axis.
    output_label : str, optional, default : "Output"
        Legend label for the replayed output line.
    measurement_label : str, optional, default : "Measurement"
        Legend label for the replayed measurement markers.
    figsize : tuple of float or None, optional, default : None
        Optional figure size. Defaults to a height that grows with the number of
        subplots.
    hover : bool, optional, default : True
        Whether to attach the interactive readout: hovering any subplot drops a
        vertical cursor across the whole stack and reports the values at that
        minute. Ignored on non-interactive backends.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled figure.
    """
    output = np.asarray(replay_results["output"])
    inputs = np.asarray(replay_results["input"])
    data_to_input = replay_results["data_to_input"]
    actions = replay_results.get("actions", []) or []

    tsteps = output.shape[0]
    t = np.arange(tsteps) * ts_min

    # --- resolve the subplot layout -----------------------------------------
    input_groups, input_colors = resolve_input_layout(data_to_input, input_groups, mask_inputs)

    # Distinct callbacks, in first-seen order, that actually logged something.
    callback_names = list(dict.fromkeys(rec["callback"] for rec in actions))

    n_rows = 1 + len(input_groups) + len(callback_names)
    fig, axes = make_figure(n_rows, figsize)
    hover_series = {}

    # --- 1. output subplot ---------------------------------------------------
    ax = axes[0]
    draw_thresholds(ax, thresholds)
    ax.plot(t, output, color="tab:blue", linewidth=1.2, label=output_label)
    hover_series[ax] = [series(output_label, t, output, KIND_DENSE)]

    measurement = replay_results.get("measurement")
    measurement_time = replay_results.get("measurement_time")
    if measurement is not None and measurement_time is not None:
        measurement = np.asarray(measurement)
        measurement_time = np.asarray(measurement_time)
        mask = np.isfinite(measurement)
        meas_t = measurement_time[mask] * ts_min
        ax.plot(
            meas_t, measurement[mask],
            linestyle="none", marker="o", markersize=3,
            color="tab:red", alpha=0.7, label=measurement_label,
        )
        hover_series[ax].append(series(measurement_label, meas_t, measurement[mask], KIND_SPARSE))

    ax.set_ylabel(output_label)
    set_panel_title(ax, "Output")
    ax.legend(fontsize="small", loc="upper right", framealpha=0.9)

    # --- 2. input subplots ---------------------------------------------------
    for row, group in enumerate(input_groups, start=1):
        hover_series[axes[row]] = draw_input_group(
            axes[row], t, inputs, group, data_to_input, input_colors)

    # --- 3. callback action subplots ----------------------------------------
    base = 1 + len(input_groups)
    for row, cb_name in enumerate(callback_names, start=base):
        ax = axes[row]
        records = [rec for rec in actions if rec["callback"] == cb_name]
        field = _resolve_action_field(records, action_field)
        ks = np.array([rec["k"] * ts_min for rec in records if field in rec])
        values = np.array([rec[field] for rec in records if field in rec], dtype=float)
        # Registered even when empty, so the cursor still tracks across a panel
        # whose callback logged nothing plottable.
        hover_series[ax] = []
        if ks.size:
            markerline, _, _ = ax.stem(ks, values, linefmt="tab:purple",
                                       markerfmt="o", basefmt=" ")
            plt.setp(markerline, color="tab:purple", markersize=4)
            hover_series[ax] = [series(field, ks, values, KIND_SPARSE)]
        ax.set_ylabel(field or "")
        set_panel_title(ax, f"Callback: {cb_name}")

    finalize(fig, axes, "Time (min)", hover_series, hover, x_fmt="t = {:.0f} min")
    return fig


def plot_replay_intervals(
    replay_results_list: list[dict],
    thresholds: list[float] | None = None,
    input_groups: list[list[int]] | None = None,
    mask_inputs: list[int] | None = None,
    action_field: str | None = None,
    ts_min: float = 1.0,
    output_label: str = "Output",
    measurement_label: str = "Measurement",
    figsize: tuple[float, float] | None = None,
    hover: bool = True,
) -> plt.Figure:
    """Plot several x0-chained ``replay()`` segments as one continuous figure.

    The "interval" workflow replays one day, carries its final state into the
    next (via ``x0``/``theta_prev``), and repeats. Each day is a separate replay;
    this utility glues them back together — laying the segments end to end on a
    single time axis — so the multi-day replay reads as one trace. It is the
    multi-segment counterpart of :func:`plot_replay` and produces the same panel
    layout (output, inputs, one panel per distinct callback), with a dotted
    vertical separator marking each join between consecutive days.

    Each segment must expose the same input-channel layout (``data_to_input``),
    taken from the first. Measurement markers are shown only when *every* segment
    carries a sensor measurement.

    Parameters
    ----------
    replay_results_list : list of dict
        The per-day dicts returned by :meth:`ReplayBG.replay`, in chronological
        order. Each must contain ``output``, ``input``, ``data_to_input`` and
        ``actions``; ``measurement``/``measurement_time`` are used when present
        on all segments.
    thresholds : list of float or None, optional, default : None
        Optional reference levels drawn as dashed horizontal lines on the output
        subplot. Two or more levels also shade the span between the outermost.
    input_groups : list of list of int or None, optional, default : None
        Optional list of channel-index groups, each becoming one overlaid
        subplot. ``None`` plots one channel per subplot, in ``data_to_input``
        order.
    mask_inputs : list of int or None, optional, default : None
        Optional list of channel indices to hide, e.g. the ``t_hour`` index.
    action_field : str or None, optional, default : None
        Name of the scalar field to plot for each callback action, as in
        :func:`plot_replay`. ``None`` resolves it per callback automatically.
    ts_min : float, optional, default : 1.0
        Minutes represented by one integration step, used to scale the time axis.
    output_label : str, optional, default : "Output"
        Legend label for the replayed output line.
    measurement_label : str, optional, default : "Measurement"
        Legend label for the replayed measurement markers.
    figsize : tuple of float or None, optional, default : None
        Optional figure size. Defaults to a height that grows with the number of
        subplots.
    hover : bool, optional, default : True
        Whether to attach the interactive readout. Ignored on non-interactive
        backends.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled figure.
    """
    if not replay_results_list:
        raise ValueError("replay_results_list must contain at least one segment")

    data_to_input = replay_results_list[0]["data_to_input"]

    # --- glue outputs and inputs end to end ----------------------------------
    outputs = [np.asarray(r["output"]) for r in replay_results_list]
    inputs_list = [np.asarray(r["input"]) for r in replay_results_list]
    tsteps_list = [o.shape[0] for o in outputs]
    starts, boundaries = segment_offsets(tsteps_list, ts_min)
    step_starts = [int(round(s / ts_min)) for s in starts]

    output = np.concatenate(outputs)
    inputs = np.concatenate(inputs_list, axis=0)
    t = np.concatenate([start + np.arange(n) * ts_min
                        for start, n in zip(starts, tsteps_list)])

    # Actions carry a step index local to their segment; offset each onto the
    # glued axis so callback panels line up with the trace.
    actions = []
    for r, step_start in zip(replay_results_list, step_starts):
        for rec in (r.get("actions") or []):
            actions.append({**rec, "k": rec["k"] + step_start})

    # Measurements are shown only when every segment produced them.
    have_meas = all(r.get("measurement") is not None
                    and r.get("measurement_time") is not None
                    for r in replay_results_list)
    meas_t, meas_v = None, None
    if have_meas:
        meas_t_parts, meas_v_parts = [], []
        for r, start in zip(replay_results_list, starts):
            m = np.asarray(r["measurement"])
            mt = np.asarray(r["measurement_time"])
            valid = np.isfinite(m)
            meas_t_parts.append(start + mt[valid] * ts_min)
            meas_v_parts.append(m[valid])
        meas_t = np.concatenate(meas_t_parts)
        meas_v = np.concatenate(meas_v_parts)

    # --- resolve the subplot layout ------------------------------------------
    input_groups, input_colors = resolve_input_layout(data_to_input, input_groups, mask_inputs)
    callback_names = list(dict.fromkeys(rec["callback"] for rec in actions))

    n_rows = 1 + len(input_groups) + len(callback_names)
    fig, axes = make_figure(n_rows, figsize)
    hover_series = {}

    # --- 1. output subplot ---------------------------------------------------
    ax = axes[0]
    draw_thresholds(ax, thresholds)
    ax.plot(t, output, color="tab:blue", linewidth=1.2, label=output_label)
    hover_series[ax] = [series(output_label, t, output, KIND_DENSE)]
    if have_meas:
        ax.plot(
            meas_t, meas_v,
            linestyle="none", marker="o", markersize=3,
            color="tab:red", alpha=0.7, label=measurement_label,
        )
        hover_series[ax].append(series(measurement_label, meas_t, meas_v, KIND_SPARSE))
    draw_segment_boundaries(ax, boundaries)
    ax.set_ylabel(output_label)
    set_panel_title(ax, f"Output ({len(replay_results_list)} segments)")
    ax.legend(fontsize="small", loc="upper right", framealpha=0.9)

    # --- 2. input subplots ---------------------------------------------------
    for row, group in enumerate(input_groups, start=1):
        hover_series[axes[row]] = draw_input_group(
            axes[row], t, inputs, group, data_to_input, input_colors)
        draw_segment_boundaries(axes[row], boundaries)

    # --- 3. callback action subplots -----------------------------------------
    base = 1 + len(input_groups)
    for row, cb_name in enumerate(callback_names, start=base):
        ax = axes[row]
        records = [rec for rec in actions if rec["callback"] == cb_name]
        field = _resolve_action_field(records, action_field)
        ks = np.array([rec["k"] * ts_min for rec in records if field in rec])
        values = np.array([rec[field] for rec in records if field in rec], dtype=float)
        hover_series[ax] = []
        if ks.size:
            markerline, _, _ = ax.stem(ks, values, linefmt="tab:purple",
                                       markerfmt="o", basefmt=" ")
            plt.setp(markerline, color="tab:purple", markersize=4)
            hover_series[ax] = [series(field, ks, values, KIND_SPARSE)]
        draw_segment_boundaries(ax, boundaries)
        ax.set_ylabel(field or "")
        set_panel_title(ax, f"Callback: {cb_name}")

    finalize(fig, axes, "Time (min)", hover_series, hover, x_fmt="t = {:.0f} min")
    return fig


def _resolve_action_field(records: list[dict], action_field: str | None) -> str | None:
    """Returns the record field to plot for one callback's actions.

    Honours ``action_field`` when the callback logs it. Otherwise picks the first
    numeric non-metadata field, which by the logging convention is the quantity
    the action delivered (e.g. ``cb_u`` for a bolus, ``ht_g`` for a rescue), so
    the utility stays agnostic to callback internals.

    Parameters
    ----------
    records : list of dict
        The action records logged by a single callback.
    action_field : str or None
        The caller's preferred field name, or ``None`` to choose automatically.

    Returns
    -------
    str or None
        The field name to plot, or ``None`` when the callback logged no numeric
        field.
    """
    if action_field is not None and any(action_field in rec for rec in records):
        return action_field
    for rec in records:
        for key, value in rec.items():
            if key not in ("k", "callback") and isinstance(value, (int, float, np.floating, np.integer)):
                return key
    return None
