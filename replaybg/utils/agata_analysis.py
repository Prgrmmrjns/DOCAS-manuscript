"""py_agata-based analysis of ReplayBG ``twin()`` and ``replay()`` outputs.

These are standalone, post-processing utilities: they take the results produced
by :meth:`ReplayBG.replay` / :meth:`ReplayBG.twin` and run the `py_agata`
package on the resulting glucose traces.

- :func:`analyze_replay` runs AGATA's full glucose-profile analysis on the
  replayed model ``output`` and, when a sensor was used, on the sensor
  ``measurement`` trace.
- :func:`analyze_twin` runs the same profile analysis on the *fitted* (predicted)
  trace **and** computes the comprehensive ``py_agata.error`` metric set,
  quantifying the twinning fit error against the reference CGM.

Both functions optionally print a compact diagnostic summary (``verbose=True``)
and write the analysis back into the saved ``.pkl`` (``path`` given), matching
the dict shape that ``replay()`` / ``twin()`` persist.

AGATA wants a homogeneous time grid expressed in minutes (the *sampling
cadence*), so traces are downsampled to that cadence before the DataFrame is
built. ReplayBG traces live at integration resolution (``integration_ts``
minutes per step, default 1); the CGM/AGATA cadence is ``ts`` minutes
(default 5).
"""

import numpy as np

from py_agata.py_agata import Agata
from py_agata.error import rmse, mard, cod, clarke, g_rmse
from py_agata.utils import glucose_vector_to_dataframe

from utils.numba_dicts import to_typed_f64_dict
from utils.save_results import save_results


def analyze_replay(
    replay_results: dict,
    ts: int = 5,
    integration_ts: int = 1,
    glycemic_target: str = "diabetes",
    verbose: bool = False,
    path: str | None = None,
    save_name: str | None = None,
) -> dict:
    """Analyze the glucose traces of a :meth:`ReplayBG.replay` run with AGATA.

    Runs :meth:`py_agata.Agata.analyze_glucose_profile` on the replayed model
    ``output`` and, when the replay used a sensor, also on the sensor
    ``measurement`` trace.

    Parameters
    ----------
    replay_results : dict
        The dict returned by :meth:`ReplayBG.replay` (must contain ``output``;
        ``measurement``/``measurement_time`` are used when present).
    ts : int, optional, default : 5
        AGATA glucose sampling cadence in minutes. The ``output`` trace is
        downsampled to this cadence before analysis.
    integration_ts : int, optional, default : 1
        Minutes represented by one integration step (``environment.ts``). Used to
        convert ``ts`` into a downsampling step and to scale the measurement
        cadence.
    glycemic_target : str, optional, default : "diabetes"
        Glycemic target set forwarded to :class:`py_agata.Agata`.
    verbose : bool, optional, default : False
        When ``True``, prints a compact metric summary.
    path : str or None, optional, default : None
        When given, the analysis is added to ``replay_results`` under the
        ``analysis`` key and the dict is re-saved to ``<path>/replay_*.pkl``.
    save_name : str or None, optional, default : None
        File name for the re-saved pickle (see
        :func:`utils.save_results.save_results`).

    Returns
    -------
    dict
        A dict with key ``output`` (the profile of the model output) and, when a
        sensor was used, ``measurement`` (the profile of the sensor trace).
    """
    agata = Agata(glycemic_target=glycemic_target)

    # The output lives at integration resolution; subsample it to the AGATA
    # cadence so the DataFrame timestamps are honest.
    step = max(1, int(round(ts / integration_ts)))
    out = np.asarray(replay_results["output"][0::step], dtype=float)
    out_df = glucose_vector_to_dataframe(out, int(ts))
    analysis = {"output": agata.analyze_glucose_profile(out_df)}

    # Sensor measurements are already at sensor cadence; recover their real
    # spacing from the integration-step indices of consecutive samples.
    if "measurement" in replay_results:
        meas = np.asarray(replay_results["measurement"], dtype=float)
        mt = np.asarray(replay_results.get("measurement_time", []))
        meas_ts = int((mt[1] - mt[0]) * integration_ts) if mt.size >= 2 else int(ts)
        meas_df = glucose_vector_to_dataframe(meas, meas_ts)
        analysis["measurement"] = agata.analyze_glucose_profile(meas_df)

    if verbose:
        _print_analysis("Replay", analysis)

    if path is not None:
        replay_results["analysis"] = analysis
        save_results(replay_results, path, save_name, prefix="replay")

    return analysis


def analyze_twin(
    twin_results: dict,
    rbg_data,
    model,
    integration_ts: int = 1,
    glycemic_target: str = "diabetes",
    verbose: bool = False,
    path: str | None = None,
    save_name: str | None = None,
) -> dict:
    """Analyze the outcome of a :meth:`ReplayBG.twin` run with AGATA.

    Simulates the model forward with the estimated parameters to obtain the
    fitted (predicted) glucose trace, then:

    - runs :meth:`py_agata.Agata.analyze_glucose_profile` on that trace, and
    - computes the comprehensive ``py_agata.error`` metric set (``rmse``,
      ``mard``, ``cod``, Clarke EGA zones, ``g_rmse``) against the reference
      CGM that was fitted (``rbg_data.y``).

    Parameters
    ----------
    twin_results : dict
        The dict returned by :meth:`ReplayBG.twin` (must contain ``theta``).
    rbg_data : object
        The data object passed to :meth:`ReplayBG.twin`. Provides the reference
        CGM (``y``), the input matrix (``u``), ``tsteps`` and the data sampling
        stride ``yts``.
    model : object
        A model instance implementing ``reset(theta_dict)``, ``step(u, t)`` and
        ``output(t)``.
    integration_ts : int, optional, default : 1
        Minutes represented by one integration step (``environment.ts``). The
        AGATA cadence is ``rbg_data.yts * integration_ts``.
    glycemic_target : str, optional, default : "diabetes"
        Glycemic target set forwarded to :class:`py_agata.Agata`.
    verbose : bool, optional, default : False
        When ``True``, prints a compact metric summary.
    path : str or None, optional, default : None
        When given, the analysis is added under the ``analysis`` key and the
        results (including ``rbg_data``) are re-saved to ``<path>/twin_*.pkl``,
        preserving the shape ``twin()`` persists.
    save_name : str or None, optional, default : None
        File name for the re-saved pickle.

    Returns
    -------
    dict
        A dict with keys ``profile`` (AGATA profile of the fitted trace) and
        ``error`` (the fit-error metric set).
    """
    theta = twin_results["theta"]

    # Simulate the fitted model forward (same reset/step/output pass the twinner
    # uses for the likelihood; mirrors utils.plot_twinning).
    model.reset(to_typed_f64_dict(theta))
    inputs = np.asarray(rbg_data.u)
    tsteps = int(rbg_data.tsteps)
    output = np.zeros(tsteps)
    output[0] = model.output(0)
    for k in range(1, tsteps):
        model.step(inputs[k], k)
        output[k] = model.output(k)

    yts = int(rbg_data.yts)
    pred = np.asarray(output[0::yts], dtype=float)
    ref = np.asarray(rbg_data.y, dtype=float)
    # Align both traces on the same homogeneous grid.
    n = min(pred.size, ref.size)
    pred, ref = pred[:n], ref[:n]

    sample_time = int(yts * integration_ts)
    ref_df = glucose_vector_to_dataframe(ref, sample_time)
    pred_df = glucose_vector_to_dataframe(pred, sample_time)

    agata = Agata(glycemic_target=glycemic_target)
    analysis = {
        "profile": agata.analyze_glucose_profile(pred_df),
        "error": {
            "rmse": rmse(ref_df, pred_df),
            "mard": mard(ref_df, pred_df),
            "cod": cod(ref_df, pred_df),
            "clarke": clarke(ref_df, pred_df),
            "g_rmse": g_rmse(ref_df, pred_df),
        },
    }

    if verbose:
        _print_analysis("Twin", analysis)

    if path is not None:
        save_results(
            {**twin_results, "rbg_data": rbg_data, "analysis": analysis},
            path, save_name, prefix="twin",
        )

    return analysis


# --- printing -------------------------------------------------------------

def _print_analysis(kind: str, analysis: dict) -> None:
    """Print a compact, human-readable summary of an analysis dict."""
    print(f"--- {kind} analysis ---")
    for key, value in analysis.items():
        if key == "error":
            _print_error(value)
        else:
            _print_profile(key, value)


def _print_profile(label: str, profile: dict) -> None:
    """Print a curated selection of AGATA glucose-profile metrics."""
    var = profile.get("variability", {})
    tir = profile.get("time_in_ranges", {})
    risk = profile.get("risk", {})
    print(f"  [{label}]")
    print(f"    mean glucose    : {var.get('mean_glucose', float('nan')):.1f} mg/dL")
    print(f"    std glucose     : {var.get('std_glucose', float('nan')):.1f} mg/dL")
    print(f"    CV              : {var.get('cv_glucose', float('nan')):.1f} %")
    print(f"    GMI             : {var.get('gmi', float('nan')):.2f} %")
    print(f"    time in target  : {tir.get('time_in_target', float('nan')):.1f} %")
    print(f"    time hypo/hyper : {tir.get('time_in_hypoglycemia', float('nan')):.1f} % / "
          f"{tir.get('time_in_hyperglycemia', float('nan')):.1f} %")
    print(f"    LBGI / HBGI     : {risk.get('lbgi', float('nan')):.2f} / "
          f"{risk.get('hbgi', float('nan')):.2f}")


def _print_error(error: dict) -> None:
    """Print the full fit-error metric block."""
    print("  [fit error]")
    print(f"    RMSE            : {error['rmse']:.2f} mg/dL")
    print(f"    MARD            : {error['mard']:.2f} %")
    print(f"    COD             : {error['cod']:.2f} %")
    print(f"    gRMSE           : {error['g_rmse']:.2f} mg/dL")
    zones = error["clarke"]
    print(f"    Clarke A/B/C/D/E: "
          f"{zones['a']:.1f} / {zones['b']:.1f} / {zones['c']:.1f} / "
          f"{zones['d']:.1f} / {zones['e']:.1f} %")
