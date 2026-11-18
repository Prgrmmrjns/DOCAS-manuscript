"""Manuscript LaTeX tables. uv run python scripts/tables.py"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts")]
import main as m

TABLES = ROOT / "manuscript" / "tables"
RESULTS = ROOT / "results"
ALPHA = 0.05


def _fmt(mean: float, std: float) -> str:
    return "---" if not np.isfinite(mean) else f"{mean:.2f} $\\pm$ {std:.2f}"


def paired_diff_test(baseline, docas, *, alpha: float = ALPHA) -> dict:
    """Two-sided paired Wilcoxon signed-rank (pre-specified; no normality gate).

    Returns paired effect size (mean baseline−DOCAS) and bootstrap 95% CI.
    """
    a = np.asarray(baseline, float)
    b = np.asarray(docas, float)
    d = a - b
    try:
        p = float(stats.wilcoxon(
            a, b, zero_method="wilcox", alternative="two-sided", method="auto").pvalue)
    except ValueError:
        p = 1.0
    significant = bool(np.isfinite(p) and p < alpha)
    direction = None
    if significant:
        direction = "lower" if float(np.mean(b)) < float(np.mean(a)) else "higher"
    rng = np.random.default_rng(0)
    if len(d) >= 2:
        boots = [float(np.mean(rng.choice(d, size=len(d), replace=True))) for _ in range(2000)]
        ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    else:
        ci = (float("nan"), float("nan"))
    return dict(
        p=p, test="wilcoxon", shapiro_p=float("nan"), normal=False,
        significant=significant, direction=direction,
        effect=float(np.mean(d)) if len(d) else float("nan"), ci=ci,
    )


def holm_correct(p_values) -> list[float]:
    """Holm–Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(p_values, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n, float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * float(p[idx]))
        adj[idx] = min(1.0, running)
    return [float(x) for x in adj]


paired_lower_test = paired_diff_test  # backward alias


def _fmt_cell(mean: float, std: float, *, sig: bool = False) -> str:
    s = _fmt(mean, std)
    return rf"\textbf{{{s}}}$^*$" if sig else s


def _metric_cells(metrics: dict, rmse_key: str, align_key: str,
                  *, mark: dict | None = None) -> list[str]:
    mark = mark or {}
    return [
        _fmt_cell(metrics[rmse_key]["mean"], metrics[rmse_key]["std"],
                  sig=bool(mark.get("rmse"))),
        _fmt_cell(metrics[align_key]["mean"], metrics[align_key]["std"],
                  sig=bool(mark.get("align"))),
    ]


def _cohort_sig(path: Path, ph: int) -> dict:
    """direction per metric after Holm correction across RMSE and AE."""
    rows = m.ohio_horizon_payload(json.loads(path.read_text()), ph).get("per_patient") or []
    if not rows:
        return {}
    pairs = (
        ("rmse", "baseline_test_rmse_mgdl", "docas_test_rmse_mgdl"),
        ("align", "baseline_alignment_error", "alignment_error"),
    )
    names, ps, dirs = [], [], []
    for name, bk, dk in pairs:
        base = [float(r[bk]) for r in rows if bk in r and dk in r]
        doc = [float(r[dk]) for r in rows if bk in r and dk in r]
        if len(base) < 2:
            names.append(name); ps.append(1.0); dirs.append(None)
            continue
        t = paired_diff_test(base, doc)
        names.append(name); ps.append(t["p"]); dirs.append(t["direction"])
    adj = holm_correct(ps)
    return {n: (d if p_adj < ALPHA else None) for n, d, p_adj in zip(names, dirs, adj)}


def _arm_marks(sig: dict, arm: str) -> dict:
    """Mark the significantly better arm (lower error)."""
    want = "lower" if arm == "docas" else "higher"
    return {k: (v == want) for k, v in sig.items()}


def _sig_caption_lines() -> list[str]:
    return [
        rf" Values are cohort mean~$\pm$~SD (metrics averaged over {len(m.SEEDS)} seeds).",
        r" Bold with $^*$ marks the significantly better arm for that metric",
        (
            rf" (two-sided Wilcoxon signed-rank, Holm-adjusted $\alpha{{=}}{ALPHA}$;"
        ),
        r" Section~\ref{subsec:ohio-protocol}).",
    ]


def _pred_table_arms(cohorts: list[tuple[str, Path]], *,
                     baseline_name: str = "Baseline",
                     docas_name: str = "DOCAS") -> list[str]:
    def cells(cohort, ph, arm):
        metrics = m.ohio_horizon_payload(json.loads(cohort.read_text()), ph).get("metrics")
        if not metrics:
            return ["---", "---"]
        mark = _arm_marks(_cohort_sig(cohort, ph), arm)
        if arm == "baseline":
            return _metric_cells(metrics, "baseline_test_rmse_mgdl", "baseline_alignment_error",
                                 mark=mark)
        return _metric_cells(
            metrics, "docas_test_rmse_mgdl", "alignment_error", mark=mark)

    lines = []
    for i, (label, cohort) in enumerate(cohorts):
        if i:
            lines.append(r"\addlinespace")
        join = lambda arm, c=cohort: " & ".join(
            c_ for ph in m.FORECAST_HORIZONS for c_ in cells(c, ph, arm))
        if label:
            lines.append(
                rf"\multicolumn{{{1 + 2 * len(m.FORECAST_HORIZONS)}}}{{@{{}}l}}"
                rf"{{\textit{{{label}}}}} \\")
        lines += [
            f"{baseline_name} & {join('baseline')} \\\\",
            f"{docas_name} & {join('docas')} \\\\",
        ]
    return lines


def _eval_cohorts(*, backbones: tuple[str, ...] = ("catboost",)) -> list[tuple[str, int, list[tuple[str, Path]]]]:
    """(dataset label, n participants, [(backbone, cohort JSON)]) for an eval table.

    A backbone is included only when its cohort was produced under the study
    protocol, so a stale run cannot silently enter the table.
    """
    tau_mode = str(m.STUDY_CFG.get("tau_mode", "fractional"))

    def on_protocol(path: Path) -> bool:
        return path.is_file() and str(
            json.loads(path.read_text()).get("protocol", "")).startswith(tau_mode)

    azt = m.RESULTS / "azt1d"
    out = []
    for label, cat, mlp in (("OhioT1DM", m.CAT_COHORT, m.MLP_COHORT),
                            ("AZT1D", azt / "cohort.json", azt / "mlp_cohort.json")):
        if not cat.is_file():
            raise FileNotFoundError(f"{label} CatBoost cohort JSON missing")
        arms = []
        if "catboost" in backbones:
            arms.append(("CatBoost", cat))
        if "mlp" in backbones and on_protocol(mlp):
            arms.append(("MLP", mlp))
        if not arms:
            continue
        n = len(json.loads(cat.read_text()).get("patient_ids") or [])
        out.append((label, n, arms))
    return out


def table_eval_tex() -> Path:
    """Held-out CatBoost accuracy and alignment on both cohorts."""
    col = "l" + "cc" * len(m.FORECAST_HORIZONS)
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{ph} min}}" for ph in m.FORECAST_HORIZONS)
    cmid = "".join(rf"\cmidrule(lr){{{2 * i + 2}-{2 * i + 3}}}"
                   for i in range(len(m.FORECAST_HORIZONS)))
    ncol = 1 + 2 * len(m.FORECAST_HORIZONS)
    body = []
    for i, (label, n, arms) in enumerate(_eval_cohorts(backbones=("catboost",))):
        if i:
            body.append(r"\addlinespace")
        body.append(rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\textbf{{{label}}} ($n={n}$)}} \\")
        body += _pred_table_arms([("", p) for _, p in arms])
    lines = [
        r"\begin{table}[pos=htbp,width=\linewidth]", r"\centering",
        r"\caption{Held-out CatBoost RMSE and alignment error on both cohorts.",
        r" Baseline: original training records only. DOCAS: original records plus",
        rf" synthetic Phase~2 rows. Cohort mean~$\pm$~SD over {len(m.SEEDS)} seeds. Bold$^*$:",
        r" significantly better arm (Section~\ref{subsec:ohio-protocol}). MLP and lag features in",
        r" Appendix~\ref{app:mlp}.",
        r"}",
        r"\label{tab:prediction}", r"\scriptsize",
        rf"\begin{{tabular}}{{{col}}}",
        r"\toprule", rf" & {head} \\", cmid,
        "Approach & " + " & ".join(["RMSE & Align."] * len(m.FORECAST_HORIZONS)) + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "table_eval.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def table_mlp_tex() -> Path:
    """MLP and lag-feature counterpart of Table~\ref{tab:prediction}."""
    col = "l" + "cc" * len(m.FORECAST_HORIZONS)
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{ph} min}}" for ph in m.FORECAST_HORIZONS)
    cmid = "".join(rf"\cmidrule(lr){{{2 * i + 2}-{2 * i + 3}}}"
                   for i in range(len(m.FORECAST_HORIZONS)))
    ncol = 1 + 2 * len(m.FORECAST_HORIZONS)
    cohorts = _eval_cohorts(backbones=("mlp",))
    if not cohorts:
        raise FileNotFoundError("MLP cohort JSONs missing")
    lag_paths = {
        "OhioT1DM": m.RESULTS / "ohio_t1dm" / "catboost_lags12_cohort.json",
        "AZT1D": m.RESULTS / "azt1d" / "catboost_lags12_cohort.json",
    }
    body = []
    for i, (label, n, arms) in enumerate(cohorts):
        if i:
            body.append(r"\addlinespace")
        body.append(rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\textbf{{{label}}} ($n={n}$)}} \\")
        body += _pred_table_arms(
            [("", p) for _, p in arms],
            baseline_name="MLP baseline", docas_name="MLP DOCAS")
        lag = lag_paths.get(label)
        if lag is not None and lag.is_file():
            body.append(r"\addlinespace")
            body += _pred_table_arms(
                [("", lag)],
                baseline_name="Lags baseline", docas_name="Lags DOCAS")
    lines = [
        r"\begin{table}[pos=htbp,width=\linewidth]", r"\centering",
        r"\caption{MLP (five seeds) and CatBoost with twelve five-minute lags (one seed).",
        r" Same protocol as Table~\ref{tab:prediction}. Cohort mean~$\pm$~SD.",
        r" Bold$^*$: significantly better arm (Section~\ref{subsec:ohio-protocol}).}",
        r"\label{tab:prediction-mlp}",
        r"\label{tab:lags}", r"\scriptsize",
        rf"\begin{{tabular}}{{{col}}}",
        r"\toprule", rf" & {head} \\", cmid,
        "Approach & " + " & ".join(["RMSE & Align."] * len(m.FORECAST_HORIZONS)) + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    out = TABLES / "table_mlp_eval.tex"
    out.write_text("\n".join(lines) + "\n")
    stale = TABLES / "table_lag_features.tex"
    if stale.is_file():
        stale.unlink()
    return out


# Post-hoc tau reproduces the target by construction; it is the alignment
# reference, not a competing DSS, so the replay table reports only the arms that
# learn their own dose response.
REPLAY_ARMS = (("no_dss", "No DSS"), ("baseline", "Baseline"), ("docas", "DOCAS"))
REPLAY_METRICS = (("TBR (\\%)", "tbr"), ("TIR (\\%)", "tir"), ("TAR (\\%)", "tar"),
                  ("Insulin (U)", "insulin_u"))


def table_replay_tex() -> Path:
    """ReplayBG CIB outcomes on the OhioT1DM cohort."""
    import replay_eval as r
    path = r.replay_cohort_path(r._SCOPE["ohio_t1dm"])
    if not path.is_file():
        raise FileNotFoundError("OhioT1DM replay cohort missing; run replay_eval")
    co = r.replay_horizon_payload(
        json.loads(path.read_text()), r.REPLAY_HORIZONS[0])["cohort"]
    gly = co["summary_post"]
    doses = co["summary"]
    n_win, n_pat = int(co["n_windows"]), int(co["n_patients"])
    body = []
    for name, key in REPLAY_METRICS:
        src = doses if key == "insulin_u" else gly
        cells = [_fmt(float(src[arm][key]["mean"]), float(src[arm][key]["std"]))
                 for arm, _ in REPLAY_ARMS]
        body.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines = [
        r"\begin{table}[pos=htbp,width=0.72\linewidth]", r"\centering",
        rf"\caption{{Post-decision outcomes for no decision support and the two "
        rf"corrective-bolus policies on {n_win} OhioT1DM windows from "
        rf"{n_pat} participants (mean~$\pm$~SD).}}",
        r"\label{tab:replay}", r"\scriptsize",
        rf"\begin{{tabular}}{{{'l' + 'c' * len(REPLAY_ARMS)}}}",
        r"\toprule",
        "Metric & " + " & ".join(lab for _, lab in REPLAY_ARMS) + r" \\",
        r"\midrule", *body,
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    out = TABLES / "table_replay.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def table_unguarded_tex() -> Path:
    """Same windows as Table tab:replay, without the hypoglycaemia guard."""
    import replay_eval as r
    path = r.unguarded_cohort_path(r._SCOPE["ohio_t1dm"])
    if not path.is_file():
        raise FileNotFoundError(
            "OhioT1DM unguarded cohort missing; "
            "run `python scripts/replay_eval.py unguarded`")
    payload = json.loads(path.read_text())
    gly = payload.get("summary_post") or payload["summary"]
    doses = payload["summary"]
    n_win = int(payload["n_windows"])
    n_pat = len(payload["patient_ids"])
    body = []
    for name, key in REPLAY_METRICS:
        src = doses if key == "insulin_u" else gly
        cells = [_fmt(float(src[arm][key]["mean"]), float(src[arm][key]["std"]))
                 for arm, _ in REPLAY_ARMS]
        body.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines = [
        r"\begin{table}[pos=htbp,width=0.72\linewidth]", r"\centering",
        rf"\caption{{The same {n_win} OhioT1DM windows from {n_pat} participants "
        rf"and the same corrective-bolus cost as Table~\ref{{tab:replay}}, but without the "
        rf"hypoglycaemia guard (no $70$~mg/dL floor, no $2\times$ margin), "
        rf"i.e.\ scored as in Prendin et al. TIR, TBR, and TAR are post-decision. "
        rf"Mean~$\pm$~SD.}}",
        r"\label{tab:unguarded}", r"\scriptsize",
        rf"\begin{{tabular}}{{{'l' + 'c' * len(REPLAY_ARMS)}}}",
        r"\toprule",
        "Metric & " + " & ".join(lab for _, lab in REPLAY_ARMS) + r" \\",
        r"\midrule", *body,
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    out = TABLES / "table_unguarded.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def table_clinical_replay_tex() -> Path:
    """ReplayBG with DOCAS trained on clinical τ (same windows as tab:replay)."""
    import replay_eval as r
    path = r.clinical_tau_cohort_path(r._SCOPE["ohio_t1dm"])
    if not path.is_file():
        raise FileNotFoundError(
            "clinical-τ replay missing; run `python scripts/replay_eval.py clinical`")
    payload = json.loads(path.read_text())
    gly = payload.get("summary_post") or payload["summary"]
    doses = payload["summary"]
    study_gly = payload.get("study_summary_post") or payload.get("study_summary") or {}
    study_doses = payload.get("study_summary") or {}
    n_win = int(payload["n_windows"])
    n_pat = len(payload["patient_ids"])
    arms = (("no_dss", "No DSS"), ("baseline", "Baseline"),
            ("docas_study", r"DOCAS (study DRC)"),
            ("docas", r"DOCAS (pump-settings DRC)"))

    def _arm_stats(arm, key):
        src_map = study_doses if key == "insulin_u" else study_gly
        own = doses if key == "insulin_u" else gly
        if arm != "docas" and src_map:
            src_arm = "docas" if arm == "docas_study" else arm
            src = src_map[src_arm][key]
        else:
            src = own[arm][key]
        return _fmt(float(src["mean"]), float(src["std"]))

    body = []
    for name, key in REPLAY_METRICS:
        cells = [_arm_stats(arm, key) for arm, _ in arms]
        body.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines = [
        r"\begin{table}[pos=htbp,width=0.92\linewidth]", r"\centering",
        rf"\caption{{Same {n_win} OhioT1DM windows and hypoglycaemia guard as "
        rf"Table~\ref{{tab:replay}} ({n_pat} participants), with DOCAS retrained "
        rf"on each patient's pump-settings DRC. Study-DRC DOCAS "
        rf"columns are the main-text arm. TIR, TBR, and TAR are post-decision. "
        rf"Mean~$\pm$~SD.}}",
        r"\label{tab:clinical-replay}", r"\scriptsize",
        rf"\begin{{tabular}}{{{'l' + 'c' * len(arms)}}}",
        r"\toprule",
        "Metric & " + " & ".join(lab for _, lab in arms) + r" \\",
        r"\midrule", *body,
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    out = TABLES / "table_clinical_replay.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def _misspec_rows(payload, ph):
    """Rows for DOCAS taught a wrong-sensitivity τ, scored against that same τ."""
    if not payload:
        return []
    metrics = payload["horizons"][str(ph)]["metrics"]
    out = []
    for scale in payload["si_scales"]:
        if float(scale) == 1.0:
            continue
        met = metrics[f"si_{float(scale):g}x"]
        out.append((rf"DRC at ${float(scale):g}\times S_I$",
                    met["alignment_error_own_tau"], met["test_rmse_mgdl"]))
    return out


def _clinical_rows(payload, ph):
    """Row for DOCAS taught a τ built only from the patient's pump settings."""
    if not payload:
        return []
    met = payload["horizons"][str(ph)]["metrics"]["clinical"]
    return [(r"Pump-settings DRC",
             met["alignment_error_own_tau"], met["test_rmse_mgdl"])]


def _budget_rows(payload, ph):
    """Ohio synthetic-budget arms already computed in ``budget_ohio.json``."""
    if not payload:
        return []
    met = payload["horizons"][str(ph)]["metrics"]
    rows = []
    for key, name in (
        ("0.1x", r"Budget $0.1\times$"),
        ("1x", r"Budget $1\times$"),
        ("100x", r"Budget $100\times$"),
    ):
        arm = met[key]
        rows.append((name, arm["alignment_error"], arm["test_rmse_mgdl"]))
    return rows


def table3_tex() -> Path:
    """Leave-one-component ablations on both cohorts."""
    import ablation as ab

    def stats_fn(vals):
        a = np.asarray(vals, float)
        return {"mean": float(np.mean(a)), "std": float(np.std(a, ddof=1))}

    def rows_for(payload, misspec, clinical, budget, ph):
        block = payload["horizons"][str(ph)]
        prows, metrics = block["per_patient"], block["metrics"]
        def_rm = stats_fn([float(r["baseline_test_rmse_mgdl"]) + float(r["default_docas"]["rmse_delta_mgdl"])
                        for r in prows])
        ae0 = metrics["default_docas"]["alignment_error"]

        rows = [
            (r"DOCAS ($20\times$ budget)", ae0, def_rm),
            *_budget_rows(budget, ph),
        ]
        if "uniform_only" in metrics:
            rows.append((r"Uniform IOB only",
                         metrics["uniform_only"]["alignment_error"],
                         metrics["uniform_only"]["test_rmse_mgdl"]))
        rows += [
            (r"DRC applied after prediction",
             metrics["tauwrap"]["alignment_error"], metrics["tauwrap"]["test_rmse_mgdl"]),
            *_misspec_rows(misspec, ph),
            *_clinical_rows(clinical, ph),
        ]
        return rows

    blocks = []
    for label, dataset in (("OhioT1DM", "ohio_t1dm"), ("AZT1D", "azt1d")):
        path = ab.ablation_cohort_path(dataset)
        if not path.is_file():
            if dataset == "ohio_t1dm":
                raise FileNotFoundError(f"{label} ablation missing")
            continue
        payload = json.loads(path.read_text())
        mp = ab.misspec_path(dataset)
        misspec = json.loads(mp.read_text()) if mp.is_file() else None
        cp = ab.clinical_tau_path(dataset)
        clinical = json.loads(cp.read_text()) if cp.is_file() else None
        bp = ab.budget_path(dataset)
        budget = json.loads(bp.read_text()) if bp.is_file() else None
        n = len(payload["patient_ids"])
        by = {ph: dict((nm, (a, r)) for nm, a, r in
                       rows_for(payload, misspec, clinical, budget, ph))
              for ph in m.FORECAST_HORIZONS}
        blocks.append((label, n, by))

    lines = [
        r"\begin{table}[pos=htbp,width=\linewidth]", r"\centering",
        r"\caption{Ablations on both cohorts (mean~$\pm$~SD, seed~$42$)."
        r" Synthetic budgets $0.1\times$, $1\times$, and $100\times$ versus"
        r" the study preset $20\times$. Uniform IOB only draws every probe"
        r" from $\mathrm{Uniform}[0,1]$ (no training-density mix)."
        r" The last row uses a pump-settings DRC"
        r" (Appendix~\ref{app:clinical-tau}).}",
        r"\label{tab:ablation}", r"\scriptsize",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
    ]
    by_label = {label: (n, by) for label, n, by in blocks}
    ohio_n, ohio = by_label["OhioT1DM"]
    azt_n, azt = by_label.get("AZT1D", (0, {30: {}, 60: {}}))
    lines += [
        rf" & \multicolumn{{4}}{{c}}{{OhioT1DM ($n={ohio_n}$)}}"
        rf" & \multicolumn{{4}}{{c}}{{AZT1D ($n={azt_n}$)}} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r" & \multicolumn{2}{c}{30 min} & \multicolumn{2}{c}{60 min}"
        r" & \multicolumn{2}{c}{30 min} & \multicolumn{2}{c}{60 min} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"Ablation & Align. & RMSE & Align. & RMSE"
        r" & Align. & RMSE & Align. & RMSE \\",
        r"\midrule",
    ]
    names = list(ohio[30])
    nan = {"mean": float("nan"), "std": float("nan")}

    def pair(block, name, ph):
        return block.get(ph, {}).get(name, (nan, nan))

    for name in names:
        (oa30, or30), (oa60, or60) = pair(ohio, name, 30), pair(ohio, name, 60)
        (aa30, ar30), (aa60, ar60) = pair(azt, name, 30), pair(azt, name, 60)
        lines.append(
            f"{name} & {_fmt(oa30['mean'], oa30['std'])} & {_fmt(or30['mean'], or30['std'])} & "
            f"{_fmt(oa60['mean'], oa60['std'])} & {_fmt(or60['mean'], or60['std'])} & "
            f"{_fmt(aa30['mean'], aa30['std'])} & {_fmt(ar30['mean'], ar30['std'])} & "
            f"{_fmt(aa60['mean'], aa60['std'])} & {_fmt(ar60['mean'], ar60['std'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    out = TABLES / "table_ablation.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def _screened_windows(h: dict) -> list[dict]:
    """Windows entering the glycaemic summary (matches replay_eval._primary_windows)."""
    import replay_eval as r
    return r._primary_windows(h["windows"])


def table_penalty_tex() -> Path:
    """TIR and delivered insulin against the shared dose penalty.

    Reporting both columns makes the table readable at matched dose: rows where
    the two arms deliver the same insulin isolate placement from dose size.
    """
    import replay_eval as r
    path = r.dose_matched_path("")
    if not path.is_file():
        raise FileNotFoundError("run replay_eval.run_dose_matched() first")
    dm = json.loads(path.read_text())
    pen_used = float(r.CIB_DOSE_PENALTY)

    def panel(key):
        fr = {arm: {float(p["penalty"]): p for p in rows}
              for arm, rows in dm[key].items()}
        out = []
        for pen in sorted(fr["baseline"]):
            b, d = fr["baseline"][pen], fr["docas"][pen]
            lab = f"{pen:g}" + (r" (reported)" if pen == pen_used else "")
            out.append(
                f"{lab} & {b['insulin_u']['mean']:.2f} & "
                f"{_fmt(b['tir']['mean'], b['tir']['std'])} & "
                f"{d['insulin_u']['mean']:.2f} & "
                f"{_fmt(d['tir']['mean'], d['tir']['std'])} & "
                f"{d['tbr']['mean']:.2f} & "
                f"{d['tir']['mean'] - b['tir']['mean']:+.2f}" + r" \\")
        return out

    m = dm["matched"]
    at_b = m["at_baseline_insulin"]
    nad = int(dm.get("nadir_min", r.NADIR_MIN))
    lines = [
        r"\begin{table}[pos=htbp,width=0.92\linewidth]", r"\centering",
        r"\caption{Sensitivity of the in-silico result to the shared quadratic dose",
        rf" penalty, on the same {dm['n_windows']} windows as Table~\ref{{tab:replay}}"
        r" at 60~min. Both arms always use the same penalty; only its level varies.",
        r" The upper panel is the reported objective of Prendin et al., which scores",
        r" each candidate dose at the 60~min horizon while four fifths of the bolus",
        rf" has yet to act; the lower panel judges it instead on the nadir the same"
        rf" audit curve implies at {nad}~min. Under the horizon objective the DOCAS"
        r" DSS spends the whole grid at weak penalties, so its advantage depends on",
        r" the penalty; projecting the nadir removes that dependence.",
        r" Because the penalty also sets how much insulin",
        r" each arm suggests, rows of equal insulin compare the arms at matched",
        rf" dose: the observational DSS reaches {at_b['insulin_u']:.2f}~U at the"
        rf" reported penalty and {m['best_over_sweep']['baseline']['insulin_u']:.2f}~U at"
        r" the weakest, never DOCAS's dose. TIR and TBR in \%, mean~$\pm$~SD over",
        r" windows; insulin in U.}",
        r"\label{tab:penalty}", r"\scriptsize",
        r"\begin{tabular}{lcccccc}", r"\toprule",
        r"& \multicolumn{2}{c}{Baseline} & \multicolumn{3}{c}{DOCAS} & \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}",
        r"Dose penalty & Insulin & TIR & Insulin & TIR & TBR & $\Delta$TIR (pp) \\",
        r"\midrule",
        r"\multicolumn{7}{l}{\textit{Dose judged at the 60~min horizon"
        r" \citep{prendin2023}, reported}} \\",
        *panel("frontier"),
        r"\midrule",
        rf"\multicolumn{{7}}{{l}}{{\textit{{Dose judged on the nadir at {nad}~min}}}} \\",
        *panel("frontier_alt_objective"),
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    out = TABLES / "table_penalty.tex"
    out.write_text("\n".join(lines) + "\n")
    return out



def _pearson(a, b) -> float:
    x, y = np.asarray(a, float), np.asarray(b, float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _ice_endpoint(row: dict, arm: str = "baseline") -> float:
    """Centred patient-level PDP at training-max IOB (mg/dL), from stored curves."""
    y = np.asarray(row["curves"]["test"][arm], float).ravel()
    return float(y[-1] - y[0])


def _iob_delta_corr(pid: str, ph: int) -> float:
    """Train-set Pearson correlation of bolus IOB with observed ΔCGM."""
    from preprocessing import horizon_steps
    data = m.load_patient(str(m.DATA_ROOT), str(pid), horizon_steps=horizon_steps(ph))
    ji = int(data.insulin_idx)
    iob = data.train_scalers[ji].inverse_transform(
        np.asarray(data.X_train, float)[:, ji: ji + 1]).ravel()
    dlt = data.cgm_scaler.inverse_transform(
        np.asarray(data.y_train, float).reshape(-1, 1)).ravel()
    return _pearson(iob, dlt)


def _cohort_json_path(dataset: str) -> Path:
    return m.RESULTS / str(dataset).lower().replace("-", "_") / "cohort.json"


def individual_alignment_rows(dataset: str = "ohio_t1dm", ph: int = 60) -> list[dict]:
    """Per-patient AE, RMSE tax, ICE endpoint, and observational IOB–ΔCGM link."""
    prev = m.DATASET
    m.set_dataset(dataset)
    path = _cohort_json_path(dataset)
    rows = m.ohio_horizon_payload(json.loads(path.read_text()), ph).get("per_patient") or []
    out = []
    for r in rows:
        ice = _ice_endpoint(r, "baseline")
        out.append(dict(
            patient_id=str(r["patient_id"]),
            baseline_alignment_error=float(r["baseline_alignment_error"]),
            alignment_error=float(r["alignment_error"]),
            ae_gain=float(r["baseline_alignment_error"] - r["alignment_error"]),
            baseline_test_rmse_mgdl=float(r["baseline_test_rmse_mgdl"]),
            docas_test_rmse_mgdl=float(r["docas_test_rmse_mgdl"]),
            rmse_cost=float(r["docas_test_rmse_mgdl"] - r["baseline_test_rmse_mgdl"]),
            insulin_span_u=float(r["insulin_span_u"]),
            ice_endpoint_mgdl=ice,
            corr_iob_delta=_iob_delta_corr(r["patient_id"], ph),
        ))
    m.set_dataset(prev)
    return out


def alignment_individual(dataset: str = "ohio_t1dm") -> dict:
    """Why some baseline ICEs fail and when alignment costs held-out RMSE.

    Numbers cited in Results §4.1 (Figure alignment_individual). Ohio primary;
    AZT1D is the same check on a larger closed-loop cohort.
    """
    out = {}
    for ph in m.FORECAST_HORIZONS:
        rows = individual_alignment_rows(dataset, ph)
        ae = [r["baseline_alignment_error"] for r in rows]
        ice = [r["ice_endpoint_mgdl"] for r in rows]
        span = [r["insulin_span_u"] for r in rows]
        cost = [r["rmse_cost"] for r in rows]
        cdelta = [r["corr_iob_delta"] for r in rows]
        better = [r for r in rows if r["rmse_cost"] < 0]
        worst = max(rows, key=lambda r: r["baseline_alignment_error"])
        rise = max(rows, key=lambda r: r["ice_endpoint_mgdl"])
        tax = max(rows, key=lambda r: r["rmse_cost"])
        long = max(rows, key=lambda r: r["insulin_span_u"])
        out[str(ph)] = dict(
            n=len(rows),
            n_rmse_better=len(better),
            rmse_better_ids=[r["patient_id"] for r in better],
            rmse_cost_min=float(min(cost)),
            rmse_cost_max=float(max(cost)),
            corr_ae_ice_end=_pearson(ae, ice),
            corr_ae_span=_pearson(ae, span),
            corr_rmse_ae=_pearson(cost, ae),
            corr_rmse_iob_delta=_pearson(cost, cdelta),
            worst_ae_id=worst["patient_id"],
            worst_ae=float(worst["baseline_alignment_error"]),
            worst_ice_end=float(worst["ice_endpoint_mgdl"]),
            worst_span=float(worst["insulin_span_u"]),
            rise_id=rise["patient_id"],
            rise_ice_end=float(rise["ice_endpoint_mgdl"]),
            rise_ae=float(rise["baseline_alignment_error"]),
            rise_corr_iob_delta=float(rise["corr_iob_delta"]),
            tax_id=tax["patient_id"],
            tax_rmse=float(tax["rmse_cost"]),
            tax_ae=float(tax["baseline_alignment_error"]),
            tax_span=float(tax["insulin_span_u"]),
            tax_ice_end=float(tax["ice_endpoint_mgdl"]),
            long_id=long["patient_id"],
            long_span=float(long["insulin_span_u"]),
            long_ae=float(long["baseline_alignment_error"]),
            long_ice_end=float(long["ice_endpoint_mgdl"]),
            long_rmse=float(long["rmse_cost"]),
        )
    if dataset == "ohio_t1dm" and "60" in out:
        gap = alignment_disagreement(60)
        extra = [r["extra_drop"] for r in gap]
        rcost = [r["rmse_cost"] for r in gap]
        tax = [r for r in gap if r["rmse_cost"] > 1.0]
        gain = [r for r in gap if r["rmse_cost"] < 0]
        by = {r["patient_id"]: r for r in gap}
        out["60"].update(
            corr_rmse_extra_drop=_pearson(rcost, extra),
            tax_n=len(tax),
            gain_n=len(gain),
            tax_mean_obs=float(np.mean([r["observed_dhi"] for r in tax])),
            tax_mean_tgt=float(np.mean([r["target_ice_q4"] for r in tax])),
            tax_mean_extra=float(np.mean([r["extra_drop"] for r in tax])),
            tax_mean_span=float(np.mean([r["insulin_span_u"] for r in tax])),
            gain_mean_obs=float(np.mean([r["observed_dhi"] for r in gain])),
            gain_mean_tgt=float(np.mean([r["target_ice_q4"] for r in gain])),
            gain_mean_extra=float(np.mean([r["extra_drop"] for r in gain])),
            gain_mean_span=float(np.mean([r["insulin_span_u"] for r in gain])),
            poles={pid: dict(
                observed_dhi=by[pid]["observed_dhi"],
                baseline_ice_q4=by[pid]["baseline_ice_q4"],
                target_ice_q4=by[pid]["target_ice_q4"],
                extra_drop=by[pid]["extra_drop"],
                rmse_cost=by[pid]["rmse_cost"],
                insulin_span_u=by[pid]["insulin_span_u"],
            ) for pid in ("544", "563", "570", "584", "540", "575") if pid in by},
            day_mismatch=alignment_pred_mismatch(60),
        )
    return out


def alignment_pred_mismatch(ph: int = 60) -> dict:
    """Ohio hours where DOCAS lowered the forecast: observed Δ vs |error| gain."""
    import visuals as v
    rows = v._alignment_down_rows(ph)
    key = v.ALIGN_EXPLAINER_WINNER
    def pack(sub):
        x = np.array([r[key] for r in sub], float)
        y = np.array([r["d_err"] for r in sub], float)
        r = float(np.corrcoef(x, y)[0, 1]) if len(sub) > 2 else float("nan")
        fell, rose = y[x < 0], y[x > 0]
        return dict(
            n=len(sub), r=r,
            n_fell=int((x < 0).sum()), n_rose=int((x > 0).sum()),
            d_err_fell=float(fell.mean()) if len(fell) else float("nan"),
            d_err_rose=float(rose.mean()) if len(rose) else float("nan"),
            frac_better_fell=float((fell > 0).mean()) if len(fell) else float("nan"),
            frac_better_rose=float((rose > 0).mean()) if len(rose) else float("nan"))
    by = {pid: pack([r for r in rows if r["patient_id"] == pid])
          for pid in ("544", "575")}
    return dict(all=pack(rows), **by)


def alignment_explainer_rank(ph: int = 60,
                             pids: tuple[str, ...] = ("544", "575")) -> dict:
    """Rank per-prediction explainers of pointwise |error| gain for 544/575."""
    import visuals as v
    from scipy import stats
    rows = v._alignment_pred_rows(ph, pids=pids)
    candidates = [
        ("cgm", "CGM"),
        ("iob", "IOB"),
        ("cho", "CHO"),
        ("y", "observed Δ"),
        ("tau", "target ICE"),
        ("ice_baseline", "baseline ICE"),
        ("ice_docas", "DOCAS ICE"),
        ("leftover", "observed Δ − target ICE"),
        ("ice_misalign", "baseline ICE − target ICE"),
        ("obs_vs_baseline", "observed Δ − baseline ICE"),
        ("cho_hi_iob", "CHO when IOB>1"),
        ("leftover_hi_iob", "leftover when IOB>1"),
    ]
    screens = [
        ("all", lambda r: True),
        ("iob_gt_1", lambda r: r["iob"] > 1.0),
    ]
    scopes = [("all", rows)]
    for pid in pids:
        scopes.append((pid, [r for r in rows if r["patient_id"] == pid]))
    out = {}
    for scr_name, scr_fn in screens:
        out[scr_name] = {}
        for scope_name, scope_rows in scopes:
            sub = [r for r in scope_rows if scr_fn(r)]
            ranked = []
            for key, label in candidates:
                if len(sub) < 3:
                    continue
                x = np.array([r[key] for r in sub], float)
                y = np.array([r["d_err"] for r in sub], float)
                if np.std(x) < 1e-9 or np.std(y) < 1e-9:
                    continue
                rho = float(stats.spearmanr(x, y).statistic)
                r = float(np.corrcoef(x, y)[0, 1])
                ranked.append(dict(key=key, label=label, n=len(sub),
                                   spearman=rho, pearson=r))
            ranked.sort(key=lambda d: abs(d["spearman"]), reverse=True)
            out[scr_name][scope_name] = ranked
    signs = {}
    for key, _ in candidates:
        s544 = next((d["spearman"] for d in out["iob_gt_1"]["544"]
                     if d["key"] == key), float("nan"))
        s575 = next((d["spearman"] for d in out["iob_gt_1"]["575"]
                     if d["key"] == key), float("nan"))
        signs[key] = (np.sign(s544) == np.sign(s575)
                      if np.isfinite(s544) and np.isfinite(s575) else False)
    pool = [d for d in out["iob_gt_1"]["all"] if signs.get(d["key"])]
    winner = pool[0] if pool else out["iob_gt_1"]["all"][0]
    out["winner"] = winner
    out["signs_agree"] = signs
    lines = [f"winner IOB>1U pooled: {winner['label']} "
             f"(Spearman {winner['spearman']:.3f}, Pearson {winner['pearson']:.3f}, "
             f"n={winner['n']})"]
    for scr_name in ("all", "iob_gt_1"):
        lines.append(f"\n=== {scr_name} ===")
        for scope_name in ["all"] + list(pids):
            lines.append(f"  {scope_name}:")
            for d in out[scr_name][scope_name][:5]:
                lines.append(f"    {d['label']:28s}  "
                             f"ρ={d['spearman']:+.3f}  r={d['pearson']:+.3f}  n={d['n']}")
    text = "\n".join(lines)
    print(text)
    out["text"] = text
    return out


def alignment_disagreement(ph: int = 60, dataset: str = "ohio_t1dm") -> list[dict]:
    """Target ICE vs observed ΔCGM on each patient's high-IOB test hours."""
    import visuals as v
    from preprocessing import horizon_steps
    prev = m.DATASET
    m.set_dataset(dataset)
    rows = individual_alignment_rows(dataset, ph)
    out = []
    for r in rows:
        pid = str(r["patient_id"])
        data = m.load_patient(str(m.DATA_ROOT), pid, horizon_steps=horizon_steps(ph))
        bundle = m._load_bundle(int(ph), pid)
        ji = int(data.insulin_idx)
        Xte = np.asarray(data.X_test, float)
        yte = data.cgm_scaler.inverse_transform(
            np.asarray(data.y_test, float).reshape(-1, 1)).ravel()
        iob = data.train_scalers[ji].inverse_transform(Xte[:, ji: ji + 1]).ravel()
        cgm = data.train_scalers[0].inverse_transform(Xte[:, 0: 1]).ravel()
        ci = int(data.feature_names.index("CHO"))
        cho = data.train_scalers[ci].inverse_transform(Xte[:, ci: ci + 1]).ravel()
        hi = iob > np.quantile(iob, 0.75)
        d_hi = float(yte[hi].mean())
        u_q4 = float(iob[hi].mean())
        span = float(r["insulin_span_u"])
        ice_b = float(v._centred_ice(bundle["baseline"], data, Xte[hi], u_q4).mean())
        t_q4 = float(np.mean([
            v._study_drop_curve(np.array([u_q4 / span]), int(ph),
                                span=span, cgm0=float(g))[0]
            for g in cgm[hi]]))
        extra = float(d_hi - t_q4)
        out.append(dict(
            cohort=str(dataset),
            patient_id=pid,
            rmse_cost=float(r["rmse_cost"]),
            ice_endpoint_mgdl=float(r["ice_endpoint_mgdl"]),
            observed_dhi=d_hi, baseline_ice_q4=ice_b, target_ice_q4=t_q4,
            extra_drop=extra, insulin_span_u=span, u_q4=u_q4,
            cgm_q4=float(cgm[hi].mean()), cho_q4=float(cho[hi].mean()),
            observed_dall=float(yte.mean()), cgm_all=float(cgm.mean()),
            cho_all=float(cho.mean()), iob_all=float(iob.mean()),
            corr_te=float(np.corrcoef(iob, yte)[0, 1]),
        ))
    m.set_dataset(prev)
    return sorted(out, key=lambda x: -x["rmse_cost"])


def alignment_disagreement_all(ph: int = 60) -> list[dict]:
    """Ohio + AZT1D high-IOB disagreement rows for the cohort figure."""
    out = []
    for dataset in ("ohio_t1dm", "azt1d"):
        out.extend(alignment_disagreement(ph, dataset))
    return out


def alignment_tax_geography(pid: str = "544", ph: int = 60, *,
                            iob_cut: float = 3.0, cgm_cut: float = 180.0) -> dict:
    """Where held-out |error| rises after alignment (saved CatBoost of one patient)."""
    from preprocessing import horizon_steps
    prev = m.DATASET
    m.set_dataset("ohio_t1dm")
    data = m.load_patient(str(m.DATA_ROOT), str(pid), horizon_steps=horizon_steps(ph))
    bundle = m._load_bundle(int(ph), str(pid))
    ji = int(data.insulin_idx)
    dy_s, ins_s, cgm_s = data.cgm_scaler, data.train_scalers[ji], data.train_scalers[0]
    Xte = np.asarray(data.X_test, float)
    y = dy_s.inverse_transform(np.asarray(data.y_test, float).reshape(-1, 1)).ravel()
    pred = lambda model: dy_s.inverse_transform(
        np.asarray(model.predict(Xte), float).reshape(-1, 1)).ravel()
    dae = np.abs(pred(bundle["docas"]) - y) - np.abs(pred(bundle["baseline"]) - y)
    iob = ins_s.inverse_transform(Xte[:, ji: ji + 1]).ravel()
    cgm = cgm_s.inverse_transform(Xte[:, 0: 1]).ravel()
    pos = np.clip(dae, 0.0, None)
    q75 = float(np.quantile(iob, 0.75))
    corner = (iob > float(iob_cut)) & (cgm > float(cgm_cut))
    q4 = iob > q75
    q1 = iob <= float(np.quantile(iob, 0.25))
    m.set_dataset(prev)
    share = lambda mask: float(pos[mask].sum() / pos.sum()) if pos.sum() else 0.0
    return dict(
        patient_id=str(pid), horizon_min=int(ph), n_test=int(len(y)),
        iob_cut=float(iob_cut), cgm_cut=float(cgm_cut),
        q4_iob=q75,
        q1_mean_dae=float(dae[q1].mean()),
        q4_mean_dae=float(dae[q4].mean()),
        q4_share_pos=share(q4),
        q4_frac=float(q4.mean()),
        corner_frac=float(corner.mean()),
        corner_mean_dae=float(dae[corner].mean()) if corner.any() else float("nan"),
        corner_share_pos=share(corner),
        mean_dae=float(dae.mean()),
    )


AZT1D_MAX_BOLUS_U_PER_DAY = 100.0  # ~2x the upper end of a plausible adult total daily dose


def azt1d_log_audit(*, max_u_per_day: float = AZT1D_MAX_BOLUS_U_PER_DAY) -> dict:
    """Flag AZT1D participants whose logged bolus totals are not physiologically possible.

    Reports cohort alignment/RMSE means with and without them, so the replication
    can be read without the few records that dominate the normalised probe axis.
    """
    import glob, re
    import pandas as pd
    per_day = {}
    for f in sorted(glob.glob(str(ROOT / "azt1d" / "CGM Records" / "Subject */Subject *.csv"))):
        sid = re.search(r"Subject (\d+)\.csv", f).group(1)
        from preprocessing import _read_events
        d = _read_events(Path(f))
        t = pd.to_datetime(d["EventDateTime"], errors="coerce")
        days = max((t.max() - t.min()).days, 1)
        per_day[sid] = float(pd.to_numeric(
            d["TotalBolusInsulinDelivered"], errors="coerce").sum()) / days
    flagged = sorted((s for s, v in per_day.items() if v > float(max_u_per_day)), key=int)
    cohort = json.loads((RESULTS / "azt1d" / "cohort.json").read_text())
    subsets = {}
    for ph in m.FORECAST_HORIZONS:
        rows = cohort["horizons"][str(ph)]["per_patient"]
        keep = [r for r in rows if str(r["patient_id"]) not in flagged]
        subsets[str(ph)] = {
            k: {"mean": float(np.mean([r[k] for r in keep])),
                "std": float(np.std([r[k] for r in keep], ddof=1)), "n": len(keep)}
            for k in ("baseline_alignment_error", "alignment_error",
                      "baseline_alignment_error_abs", "alignment_error_abs",
                      "baseline_test_rmse_mgdl", "docas_test_rmse_mgdl", "insulin_span_u")}
    return dict(bolus_u_per_day=per_day, flagged=flagged,
                threshold_u_per_day=float(max_u_per_day), subsets=subsets)


def tau_scale() -> dict:
    """Magnitude of the target itself, so alignment error has a reference scale."""
    from core import calibrated_target
    out = {}
    for ph in m.FORECAST_HORIZONS:
        R = calibrated_target(int(ph), span=1.0)
        for span, tag in ((m.ABS_DOSE_GRID_U, "abs5u"), (18.74, "median_span")):
            u = np.linspace(0.0, 1.0, int(m.N_GRID_POINTS))
            tau = -180.0 * np.asarray(R(u * span), float).ravel()
            out[f"ph{ph}_{tag}"] = dict(
                max_mgdl=float(abs(tau[-1])), rms_mgdl=float(np.sqrt(np.mean(tau ** 2))))
    return out


def breakeven_lambda() -> dict:
    """Break-even ΔRMSE/ΔAE: DOCAS is preferable above this exchange rate."""
    out = {}
    for name, path in (("ohio", RESULTS / "ohio_t1dm" / "cohort.json"),
                       ("azt1d", RESULTS / "azt1d" / "cohort.json")):
        if not path.is_file():
            continue
        cohort = json.loads(path.read_text())
        for ph, hz in (cohort.get("horizons") or {}).items():
            rows = hz.get("per_patient") or []
            if not rows:
                continue
            d_rmse = np.mean([r["docas_test_rmse_mgdl"] - r["baseline_test_rmse_mgdl"]
                              for r in rows])
            d_ae = np.mean([r["baseline_alignment_error"] - r["alignment_error"]
                            for r in rows])
            # per-patient break-even, so we can report how many patients agree
            per = [(r["docas_test_rmse_mgdl"] - r["baseline_test_rmse_mgdl"]) /
                   max(r["baseline_alignment_error"] - r["alignment_error"], 1e-9)
                   for r in rows]
            out[f"{name}_ph{ph}"] = dict(
                lambda_breakeven=float(max(d_rmse / d_ae, 0.0)),
                lambda_breakeven_max_patient=float(max(per)),
                mean_rmse_cost=float(d_rmse), mean_align_gain=float(d_ae))
    return out


def write_all() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    table_eval_tex()
    table_replay_tex()
    table3_tex()
    for fn, label in ((table_mlp_tex, "MLP"),
                      (table_unguarded_tex, "unguarded"),
                      (table_clinical_replay_tex, "clinical-replay")):
        try:
            fn()
        except FileNotFoundError as e:
            print(f"  skip {label} table: {e}", flush=True)


def build_manuscript() -> None:
    import visuals
    write_all()
    visuals.write_all_figures()


if __name__ == "__main__":
    build_manuscript()
