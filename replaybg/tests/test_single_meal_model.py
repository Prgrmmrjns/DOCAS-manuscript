"""End-to-end twinning-replicability test for the single-meal T1D model.

MAP twinning must be reproducible: running :meth:`Twinner.twin` twice with the
same configuration returns byte-identical parameters. This guards against
non-determinism creeping into the optimisation (unseeded sampling,
order-dependent parallelism, model state leaking between runs).

The check is done **per** ``n_starts`` setting, not across settings. Two runs
at ``n_starts=1`` must match each other, and two runs at ``n_starts=4`` must
match each other. Crucially, ``n_starts=1`` and ``n_starts=4`` are *not*
expected to agree with one another: each start ``i`` is seeded by its index, so
``n_starts=4`` explores starts 0-3 and keeps a better optimum than start 0
alone — that is correct behavior, not a bug.

The test also pins the twinned parameters as **golden values** (``GOLDEN_X``),
so the fit can't silently change between suite runs. Note that
:meth:`Twinner.twin` returns a single best vector regardless of ``n_starts``
(it keeps the lowest-objective start), so there is one golden vector per
setting. Regenerate ``GOLDEN_X`` (rerun this test and repaste ``result["x"]``)
if the physiological model, the optimiser, or a numerical dependency
(scipy/numpy/BLAS/Numba) legitimately changes.

The run fits the full 10-parameter prior over 13 hours of data
(``example/data_single_meal.parquet``, 157 5-minute samples), so it pays a
one-time Numba JIT cost and is heavier than the unit tests.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from distributions import Gamma, LogNormal, Normal, Uniform
from data.single_meal_t1d_data import SingleMealT1DData
from model.single_meal_t1d import SingleMealT1DModel
from twinner.twinner import Twinner


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example"

# Golden twinned parameters (order matches _prior(): Gb, SG, p2, f, ka2, kd,
# kempt, SI, kabs, beta). One best-fit vector per n_starts setting. Captured
# from a real run; see the module docstring for when/how to regenerate.
GOLDEN_X = {
    1: np.array([
        118.47977090583834, 0.01897463544409741, 0.10991966371112584,
        0.8552028954054943, 0.010766231400495799, 0.014665112234466256,
        0.1096117986839381, 0.0006314401730194213, 0.01696724590290372,
        0.0,
    ]),
    # On this trace start 0 also happens to be the best of starts 0-3, so the
    # two settings currently agree; that is incidental, not a requirement.
    4: np.array([
        118.47977090583834, 0.01897463544409741, 0.10991966371112584,
        0.8552028954054943, 0.010766231400495799, 0.014665112234466256,
        0.1096117986839381, 0.0006314401730194213, 0.01696724590290372,
        0.0,
    ]),
}


def _prior():
    """The 10-parameter single-meal prior used for the replicability check."""
    return {
        "Gb": {"prior": Normal(mu=119.13, sigma=7.11), "min": 70, "max": 150},
        "SG": {"prior": LogNormal(mu=-3.8, sigma=0.5), "min": 0, "max": 0.5},
        "p2": {"prior": Normal(mu=0.11, sigma=0.004), "min": 0, "max": 0.5},
        "f": {"prior": Normal(mu=0.8, sigma=0.05), "min": 0, "max": 1},
        "ka2": {"prior": LogNormal(mu=-4.2875, sigma=0.4274), "min": 0, "max": 0.5},
        "kd": {"prior": LogNormal(mu=-3.5090, sigma=0.6187), "min": 0, "max": 0.5},
        "kempt": {"prior": LogNormal(mu=-1.9646, sigma=0.7069), "min": 0, "max": 0.75},
        "SI": {"prior": Gamma(alpha=3.3, beta=1 / 5e-4), "min": 0, "max": 0.1},
        "kabs": {"prior": LogNormal(mu=-5.4591, sigma=1.4396), "min": 0, "max": 0.5},
        "beta": {"prior": Uniform(a=0, b=60), "min": 0, "max": 60, "integer": True},
    }


@pytest.fixture
def sm_setup(env):
    """A single-meal data object + fresh model over the fake single-meal trace.

    Loads ``example/data_single_meal.parquet`` — a 13-hour, 157-sample window
    (06:00-19:00) holding two bolused meals, at 07:00 and 13:00, which makes the
    kinetic parameters identifiable.
    """
    df = pd.read_parquet(EXAMPLE_DIR / "data_single_meal.parquet")
    df["t"] = pd.to_datetime(df["t"])

    rbg_data = SingleMealT1DData(data=df, body_weight=100, environment=env)
    model = SingleMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps)
    return model, rbg_data


@pytest.mark.slow
@pytest.mark.parametrize("n_starts", [1, 4])
def test_twin_is_replicable(sm_setup, n_starts):
    model, rbg_data = sm_setup
    prior = _prior()

    twinner = Twinner(parallelize=False, n_starts=n_starts, verbose=False)
    r1 = twinner.twin(model, rbg_data, prior)
    r2 = twinner.twin(model, rbg_data, prior)

    # Reproducible within this run: two twins give byte-identical parameters.
    np.testing.assert_array_equal(r1["x"], r2["x"])
    assert r1["fun"] == r2["fun"]

    # Reproducible across suite runs: parameters match the pinned golden values.
    np.testing.assert_allclose(r1["x"], GOLDEN_X[n_starts], rtol=1e-6)
