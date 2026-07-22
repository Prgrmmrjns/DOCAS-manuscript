"""End-to-end twinning-replicability test for the multi-meal *extended* model.

The extended counterpart of ``test_multi_meal_model.py``. MAP twinning must be
reproducible: running :meth:`Twinner.twin` twice with the same configuration
returns byte-identical parameters. This guards against non-determinism creeping
into the optimisation (unseeded sampling, order-dependent parallelism, model
state leaking between runs).

The check is done **per** ``n_starts`` setting, not across settings. Two runs
at ``n_starts=1`` must match each other, and two runs at ``n_starts=4`` must
match each other. ``n_starts=1`` and ``n_starts=4`` are *not* expected to agree
with one another: each start ``i`` is seeded by its index, so ``n_starts=4``
explores starts 0-3 and keeps a better optimum than start 0 alone — correct
behavior, not a bug.

The test also pins the twinned parameters as **golden values** (``GOLDEN_X``),
so the fit can't silently change between suite runs. :meth:`Twinner.twin`
returns a single best vector regardless of ``n_starts``, so there is one golden
vector per setting. Regenerate ``GOLDEN_X`` (rerun this test and repaste
``result["x"]``) if the physiological model, the optimiser, or a numerical
dependency (scipy/numpy/BLAS/Numba) legitimately changes.

Prior: the 18-parameter multi-meal prior plus seven second-occurrence channels
(``kabs_B2, kabs_L2, kabs_S2, beta_B2, beta_L2, beta_S2, SI_B2``) → 25
parameters. Unlike the other two tests, this one uses the *entire*
``data_two_day_extended`` trace (both days), so the second-day B2/S2 meals are
present and their channels are exercised. (There is no L2 meal in the data, so
``kabs_L2`` / ``beta_L2`` stay unconstrained — inherent to the trace.)

The run fits 25 parameters over ~29 hours of data
(``example/data_two_day_extended.parquet``, 349 5-minute samples), so it is the
heaviest test in the suite and pays a one-time Numba JIT cost.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from distributions import Gamma, LogNormal, Normal, Uniform
from data.multi_meal_extended_t1d_data import MultiMealExtendedT1DData
from model.multi_meal_extended_t1d import MultiMealExtendedT1DModel
from twinner.twinner import Twinner


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example"

# Golden twinned parameters (order matches _prior()). One best-fit vector per
# n_starts setting. Captured from a real run; see the module docstring for
# when/how to regenerate.
GOLDEN_X = {
    1: np.array([
        131.8714520676213, 0.06287978816766188, 0.940635925674035,
        0.029574489029689014, 0.024977066238940193, 0.0021383477200649526,
        0.05838488389684813, 0.0026491050350779116, 0.0012957054981062766,
        0.0013809007906737778, 0.03013731058785782, 0.05365931279097214,
        0.05123004109425258, 0.0018091168464285832, 2.0, 1.0, 1.0, 0.0,
        0.052549845229120144, 0.0005359079687260058, 0.036020775240217545,
        0.0, 33.0, 4.0, 0.0013580413916806536,
    ]),
    4: np.array([
        129.95456320237116, 0.03315832339046529, 0.6353625577181835,
        0.030094614613511265, 0.040938816935882694, 0.00325151640013785,
        0.10754668981409032, 0.0011464085175792278, 0.0006408548241982597,
        0.0007372106768722687, 0.023950743761488584, 0.029200158749169272,
        0.03599524725626991, 0.0027818395889187765, 5.0, 0.0, 3.0, 0.0,
        0.029307714716893075, 0.0005359079227001439, 0.0244489656872682,
        0.0, 26.0, 2.0, 0.0008410367982316624,
    ]),
}


def _prior():
    """The 25-parameter multi-meal-extended prior used for the check.

    The 18-parameter multi-meal prior plus the seven second-occurrence channels
    (``kabs_B2, kabs_L2, kabs_S2, beta_B2, beta_L2, beta_S2, SI_B2``).
    """
    return {
        "Gb": {"prior": Normal(mu=119.13, sigma=7.11), "min": 70, "max": 150},
        "SG": {"prior": LogNormal(mu=-3.8, sigma=0.05), "min": 0, "max": 0.5},
        "f": {"prior": Normal(mu=0.8, sigma=0.05), "min": 0, "max": 1},
        "p2": {"prior": Normal(mu=0.11, sigma=0.05), "min": 0, "max": 0.5},
        "ka2": {"prior": LogNormal(mu=-4.2875, sigma=0.4274), "min": 0, "max": 0.5},
        "kd": {"prior": LogNormal(mu=-3.5090, sigma=0.6187), "min": 0, "max": 0.5},
        "kempt": {"prior": LogNormal(mu=-1.9646, sigma=0.7069), "min": 0, "max": 0.75},
        "SI_B": {"prior": Gamma(alpha=3.3, beta=1 / 5e-4), "min": 0, "max": 0.1},
        "SI_L": {"prior": Gamma(alpha=3.3, beta=1 / 5e-4), "min": 0, "max": 0.1},
        "SI_D": {"prior": Gamma(alpha=3.3, beta=1 / 5e-4), "min": 0, "max": 0.1},
        "kabs_B": {"prior": LogNormal(mu=-5.4591, sigma=1.4396), "min": 0, "max": 0.5},
        "kabs_L": {"prior": LogNormal(mu=-5.4591, sigma=1.4396), "min": 0, "max": 0.5},
        "kabs_D": {"prior": LogNormal(mu=-5.4591, sigma=1.4396), "min": 0, "max": 0.5},
        "kabs_S": {"prior": LogNormal(mu=-5.4591, sigma=1.4396), "min": 0, "max": 0.5},
        "beta_B": {"prior": Uniform(a=0, b=60), "min": 0, "max": 60, "integer": True},
        "beta_L": {"prior": Uniform(a=0, b=60), "min": 0, "max": 60, "integer": True},
        "beta_D": {"prior": Uniform(a=0, b=60), "min": 0, "max": 60, "integer": True},
        "beta_S": {"prior": Uniform(a=0, b=60), "min": 0, "max": 60, "integer": True},
        "kabs_B2": {"prior": LogNormal(mu=-5.4591, sigma=1.4396), "min": 0, "max": 0.5},
        "kabs_L2": {"prior": LogNormal(mu=-5.4591, sigma=1.4396), "min": 0, "max": 0.5},
        "kabs_S2": {"prior": LogNormal(mu=-5.4591, sigma=1.4396), "min": 0, "max": 0.5},
        "beta_B2": {"prior": Uniform(a=0, b=60), "min": 0, "max": 60, "integer": True},
        "beta_L2": {"prior": Uniform(a=0, b=60), "min": 0, "max": 60, "integer": True},
        "beta_S2": {"prior": Uniform(a=0, b=60), "min": 0, "max": 60, "integer": True},
        "SI_B2": {"prior": Gamma(alpha=3.3, beta=1 / 5e-4), "min": 0, "max": 0.1},
    }


@pytest.fixture
def mme_extended_setup(env):
    """A multi-meal-extended data object + fresh model over the full trace.

    Loads the entire ``example/data_two_day_extended.parquet`` (~29 h, both
    days), so the second-day B2/S2 meals are present and their channels are
    exercised. The extended model needs ``t_start`` (minutes past midnight of
    the first sample) to align its time-of-day insulin sensitivity.
    """
    df = pd.read_parquet(EXAMPLE_DIR / "data_two_day_extended.parquet")
    df["t"] = pd.to_datetime(df["t"])

    rbg_data = MultiMealExtendedT1DData(data=df, body_weight=100, environment=env)
    t_start = int(
        (df["t"].iloc[0] - df["t"].iloc[0].normalize()).total_seconds() / 60
    )
    model = MultiMealExtendedT1DModel(
        u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps, t_start=t_start
    )
    return model, rbg_data


@pytest.mark.slow
@pytest.mark.parametrize("n_starts", [1, 4])
def test_twin_is_replicable(mme_extended_setup, n_starts):
    model, rbg_data = mme_extended_setup
    prior = _prior()

    twinner = Twinner(parallelize=False, n_starts=n_starts, verbose=False)
    r1 = twinner.twin(model, rbg_data, prior)
    r2 = twinner.twin(model, rbg_data, prior)

    # Reproducible within this run: two twins give byte-identical parameters.
    np.testing.assert_array_equal(r1["x"], r2["x"])
    assert r1["fun"] == r2["fun"]

    # Reproducible across suite runs: parameters match the pinned golden values.
    np.testing.assert_allclose(r1["x"], GOLDEN_X[n_starts], rtol=1e-6)
