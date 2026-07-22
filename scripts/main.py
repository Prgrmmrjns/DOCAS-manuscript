"""DOCAS entrypoint. Edit knobs / uncomment steps below, then: python scripts/main.py"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "replaybg"))

from docas import DOCAS  # PyPI package
import ohio_t1dm_eval

DOCAS.LOAD = False
DOCAS.SEED = 42
DOCAS.SPAN = 10.0
DOCAS.Y0_LO, DOCAS.Y0_HI = 170.0, 280.0
DOCAS.N_F0 = 300
DOCAS.N_U = 21
DOCAS.SYNTH_WEIGHT = 40.0
DOCAS.ALIGN_PASSES = 2
DOCAS.TUNE_CTX_N = 80
DOCAS.U_GRID = np.linspace(0.0, 1.0, 10)
print(f"[DOCAS] τ=f0·[1−R_h] (ReplayBG prior mode)  "
      f"n_f0={DOCAS.N_F0} n_u={DOCAS.N_U} w_synth={DOCAS.SYNTH_WEIGHT} "
      f"passes={DOCAS.ALIGN_PASSES}", flush=True)


def main() -> None:
    ohio_t1dm_eval.set_backend("lgbm")
    # ohio_t1dm_eval.run()
    # ohio_t1dm_eval.run_monotonic()
    # ohio_t1dm_eval.run_lstm()
    # import replay_bg; replay_bg.run()


if __name__ == "__main__":
    main()
