"""Where the CreditTrading checkout is, for the scripts in this folder.

WHY: this study moved on 2026-11-24 from a CreditTrading branch to this separate research
repository. The scripts
import the shared harness (`scripts/research/harness.py`, which must hold `evaluate_window`),
`quantt.broker.alpaca`, `scripts/cef/band_frontier.py` and the prior study
`results/ideas/levered_etf_reversal/` from that checkout, so they no longer sit inside it.
`desk/README.md` says to name it on PYTHONPATH. There is no default: a guessed checkout could
supply an older harness or a different estimator without saying so.
"""
from __future__ import annotations

import sys
from pathlib import Path


def credittrading_root() -> Path:
    for entry in sys.path:
        p = Path(entry) if entry else None
        if p and (p / "scripts/research/harness.py").is_file() and (p / "quantt").is_dir():
            return p.resolve()
    raise SystemExit("set PYTHONPATH to a CreditTrading checkout (one holding "
                     "scripts/research/harness.py and quantt/): see desk/README.md")
