"""Measure several scans unattended, one GPU each, from a checkout.

Copy scans.example.ini, fill in the scans, then check and start them. The runs
go on in the background after the console is closed.
    python scripts/measure_scans.py scans.ini

What the runs have done, and how much is left.
    python scripts/measure_scans.py scans.ini --status

Stop them. Starting again continues where they ended.
    python scripts/measure_scans.py scans.ini --stop
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from actis.scans import main

if __name__ == "__main__":
    raise SystemExit(main())
