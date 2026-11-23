"""Run from a checkout with python scripts/contour_lab.py [IMAGE ...] to open the contour lab."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from actis.contourlab.server import main

if __name__ == "__main__":
    raise SystemExit(main())
