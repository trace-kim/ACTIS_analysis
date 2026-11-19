"""Run from a checkout with python scripts/benchmark_tiff.py SCAN.tif --reference HOLES.tif."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from actis.tiff_benchmark import main

if __name__ == "__main__":
    raise SystemExit(main())
