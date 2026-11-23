"""Benchmark a multipage scan from a checkout.

Run every method on every accepted tile of every page.
    python scripts/benchmark_tiff.py scan.tif --reference holes.tif

Time one method on every accepted tile.
    python scripts/benchmark_tiff.py scan.tif --reference holes.tif --method pooled --all-tiles
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from actis.tiff_benchmark import main

if __name__ == "__main__":
    raise SystemExit(main())
