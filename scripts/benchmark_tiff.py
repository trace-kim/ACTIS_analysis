"""Benchmark a multipage scan from a checkout.

Run every method on every accepted tile of every page. The run ends by writing
every tile's value and opening a map of them in the browser.
    python scripts/benchmark_tiff.py scan.tif --reference holes.tif --json timing.json

Time one method on every accepted tile.
    python scripts/benchmark_tiff.py scan.tif --reference holes.tif --method pooled

Leave the browser closed, or write no map at all.
    python scripts/benchmark_tiff.py scan.tif --reference holes.tif --no-open
    python scripts/benchmark_tiff.py scan.tif --reference holes.tif --no-map
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from actis.tiff_benchmark import main

if __name__ == "__main__":
    raise SystemExit(main())
