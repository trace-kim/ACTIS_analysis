"""Measure every stripe of a scan from a checkout.

Raw brightness of every accepted tile, stripe by stripe.
    python scripts/measure_stripes.py D:\\ACTIS\\scan --reference holes.tiff --output D:\\ACTIS\\scan-raw

A short trial on the first pages of one stripe.
    python scripts/measure_stripes.py D:\\ACTIS\\scan --reference holes.tiff --output trial --stripe 1 --max-pages 50
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from actis.stripes import main

if __name__ == "__main__":
    raise SystemExit(main())
