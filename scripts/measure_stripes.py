"""Measure every stripe of a scan from a checkout.

Raw brightness in 10 um tiles and 100 um bins, stripe by stripe.
    python scripts/measure_stripes.py D:\\ACTIS\\scan --reference holes.tiff --output D:\\ACTIS\\scan-raw

A short trial on the first pages of one stripe, on the second GPU.
    python scripts/measure_stripes.py D:\\ACTIS\\scan --reference holes.tiff --output trial --stripe 1 --max-pages 50 --gpu 1

Bins of 50 x 50 um, every accepted tile averaged.
    python scripts/measure_stripes.py D:\\ACTIS\\scan --reference holes.tiff --output D:\\ACTIS\\scan-50 --bin-y-um 50 --bin-x-um 50 --no-reject
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from actis.stripes import main

if __name__ == "__main__":
    raise SystemExit(main())
