# ACTIS reference-image metrology

Measure apparent hole dimensions and relative placement in eight 16-bit grayscale PNGs acquired over the same EUV mask area. The benchmark compares raw brightness, local contours, a pixel-integrated aperture model, a differential fit, integrated intensity, lattice harmonics and pooled aperture moments. It discovers square or staggered arrays and estimates hole shape and orientation directly from the images.

## Run on Windows

Use a Python 3.10+ environment. These PowerShell commands use a repository-local interpreter and work without activating it.

~~~powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[fast,test]"
.\.venv\Scripts\actis analyze-reference --input "D:\ACTIS\reference-pngs" --output "D:\ACTIS\reports\run-01"
~~~

Point `--input` to a folder containing exactly eight single-channel, 16-bit PNGs of the same dimensions. File names are naturally sorted and recorded with SHA-256 hashes. The first acquisition defines the coordinate frame. Use a new or empty output directory. A completed report opens locally in any browser.

~~~powershell
Start-Process "D:\ACTIS\reports\run-01\index.html"
~~~

The default mask scale is 21 nm/pixel and the pilot tile side is 5 µm. Override these with `--pixel-nm` and `--tile-um`. The described reference images are 2,048 rows by 676 columns. Other dimensions are recorded in the report and must agree across repeats. The `fast` extra installs the compiled CPU path for pooled aperture moments. A NumPy implementation is available in the base install.

For the optional CUDA fitting backend, install the GPU extra in the same environment and select a device. It uses bounded CuPy batches for the aperture and differential fits. The other estimators run on CPU. The reference study uses one selected GPU per process. Runtime records include the actual backend and device. CPU remains available with `--backend cpu`. [CuPy installation](https://docs.cupy.dev/en/stable/install.html)

~~~powershell
.\.venv\Scripts\python -m pip install -e ".[gpu]"
$env:CUDA_VISIBLE_DEVICES = "0"
.\.venv\Scripts\actis analyze-reference --input "D:\ACTIS\reference-pngs" --output "D:\ACTIS\reports\run-gpu" --backend cuda
~~~

## Try a complete study without mask data

~~~powershell
.\.venv\Scripts\actis synthetic --output results\example --backend cpu
.\.venv\Scripts\python -m pytest
~~~

The synthetic command writes eight PNGs, ground truth and a full report. Defaults use a staggered array of ellipses. Options include `--shape circle|square|ellipse`, `--lattice square|staggered`, `--cd-nm`, `--pitch-px`, `--angle-deg`, `--width`, `--height` and `--seed`. Shape and pitch configure the **generator**. The analysis recovers them blind. Generated PNGs carry a provenance tag so a subsequent reference-folder run still identifies them as simulated.

## Read the report

- **Native page** reproduces one raw arithmetic mean per acquisition. **Common field** uses the overlapping mask area. **Pilot tile** uses complete fixed squares. The eight crops cover the same 14.196 × 43.008 µm field at nominal scale. A full 100 × 100 µm historical bin requires a larger measured area.
- **3σ** is three times the sample standard deviation across acquisitions. The percentage is `100 × 3s / |mean|`. Each bin first gets one value per acquisition. The study then averages the binwise scores, giving each valid bin equal weight. The 56 possible three-acquisition subsets appear separately. Their overlap limits interpretation.
- The primary shape-specific CD is circle diameter, square side or ellipse major diameter. Width, height and area-equivalent diameter are also exported. An ambiguous contour uses its area-equivalent diameter. Harmonics report a tile dimension. The flux observable uses DN·px² unless independent bright and dark reference levels are supplied to the estimator.
- **Registration** includes global acquisition shifts and local residual X/Y in nominal mask nm. Positive X points right and positive Y down. Held-out holes score local placement after a translation-only fit on the other half. Repeated patterns can leave a whole-pitch ambiguity. Absolute pattern placement needs a mask coordinate reference.
- **Coverage** accompanies the scores. Per-site or per-tile scores need a finite valid observation in every repeat. A strict shared-site table compares eligible hole-based CD methods on the same sites. Read the excluded-site counts and contour overlays together with the precision results.
- **Synthetic response** tests global CD, local CD, placement, gain, offset, blur, noise, clipping and ellipse aspect ratio. Each discovered region gets a synthetic size sweep at its fitted orientation and lattice vectors, a separate 70 nm challenge, and subpixel phase tests rendered before detector sampling. The comparison scales relative scatter by simulated fractional CD response. Actual offset, optical transfer and detector response remain unknown. Good reproducibility and a useful simulated slope support further qualification with matched mask CD-SEM or AFM.

| Output | Contents |
|---|---|
| `index.html` and `images/` | Offline report, native images, overlays, detail crops, precision maps and placement vectors |
| `holes.csv.gz` | Individual coordinates, shape, axes, CD, flags, quality, contour vertices and model diagnostics in native pixels, with held-out placement residuals in nm |
| `native_tiles.csv` | Raw and harmonic measurements with reference tile indices, phase and fit status |
| `tile_values.csv`, `tile_scores.csv`, `precision.csv` | Per-acquisition matched aggregates, binwise scores and summary metrics |
| `triplets.csv`, `stress.csv`, `comparison.csv` | Three-acquisition subsets, known-geometry challenges and sensitivity-adjusted comparisons |
| `run.json`, `performance.json` | Acquisition ordering, raw intensity profiles, shape evidence, translations, software versions, hardware and timings |

The CSV acquisition indices start at zero. Overlay positions and fitted contours use native (column, row) coordinates. Tiles use the first acquisition's coordinates. The color limits and grayscale stretch affect display only.

## Throughput and production boundary

The report times input, discovery, alignment, each estimator and rendering separately. Optional `--io-sample "D:\ACTIS\stripes\2_1_1.tiff"` measures sequential file reading on the workstation. The operating system can cache a repeated read. The 1,500-page projections use 4,227,072,000 decoded pixels per board stack. Reference PNG bit depth, pattern density, TIFF compression and parallelism affect transfer to production. The exhaustive method comparison is a qualification run. Evaluate the selected method's throughput separately.

Run the resident-data benchmark to measure the raw, flux, pooled-moment and harmonic tile kernels with bounded concurrency. It reports measured input and valid-area throughput, checksum agreement with serial processing, and comparison against the HDD ingest rate. Warm kernels reuse reference geometry. Read, decode, repeat correspondence, reporting and a new field's geometry discovery have separate costs.

~~~powershell
.\.venv\Scripts\actis benchmark --input "D:\ACTIS\reference-pngs" --output "D:\ACTIS\reports\timing-01.json" --workers 4 --seconds 5 --io-sample "D:\ACTIS\stripes\2_1_1.tiff"
~~~

The TIFF contract is recorded for a later implementation. Arrays have shape up to (1500, 2048, 1376) in (page, row, column) order at 8 bits per pixel. A file name encodes stripe, vertical segment and board. The four simultaneous boards have 32-column overlaps and a combined width of 5,408 pixels. Neighboring stripes overlap by 32 columns. Stripe 1 scans down from the top. Even stripes scan up with reversed segment/page order and vertically flipped pages. The far end can overrun the nominal mask boundary by a fractional page, giving neighboring stripes different Y origins. The reference benchmark reads PNGs. TIFF stitching, full-mask streaming and workbench/UI development belong to the next phase.

The equipment evidence, method assumptions and physical calibration requirements are documented in [the research notes](docs/metrology.md).
