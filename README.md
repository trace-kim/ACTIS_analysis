# ACTIS reference-image metrology

Measure apparent hole dimensions and relative placement in eight 16-bit grayscale PNGs acquired over the same EUV mask area. The benchmark compares raw brightness, local contours, a pixel-integrated aperture model, a differential fit, integrated intensity, lattice harmonics and pooled aperture moments. It discovers square or staggered arrays and estimates hole shape and orientation directly from the images.

## Run on Windows

Use a Python 3.10+ environment. These PowerShell commands use a repository-local interpreter and work without activating it.

~~~powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[fast,test]"
.\.venv\Scripts\actis analyze-reference --input "D:\ACTIS\reference-pngs" --output "D:\ACTIS\reports\run-01"
~~~

Point `--input` to a folder containing exactly eight single-channel, 16-bit PNGs of the same dimensions. File names are naturally sorted and recorded with SHA-256 hashes. The first acquisition defines the coordinate frame. Use a new or empty output directory. A completed report opens locally in a modern browser.

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

## Inspect lattice harmonics in one image

For a single-image lattice harmonic walkthrough, run the following command from the checkout. Open the resulting HTML locally, select a tile and use the step selector to follow its native detector samples through the Fourier measurement, fit and CD qualification.

~~~powershell
.\.venv\Scripts\python scripts\inspect_harmonics.py "D:\ACTIS\field.tif" -o "D:\ACTIS\field-harmonics.html"
~~~

The inspector accepts single-channel 8- and 16-bit bright-opening PNGs and TIFFs. It defaults to 128 px tiles and automatic shape discovery. Set `--tile-px` for another tile side, `--page` for a zero-based TIFF page, or `--pixel-nm` for calibrated units. The report embeds its image pyramid and measurements in one file. Hover over a pixel for its native DN, click a Fourier order to see its pixel contributions, or use **Hole detail** to compare the fitted tile-average opening with the image.

## Find contours interactively

The contour lab is a window for trying combinations of image-processing steps on real images until the contours sit where they should. It opens in the browser and is served from this PC only. It needs nothing beyond the base install.

~~~powershell
.\.venv\Scripts\python -m actis.contourlab "D:\ACTIS\reference-pngs\ref-01.png" "D:\ACTIS\stripes\2_1_1.tiff"
~~~

The command prints an address, opens it, and keeps running until Ctrl+C. Images can also be opened from the window (**Open images…**, Ctrl+O) or dropped on it. `.\.venv\Scripts\actis contours` and `python scripts\contour_lab.py` start the same window. Plain TIFFs are read with Pillow. For compressed or very large stacks, install the faster reader with `pip install -e ".[tiff]"`.

- **Tabs.** Every image gets a tab. With **all tabs** selected, every tab runs the same workflow, so switching tabs shows it on the other images, at the same place and zoom when the images have the same size. With **this tab**, the tab in front keeps a workflow of its own. **Duplicate tab** opens the same image again to compare two workflows on it.
- **TIFF pages.** A multi-page TIFF is shown one page at a time, with arrows and a page box. **Stack** joins consecutive pages below each other into one strip. **reverse order** and **flip pages** undo the storage order of stripes scanned upwards.
- **Region.** A workflow runs on the whole image, or on a region drawn with Shift + drag. A small region answers at once while options are tried. An image above 6 million pixels starts with a region at its top (the first page of a strip), and a region can hold at most 24 million pixels.
- **The image.** The wheel zooms at the pointer and dragging moves the image. The line under the image gives the pointer's position, the grey level there, the page and row in a strip, and the diameter of the contour under it. Clicking a contour lists its measures. **Pin** keeps the current contours as a second layer while the workflow is changed.
- **The workflow.** Steps run from top to bottom. Each is a module of one of three kinds: *Image* modules change the grey levels, *Mask* modules mark the pattern, *Contours* modules make contours and choose, move or smooth them. Every module and option is explained in the panel. **Show: selected step** displays what the workflow had made after the step clicked in the list.
- **Results.** Contours export as CSV (one row of measures per contour), JSON (with points and the workflow) or NPZ. Lengths are also given in nanometres using the pixel size in the top bar.

Contours are traced between the pixels: a threshold keeps the image minus its level, and the outline is the zero line of that on the cubic spline through the pixel values. A four-pixel hole gets a smooth outline with no upscaling.

**Examples** holds twelve workflows to start from. The table gives their results on synthetic acquisitions from this repository's renderer: 256 × 224 px, eight repeats with gain, offset, illumination, blur, shift and noise varied. The three cases are (A) 80 nm squares at 6.77 px pitch, (B) 100 nm circles at 12 px, staggered, and (C) 70 nm squares at 6 px with stronger blur. Repeatability is the median over sites of the standard deviation of the equal-area diameter across the eight repeats. Size response is the slope of measured against true diameter, where 1 means a size difference is followed one to one.

| Example | Holes found | Repeatability, nm (A / B / C) | Size response (A / B / C) |
|---|---|---|---|
| Bright holes: local half level | 100 % | 0.80 / 0.90 / 2.37 | 1.09 / 1.05 / 1.36 |
| Bright holes: deblurred | 100 % | 0.76 / 0.61 / 2.14 | 1.07 / 1.07 / 1.13 |
| Bright holes: each hole's own half maximum | 100 % | 1.01 / 1.12 / 2.24 | 0.78 / 0.82 / 0.51 |
| Bright holes: one level for the image (Otsu) | 99–100 % | 1.23 / 2.65 / 4.16 | 1.21 / 1.19 / 1.67 |
| Bright holes: steepest gradient | 100 % | 2.08 / 1.80 / 2.31 | 0.92 / 1.00 / 0.49 |
| Edges without a threshold | 99.5–100 % | 0.97 / 0.91 / 1.84 | 0.68 / 0.87 / 0.29 |
| Noisy pages | 100 % | 0.77 / 1.43 / 3.27 | 1.14 / 1.05 / 1.67 |
| Pixel mask route (upscaled) | 100 % | 1.22 / 0.91 / 2.03 | 0.73 / 0.83 / 0.34 |
| Benchmark radial contour | 93–96 % | 1.56 / 1.00 / 2.34 | 0.73 / 0.85 / 0.35 |

*Any pattern* (lines, pads, mixed pages), *Dark features on a bright background* and *One ellipse per hole* use the local half level and score as it does on these arrays. On an 8-bit version of case A with a contrast of ten noise sigma, all holes are still found; repeatability is 2.8 nm for the local half level and 4.6 nm for the benchmark's radial contour. Every example except the benchmark's radial contour gives no contours on a page of noise alone. These numbers describe the renderer's optics, which have no calibration to the instrument. Use them to choose where to start, and judge the contours on real images.

The size of the tiles the local levels come from (`window_px` of the Mid-level threshold, 128 px by default) decides how much of a regional difference is taken for illumination. Smaller tiles follow illumination more closely and follow a regional difference in hole size less.

A workflow saved from the window (**Save…**) runs without it:

~~~python
from actis.contourlab import load_image, load_workflow, measure, workflow_contours

contours = workflow_contours(load_image("2_1_1.tiff", page=17), load_workflow("my workflow.json"))
diameters_px = measure(contours)["diameter"]  # one per contour; contours.split() gives the points
~~~

## Try a complete study without mask data

~~~powershell
.\.venv\Scripts\actis synthetic --output results\example --backend cpu
.\.venv\Scripts\python -m pytest
~~~

The default demo uses a 256 × 192 pixel crop of a dense, axis-aligned square array. The nominal mask opening is 80 nm wide. At 21 nm/pixel its width is 3.81 pixels, and its assumed pitch is 6.77 pixels. At this density a full 676 × 2,048 field would contain about 30,200 sites before edge exclusion. The compact demo uses 1.25 µm tiles and processes all eight simulated exposures through the same estimators as the real-data command. The dimensions, pitch choice and image-formation assumptions are described in the [metrology notes](docs/metrology.md#demo-imagery).

The full-size example in `results/demo` uses 688 columns, 2,048 rows, the same 6.77 px pitch and 5 µm tiles. This field contains more than 30,000 sites with a resolved square aperture shape. Run the following command to reproduce it in a new output directory.

~~~powershell
.\.venv\Scripts\actis synthetic --width 688 --height 2048 --pitch-px 6.77 --tile-um 5 --backend cpu --output results\full-field-example
~~~

Two areas of the mask have wider and narrower openings, with smaller random variation between neighboring sites. These dimensions stay fixed across exposures while the optical blur, detector noise, gain, offset, illumination gradient and registration shift vary. `synthetic-truth.npz` records the actual site dimensions, coordinates and acquisition settings. Generated PNGs carry a simulation tag that survives reanalysis with `analyze-reference`.

Set `--cd-variation-nm 0` for a uniform mask. Other options include `--shape square|circle|ellipse`, `--lattice square|staggered`, `--pitch-px`, `--cd-nm`, `--angle-deg`, `--sigma-xy-px X Y`, `--width`, `--height` and `--seed`. Shape and pitch configure the generator. The analysis infers both from pixel data.

## Read the report

The **Comparison** tab opens with two methods on the same acquisition. Each pane has its own method and filename selectors. To assign acquisitions from the sidebar, choose **A** or **B**. Use **Overlay** to blend the registered images and compare solid A and dashed B annotations.

The viewer opens on **Source PNG** at **1:1**, with one detector sample per physical screen pixel. Magnification uses nearest-neighbor sampling. The pointer readout shows each pane's native pixel coordinates and exact 16-bit DN. The download preserves the input PNG bytes and metadata. Switch to **Shared contrast** when the detector uses a narrow part of its 16-bit range. This applies one display stretch across all acquisitions. Scroll over either image to zoom, drag to pan, or use **Fit** and the field navigator to move around the acquisition.

Click a hole to see both measurements, units, fitted dimensions, flags and any held-out placement residual. **Find site** accepts the region and site ID from the hole export. **Largest difference** jumps to the largest absolute discrepancy among qualified pairs with matching quantities and spatial support. **Shared scale** uses common CD color limits. **Method scale** reveals smaller spatial variation within one method and retains the same limits across acquisitions.

Hole methods draw their measured contours or center markers. For tile methods the rings show which reference sites share one tile estimate. Dashed tile boundaries mark its footprint. **Excluded** shows failed measurements, and the inspector gives the reason. **Image only** lets you examine the detector image alone.

The **Precision** tab contains repeatability, coverage and matched spatial comparisons. **Run details** contains acquisition signal profiles, registration, timings and downloadable measurement tables.

### Diagnose an empty comparison

The detector looks for bright openings on a darker background in the first acquisition. On the workstation, run the following read-only command against the same input folder to check discovery before fitting the stack. It prints one line of counts and rejection reasons. It runs offline and writes no report.

~~~powershell
.\.venv\Scripts\actis diagnose-reference --input "D:\ACTIS\reference-pngs"
~~~

`bright_candidates` counts detected centers. `regions` counts regions with a qualified lattice. With zero regions, the pipeline skips hole fits and region-based tiles. Whole-acquisition mean DN remains available. For runs with an accepted lattice, the method selector distinguishes an empty support from measurements that were attempted and excluded. Each pane shows its exclusion reasons. `run.json` records discovery evidence and the valid, eligible and excluded counts for each acquisition and method.

- **Native page** reproduces one raw arithmetic mean per acquisition. **Common field** uses the overlapping mask area. **Pilot tile** uses complete fixed squares. A 676 × 2,048 source field covers 14.196 × 43.008 µm at the nominal scale. A full 100 × 100 µm historical bin requires a larger measured area.
- **3σ** is three times the sample standard deviation across acquisitions. The percentage is `100 × 3s / |mean|`. Each bin first gets one value per acquisition. The study then averages the binwise scores, giving each valid bin equal weight. The 56 possible three-acquisition subsets appear separately. Their overlap limits interpretation.
- The primary shape-specific CD is circle diameter, square side or ellipse major diameter. Width, height and area-equivalent diameter are also exported. An ambiguous contour uses its area-equivalent diameter. Harmonics report a tile dimension. The flux observable uses DN·px² unless independent bright and dark reference levels are supplied to the estimator.
- **Registration** includes global acquisition shifts and local residual X/Y in nominal mask nm. Positive X points right and positive Y down. Held-out holes score local placement after a translation-only fit on the other half. Repeated patterns can leave a whole-pitch ambiguity. Absolute pattern placement needs a mask coordinate reference.
- **Coverage** accompanies the scores. Per-site or per-tile scores need a finite valid observation in every repeat. A strict shared-site table compares eligible hole-based CD methods on the same sites. Read the excluded-site counts and contour overlays together with the precision results.
- **Synthetic response** tests global CD, local CD, placement, gain, offset, blur, noise, clipping and ellipse aspect ratio. Each discovered region gets a synthetic size sweep at its fitted orientation and lattice vectors, a separate 70 nm challenge, and subpixel phase tests rendered before detector sampling. The comparison scales relative scatter by simulated fractional CD response. Actual offset, optical transfer and detector response remain unknown. Good reproducibility and a useful simulated slope support further qualification with matched mask CD-SEM or AFM.

| Output | Contents |
|---|---|
| `index.html`, `report.js`, `report.css`, `images/` and `data/` | Portable comparison viewer, native image planes, 16-bit downloads, measurement layers and diagnostic charts. Keep these together when copying the report. |
| `holes.csv.gz` | Individual coordinates, shape, axes, CD, flags, quality, contour vertices and model diagnostics in native pixels, with held-out placement residuals in nm |
| `native_tiles.csv` | Raw and harmonic measurements with reference tile indices, phase and fit status |
| `tile_values.csv`, `tile_scores.csv`, `precision.csv` | Per-acquisition matched aggregates, binwise scores and summary metrics |
| `triplets.csv`, `stress.csv`, `comparison.csv` | Three-acquisition subsets, known-geometry challenges and sensitivity-adjusted comparisons |
| `run.json`, `performance.json` | Acquisition ordering, raw intensity profiles, shape evidence, translations, software versions, hardware and timings |

The CSV acquisition indices start at zero. Fitted contours are exported in native (column, row) coordinates. The viewer subtracts each acquisition's measured translation to draw the image and contours in the reference coordinate system. Tiles already use that reference system. Color limits, grayscale stretching and display interpolation only affect viewing.

## Time all seven methods on one multi-page TIFF

From the checkout, install the CUDA extra and supply a holes-only reference page at the scan's pixel scale and orientation. The scan and reference may have different dimensions and bit depths. Both support 8- and 16-bit grayscale.

~~~sh
python -m pip install -e ".[gpu]"
python scripts/benchmark_tiff.py scan.tiff --reference holes.tiff --json timing.json
~~~

If the reference is inside the scan, use `--reference-page 17` in place of `--reference holes.tiff` to use the eighteenth page. The benchmark still processes every page. A separate reference can also be a grayscale PNG or a numbered page in another TIFF.

By default the benchmark screens the entire stack, then times all seven methods on the same **32 accepted tiles**, drawn uniformly across the stack with a fixed random seed. If fewer qualify, it measures them all. The output gives both the accepted and measured counts. Method timings and means describe the measured sample. Use `--sample-tiles N` to change its size, or `--max-pages 10` for a short trial. For exhaustive metrology use `--all-tiles`. That mode fits each accepted tile and can take days on dense stacks.

The console identifies the CUDA device and reports the current stage every 15 seconds, including during reference preparation and individual fits. With `--json timing.json`, the benchmark writes ongoing results to `timing.partial.json` at completed pages or tiles and replaces them with the final `timing.json` on success. An interrupted run retains its most recent partial report. Use a new output name for a new run.

The selector takes one exemplar from each detected reference lattice and tests every complete scan tile against those exemplars, independent of page position. A batched GPU FFT estimates translation. All four quadrants of the overlapping pixels must then match at the same shift. The score is the lowest quadrant correlation after local gain and offset normalization. A blank or striped quadrant can therefore disqualify a tile whose overall correlation is high. This tests for hole arrays represented in the reference. Inspect labeled examples from the instrument when choosing `--min-corr` (default 0.80). The default tile side is 128 pixels. Choose an even `--tile-size` spanning at least five hole pitches. The output counts pixels skipped in incomplete edge tiles.

The script prefetches the next TIFF page while processing bounded tile batches on one CUDA device. It reports scan wall time, pages per second, warm-up, decode, selection, and times and valid counts for all seven methods. Raw, harmonic and pooled counts refer to tiles. Contour, flux, model and differential counts refer to sites. GPU work is synchronized at timing boundaries. TIFF decoding, reference discovery, record assembly and the harmonic nonlinear fit use the CPU. Both aperture fits share contour seeds and blur profiling. Their individual times include that shared work, and the JSON also gives exclusive times. If the effective blur is independently calibrated, `--psf-sigma-px X Y` skips profiling. Use `--shape` only when the reference hole shape is known.

## Throughput and production boundary

The report times input, discovery, alignment, each estimator and rendering separately. Optional `--io-sample "D:\ACTIS\stripes\2_1_1.tiff"` measures sequential file reading on the workstation. The operating system can cache a repeated read. The 1,500-page projections use 4,227,072,000 decoded pixels per board stack. Reference PNG bit depth, pattern density, TIFF compression and parallelism affect transfer to production. The exhaustive method comparison is a qualification run. Evaluate the selected method's throughput separately.

Run the resident-data benchmark to measure the raw, flux, pooled-moment and harmonic tile kernels with bounded concurrency. It reports measured input and valid-area throughput, checksum agreement with serial processing, and comparison against the HDD ingest rate. Warm kernels reuse reference geometry. Read, decode, repeat correspondence, reporting and a new field's geometry discovery have separate costs.

~~~powershell
.\.venv\Scripts\actis benchmark --input "D:\ACTIS\reference-pngs" --output "D:\ACTIS\reports\timing-01.json" --workers 4 --seconds 5 --io-sample "D:\ACTIS\stripes\2_1_1.tiff"
~~~

The production board stacks have shape up to (1500, 2048, 1376) in (page, row, column) order at 8 bits per pixel. A file name encodes stripe, vertical segment and board. The four simultaneous boards have 32-column overlaps and a combined width of 5,408 pixels. Neighboring stripes overlap by 32 columns. Stripe 1 scans down from the top. Even stripes scan up with reversed segment/page order and vertically flipped pages. The far end can overrun the nominal mask boundary by a fractional page, giving neighboring stripes different Y origins. The single-file TIFF benchmark measures native pages. Board and stripe stitching belong to the next phase.

The equipment evidence, method assumptions and physical calibration requirements are documented in [the research notes](docs/metrology.md).
