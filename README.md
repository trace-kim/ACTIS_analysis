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

The contour lab is a window for trying methods on real images until the contours sit where they should. It opens in the browser and is served from this PC only. It needs nothing beyond the base install.

~~~powershell
.\.venv\Scripts\python -m actis.contourlab "D:\ACTIS\reference-pngs\ref-01.png" "D:\ACTIS\stripes\2_1_1.tiff"
~~~

The command prints an address, opens it, and keeps running until Ctrl+C. Images can also be opened from the window (**+ Open**, Ctrl+O) or dropped on it. `.\.venv\Scripts\actis contours` and `python scripts\contour_lab.py` start the same window. Plain TIFFs are read with Pillow. For compressed or very large stacks, install the faster reader with `pip install -e ".[tiff]"`.

- **Left: the method.** A method is a short list of steps that run from top to bottom. **Choose a method…** offers eleven to start from. Click a step to change its settings; the **?** beside a setting says what it does. **More** opens and saves method files, gives a tab a method of its own, and duplicates a tab to compare two methods on one image.
- **Middle: the image.** Every image gets a tab, and all tabs run the same method, so switching tabs compares images at the same place and zoom. The wheel zooms at the pointer and dragging moves the image. A multi-page TIFF gets a page bar under the image; **Stack** joins pages into one strip. **Run on** (bottom left) limits the method to what the window shows or to a box drawn with Shift + drag; an image above 6 million pixels starts with its first page.
- **Right: the numbers.** How many contours, and one measure at a time: diameter, diameter against the eight nearest contours, width, height, elongation or brightness, each with its typical value, spread and histogram. The same measure colours the contours, so the histogram is the key to the colours. Click a contour for its own numbers. **Keep these as a reference** leaves the contours on screen in green while another method is tried. Contours save as CSV, JSON or NPZ, and the view as a PNG.

Contours are drawn 2.5 px wide with a dark edge (**Display** changes the width and the colour). Zoomed out, each contour becomes a disc of its own size and colour, which makes a map of the measure across the page. The graphics card draws them (WebGL 2), so a page of 61,000 contours moves at the display's frame rate; a browser without it falls back to a slower canvas drawing.

Contours are traced between the pixels: a threshold keeps the image minus its level, and the outline is the zero line of that on the cubic spline through the pixel values. A four-pixel hole gets a smooth outline with no upscaling.

### The methods

The method the window opens with, **Shared half level**, outlines every feature half way between its top and the dark level around it. Both levels are the medians over the feature's 64 nearest features of like contrast, so no tile is laid over the image: faint holes beside a bright pad, sparse holes, lines and pads each get the level of their own kind. A bump counts as a feature only if it stands out from its surroundings and has an edge within reach, measured against the noise at its own brightness, so empty areas give no contours, dark or bright. **Own half level** measures each feature against its own top alone. **Local tiles** is the method the lab opened with before.

`python scripts/evaluate_contour_flows.py` scores every method on synthetic images from this repository's renderer. The hole arrays are 256 × 224 px, eight acquisitions with gain, offset, illumination, blur, shift and noise varied: (A) 80 nm squares at 6.77 px pitch, (B) 100 nm circles at 12 px, staggered, (C) 70 nm squares at 6 px with stronger blur. Repeatability is the median over sites of the standard deviation of the equal-area diameter across the eight acquisitions. Size response is the slope of measured against true diameter: 1 means a size difference is followed one to one. The last column is case A in 8 bits at a contrast of about ten noise sigma.

| Method | Holes found, % (A / B / C) | Repeatability, nm | Size response | Noisy 8-bit A: found %, nm |
|---|---|---|---|---|
| Shared half level | 100 / 100 / 100 | 1.06 / 0.75 / 1.97 | 0.76 / 0.97 / 0.67 | 100, 3.2 |
| Own half level | 100 / 100 / 100 | 1.21 / 0.96 / 2.07 | 0.60 / 0.83 / 0.38 | 100, 4.8 |
| Optical low-pass first | 100 / 100 / 100 | 1.15 / 1.01 / 1.91 | 0.80 / 1.00 / 0.64 | 100, 3.3 |
| Deblurred half level | 100 / 100 / 100 | 1.48 / 1.38 / 1.97 | 1.00 / 1.04 / 0.67 | 100, 4.5 |
| Steepest edge | 100 / 100 / 90.6 | 1.13 / 0.93 / 2.23 | 0.55 / 0.86 / 0.29 | 89, 5.1 |
| Local tiles | 100 / 100 / 100 | 1.26 / 0.87 / 2.45 | 1.10 / 1.07 / 1.44 | 100, 4.8 |
| One level for the page | 100 / 100 / 92.1 | 1.55 / 2.60 / 4.48 | 1.27 / 1.20 / 1.88 | 96, 9.9 |
| Pixel mask route | 100 / 100 / 100 | 1.82 / 0.92 / 2.52 | 0.61 / 0.84 / 0.34 | 100, 5.0 |
| Benchmark radial contour | 100 / 100 / 98.8 | 1.95 / 1.02 / 2.47 | 0.60 / 0.85 / 0.33 | 100, 5.4 |

The second test is a mixed 8-bit page of 1,024 × 768 px with uneven illumination and shot noise: dense small holes, wide holes, faint holes 40 px apart, lines, a pad of 300 × 260 px with faint holes 12 px beside it, a bright area with dark dots in it, and an empty dark area.

| Method | Faint holes by the pad found, % | Dark dots found, % | Contours in empty areas |
|---|---|---|---|
| Shared half level, Own half level, Optical low-pass first, Deblurred half level, Steepest edge | 100 | 100 | 0 |
| Local tiles | 82 | 0 | 87 |
| One level for the page | 100 | 0 | 0 |
| Pixel mask route | 73 | 0 | 44 |

Every method finds the other holes, the lines and the pad on that page, except that the benchmark's radial contour is for hole arrays only. *Local tiles*, *One level for the page* and *Pixel mask route* leave out inner boundaries by their settings, which is why they have no dark dots. No method gives a contour on a page of noise alone.

What these numbers do not show:

- The renderer's optics have no calibration to the instrument. Use the numbers to choose where to start, and judge the contours on real images.
- *Shared half level* follows a size difference less than one to one, where *Local tiles* follows it more than one to one. The number of features that share their levels (64) sets this: more follows size more closely and illumination less.
- Faint features within about 15 px of a wide bright one can pull its outline towards them by a pixel or two.
- *Optical low-pass first* cuts off the spatial frequencies beyond those the image holds pattern at. The renderer blurs with a Gaussian, which leaves pattern up to the sampling limit, so on these images the step removes nothing or only the corners of the spectrum. On images with a cutoff inside the sampled band (tested with one imposed at 0.27 cycles per pixel) it finds the cutoff and halves the noise. Whether ACTIS images have one shows in what the step reports. On small or noisy images set the cutoff by hand, the same for every image.

A method saved from the window (**More**, *Save the method to a file*) runs without it:

~~~python
from actis.contourlab import load_image, load_workflow, measure, workflow_contours

contours = workflow_contours(load_image("2_1_1.tiff", page=17), load_workflow("my method.json"))
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

If the reference is inside the scan, use `--reference-page 17` in place of `--reference holes.tiff` to use the eighteenth page. A separate reference can also be a grayscale PNG or a numbered page in another TIFF.

The default run screens **every page** and measures **all seven methods on every accepted tile**. The following explicit flags select the same full-file run.

~~~sh
python scripts/benchmark_tiff.py scan.tiff --reference holes.tiff --all-pages --all-tiles --json full-timing.json
~~~

### What a run leaves behind

The console prints one row per method: time, valid and attempted counts, and the mean. The run also writes three files.

| File | Contents |
|---|---|
| `timing.json` | Times per method and per stage, counts, the blur found on the pages and the settings of the run. `total_seconds` includes reference preparation, reading, screening and measurement. |
| `timing.cd-map.npz` | Every tile's value. One array per method, shaped (page, tile row, tile column), with NaN where a tile was not accepted or the method gave no valid value. |
| `timing.cd-map.html` | The map of those values, one self-contained page. It opens in the browser when the run ends. |

Without `--json` the two map files take the TIFF's name and go to the current folder. `--map results/scan.html` chooses another place, `--no-map` skips both files, and `--no-open` leaves the browser closed. A run whose output is redirected to a file does not open the browser either.

A tile's value is the tile mean brightness for `raw`, one fitted size for `harmonic` and `pooled`, and the mean over the tile's valid openings for `contour`, `model`, `differential` and `flux`. For those four the file also holds `<method>_openings`, the number of valid openings behind each mean. `reference` gives the matched reference lattice of each tile (-1 for none), `page_index` the zero-based TIFF page of each row, and `blur_sigma_px` and `blur_usable` the blur of each page and reference lattice.

~~~python
import numpy as np

tiles = np.load("timing.cd-map.npz")
contour_px = tiles["contour"]  # (page, tile row, tile column)
page_medians = np.nanmedian(contour_px.reshape(len(contour_px), -1), axis=1)
~~~

The map has four parts.

- **Methods.** Time, tiles with a value, typical value and tile-to-tile spread of all seven methods in one table. Selecting a row shows that method everywhere else.
- **Every tile of the file.** The pages laid end to end, coloured by how far each tile is from the method's typical value, above a line of page medians. The wheel zooms, dragging moves, and pointing at a tile gives its value in all seven methods.
- **One page, all methods.** The page that was clicked, as seven small grids, with its numbers in a table on request.
- **How tile values spread**, and **the blur found on each page**.

When the reference holds several hole arrays, a selector at the top shows one array at a time, so a tile is compared with tiles of its own array. `reference_shapes` and `reference_pitch_px` in the JSON list them.

### Speed

A full run on a local RTX 4060 Ti (16 GB) and Ryzen 5 7500F took 2 min 20 s from command launch to exit. The 4.2 GB uncompressed synthetic TIFF had 1,500 grayscale 8-bit pages of 2,048 rows by 1,376 columns, the size of a production board stack. All seven methods ran on all 232,040 accepted tiles out of 240,000 screened, which hold 59.4 million openings. The seconds below count each stage once.

| Stage | Seconds |
|---|---|
| Read and decode | 1.8 |
| Screen tiles against the reference | 6.9 |
| Raw brightness | 0.3 |
| Contours | 19.9 |
| Blur search and fit preparation, shared by the two aperture fits | 28.3 |
| Aperture model fits | 16.5 |
| Differential fits | 7.0 |
| Integrated intensity | 6.7 |
| Lattice harmonics | 18.1 |
| Pooled moments | 27.2 |

Once you have chosen a method, use `--method` to measure it alone on every accepted tile. The other choices are `raw`, `contour`, `flux`, `harmonic`, `model` and `differential`.

~~~sh
python scripts/benchmark_tiff.py scan.tiff --reference holes.tiff --method pooled --json pooled.json
~~~

The benchmark uses one CUDA device, chosen with `--device`. On a workstation with several, start one run per device on different files.

### How the methods run

The script prefetches the next TIFF page while it measures the current one. A page's complete tiles are screened together, and its accepted tiles are then measured together by every method. GPU work is synchronized at timing boundaries. TIFF decoding, reference discovery and the bookkeeping of results use the CPU.

Radial contours, flux apertures and the two aperture fits run as CUDA kernels over all openings of a page, for pitches up to 27 pixels. Larger pitches use the original per-tile estimators. Every opening keeps its own contour seed, position, fitted size and validity tests. The lattice-harmonic fit also runs on the GPU for all tiles of a page at once. It solves the same bounded least-squares problem from the same three starts as the CPU reference in `actis.harmonics`, and the tests hold the two to the same accepted tiles and sizes.

### Blur

The aperture model and the differential fit compare each opening with a blurred template of a known shape, so they need the blur of the optics. The other five methods do not take a blur. No blur has to be supplied. On each page the benchmark tries a short list of blur settings on up to ten openings from every accepted tile and keeps the setting whose templates reproduce those pixels best. Every tile of the page is then fitted with it. The console, the JSON and the map report what was found. A page whose best settings disagree about the size, or whose blur lies at the end of the search range, gives no aperture-fit values. `--psf-sigma-px X Y` replaces the search with a fixed blur.

The fitted size depends on the blur the fit assumes, and the search lands on a small set of values. The synthetic stack above shows the effect. Its mask is the same on every page, and its blur rises in two steps of about 0.1 px, after page 500 and after page 1000.

| Method | Change of page median at the first step | At the second step |
|---|---|---|
| Contour | -0.3% | +0.2% |
| Aperture model | +0.3% | -2.8% |
| Differential fit | +0.3% | -2.5% |
| Lattice harmonics | -0.6% | -0.9% |
| Pooled moments | -0.8% | -0.7% |

The blur search returned 0.85 by 1.01 px on both sides of the first step (true 0.78 by 0.96, then 0.88 by 1.04) and 1.09 by 1.25 px after the second (true 1.00 by 1.14). This is one synthetic stack with a Gaussian blur. On real pages, read the blur chart of the map beside the page medians of the two aperture fits: a jump in both at the same page is the blur search, not the mask.

### Sampling

The console identifies the CUDA device and reports the current stage every 15 seconds, including during reference preparation. With `--json timing.json`, the benchmark writes ongoing results to `timing.partial.json` at completed pages and replaces them with the final `timing.json` on success. An interrupted run retains its most recent partial report. Use a new output name for a new run.

Only the explicit options `--sample-pages N` and `--max-pages N` limit the pages screened. An explicit `--sample-tiles N` selects a reproducible uniform sample of accepted tiles. The sampled tiles of a page are measured together and share that page's blur. For a tile sample, the console and JSON give an estimated time to run each method on every accepted tile on the screened pages. The combined `estimated_all_tiles_method_seconds` counts shared fit setup once. Reference preparation, TIFF reads and tile selection have separate timings. With either page limit, counts and projections cover only the pages screened.

### How tiles are chosen

The selector takes one exemplar from each detected reference lattice and tests every complete scan tile against those exemplars, independent of page position. A batched GPU FFT estimates translation. All four quadrants of the overlapping pixels must then match at the same shift. The score is the lowest quadrant correlation after local gain and offset normalization. A blank or striped quadrant can therefore disqualify a tile whose overall correlation is high. This tests for hole arrays represented in the reference. Inspect labeled examples from the instrument when choosing `--min-corr` (default 0.80). The default tile side is 128 pixels. Choose an even `--tile-size` spanning at least five hole pitches. The output counts pixels skipped in incomplete edge tiles. Use `--shape` only when the reference hole shape is known. The JSON lists the shape found for each reference lattice in `reference_shapes`. With the shape `ambiguous`, the two aperture fits and the lattice harmonics give no sizes.

## Measure every stripe of a scan

`measure_stripes.py` runs a method on all board stacks of a scan, in tiles of about 10 µm, and averages the tiles into bins of about 100 µm for a map. The method is raw brightness unless `--method` names others. `--method all` runs all seven on the same tiles in one pass over the scan. It needs the CUDA extra and a holes-only reference, like the TIFF benchmark above. To measure several scans unattended, each on its own GPU, see [Measure several scans unattended](#measure-several-scans-unattended).

~~~sh
python scripts/measure_stripes.py D:\ACTIS\scan --reference holes.tiff --output D:\ACTIS\scan-raw
~~~

The scan folder holds the header and the board stacks, which may lie at any depth below it, for example in `RAW\S1`. A board stack is named `<stripe>_<vertical segment>_<board>.tiff`, so `1_1_1.tiff` to `1_1_4.tiff` are the four boards of the first segment of stripe 1. Only the file names are read, not the folder names. A name that occurs twice under the scan folder stops the run and names both files.

The pixel size comes from the scan's header, read with `actis.header` (`find_header` and `parse_header`, whose `pixel_x` and `pixel_y` are in µm). That module is the workstation's own and is not part of this repository. `--pixel-nm 21` gives the size for both directions instead, for a scan without a readable header or a PC without the module. `stripes.json` records the size used and where it came from.

### Boards are merged first

Page by page, the four boards of a segment are merged into one page of 5,408 columns (4 × 1,376 − 3 × 32), and that page is what gets screened and measured. Neighbouring boards image the same 32 columns. Each board gives up 16 of them, so every column is measured once and comes from the board it lies deeper inside. Boards go left to right in number order. `--overlap` and `--boards` set other numbers.

For every page the run also compares the 32 columns as each of the two boards recorded them. A correlation near 1 means the boards overlap as assumed. A low value on pages that show pattern at the seam means they do not, by board order, overlap width or alignment. The console prints one number per seam: the level the best tenth of the segment's pages reach, so empty pages do not pull it down.

### Tiles and bins

The merged pages of a segment are stacked into one strip, in the order and orientation of the files, and the strip is cut into square tiles. Reversed page order and flipped pages of a stripe scanned upwards are left to the map.

- **Tiles.** `--tile-um 10` sets the side, which becomes the nearest even number of pixels at the scan's pixel size: 476 px at 21 nm. Across the stripe the tiles are centred. 5,408 columns hold 11 tiles and leave 86 columns at each edge, which no tile uses, so the 32 columns a stripe shares with each neighbour are measured by neither. Along the stripe a tile may run from one page into the next. The rows at the end of a segment that do not fill a tile are not used.
- **Screening.** Every tile is compared with the reference as before, and only tiles that match it are measured.
- **Bins.** `--bin-y-um 100` sets the length along the stripe and `--bin-x-um` the width across it, each made a whole number of tiles: 10 × 11 tiles at the defaults. A bin is at least one tile and never wider than the stripe, and no bin reaches into a neighbouring stripe. Without `--bin-x-um` a bin is as wide as every tile that fits across, 11 tiles or 110 µm at 21 nm per pixel. A narrower width that leaves tiles over centres the bins, and the tiles left at the edges are in no bin. Bins begin at the first tile row of the segment. The last bin row of a segment holds the tile rows that are left, so it can be shorter.
- **A bin's value.** The mean of its accepted tiles, after leaving out each tile more than 1.5 standard deviations from the mean of the bin's tiles. The mean and the sample standard deviation are taken once, over all accepted tiles of the bin. `--reject-sigma 2` sets another threshold, and `--no-reject` averages every accepted tile. A bin with one tile keeps it. Each method is filtered on its own values.

The threshold cannot reject anything in very small bins. No tile of n can lie more than (n − 1)/√n standard deviations from their mean. That is 1.5 for four tiles, so a 2 × 2 bin never loses a tile at 1.5. A bin of 110 tiles can reach 10.4.

The console prints the pixel size, the tile and bin sizes in µm and the GPU before the first segment. Every method then measures the reference tile itself, and the console prints what each found there. A method that finds nothing on the reference finds nothing in the scan, since only tiles like the reference are measured.

`pooled` finds nothing on dense arrays. Around each opening it uses a window of the largest odd number of pixels up to the pitch, and at least 5. On synthetic 80 nm squares (3.8 px) at a pitch of 6.7 px that window is 5 × 5 px, its border lies on the opening's own blurred edge, and every opening fails its checks: no value at tiles of 128, 238 or 476 px. With openings farther apart, as at the 8 px pitch of the tests, it measures.

Cutting tiles across page boundaries assumes that a page continues the one before it without a gap or a repeat, which has not been checked on real pages. A tile across a boundary where the hole lattice jumps fails the screen. On a synthetic stack that repeats one page, with a jump of half a pitch at every boundary, tiles within pages passed 100% and tiles across boundaries 19%. Each finished segment's line therefore gives the accepted share for tile rows within one page and across two, and `stripes.json` holds the counts under `accepted_by_rows`. Clearly fewer across boundaries means the pages do not join.

~~~sh
python scripts/measure_stripes.py D:\ACTIS\scan --reference holes.tiff --output D:\ACTIS\scan-50 --bin-y-um 50 --bin-x-um 50 --reject-sigma 2
~~~

### What a run leaves behind

| File | Contents |
|---|---|
| `stripes.json` | Format `actis-stripes-2`. The settings with the pixel size in µm, the header and where the pixel size came from, the tile side in pixels, the GPU (number, name and PCI bus), what each method measured on each reference tile (`reference_values`, null for nothing, and `reference_errors`), and one entry per segment: its board files, the result file, the strip file under `edges` if strips are kept, page count, board and merged page size, tile grid, first tile column, rows left over, bin size in tiles, bin grid, seam columns, accepted and screened tiles, per method the mean over tiles, the mean over bins, the tiles averaged and the batches it failed in (`failed_batches`, with the first `error` and its `traceback`), and the seconds per stage. A failed segment has `error`, and `traceback` when the error is not about its files. `incomplete` lists segments without all boards, which are not measured. |
| `<stripe>_<segment>.npz` | The tile and bin values of one segment. |
| `<stripe>_<segment>_edges.npz` | With `--edge-strips`: the pixels of the outer columns of every merged page of the segment. |

| Array in the `.npz` | Shape | Contents |
|---|---|---|
| `bin_raw`, or `bin_` and the name of each method that ran | (bin row, bin column) | The bin's value: the mean of the tiles it averaged. NaN where it averaged none. |
| `bin_<method>_tiles` | (bin row, bin column) | How many tiles each bin averaged. |
| `tile_raw`, or `tile_` and the name of each method | (tile row, tile column) | The tile's value. NaN where the tile was not accepted or gave no valid value. |
| `tile_<method>_used` | (tile row, tile column) | Whether the tile is in its bin's value: accepted, valid, in a bin and not left out as too far from the bin's mean. |
| `tile_reference` | (tile row, tile column) | The reference lattice the tile matched, −1 for a tile that was not accepted. |
| `bin_rows`, `bin_columns` | (bin row, 2), (bin column, 2) | The first and the end tile row, and tile column, of each bin. |
| `page_index` | (page) | The zero-based TIFF page of each merged page. |
| `seam_correlation` | (page, seam) | Agreement of neighbouring boards in their shared columns. NaN where those columns are flat. |
| `stripe`, `vertical`, `tile_size`, `tile_x0`, `page_rows`, `pixel_um`, `reject_sigma`, `overlap_px`, `boards`, `board_columns` | single numbers, `pixel_um` (x, y) | What the map needs to place a tile without `stripes.json`. `reject_sigma` is NaN for `--no-reject`. |

With `T` the tile size, tile `[r, c]` covers columns `tile_x0 + c × T` to `tile_x0 + (c + 1) × T` of the merged pages and rows `r × T` to `(r + 1) × T` of the strip, where strip row `y` is row `y mod page_rows` of merged page `page_index[y // page_rows]`. The methods that measure each opening also store `tile_<method>_openings`. The two aperture fits store `blur_sigma_px` and `blur_usable` for each batch of tile rows measured together, about one page's worth, with `blur_tile_row` giving the first tile row of each batch.

The tile values are kept so that bins of another size, or another threshold, can be computed without reading the scan again:

~~~python
import json
from pathlib import Path

import numpy as np

from actis.stripes import bin_tiles

run = Path(r"D:\ACTIS\scan-raw")
for segment in json.loads((run / "stripes.json").read_text(encoding="utf-8"))["segments"]:
    if segment["status"] == "complete":
        data = np.load(run / segment["result"])
        bins = data["bin_raw"]  # (bin row, bin column)
        # Bins of 5 x 11 tiles from tile column 0, leaving out tiles beyond 2 sigma.
        values, averaged, used = bin_tiles(data["tile_raw"], (5, 11), 0, reject_sigma=2.0)
~~~

### Edge strips for the shift between stripes

Neighbouring stripes share 32 columns, and a stripe does not begin at the height where its neighbour ended. This measurement does not align stripes. It keeps every stripe as stored. `--edge-strips` saves the pixels an alignment needs, so the scan is not read a second time: the first 32 and the last 32 columns of every merged page. `--edge-strips 64` saves 64 columns per edge.

~~~sh
python scripts/measure_stripes.py D:\ACTIS\scan --reference holes.tiff --output D:\ACTIS\scan-raw --edge-strips
~~~

| Array in `<stripe>_<segment>_edges.npz` | Shape | Contents |
|---|---|---|
| `left` | (page, row, column) | Columns 0 to 31 of each merged page, which board 1 recorded. |
| `right` | (page, row, column) | The last 32 columns of each merged page, which board 4 recorded. At the default tile size no tile covers these columns or those of `left`. |
| `page_index` | (page) | The zero-based TIFF page of each row, as in the value file. |
| `stripe`, `vertical` | single numbers | The segment. |

The strips have the pixel type of the TIFFs and the page order and orientation of the files, like the values. Saving them does not change the values.

A 1,500-page segment adds 188 MB, 1.1% of what was read. The file is not compressed. Writing two synthetic strips of that size took 0.2 s. Compressing them took 6 s and saved a quarter to a third.

An output folder holds a run with strips or a run without. Adding strips later needs a new output folder and a second pass over the scan.

### Long runs

A segment's files are written when the segment is finished. Running the same command again on the same output folder continues a stopped run: complete segments stay, failed ones are tried again. A segment that cannot be read is recorded as failed with the reason, and the run goes on. So is a segment that fails for any other reason, with the traceback in `stripes.json`, unless the GPU no longer works afterwards: then the run stops with a message, and running it again continues. Other settings need a new output folder.

A method that raises an error loses only its own values, in the batch of tile rows where it happened, about one page. That batch is measured again one method at a time, and the other methods keep theirs. The segment's line says how many batches each method failed in and the first error, and `stripes.json` keeps the traceback.

`--stripe 1 3` measures only those stripes, and `--segment 3_1 4_2` only those segments. `--max-pages 50` stops each segment after 50 pages for a trial. `--method raw pooled` runs several methods on the same accepted tiles, and `--method all` all seven.

`--gpu 1` measures on the second GPU (`--device` is the same option). The numbers are those `nvidia-smi` shows: the command numbers GPUs by PCI bus, while CUDA's own default puts the fastest first, so the two could disagree on a PC with different GPUs. A `CUDA_DEVICE_ORDER` already set in the environment is kept. One run uses one GPU. To use two at once, start two runs on different stripes into different output folders, for example `--stripe 1 2 3 --gpu 0` and `--stripe 4 5 6 --gpu 1`. Both read the same disk, so this gains little while the disk sets the pace (see below).

### Speed

These numbers were measured with the earlier tiling, 128-px tiles cut page by page. They have not been measured again for 10 µm tiles. One synthetic segment of production size took 103 to 109 s for raw brightness on an RTX 4060 Ti, read from a SATA hard disk with nothing in the file cache. It was four 8-bit board stacks of 1,500 pages, 16.9 GB, with 1,008,000 tiles screened and 912,000 measured. The disk set the pace: the reader was busy for all but one second, at about 160 MB/s. The GPU needed 16 s of it, 2.5 s to upload and merge, 13 s to screen and 0.5 s for the brightness.

Each segment's `seconds` in `stripes.json` show where its time went. `read` is how long the background reader worked. `read_wait` is how long the measurement stood still waiting for it, which the console prints as waiting for the disk. `merge`, `selection` and the method names are GPU stages. `save` is the time to write the segment's files.

- **Pages are measured one merged page at a time.** Screening costs 11.7 µs per tile with the 672 tiles of one merged page in a batch, and 10.9 µs with several pages in a batch, so larger batches would gain about 7% of the screening time.
- **Files are read in blocks.** The reader takes about 64 MB of one board, then the same pages of the next board, one block ahead of the measurement. Reading one page of each board in turn took 140 to 150 s on the same files.

The four files of that test each lay in a few pieces on the disk, as copied files do. Written four at a time, in 169 pieces each, they took 118 and 141 s. These numbers come from one PC and synthetic stacks.

All seven methods at once, on 10 µm tiles (476 px), took 0.33 s per merged page on the same GPU: synthetic pages of 2,048 × 5,408 px of 80 nm squares at a pitch of 6.7 px, about 4,900 openings per tile, read from the file cache. That is about 8 minutes of GPU time for a 1,500-page segment, less than reading the segment from a hard disk takes. Per page, the contour seeds and the two aperture fits took 0.18 s, `pooled` 0.06 s, `harmonic` 0.03 s, `flux` 0.02 s and screening 0.015 s. Real pages with fewer openings per tile take less.

## Measure several scans unattended

`measure_scans.py` measures several scans at once, each on its own GPU, and goes on in the background after the console is closed, for a run over a weekend. Every scan is a stripe run as above, with any of its options.

Copy `scripts\scans.example.ini`, then fill in one section per scan: the scan folder, a new folder for its results, and its GPU, numbered as `nvidia-smi` numbers them. Settings under `[DEFAULT]` apply to every scan, and a scan's own section can set them again. Every key is a `measure_stripes.py` option without its dashes. `yes` gives an option without a value, `no` leaves it out, and paths are written as they are.

~~~ini
[DEFAULT]
reference = D:\ACTIS\reference\holes.tiff
method = all
edge-strips = 32

[scan-A]
root = D:\ACTIS\scan-A
output = D:\ACTIS\results\scan-A
gpu = 0

[scan-B]
root = D:\ACTIS\scan-B
output = D:\ACTIS\results\scan-B
gpu = 1
~~~

Check the scans and start them:

~~~powershell
.\.venv\Scripts\python scripts\measure_scans.py D:\ACTIS\scans.ini
~~~

The command first runs a trial on the first 30 pages of a segment in the middle of every scan, all scans at once, each on its GPU. For every scan it prints the GPU, the pixel size and where it came from, the tiles accepted, the seams, the methods that gave values, the pace, and how long the rest takes at that pace. It takes a minute or two. Then it starts one supervisor per scan in the background and returns.

- **Problems stop the start**, so nothing runs: a scan folder that does not exist, an output folder that holds other files or a run with other settings, a trial that fails (a header that cannot be read, a GPU that does not exist), or a drive without room for the results.
- **Warnings ask first**, and `--yes` starts without asking: two scans on one GPU, a method that gave no value or failed in the trial, or no tile that matched the reference in the trial's pages. Those pages may lie outside the pattern, so this is a warning and not a problem.

What the runs have done, at any time, and stopping them:

~~~powershell
.\.venv\Scripts\python scripts\measure_scans.py D:\ACTIS\scans.ini --status
.\.venv\Scripts\python scripts\measure_scans.py D:\ACTIS\scans.ini --stop
~~~

`--status` says for every scan whether it is running, finished or stopped, and why. It also gives the segments complete, the segment being measured, the time left at the pace so far, every segment that did not complete with the reason, the methods that failed in some batches, and how many tiles each method put into bins. Starting again continues the runs where they ended, also after `--stop` or a restart of the PC: complete segments stay. `--only scan-B` acts on one scan of the file.

### What a supervisor does

- **One process per segment.** Each segment is measured by its own stripe run, so every segment starts with a fresh GPU. An error that leaves the GPU unusable, or a crash of the process, costs only that segment.
- **Every segment gets an outcome.** It is *complete* or *failed* (the reason comes from `stripes.json`). It is *crashed* if its process ended before recording it (with the exit code and the last lines it printed), and *stalled* if its progress did not change for 30 minutes (`--stall-minutes`), in which case its process is stopped.
- **One more try.** At the end, the segments that did not complete are measured once more (`--retries`).
- **Pauses and stops.** After three segments in a row that did not complete, it waits ten minutes before each next one, so that a disk that is away for a while costs few segments. After five processes in a row that ended without recording their segment, something outside the segments is wrong, such as the GPU or the header, and the supervisor stops with that reason.
- **No sleep.** It keeps Windows from going to sleep while it runs.

The logs are in a folder beside the config file, named after it (`scans-logs` for `scans.ini`; `--logs` sets another). For every scan it holds `<scan>.log`, with everything the stripe runs printed and the time of every line, the supervisor's state `<scan>.json`, and the trial's output in `trial-<scan>` and `trial-<scan>.log`.

### Before leaving

- Signing out of Windows ends the runs. Disconnecting a remote desktop session does not. A restart for Windows Update ends them too, so pause updates for the weekend. After a restart, run the start command again.
- Started over SSH, the runs may end with the session if Windows keeps them in the session's job. The command says so when that happens. This has not been tried over SSH.
- Scans read from one hard disk share its speed. The trials run at the same time, so their pace already includes that, roughly.
- The time left is an estimate from 30 pages, and from pages already in the file cache when the start command is run a second time.

This was tested on this development PC: three synthetic scans of eight segments each, all on its one GPU, started, checked, stopped part way and started again. It has not run on the workstation, its four GPUs or real scans.

## Throughput and production boundary

The report times input, discovery, alignment, each estimator and rendering separately. Optional `--io-sample "D:\ACTIS\stripes\2_1_1.tiff"` measures sequential file reading on the workstation. The operating system can cache a repeated read. The 1,500-page projections use 4,227,072,000 decoded pixels per board stack. Reference PNG bit depth, pattern density, TIFF compression and parallelism affect transfer to production. The exhaustive method comparison is a qualification run. Evaluate the selected method's throughput separately.

Run the resident-data benchmark to measure the raw, flux, pooled-moment and harmonic tile kernels with bounded concurrency. It reports measured input and valid-area throughput, checksum agreement with serial processing, and comparison against the HDD ingest rate. Warm kernels reuse reference geometry. Read, decode, repeat correspondence, reporting and a new field's geometry discovery have separate costs.

~~~powershell
.\.venv\Scripts\actis benchmark --input "D:\ACTIS\reference-pngs" --output "D:\ACTIS\reports\timing-01.json" --workers 4 --seconds 5 --io-sample "D:\ACTIS\stripes\2_1_1.tiff"
~~~

The production board stacks have shape up to (1500, 2048, 1376) in (page, row, column) order at 8 bits per pixel. A file name encodes stripe, vertical segment and board. The four simultaneous boards have 32-column overlaps and a combined width of 5,408 pixels. Neighboring stripes overlap by 32 columns. Stripe 1 scans down from the top. Even stripes scan up with reversed segment/page order and vertically flipped pages. The far end can overrun the nominal mask boundary by a fractional page, giving neighboring stripes different Y origins. The single-file TIFF benchmark measures native pages. The stripe measurement above merges the boards. Placing the stripes on the mask belongs to the next phase.

The equipment evidence, method assumptions and physical calibration requirements are documented in [the research notes](docs/metrology.md).
