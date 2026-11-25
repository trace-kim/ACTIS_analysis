# ACTIS scan data: what is known and what to verify

Written on 2026-11-25 on the development PC, which holds no real ACTIS data. Everything below comes from what the user said in working sessions, from what this repository records, or from arithmetic on those two. Nobody on the development side has opened a real board stack or a real `header.xml`. You, reading this on the workstation, can.

This document gives you the starting picture, says how far each part can be trusted, and lists what to look at before building the tools that read the metadata and join it to the measurement results.

## The short version

- A **scan** is a folder with a `header.xml` and a `RAW` folder. `RAW` holds one folder per stripe. A stripe folder holds multi-page TIFFs named `<stripe>_<vertical>_<board>.tiff`.
- A **stripe** is one pass of the instrument along the mask. Four **boards** record it side by side, each into its own TIFF. Boards 1 to 4 go left to right, and neighbouring boards share 32 columns.
- A board TIFF has up to 1,500 **pages** of 2,048 rows by 1,376 columns, 8-bit grayscale. Pages follow one another along the scan direction.
- Stripe 1 is scanned from the top of the mask downward and stored as scanned. Stripe 2 is scanned upward and stored flipped, and so on in turn. Neighbouring stripes share 32 columns. They do not begin at the same height, so each needs a vertical shift, which the header is expected to give only roughly.
- `header.xml` is expected to hold the pattern position, the flip, the pixel size in nm and perhaps the shift. 21 nm per pixel is only a typical value, and each scan's own value must come from its header. **The header's structure is unknown on the development side.** Reading it is the first job.
- `actis.stripes` already measures every stripe and writes one array of tile values per segment. Nothing yet reads the header, turns a stripe upright, finds the shift or places a tile on the mask.

## Where each fact comes from

Every table below names a source for each row.

| Source | Meaning | How far to trust it |
|---|---|---|
| **User** | Said by the user in a session, with the date. Quoted where the wording matters. | The best there is, but it is a description from memory, not a specification. |
| **Brief** | In this repository since its first commit (2026-11-18). Another agent wrote it from a brief by the user that the author of this document has not seen. | Second-hand. Check it before relying on it. |
| **Code** | What the code here assumes or does. | Checked on synthetic stacks only, unless a real run is named. |
| **Derived** | Arithmetic on other rows. | As good as its inputs. |
| **Real run** | Reported by the user from a run on the workstation. | Real, but so far only a few remarks, no logs. |

## Folder and file layout

```
<scan root>\
    header.xml
    RAW\
        S1\                 stripe folder (the name is the user's example)
            1_1_1.tiff      stripe 1, vertical segment 1, board 1
            1_1_2.tiff
            1_1_3.tiff
            1_1_4.tiff
            1_2_1.tiff      vertical segment 2, if the stripe has one
            ...
        S2\
            2_1_1.tiff
            ...
```

| Fact | Source |
|---|---|
| The scan root holds `header.xml`, and "my root has a RAW folder and individual stripe folders are within RAW". | User, 2026-11-25 |
| "stripe 1 can be stored in subfolder called 'S1'". This was given as an example, so the naming rule for stripe folders is not established. | User, 2026-11-24 |
| File names are `{stripe_num}_{vertical_num}_{board_num}.tiff`, for example `1_1_1.tiff` to `1_1_4.tiff`. | User, 2026-11-24 |
| `actis.stripes.find_segments` finds the stacks by file name alone, at any depth under the folder it is given. The name must match `(\d+)_(\d+)_(\d+)\.tiff?` in full, in any letter case. Folder names are not read. A name that occurs twice under the folder stops the run. | Code |
| A set of files with the same stripe and vertical number is called a **segment** in the code and the README. A segment with a board missing is listed as incomplete and not measured. | Code |

Not recorded anywhere here: how many stripes a scan has, how many vertical segments a stripe has, whether `RAW` or the root hold other files, and the real paths on the workstation. The `D:\ACTIS\...` paths in the README are placeholders.

## One board stack

| Fact | Source |
|---|---|
| A multi-page TIFF. A page is 2,048 rows by 1,376 columns, 8 bits per pixel, grayscale. | Brief |
| Up to 1,500 pages, so the array is at most (1500, 2048, 1376) in (page, row, column) order. | Brief |
| A file is "about 2~4 GB". | User, 2026-11-23 |
| The pages of a file can be stacked "to create a one long strip of image": they follow one another along the scan. | User, 2026-11-20 |
| A scan is "hundreds or thousands of tiff files". | User, 2026-11-23 |
| 1,500 uncompressed pages are 4,227,072,000 bytes, which is 68 MB below the 4 GiB limit of a classic TIFF. | Derived |
| Reading a 2 GB file from the source hard disk was given as 30 to 60 s. | Brief (`throughput.py`) |
| A real stack was read to the end through `tifffile` by the TIFF benchmark, which accepts only single-channel, unsigned 8- or 16-bit grayscale pages. The reported rate and time of that run give about 1,500 pages. | Real run, 2026-11-23 |

A guess from the size arithmetic, not something anyone said: the 1,500-page limit, and with it the split of a stripe into vertical segments, may exist to keep a file inside a classic TIFF. Whether the files are classic TIFF or BigTIFF will tell.

Not recorded: compression, the TIFF tags (a description or a time stamp per page would be useful), whether the last segment of a stripe is shorter, and whether the four boards of a segment always have the same number of pages. If they do not, `actis.stripes` measures the pages all four have and writes a note into `stripes.json`.

## Inside a stripe: the four boards

| Fact | Source |
|---|---|
| "each stripe is measured with 4 boards, where each board has their own tiff file." | User, 2026-11-24 |
| "Board 1 to 4 go left to right. That never changes." | User, 2026-11-25 |
| "The boards have 32 pixel overlap between boards." The shared columns "must be processed only once". | User, 2026-11-24 |
| The combined width is 4 × 1,376 − 3 × 32 = 5,408 columns. | Brief, Derived |
| Column `j` of board `k` (`k` = 1 to 4) is stripe column `(k − 1) × 1,344 + j`. | Derived |
| The merge takes 16 of the 32 shared columns from each of the two boards, so the seams fall at stripe columns 1,360, 2,704 and 4,048. | Code (`merge_boards`) |

Three things are assumed here and have not been checked on real pages:

- The overlap is exactly 32 columns at every seam.
- The boards are aligned in rows, so the same row of two boards shows the same height on the mask.
- The boards respond alike. Four separate boards may well differ in gain and dark level, which would show as steps in a brightness map at the seams.

`actis.stripes` tests the first two for every page: it correlates the 32 shared columns as each of the two boards recorded them, at zero offset, and stores the result as `seam_correlation`. Near 1 on patterned pages means the assumption holds. No seam number from a real run has reached the development side yet.

## Between stripes

In the user's words, 2026-11-25:

> stripe 1 isn't flipped measured from top to bottom, stripe 2 is flipped measured from bottom to top and so on. The stripes are not 'aligned' though even if you flip stripe 2, as the start point of stripe 2 and end point of stripe 1 do not match and requires a shift in position. The header may contain the shift value, but likely not completely accurate. May require comparing the overlapping 32 pixel stripes at the edges to obtain the 'real' shift and compensate for it.

In the brief:

> Neighboring stripes overlap by 32 columns. Stripe 1 scans down from the top. Even stripes scan up with reversed segment/page order and vertically flipped pages. The far end can overrun the nominal mask boundary by a fractional page, giving neighboring stripes different Y origins.

Read together, this is the picture the code and this document work from. The parts marked "assumed" are the development side's reading, not a statement by the user.

```
              stripe 1                                 stripe 2
              scanned top to bottom                    scanned bottom to top
              stored as scanned                        stored flipped

 mask top  +------+------+------+------+           +------+------+------+------+
           |  b1  |  b2  |  b3  |  b4  |           |  b1  |  b2  |  b3  |  b4  |
  page 0   |      |      |      |      |           |      |      |      |      |  last page
  page 1   |      |      |      |      |           |      |      |      |      |  ...
  ...      |      |      |      |      |           |      |      |      |      |  page 1
  last     |      |      |      |      |           |      |      |      |      |  page 0
 mask      +------+------+------+------+           +------+------+------+------+
 bottom       neighbouring boards share               its left 32 columns are the right 32
              32 columns                              of stripe 1, at another height
```

| Point | Status |
|---|---|
| Odd stripes run downward and are stored upright. Even stripes run upward and are stored flipped. | User and Brief agree |
| In an even stripe the first stored page is the first one scanned, at the bottom of the mask, and row 0 of a page is its lowest row on the mask. Stacking all pages in stored order and flipping the whole strip top to bottom turns the stripe upright. | Assumed from the brief's "reversed segment/page order and vertically flipped pages" |
| Vertical segments are numbered in the order they were scanned, so in an even stripe segment 1 is at the bottom. | User, 2026-11-25, confirming the development side's reading |
| Stripe `N + 1` lies to the right of stripe `N`, so the last 32 columns of board 4 in one stripe show what the first 32 columns of board 1 show in the next. | User, 2026-11-25, confirming the development side's reading. The edge strips of `actis.stripes` are built on it |
| Neighbouring stripes share exactly 32 columns. | Brief. The user also speaks of "the overlapping 32 pixel stripes at the edges" |
| The stripes need a vertical shift against each other. | User |
| That shift is one number for a pair of stripes. | Assumed. It could change along the stripe |

Nothing between stripes is implemented. `actis.stripes` keeps every stripe exactly as stored: no flip, no reversed page order, no shift, no comparison with a neighbour.

## Pixel scale and the sizes that follow

21 nm per pixel at mask scale is a typical value, not a constant. In the user's words (2026-11-25), it "is a typical value", each scan "could differ" slightly, and "the actual value must be obtained from the header.xml file".

- **The pixel size belongs to a scan.** Read it from that scan's header and keep it with that scan's results. Two scans cannot be assumed to share a value.
- **The code does not do this yet.** 21 is the default of every `--pixel-nm` option, and nothing reads the header. Treat 21 as a fallback for a plausibility check, never as the value to compute positions or sizes with.
- **Along and across the scan.** Whether the scale is the same in both directions is not recorded. The two could differ, because one comes from the optics and the other from the stage speed and line rate.

At the typical 21 nm per pixel:

| Quantity | Pixels | Size |
|---|---|---|
| Width of a board page | 1,376 | 28.896 µm |
| Width of a merged stripe | 5,408 | 113.568 µm |
| Step from one stripe to the next, if they share 32 columns | 5,376 | 112.896 µm |
| Height of a page | 2,048 | 43.008 µm |
| Length of a 1,500-page segment | 3,072,000 | 64.512 mm |
| Default tile | 128 | 2.688 µm |
| Reference PNG | 676 × 2,048 | 14.196 × 43.008 µm |

Use these to check what the header says. Stripe positions that step by about 113 µm, for instance, would confirm both the scale and the 32 shared columns.

## What the pages show

| Fact | Source |
|---|---|
| Openings are bright on a darker background. | Brief, and every detector in the code relies on it |
| The features measured are arrays of contact holes, on a square or a staggered lattice. The code finds the lattice, the pitch and the hole shape (circle, square or ellipse) from the image. | Brief, Code |
| A hole is only about four pixels across. | Brief. See the note below |
| Board pages are mixed: the hole array does not fill every page. | Brief (the commit that added the TIFF benchmark calls them "mixed-pattern TIFF stacks") |
| "My reference image is a single page within a tiff file." | User, 2026-11-23 |

On the hole size: the synthetic images of this repository use 80 nm openings (3.81 px) at a pitch of 6.77 px. The 80 nm comes from a published Lasertec example, not from the user's mask. The pitch was worked out from a site density reported in the brief, about 30,200 holes in a 676 × 2,048 field. The real size, pitch, shape and lattice are not written down anywhere here, and neither is what else the pages hold (other arrays, lines, pads, marks, blank areas).

Because pages are mixed, measuring needs a **holes-only reference page** at the scan's scale and orientation. Every tile is compared with it, and only tiles that match it (correlation of at least 0.80 in each of the tile's four quadrants, by default) are measured. All other tiles get NaN. The reference is passed as `--reference <file>` with `--reference-page N`, counted from zero, and it may be a page of one of the scan's own stacks. Which file and page the user uses is not recorded.

### The eight reference PNGs

A second, separate data set exists: eight 16-bit grayscale PNGs of 2,048 rows by 676 columns, all of the same area of the mask, taken with that area at different positions on the TDI detector (Brief). `actis analyze-reference` measures how well each method repeats across them. Their relation to a stripe scan (which mask, where on it) is not recorded.

The brief also mentions a "historical" bin of 100 × 100 µm. That is the only hint here of the bin size earlier CD maps used. Ask the user before choosing a bin for the map.

## header.xml

What is known is short.

| Fact | Source |
|---|---|
| It lies in the scan root and "contains metadata about the measurement". | User, 2026-11-24 |
| "The pattern position, image flipud, pixel size in nm etc. information can be retrieved from the metadata file and would be used together to create the map." | User, 2026-11-24 |
| "The header may contain the shift value, but likely not completely accurate." | User, 2026-11-25 |
| No code reads it. `stripes.json` stores its path under `header`, or `null` when the folder given to the run does not hold it. The tests stand in for it with an empty `<header/>`. | Code |

Everything else is unknown: element names, nesting, units, the origin and directions of its coordinates, and whether it describes the scan as a whole, each stripe, each segment or each page.

What the map needs from it, in order of need:

| Wanted | Why |
|---|---|
| Pixel size of this scan, with its unit, along and across the scan | Turns pixels into nm. 21 is only typical, and the value differs from scan to scan |
| Position of each stripe across the scan direction | Places a stripe in X. It should grow with the stripe number, since stripes advance to the right |
| Start position and scan direction (or flip flag) of each stripe | Places a stripe in Y and says which stripes to turn upright |
| The shift between stripes, if it is stored | The coarse part of the vertical alignment |
| Page or row counts per stripe and segment | Checks the header against the files |
| Board count, board width and overlaps | Checks the 4, 1,376 and 32 the code assumes |
| The pattern position: where the pattern or its origin lies, in which coordinate system | Turns scan coordinates into mask or pattern coordinates |
| Origin, axis directions and units of all coordinates | Without these the positions cannot be joined to the pixels |
| Anything per page, such as a stage position | Would make the shift measurable along the stripe |

## The measurement results that exist

`actis.stripes` (`python scripts/measure_stripes.py <scan root> --reference <file> --reference-page N --output <folder>`) writes into its output folder. The README section "Measure every stripe of a scan" has the full description. The essentials for joining with the header:

| File | Contents |
|---|---|
| `stripes.json` | `format` (`"actis-stripes-1"`), `root`, `header`, `settings`, `reference_shapes`, `reference_pitch_px`, `incomplete`, and `segments`. Each segment entry has `stripe`, `vertical`, `files` (relative to `root`), `status`, and when complete `result`, `pages`, `board_shape`, `page_shape`, `tile_grid`, `seam_columns`, `seam_correlation_p90`, `tested_tiles`, `accepted_tiles`, `methods`, `seconds` and `notes`. |
| `<stripe>_<vertical>.npz` | One float32 array per method that ran (`raw` by default), shaped (page, tile row, tile column), NaN where the tile was not accepted. `reference` (int8, −1 where not accepted), `page_index`, `seam_correlation` shaped (page, seam), and the single numbers `stripe`, `vertical`, `tile_size`, `overlap_px`, `boards`, `board_columns`. |
| `<stripe>_<vertical>_edges.npz` | Only with `--edge-strips`: `left` and `right`, shaped (page, row, column), the first and the last 32 columns of every merged page in the pixel type of the TIFFs, plus `page_index`, `stripe`, `vertical`. |

`raw` is the mean brightness of a tile in DN. The other methods give sizes in pixels, except `flux` in DN px².

### Where a tile lies

With `T` the tile size (128), a tile `[p, r, c]` of a segment covers, on the merged page as stored:

- rows `r × T` to `(r + 1) × T` of page `page_index[p]`,
- stripe columns `c × T` to `(c + 1) × T`.

So its centre is at stripe column `(c + 0.5) × T` and, within its segment, at strip row `p × 2,048 + (r + 0.5) × T`. For a stripe of several segments, add the rows of the segments before it.

- **Across the stripe.** 42 tiles cover columns 0 to 5,375. The last 32 columns belong to no tile. Stripes advance to the right, so if they share exactly 32 columns, those columns are the first 32 of the next stripe and the tile columns of neighbouring stripes continue each other without a gap.
- **Turning a stripe upright.** 2,048 rows are exactly 16 tiles of 128. Flipping a strip therefore moves whole tiles: for the values of a stripe stacked over all its pages, `values[::-1, ::-1, :]` is the same strip upside down. This holds only while the tile size divides 2,048.
- **The shift.** A shift between stripes is not a multiple of the tile size. Place every tile by the coordinate of its centre instead of forcing the stripes onto one grid.

Then, for a tile centre at stripe column `x` and upright strip row `y`:

```
X = X of the stripe (header) + x × pixel size
Y = Y of the stripe (header) + shift of the stripe + y × pixel size
```

### What the results do not hold

- **No pixels**, apart from the optional edge strips.
- **No position on the mask** and nothing from the header.
- **Not the offset of each tile's pattern.** The screening finds how far every accepted tile is shifted against the reference tile (`select_tiles` returns it), but the files do not keep it.
- **Not every tile.** Only tiles that matched the reference have values.

## What real runs have shown so far

Very little has come back, and no logs or result files.

| Date | What was reported |
|---|---|
| 2026-11-23 | The first TIFF benchmark took "more than 3 days" on one real stack. After a first rework it took about 1,300 s at 1.17 pages per second. It was rebuilt after that (commit `fa14154`), and no timing of the rebuilt version on real data has been reported. |
| 2026-11-25 | The first stripe run found no files, because it looked one level below the root and the stacks are in `RAW\<stripe folder>`. Fixed in commit `24a9492`. |
| 2026-11-25 | A stripe run for raw brightness, started on the `RAW` folder with the version before that fix, "took about 150 s to run until segment 3_1". Its finished-segment lines (page counts, accepted tiles, seam numbers, seconds spent waiting for the disk) were asked for and not received. |

The output folder of that last run should still be on the workstation. Its `stripes.json` and `.npz` files are the first real results, and they already hold real page counts, seam correlations and accepted-tile counts. Because the run was given the `RAW` folder, its `header` entry is `null` and its file paths are relative to `RAW`. Whether it was run with `--edge-strips` is not known.

## What to check first on the real data

In this order, because each step leans on the ones before. Nothing here needs the GPU except where `actis.stripes` is named.

1. **Inventory.** List the scan root and `RAW`: stripe folders and their names, files per folder, how many stripes and vertical segments, file sizes, and anything that is not a board stack.
2. **One TIFF.** Classic or BigTIFF, compression, photometric interpretation, bits per sample, page count, the shape of the first and the last page, and all tags of one page.
3. **The header.** Its encoding and its tree: every element name, its attributes and how often it occurs, and the full entries of two neighbouring stripes. Then go through the table of wanted items above and note which exist.
4. **Header against files.** Do stripe and segment counts agree, do page counts agree, is the pixel size near 21 nm, are the board width and the overlaps there as 1,376 and 32?
5. **Board seams.** On patterned pages, is `seam_correlation` near 1? If not, find the offset in rows and columns at which the shared columns agree. Compare the mean brightness of the shared columns between the two boards for gain and dark-level differences.
6. **Page to page.** Does the last row of a page continue into the first row of the next without a gap or a repeat? The same from the last page of one vertical segment to the first page of the next.
7. **The flip.** Take an odd and an even stripe that both show a recognisable, non-repeating feature, such as the edge of the patterned area. Confirm that stacking the even stripe in stored order and flipping the strip gives the same picture as the odd one.
8. **The shift between stripes.** After turning the even stripe upright, compare the last 32 columns of one stripe with the first 32 of the stripe to its right over a range of row offsets. A clear peak gives the shift and confirms the 32 shared columns. No peak: try other widths. Compare the shift with the header's value, and see whether it stays the same along the stripe.
9. **Odd against even in the results.** One reference page screens all stripes, and the even ones are stored upside down. Compare the share of accepted tiles in odd and in even stripes. A clearly lower share in even stripes means the pattern does not look the same upside down and those stripes need a flipped reference.
10. **What a page holds.** Look at a holes-only page, a mixed page and a blank one, and write down the hole pitch, size and shape, and what else appears.

On step 8: a hole array repeats, so comparing two strips of it fixes the shift only up to a whole number of pitches. The coarse part has to come from the header or from something that does not repeat, such as the first and last patterned row of each stripe. The strips then refine it.

## What the user wants built

For this stage, in the user's words (2026-11-25): tools for "reading the metadata file, extracting info, matching them with the measurement results". The goal behind it (2026-11-24) is that the results of `actis.stripes` "would later be loaded to create a CD map", with "the pattern position, image flipud, pixel size in nm etc." taken from the header. The stripe measurement is to grow "to apply other methods as well"; raw brightness is only the first.

Constraints the user has stated:

| Constraint | Source |
|---|---|
| Speed decides. A method "must finish within minutes since I need to run the to be accpeted method on hundreds or thousands of tiff files". | 2026-11-23 |
| No method may be handed a blur. "No blur value can be given before running the pipeline in real data processing. If the method needs blur, it needs to somehow get it from the images themself." If it differs between images, find it "per page". The user does not trust blur-dependent sizes much. | 2026-11-23 |
| Sharing one blur across the four boards of a merged page is "okay for now, we should check later". | 2026-11-25 |
| No matplotlib for showing images: "my images are huge and displaying the images/contours using matplotlib will take forever to load anything." The viewers in this repository are browser pages drawn with canvas and WebGL. | 2026-11-23 |
| Windows that people read must be readable. The first contour lab was rejected for: "Loads of small text, numbers hidden in different places, overwhelming method descriptions packed into small areas". | 2026-11-23 |
| Asked for what you think is best, design it for this instrument. Do not carry over a method or a layout from earlier work without thought. | 2026-11-23 |

Two pointers:

- **The user's own earlier code.** The user wrote a raw-intensity pipeline before this repository's ("My previous implementation that I did on my own"), which loaded whole images onto the GPU and picked tiles by phase cross-correlation. If it is on the workstation, it may already show how the header was read and how stripes were put together. Ask.
- **Keep `actis-stripes-1` loadable.** The map is to be built from these files. Add to the format, or change it deliberately with a new format name, but do not break the reading of existing runs.

Suggestions from the development side, not instructions from the user:

- **Keep real data and real metadata out of git.** `.gitignore` covers `*.tif`, `*.tiff`, `*.png` and `results/`. It does not cover `header.xml`, `.npz`, `.json` or `.csv`. Ask the user before committing a real header, even as a test fixture. A fixture with the real structure and invented values serves the tests as well.
- **Write back what you find.** The development PC cannot see the data, and code travels between the two machines only through git. When a row of this document is confirmed or turns out wrong, change it here and commit it, so the next session on either machine starts from the facts.
