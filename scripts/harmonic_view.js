"use strict";
const $ = (id) => document.getElementById(id);
const study = JSON.parse($("study").textContent);
const { width, height, tile_px: tileSide, regions, tiles, chunk } = study;
const fmt = (x, digits = 4) => (Number.isFinite(x) ? Number(x.toPrecision(digits)).toString() : "—");
const vector = (a) => a?.map((v) => fmt(v, 6)).join(", ") ?? "—";
const count = (a) => a?.filter(Boolean).length ?? 0;
const human = (s) => s.replaceAll("_", " ");
const esc = (s) =>
  String(s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
  );
const steps = [
  "Detect bright openings",
  "Index the lattice",
  "Window the tile",
  "Sample Fourier orders",
  "Recover displacement",
  "Select signed orders",
  "Fit size and blur",
  "Qualify the CD",
];
const layerNames = {
  raw: "Native image",
  smooth: "Smoothed image",
  low: "Local background (P10)",
  high: "Local bright level (P98)",
  threshold: "Detection threshold",
  mask: "Connected islands",
  core: "Centroid cores",
  hann: "Hann weights",
  weighted: "Weighted signal",
  real: "Order contribution · real",
  imag: "Order contribution · imaginary",
};
const tileLayer = (mode) => ["hann", "weighted", "real", "imag"].includes(mode);
const orderLayer = (mode) => mode === "real" || mode === "imag";
const layerHints = {
  raw: "Readout gives the original detector number (DN). Contrast stretch uses the field’s 0.5th and 99.5th percentiles.",
  smooth: "Gaussian-smoothed intensity in DN, σ = 0.65 px.",
  low: "Interpolated local 10th percentile of the smoothed intensity, in DN.",
  high: "Interpolated local 98th percentile of the smoothed intensity, in DN.",
  threshold: "Required excursion above local background, in DN.",
  mask: "White pixels exceed the local threshold. A connected component supplies one candidate opening.",
  core: "White pixels reach at least half their component’s peak excursion and contribute to its weighted centroid.",
  hann: "The separable Hann weight runs from zero at the tile boundary to approximately one in its center.",
  weighted:
    "Blue is negative, amber is positive. The readout gives (DN − weighted mean) × Hann weight / sum of weights.",
  real: "Per-pixel real contribution to the selected order before anchor rotation. Amber adds signal and blue subtracts it.",
  imag: "Per-pixel imaginary contribution to the selected order before anchor rotation. Amber adds signal and blue subtracts it.",
};
const tileAt = new Map(tiles.map((t, i) => [`${t.tile_i},${t.tile_j}`, i]));
let regionIndex = 0,
  tileIndex = -1,
  row = null,
  trace = {},
  reg = null,
  stage = 0,
  order = 0;
const cv = $("fieldview"),
  ctx = cv.getContext("2d");
const cam = { x: width / 2, y: height / 2, z: 1 };
let vw = 0,
  vh = 0,
  frame = 0,
  pointer = null,
  pinned = null,
  gesture = null,
  problem = "";
const cache = new Map(),
  working = new Map(),
  failures = new Set();
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
function requestDraw() {
  if (!frame) frame = requestAnimationFrame(draw);
}
function getChunk(name, level, x, y) {
  const key = `${name}-${level}-${x}-${y}`;
  const cached = cache.get(key);
  if (cached) {
    cache.delete(key);
    cache.set(key, cached);
    return cached;
  }
  if (working.has(key) || failures.has(key)) return;
  const element = $(key);
  if (!element) return;
  working.set(key, true);
  (async () => {
    const bytes = Uint8Array.from(atob(element.textContent), (c) => c.charCodeAt(0));
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate"));
    const buffer = await new Response(stream).arrayBuffer();
    const values = name === "raw" && study.bits === 16 ? new Uint16Array(buffer) : new Uint8Array(buffer);
    const [w, h] = study.layers[name].levels[level];
    cache.set(key, {
      values,
      w: Math.min(chunk, w - x * chunk),
      h: Math.min(chunk, h - y * chunk),
      image: null,
    });
    while (cache.size > 128) cache.delete(cache.keys().next().value);
  })()
    .catch((e) => {
      failures.add(key);
      problem = `Image decoding failed (${e.message}). Open this file in a current desktop browser.`;
    })
    .finally(() => {
      working.delete(key);
      requestDraw();
    });
}
function weight(x, y) {
  const [x0, y0] = trace.lo,
    [x1, y1] = trace.hi;
  return (
    ((1 - Math.cos((2 * Math.PI * (x - x0)) / (x1 - x0 - 1))) *
      (1 - Math.cos((2 * Math.PI * (y - y0)) / (y1 - y0 - 1)))) /
    4
  );
}
function fourierWeight(mode, x, y) {
  const [fx, fy] = trace.freq[order],
    phi = 2 * Math.PI * (fx * (x - trace.lo[0]) + fy * (y - trace.lo[1]));
  return mode === "real" ? Math.cos(phi) : -Math.sin(phi);
}
function chunkImage(data, mode, x0, y0, step) {
  const spatial = tileLayer(mode);
  const variant = spatial ? `${mode}-${tileIndex}-${order}` : mode + $("stretch").checked;
  if (data.image && data.variant === variant) return data.image;
  const canvas = document.createElement("canvas");
  canvas.width = data.w;
  canvas.height = data.h;
  const gc = canvas.getContext("2d"),
    img = gc.createImageData(data.w, data.h);
  let black = 0,
    white = 255;
  if (mode === "raw") {
    black = $("stretch").checked ? study.layers.raw.black : 0;
    white = $("stretch").checked ? study.layers.raw.white : 2 ** study.bits - 1;
  }
  const range = spatial ? Math.max(...trace.weighted_range.map(Math.abs), 1e-15) : 1;
  for (let i = 0; i < data.values.length; i++) {
    const v = data.values[i];
    let g = clamp(((v - black) * 255) / Math.max(1, white - black), 0, 255),
      r = g,
      b = g;
    if (spatial) {
      const x = x0 + (i % data.w) * step,
        y = y0 + Math.floor(i / data.w) * step,
        win = weight(x, y);
      if (mode === "hann") {
        r = g = b = 255 * win;
      } else {
        const factor = orderLayer(mode) ? fourierWeight(mode, x, y) : 1;
        const q = clamp(((v - trace.mean) * win * factor) / (trace.window_sum * range), -1, 1);
        r = 20 + Math.max(0, q) * 230;
        g = 35 + Math.abs(q) * 145;
        b = 40 + Math.max(0, -q) * 210;
      }
    }
    img.data[4 * i] = r;
    img.data[4 * i + 1] = g;
    img.data[4 * i + 2] = b;
    img.data[4 * i + 3] = 255;
  }
  gc.putImageData(img, 0, 0);
  data.image = canvas;
  data.variant = variant;
  return canvas;
}
function visitSites(r, box, fn) {
  const [x0, y0, x1, y1] = box;
  for (let y = Math.floor(Math.max(0, y0) / chunk); y <= Math.floor(Math.min(height, y1) / chunk); y++)
    for (let x = Math.floor(Math.max(0, x0) / chunk); x <= Math.floor(Math.min(width, x1) / chunk); x++)
      for (const [a, b] of r.sites[`${x},${y}`] || []) if (a >= x0 && a < x1 && b >= y0 && b < y1) fn(a, b);
}
function draw() {
  frame = 0;
  const dpr = window.devicePixelRatio || 1;
  vw = cv.clientWidth;
  vh = cv.clientHeight;
  if (cv.width !== Math.round(vw * dpr) || cv.height !== Math.round(vh * dpr)) {
    cv.width = Math.round(vw * dpr);
    cv.height = Math.round(vh * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#102024";
  ctx.fillRect(0, 0, vw, vh);
  const left = cam.x - vw / (2 * cam.z),
    top = cam.y - vh / (2 * cam.z);
  const right = left + vw / cam.z,
    bottom = top + vh / cam.z;
  const mode = $("layer").value,
    spatial = tileLayer(mode),
    name = spatial ? "raw" : mode;
  const levels = study.layers[name].levels;
  const level = spatial ? 0 : clamp(Math.floor(Math.log2(1 / (cam.z * dpr))), 0, levels.length - 1);
  const step = 2 ** level,
    block = chunk * step;
  ctx.translate(-left * cam.z, -top * cam.z);
  ctx.scale(cam.z, cam.z);
  ctx.imageSmoothingEnabled = cam.z < 1;
  let box = [0, 0, width, height];
  if (spatial) box = [...trace.lo, ...trace.hi];
  ctx.save();
  ctx.beginPath();
  ctx.rect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
  ctx.clip();
  for (
    let y = Math.max(0, Math.floor(Math.max(box[1], top) / block));
    y < Math.ceil(Math.min(box[3], bottom) / block);
    y++
  )
    for (
      let x = Math.max(0, Math.floor(Math.max(box[0], left) / block));
      x < Math.ceil(Math.min(box[2], right) / block);
      x++
    ) {
      const data = getChunk(name, level, x, y);
      if (data)
        ctx.drawImage(
          chunkImage(data, mode, x * block, y * block, step),
          x * block,
          y * block,
          data.w * step,
          data.h * step,
        );
    }
  ctx.restore();
  if ($("boxes").checked) {
    ctx.lineWidth = 1 / cam.z;
    for (
      let j = Math.max(0, Math.floor(top / tileSide));
      j < Math.ceil(Math.min(height, bottom) / tileSide);
      j++
    )
      for (
        let i = Math.max(0, Math.floor(left / tileSide));
        i < Math.ceil(Math.min(width, right) / tileSide);
        i++
      ) {
        const t = tiles[tileAt.get(`${i},${j}`)];
        if (!t) continue;
        ctx.strokeStyle = t.status === "ok" ? "#58cdad" : "#f3b36c";
        ctx.strokeRect(i * tileSide, j * tileSide, tileSide, tileSide);
      }
    if (row) {
      ctx.lineWidth = 2.7 / cam.z;
      ctx.strokeStyle = "#fff";
      ctx.strokeRect(row.x0 + 0.5, row.y0 + 0.5, tileSide, tileSide);
    }
  }
  if (reg && stage === 1) {
    ctx.lineWidth = 1.5 / cam.z;
    ctx.strokeStyle = "#76e9ec";
    ctx.setLineDash([5 / cam.z, 4 / cam.z]);
    for (const [x, y, X, Y] of reg.diagnostics.shape_patch_bounds_xyxy || [])
      ctx.strokeRect(x, y, X - x, Y - y);
    ctx.setLineDash([]);
  }
  const sitesOn = $("sites").checked,
    outlinesOn = $("outlines").checked && row?.status === "ok";
  if (reg && (sitesOn || outlinesOn) && reg.pitch * cam.z >= 3) {
    const b = reg.basis,
      [ax, ay] = reg.diagnostics.anchor_xy,
      det = b[0][0] * b[1][1] - b[0][1] * b[1][0];
    visitSites(reg, [left - 2, top - 2, right + 2, bottom + 2], (x, y) => {
      if (sitesOn) {
        ctx.fillStyle = "#6af7ca";
        ctx.beginPath();
        ctx.arc(x + 0.5, y + 0.5, Math.min(reg.pitch * 0.09, 2 / cam.z), 0, 2 * Math.PI);
        ctx.fill();
      }
      if (outlinesOn && x >= row.x0 && x < row.x1 && y >= row.y0 && y < row.y1) {
        const i = Math.round(((x - ax) * b[1][1] - (y - ay) * b[1][0]) / det);
        const j = Math.round((-(x - ax) * b[0][1] + (y - ay) * b[0][0]) / det);
        ctx.save();
        ctx.translate(
          ax + i * b[0][0] + j * b[1][0] + row.dx_px + 0.5,
          ay + i * b[0][1] + j * b[1][1] + row.dy_px + 0.5,
        );
        ctx.rotate(reg.angle);
        ctx.lineWidth = 1.4 / cam.z;
        ctx.strokeStyle = "#ffbf70";
        ctx.beginPath();
        if (reg.shape === "square") ctx.rect(-row.cd_x_px / 2, -row.cd_y_px / 2, row.cd_x_px, row.cd_y_px);
        else ctx.ellipse(0, 0, row.cd_x_px / 2, row.cd_y_px / 2, 0, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.restore();
      }
    });
  }
  if (stage === 1 && reg) {
    const [ax, ay] = reg.diagnostics.anchor_xy;
    ctx.lineWidth = 2 / cam.z;
    reg.basis.forEach(([dx, dy], i) => {
      ctx.strokeStyle = i ? "#ffc277" : "#8aeed4";
      ctx.beginPath();
      ctx.moveTo(ax + 0.5, ay + 0.5);
      ctx.lineTo(ax + dx + 0.5, ay + dy + 0.5);
      ctx.stroke();
    });
  }
  $("scale").textContent = `${fmt(cam.z, 3)}× · ${fmt(vw / cam.z, 5)} px across`;
  updateProbe();
}
function world(p) {
  return [cam.x + (p[0] - cv.clientWidth / 2) / cam.z, cam.y + (p[1] - cv.clientHeight / 2) / cam.z];
}
function atEvent(e) {
  const b = cv.getBoundingClientRect();
  return [e.clientX - b.left, e.clientY - b.top];
}
function updateProbe() {
  if (problem) {
    $("probe").textContent = problem;
    return;
  }
  const pt = pointer ? world(pointer) : pinned;
  if (!pt) {
    $("probe").textContent = "Move over the image to inspect detector samples.";
    return;
  }
  const [x, y] = pt.map(Math.floor);
  if (x < 0 || x >= width || y < 0 || y >= height) {
    $("probe").textContent = "Outside the image field";
    return;
  }
  const tx = Math.floor(x / chunk),
    ty = Math.floor(y / chunk);
  function sample(name) {
    const c = getChunk(name, 0, tx, ty);
    return c?.values[(y % chunk) * (c?.w || 0) + (x % chunk)];
  }
  const v = sample("raw"),
    mode = $("layer").value;
  let msg = `x ${x}   y ${y}   ·   ${v ?? "…"} DN`;
  if (study.white_is_zero && v !== undefined) {
    msg = `x ${x}   y ${y}   ·   ${2 ** study.bits - 1 - v} stored DN   ·   ${v} bright-high DN`;
  }
  if (tileLayer(mode) && x >= trace.lo[0] && y >= trace.lo[1] && x < trace.hi[0] && y < trace.hi[1]) {
    const win = weight(x, y),
      factor = orderLayer(mode) ? fourierWeight(mode, x, y) : 1;
    msg += `   ·   Hann ${fmt(win, 6)}   ·   contribution ${fmt(((v - trace.mean) * win * factor) / trace.window_sum, 6)} DN`;
  } else if (study.layers[mode] && mode !== "raw") {
    const layer = study.layers[mode],
      val = sample(mode);
    msg +=
      mode === "core" || mode === "mask"
        ? `   ·   mask ${val === undefined ? "…" : Number(val > 0)}`
        : `   ·   display ≈ ${fmt(layer.black + ((layer.white - layer.black) * val) / 255, 6)} DN`;
  }
  $("probe").textContent = msg;
}
function fitBox(x, y, w, h) {
  cam.x = x + w / 2;
  cam.y = y + h / 2;
  cam.z = 0.94 * Math.min(cv.clientWidth / w, cv.clientHeight / h);
  requestDraw();
}
function fitTile() {
  if (row) fitBox(...trace.lo, trace.hi[0] - trace.lo[0], trace.hi[1] - trace.lo[1]);
}
function zoom(factor, p = [cv.clientWidth / 2, cv.clientHeight / 2]) {
  const [x, y] = world(p);
  cam.z = clamp(
    cam.z * factor,
    Math.min(cv.clientWidth / width, cv.clientHeight / height) * 0.25,
    Math.max(128, cam.z),
  );
  cam.x = x - (p[0] - cv.clientWidth / 2) / cam.z;
  cam.y = y - (p[1] - cv.clientHeight / 2) / cam.z;
  requestDraw();
}
cv.addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    pointer = atEvent(e);
    zoom(Math.exp(-clamp(e.deltaY, -300, 300) * 0.003), pointer);
  },
  { passive: false },
);
cv.onpointerdown = (e) => {
  if (e.button !== 0) return;
  cv.focus({ preventScroll: true });
  pointer = atEvent(e);
  gesture = { p: pointer, x: cam.x, y: cam.y, moved: false };
  cv.setPointerCapture(e.pointerId);
};
cv.onpointermove = (e) => {
  pointer = atEvent(e);
  if (gesture) {
    const dx = pointer[0] - gesture.p[0],
      dy = pointer[1] - gesture.p[1];
    if (Math.hypot(dx, dy) > 3) gesture.moved = true;
    if (gesture.moved) {
      cam.x = gesture.x - dx / cam.z;
      cam.y = gesture.y - dy / cam.z;
    }
  }
  requestDraw();
};
cv.onpointerup = (e) => {
  if (gesture && !gesture.moved) {
    pinned = world(atEvent(e));
    const [x, y] = pinned,
      id = tileAt.get(`${Math.floor(x / tileSide)},${Math.floor(y / tileSide)}`);
    if (id !== undefined) {
      if (id !== tileIndex) chooseTile(id, false);
    } else {
      const ri = regions.findIndex(
        (r) => x >= r.bounds[0] && x < r.bounds[2] && y >= r.bounds[1] && y < r.bounds[3],
      );
      if (ri >= 0 && ri !== regionIndex) chooseRegion(ri);
    }
  }
  gesture = null;
  requestDraw();
};
cv.onpointercancel = () => {
  gesture = null;
};
cv.onpointerleave = () => {
  if (!gesture) pointer = null;
  requestDraw();
};
cv.onkeydown = (e) => {
  if (["+", "=", "-", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key)) {
    e.preventDefault();
    if (e.key === "+" || e.key === "=") zoom(1.5);
    else if (e.key === "-") zoom(1 / 1.5);
    else {
      cam.x += (({ ArrowLeft: -1, ArrowRight: 1 }[e.key] || 0) * 80) / cam.z;
      cam.y += (({ ArrowUp: -1, ArrowDown: 1 }[e.key] || 0) * 80) / cam.z;
      requestDraw();
    }
  }
};
function table(headers, rows) {
  return (
    `<div class="scroll"><table><thead><tr>${headers.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>` +
    rows.map((r) => `<tr>${r.map((v) => `<td>${esc(v ?? "—")}</td>`).join("")}</tr>`).join("") +
    "</tbody></table></div>"
  );
}
function facts(items) {
  return `<div class="facts">${items.map(([k, v]) => `<div class="fact"><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join("")}</div>`;
}
const formula = (v) => `<div class="formula">${esc(v)}</div>`;
const note = (v) => `<div class="note">${esc(v)}</div>`;
const fold = (name, content) => `<details><summary>${esc(name)}</summary>${content}</details>`;
const mag = (i) => Math.hypot(trace.anchored_re[i], trace.anchored_im[i]);
const radius = (i) => Math.hypot(...trace.freq[i]);
function indices(a) {
  return a?.flatMap((v, i) => (v ? [i] : [])) || [];
}
function hk(i) {
  const [fx, fy] = trace.freq[i];
  return reg.basis.map(([x, y]) => Math.round(x * fx + y * fy)).join(", ");
}
function selectedOrders() {
  return `<label>Inspect order <select id="order">${trace.freq
    .map((f, i) => `<option value="${i}" ${order === i ? "selected" : ""}>(${hk(i)})</option>`)
    .join("")}</select></label>`;
}
function traceMissing() {
  if (!reg) return note("Discovery found no qualified lattice. A tile CD requires an indexed region.");
  if (!row)
    return note(
      "No complete analysis tile fits inside this region on the image-aligned grid. Check the region extent and tile side.",
    );
  return note(
    `This calculation stopped at “${human(row.status)}”. The qualification step shows the measured values and limits.`,
  );
}
function coordinate(x, y) {
  return [fmt(x, 6), fmt(y, 6)].join(", ");
}
function discoveryStep() {
  const d = study.discovery;
  let out =
    "<p>The detector smooths the native image, estimates the local background and bright level, then labels connected bright pixels. Switch image layers to follow one opening through these operations.</p>";
  out += facts([
    ["Image", `${width} × ${height} px`],
    ["Polarity", "Bright openings"],
    ["Noise estimate", `${fmt(d.noise)} DN`],
    ["Components", d.components],
    ["Candidate centers", d.candidate_centers],
    ["Indexed regions", regions.length],
  ]);
  out += formula(
    "excursion = Gaussian(image, σ = 0.65 px) − P10\nthreshold = max(0.47 × max(P98 − P10, 4.5 × noise), 2.1 × noise)",
  );
  out +=
    "<p>Local percentiles are interpolated between blocks spanning roughly 64 px. Noise comes from the median absolute deviation of mixed pixel differences divided by 1.349. Pixels with excursion above threshold enter the island mask.</p>";
  out += formula("center = Σ(core pixel position × excursion) / Σ(excursion)");
  out +=
    "<p>Within each component, pixels with at least half the peak excursion form the core. Discovery keeps interior components with ≥ 4 core pixels, axis ratio &lt; 8 and core area below 6.5% of the field.</p>";
  if (d.rejected.length)
    out +=
      `<h3>Rejected regions</h3>` +
      table(
        ["Bounds (exclusive end)", "Candidates", "Reason"],
        d.rejected.map((r) => [r.bounds.join(", "), r.candidate_centers, human(r.reason)]),
      );
  return out;
}
function latticeStep() {
  if (!reg) return traceMissing();
  const d = reg.diagnostics,
    bt = d.basis_trace,
    rows = Object.entries(d.heldout_shape_rms || {}).map(([name, v]) => [
      name,
      fmt(v),
      fmt(d.spatial_holdout_rms?.[name]),
    ]);
  let out =
    "<p>Neighbor displacements propose two independent primitive vectors. Discovery refines a shared anchor and basis by fitting integer lattice indices to the detected centers.</p>";
  out += facts([
    ["Lattice", reg.lattice],
    ["Pitch", `${fmt(reg.pitch)} px`],
    ["Indexed region centers", reg.n_sites],
    ["Inlier fraction", fmt(d.lattice_inlier_fraction)],
    ["Anchor (x, y)", vector(d.anchor_xy)],
    ["Opening family", reg.shape],
    ["Shape source", study.shape === "auto" ? "Automatic" : "User supplied"],
    ["Opening angle", `${fmt((reg.angle * 180) / Math.PI)}°`],
  ]);
  out += formula(
    `site(i, j) = anchor + i × a + j × b\na = (${vector(reg.basis[0])}) px    b = (${vector(reg.basis[1])}) px`,
  );
  out +=
    "<p>The two basis vectors start at the fitted anchor in the field view. A center counts as an inlier within 0.19 pitch of a lattice site. A region needs at least 45% inliers.</p>";
  out += fold(
    "Neighbor modes and basis refinements",
    table(
      ["Mode", "Δx (px)", "Δy (px)"],
      bt.neighbor_modes.map((v, i) => [i + 1, ...v.map((x) => fmt(x))]),
    ) +
      table(
        ["Fit", "a (px)", "b (px)", "Inliers"],
        [
          ["Initial", ...bt.initial_basis.map(vector), "—"],
          ...bt.refinements.map((v, i) => [i + 1, ...v.basis.map(vector), v.inliers]),
        ],
      ),
  );
  out += "<h3>Opening shape evidence</h3>";
  out += facts([
    ["Median core area", `${fmt(d.median_component_area_px)} px²`],
    ["Apparent axis ratio", fmt(d.apparent_axis_ratio)],
    ["Fourfold moment", fmt(d.fourfold_moment)],
    ["Moment hypothesis", d.moment_shape],
    ["Best hypothesis", reg.best],
    ["Automatic decision", d.shape_resolved ? "Resolved" : "Ambiguous"],
  ]);
  if (rows.length) {
    out +=
      "<p>Signed orders from the first dashed patch train the shape models. Every third order is reserved for scoring. A second dashed patch, when available, tests the same shape and blur with a new gain.</p>";
    out +=
      table(["Shape", "Reserved-order RMS", "Spatial RMS"], rows) +
      facts([
        ["Reserved orders", d.tested_orders],
        ["Separation of best scores", fmt(d.shape_margin)],
        ["Noise fraction", fmt(d.noise_fraction)],
        ["Spatial holdout", human(d.spatial_holdout_status)],
      ]);
  } else
    out +=
      "<p>Reserved-order shape comparison supplied no qualified result. The automatic decision uses the component moments.</p>";
  return out;
}
function windowStep() {
  if (trace.mean === undefined) return traceMissing();
  return (
    "<p>The estimator takes the complete native-pixel tile and subtracts its Hann-weighted mean. Tapering the edges limits leakage into neighboring Fourier orders. Inspect the weights or the signed contribution of any pixel in the field viewer.</p>" +
    facts([
      ["Pixel bounds [start, end)", `${vector(trace.lo)} → ${vector(trace.hi)}`],
      ["Sites", row.n_sites],
      ["Weighted mean", `${fmt(trace.mean, 7)} DN`],
      ["Sum of weights", fmt(trace.window_sum, 7)],
    ]) +
    formula("W(x,y) = Hann(x) × Hann(y)\nμ = Σ(I × W) / ΣW\nP(x,y) = (I(x,y) − μ) × W(x,y) / ΣW") +
    facts([
      ["Smallest contribution", `${fmt(trace.weighted_range[0])} DN`],
      ["Largest contribution", `${fmt(trace.weighted_range[1])} DN`],
    ]) +
    "<p>Tile-local x and y start at zero at the first pixel center. The absolute coordinates in the readout count columns from the left and rows from the top.</p>"
  );
}
function fourierStep() {
  if (!trace.anchored_re) return traceMissing();
  let out =
    "<p>The reciprocal basis maps each integer pair (h, k) to a native frequency. The estimator samples the windowed signal at that frequency and at two nearby off-lattice points.</p>";
  out += formula("f = (h, k) × inverse(basis)ᵀ\nAraw(f) = Σ P(x,y) × exp[−2πi (fx x + fy y)]");
  out += facts([
    ["Sampler", trace.sampler],
    ["Candidate orders", trace.freq.length],
    ["Sideband floor", `${fmt(trace.noise)} DN`],
  ]);
  out +=
    "<p>One half-plane of orders through fifth order is tested. Both frequency components must have magnitude below 0.445 cycles/px. The background image below is a native-bin FFT displayed as log(1 + magnitude / noise). Markers show the exact sampling coordinates.</p>";
  out += `<canvas id="spectrum" width="460" height="460" aria-label="Fourier magnitude with selectable sampled orders"></canvas>
    <p class="legend">Horizontal fₓ and downward fᵧ, from −0.5 to +0.5 cycles/px. Green rings mark sampled orders. Amber marks the selected order and blue its two sidebands. Click a ring to inspect it.</p>`;
  out +=
    selectedOrders() +
    facts([
      ["f (cycles/px)", vector(trace.freq[order])],
      ["Raw Re + i Im (DN)", coordinate(trace.raw_re[order], trace.raw_im[order])],
      ["|A| (DN)", fmt(mag(order))],
      ["Sideband 1 (DN)", fmt(trace.side1[order])],
      ["Sideband 2 (DN)", fmt(trace.side2[order])],
    ]);
  out += formula(
    "sideband offsets = (+3.6/w, +2.4/h), (−2.3/w, +3.7/h)\nnoise = max(10⁻¹², median over orders of min(|side₁|, |side₂|))",
  );
  out +=
    "<h3>Pixel contributions to this order</h3><p>Select the real or imaginary contribution layer to inspect the terms in the Fourier sum. Each tile pixel contributes P × cos(2π f · r) to the real part and −P × sin(2π f · r) to the imaginary part.</p>";
  if (trace.sampler !== "Direct DFT")
    out +=
      "<p>This tile uses cubic FFT interpolation to approximate that sum. The numeric coefficients above come from the interpolation.</p>";
  return out;
}
function spectrum() {
  const canvas = $("spectrum");
  if (!canvas) return;
  const gc = canvas.getContext("2d"),
    n = canvas.width;
  const w = trace.hi[0] - trace.lo[0],
    h = trace.hi[1] - trace.lo[1];
  // Bin centers for odd/even fftshift axes differ by half a display pixel.
  const place = ([x, y]) => [
    (x + (Math.floor(w / 2) + 0.5) / w) * n,
    (y + (Math.floor(h / 2) + 0.5) / h) * n,
  ];
  const im = new Image();
  im.onload = () => {
    if (!canvas.isConnected) return;
    gc.imageSmoothingEnabled = false;
    gc.drawImage(im, 0, 0, n, n);
    for (let i = 0; i < trace.freq.length; i++) {
      const [x, y] = place(trace.freq[i]);
      gc.strokeStyle = i === order ? "#ffd27a" : "#60dfc0";
      gc.lineWidth = i === order ? 3 : 1.5;
      gc.beginPath();
      gc.arc(x, y, i === order ? 7 : 4, 0, 2 * Math.PI);
      gc.stroke();
    }
    const [fx, fy] = trace.freq[order];
    for (const [dx, dy] of [
      [3.6 / w, 2.4 / h],
      [-2.3 / w, 3.7 / h],
    ]) {
      const [x, y] = place([
        ((((fx + dx + 0.5) % 1) + 1) % 1) - 0.5,
        ((((fy + dy + 0.5) % 1) + 1) % 1) - 0.5,
      ]);
      gc.strokeStyle = "#73c4ff";
      gc.strokeRect(x - 3, y - 3, 6, 6);
    }
  };
  im.src = "data:image/png;base64," + trace.spectrum;
  canvas.onclick = (e) => {
    const b = canvas.getBoundingClientRect(),
      x = ((e.clientX - b.left) * n) / b.width,
      y = ((e.clientY - b.top) * n) / b.height;
    let best = Infinity,
      idx = order;
    trace.freq.forEach((f, i) => {
      const [a, c] = place(f),
        d = Math.hypot(a - x, c - y);
      if (d < best) {
        idx = i;
        best = d;
      }
    });
    order = idx;
    renderWalk();
    requestDraw();
  };
}
function phaseStep() {
  if (!trace.anchored_re) return traceMissing();
  let out =
    "<p>The lattice anchor removes the tile’s arbitrary phase origin. Two independent fundamental directions then determine a common x/y translation by weighted least squares.</p>";
  out += formula(
    "A = Araw × exp[+2πi f · anchorlocal]\nf · d ≈ −arg(A) / (2π)     weight in squared loss = |A|²",
  );
  out += facts([
    ["Local anchor (px)", vector(trace.anchor_local)],
    ["Fundamentals", count(trace.fundamental)],
    ["Direction rank", trace.rank],
    ["Fitted d (px)", vector(trace.displacement)],
    ["Weighted phase RMS", `${fmt(trace.phase_rms)} rad`],
  ]);
  out += `<p>Fundamentals must satisfy |f| ≤ ${fmt(1.26 / reg.pitch)} cycles/px and |A| &gt; ${fmt(5 * trace.noise)} DN. Translation must be within ${fmt(0.35 * reg.pitch)} px, with weighted phase RMS ≤ 0.18 rad.</p>`;
  const ids = indices(trace.fundamental);
  out +=
    selectedOrders() +
    facts([
      ["Raw phase", `${fmt(Math.atan2(trace.raw_im[order], trace.raw_re[order]))} rad`],
      ["Anchored Re, Im (DN)", coordinate(trace.anchored_re[order], trace.anchored_im[order])],
      ["Anchored phase", `${fmt(Math.atan2(trace.anchored_im[order], trace.anchored_re[order]))} rad`],
    ]);
  out += table(
    ["(h, k)", "Weight |A|²", "Anchor phase (rad)", "Residual phase (rad)"],
    ids.map((i, j) => [
      hk(i),
      fmt(mag(i) ** 2),
      fmt(Math.atan2(trace.anchored_im[i], trace.anchored_re[i])),
      fmt(trace.phase_err?.[j]),
    ]),
  );
  if (trace.phase_err)
    out += plot(
      ids.map((i, j) => [j, Math.atan2(trace.anchored_im[i], trace.anchored_re[i]), trace.phase_err[j]]),
      "Fundamental index",
      "Phase (rad)",
      "Anchor phase",
      "Residual phase",
    );
  if (row.phase_status !== "ok") out += traceMissing();
  return out;
}
function selectionStep() {
  if (!trace.signed) return traceMissing();
  const ids = indices(trace.good),
    maxOrder = Math.max(...ids.map(radius)),
    boundary = Math.min(0.46, 1.22 * maxOrder);
  let out =
    "<p>Phase correction makes the real coefficient a signed measure of the aperture response. The strongest visible orders establish the usable bandwidth. Weak and near-zero orders inside that bandwidth also constrain the fitted size.</p>";
  out += formula(
    "S(f) = Re[A(f) × exp(+2πi f · d)]\nstrong if |A| > max(4 × noise, 0.008 × max |A|)\nin fit if |f| < min(0.46, 1.22 × highest strong frequency)",
  );
  out += facts([
    [
      "Strong threshold",
      `${fmt(Math.max(4 * trace.noise, 0.008 * Math.max(...trace.freq.map((_, i) => mag(i)))))} DN`,
    ],
    ["Strong orders", ids.length],
    ["Strongest reach", `${fmt(maxOrder)} cycles/px`],
    ["Fit band edge", trace.include ? `${fmt(boundary)} cycles/px` : "Not reached"],
    ["Fitted orders", trace.include ? count(trace.include) : "Not reached"],
  ]);
  out += plot(
    trace.freq.map((_, i) => [radius(i), trace.signed[i], trace.include?.[i] ? trace.signed[i] : null]),
    "Frequency radius (cycles/px)",
    "Signed signal (DN)",
    "All orders",
    "Inside fit band",
  );
  out += table(
    ["(h, k)", "|f| (cy/px)", "|A| (DN)", "Signed (DN)", "Strong", "In fit"],
    trace.freq.map((f, i) => [
      hk(i),
      fmt(radius(i)),
      fmt(mag(i)),
      fmt(trace.signed[i]),
      trace.good[i] ? "Yes" : "No",
      trace.include ? (trace.include[i] ? "Yes" : "No") : "—",
    ]),
  );
  if (!trace.fit) out += traceMissing();
  return out;
}
function plot(data, xlabel, ylabel, key1, key2) {
  const w = 560,
    h = 210,
    ox = 68,
    ex = 543,
    oy = 161,
    ey = 25;
  const xs = data.map((v) => v[0]),
    ys = data.flatMap((v) => v.slice(1)).filter(Number.isFinite);
  if (!xs.length || !ys.length) return "";
  const lo = Math.min(...xs),
    hi = Math.max(...xs),
    min = Math.min(0, ...ys),
    max = Math.max(0, ...ys);
  const X = (v) => ox + ((v - lo) / (hi - lo || 1)) * (ex - ox),
    Y = (v) => oy - ((v - min) / (max - min || 1)) * (oy - ey);
  const text = (x, y, s, align = "start") =>
    `<text x="${x}" y="${y}" text-anchor="${align}" font-size="11" fill="#415957">${esc(s)}</text>`;
  let svg =
    `<svg class="plot" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(ylabel + " versus " + xlabel)}">` +
    text(10, 16, ylabel) +
    text((ox + ex) / 2, 201, xlabel, "middle");
  for (let i = 0; i < 4; i++) {
    const v = min + ((max - min) * i) / 3,
      y = Y(v);
    svg += `<path d="M${ox} ${y}H${ex}" stroke="#cbdad3"/>` + text(ox - 6, y + 4, fmt(v, 3), "end");
  }
  for (let i = 0; i < 4; i++) {
    const v = lo + ((hi - lo) * i) / 3;
    svg += text(X(v), 180, fmt(v, 3), "middle");
  }
  for (const [x, a, b] of data) {
    if (Number.isFinite(a))
      svg += `<circle cx="${X(x)}" cy="${Y(a)}" r="3.6" fill="#117769"><title>${esc(key1 + " · " + fmt(x) + ", " + fmt(a))}</title></circle>`;
    if (Number.isFinite(b))
      svg += `<circle cx="${X(x)}" cy="${Y(b)}" r="5.1" fill="none" stroke="#aa5d17" stroke-width="1.5"><title>${esc(key2 + " · " + fmt(x) + ", " + fmt(b))}</title></circle>`;
  }
  return (
    svg +
    `</svg><p class="legend"><i class="swatch" style="background:#117769"></i>${esc(key1)} <i class="swatch" style="border:2px solid #aa5d17"></i>${esc(key2)}</p>`
  );
}
function fitStep() {
  const f = trace.fit;
  if (!f) return traceMissing();
  const ids = indices(trace.include);
  let out =
    "<p>Three starting sizes compete in a bounded nonlinear least-squares fit. The model predicts each signed order from opening geometry, directional blur, pixel integration and a fitted contrast gain.</p>";
  out += formula(
    "target = S / max |S| over included orders\nσ = max(noise / max |S|, 0.008)\nmodel = gain × aperture form factor × Gaussian transfer × pixel transfer\ncost = ½ Σ [(model − target) / σ]²",
  );
  out += facts([
    ["Signal scale", `${fmt(f.scale)} DN`],
    ["Normalized σ", fmt(f.error[0])],
    ["Best start", `${f.winner + 1} of 3`],
    ["Fitted axes", `${fmt(f.width)} × ${fmt(f.height)} px`],
    ["Blur σ (x, y)", `${coordinate(row.blur_x_px, row.blur_y_px)} px`],
    ["Gain", fmt(f.gain)],
  ]);
  out += plot(
    ids.map((i, j) => [radius(i), f.target[j], f.predicted[j]]),
    "Frequency radius (cycles/px)",
    "Normalized signed amplitude",
    "Measured",
    "Model",
  );
  out += formula(
    (reg.shape === "square"
      ? "form = sinc(w f∥) × sinc(h f⊥)"
      : "z = π √[(w f∥)² + (h f⊥)²]; form = 2 J₁(z) / z") +
      "\nGaussian = exp[−2π² ((σx fx)² + (σy fy)²)]\npixel = sinc(fx) × sinc(fy)",
  );
  out +=
    "<p>Frequencies parallel and perpendicular to the opening are rotated by its fitted orientation. Circles and squares share one size parameter. Bounds span 0.075–1.20 pitch for size, 0.004–0.35 pitch for each blur σ, and −2 to 4 for log gain.</p>";
  out += fold(
    "Order-by-order model and residuals",
    table(
      ["(h, k)", "Target", "Aperture", "Gaussian", "Pixel", "Model", "Model − target", "Residual / σ"],
      ids.map((i, j) => [
        hk(i),
        ...[
          f.target[j],
          f.form[j],
          f.mtf[j],
          f.pixel[j],
          f.predicted[j],
          f.predicted[j] - f.target[j],
          f.residual[j],
        ].map((v) => fmt(v, 6)),
      ]),
    ),
  );
  const sizeLabels = reg.shape === "ellipse" ? ["Width", "Height"] : ["Size"];
  out += fold(
    "Which orders distinguish size from blur?",
    "<p>Each derivative gives the change in a weighted residual per unit change in a fit parameter. Size and blur are fractions of pitch. The last columns remove the part of each size derivative that blur and log gain could explain.</p>" +
      table(
        [
          "(h, k)",
          ...sizeLabels,
          "Blur x",
          "Blur y",
          "Log gain",
          ...sizeLabels.map((s) => s + " after projection"),
        ],
        ids.map((i, j) => [
          hk(i),
          ...f.jacobian[j].map((v) => fmt(v)),
          ...f.projected_size[j].map((v) => fmt(v)),
        ]),
      ),
  );
  out += "<h3>Optimizer evaluations</h3>";
  for (const [i, run] of f.starts.entries()) {
    out += fold(
      `Start ${i + 1} · cost ${fmt(run.cost, 6)}${i === f.winner ? " · selected" : ""}`,
      `<p>${esc(run.message)} Each row is one evaluated trial point.</p>` +
        table(
          ["Trial", "Width (px)", "Height (px)", "σx (px)", "σy (px)", "Gain", "Cost"],
          run.evaluations.map((r, j) => [j + 1, ...r.map((v) => fmt(v, 6))]),
        ),
    );
  }
  return out;
}
function gates() {
  const t = trace;
  const rows = [];
  function add(name, val, limit, pass) {
    rows.push([name, val, limit, pass ? "Pass" : "Fail"]);
  }
  add(
    "Native footprint",
    `${vector(t.lo)} → ${vector(t.hi)}`,
    "Inside image",
    t.lo.every((x) => x >= 0) && t.hi[0] <= width && t.hi[1] <= height,
  );
  if (row.status === "incomplete_native_footprint") return rows;
  add("Pattern population", row.n_sites, `≥ ${fmt(t.minimum_sites, 6)}`, row.n_sites >= t.minimum_sites);
  add(
    "Tile span",
    Math.min(t.hi[0] - t.lo[0], t.hi[1] - t.lo[1]) + " px",
    `≥ ${fmt(5 * reg.pitch)} px`,
    Math.min(t.hi[0] - t.lo[0], t.hi[1] - t.lo[1]) >= 5 * reg.pitch,
  );
  if (row.status === "insufficient_pattern_coverage") return rows;
  add("Candidate orders", t.freq.length, "≥ 5", t.freq.length >= 5);
  if (!t.fundamental) return rows;
  add("Fundamentals", count(t.fundamental), "≥ 2", count(t.fundamental) >= 2);
  add("Independent directions", t.rank, "2", t.rank === 2);
  if (!t.displacement) return rows;
  add(
    "Displacement",
    `${fmt(Math.hypot(...t.displacement))} px`,
    `≤ ${fmt(0.35 * reg.pitch)} px`,
    Math.hypot(...t.displacement) <= 0.35 * reg.pitch,
  );
  add("Phase RMS", fmt(t.phase_rms), "≤ 0.18 rad", t.phase_rms <= 0.18);
  if (row.phase_status !== "ok") return rows;
  add("Opening shape", reg.shape, "Resolved or supplied", reg.shape !== "ambiguous");
  if (!t.good) return rows;
  const min = reg.shape === "ellipse" ? 7 : 5,
    reach = Math.max(...indices(t.good).map(radius));
  add("Strong order count", count(t.good), `≥ ${min}`, count(t.good) >= min);
  if (count(t.good) >= min)
    add("Strong order reach", fmt(reach), `≥ ${fmt(1.8 / reg.pitch)} cy/px`, reach >= 1.8 / reg.pitch);
  if (!t.include) return rows;
  add("Orders inside fit band", count(t.include), "≥ 6", count(t.include) >= 6);
  if (!t.fit) return rows;
  add("Optimizer", t.fit.success ? "Converged" : "Failed", "Converged", t.fit.success);
  add("CD bound", t.fit.cd_at_bound ? "Active" : "Interior", "Interior", !t.fit.cd_at_bound);
  add("Jacobian condition", fmt(row.condition), "≤ 10,000", row.condition !== null && row.condition <= 1e4);
  add("Projected size sensitivity", fmt(t.fit.sensitivity), "≥ 7", t.fit.sensitivity >= 7);
  add("Normalized model RMS", fmt(row.fit_rms), "≤ 0.095", row.fit_rms <= 0.095);
  return rows;
}
function qualificationStep() {
  if (!row) return traceMissing();
  let out = "";
  if (row.status === "ok") {
    const nm = study.pixel_nm === null ? "" : ` · ${fmt(row.cd_px * study.pixel_nm, 5)} nm`;
    out += `<div class="result">${fmt(row.cd_px, 6)} px${nm}</div><p>This tile’s effective opening CD is the larger fitted aperture axis. The model pools the periodic response of ${row.n_sites} detected sites. Select <b>Hole detail</b> to inspect the fitted aperture at individual lattice positions.</p>`;
    out += facts([
      ["Width along aperture axis", `${fmt(row.cd_x_px, 6)} px`],
      ["Perpendicular width", `${fmt(row.cd_y_px, 6)} px`],
      ["Area-equivalent diameter", `${fmt(row.area_diameter_px, 6)} px`],
      ["Pixel pitch", study.pixel_nm === null ? "Unspecified" : `${fmt(study.pixel_nm)} nm/px`],
    ]);
    out += formula(
      `CD = max(width, height)\narea-equivalent diameter = ${reg.shape === "square" ? "2 √(width × height / π)" : "√(width × height)"}${study.pixel_nm === null ? "" : `\n${fmt(row.cd_px, 7)} px × ${study.pixel_nm} nm/px = ${fmt(row.cd_px * study.pixel_nm, 7)} nm`}`,
    );
  } else out += note(`CD withheld · ${human(row.status)}`);
  out += "<h3>Qualification checks</h3>" + table(["Check", "Measured", "Limit", "Result"], gates());
  out +=
    "<p>The condition number uses column-normalized fit derivatives. Projected sensitivity measures how much size information survives after allowing blur and gain to change. A high condition number, weak sensitivity or poor model agreement can withhold the CD.</p>";
  const reasons = new Map();
  for (const t of tiles) reasons.set(t.status, (reasons.get(t.status) || 0) + 1);
  out += fold(
    "Outcomes across the field",
    table(
      ["Status", "Tiles"],
      [...reasons].map(([k, v]) => [human(k), v]),
    ),
  );
  out +=
    "<p>One image can reveal weak orders, phase errors and size–blur ambiguity. Establishing physical accuracy requires independently measured openings. Repeat images are needed to assess repeatability.</p>";
  return out;
}
const panes = [
  discoveryStep,
  latticeStep,
  windowStep,
  fourierStep,
  phaseStep,
  selectionStep,
  fitStep,
  qualificationStep,
];
function renderWalk() {
  $("walk").innerHTML = `<h2>${steps[stage]}</h2>` + panes[stage]();
  if ($("order"))
    $("order").onchange = (e) => {
      order = Number(e.target.value);
      renderWalk();
      requestDraw();
    };
  spectrum();
}
function setLayer(mode) {
  if (tileLayer(mode) && trace.mean === undefined) mode = "raw";
  $("layer").value = mode;
  $("layerhint").textContent = layerHints[mode];
  if (study.white_is_zero && mode === "raw")
    $("layerhint").textContent += " This WhiteIsZero TIFF is inverted for bright-opening analysis. The readout shows both DN conventions.";
  $("stretch").disabled = mode !== "raw";
  requestDraw();
}
function chooseTile(idx, recenter) {
  tileIndex = idx;
  row = tiles[idx] || null;
  trace = row ? JSON.parse($(`tile-${idx}`).textContent) : {};
  order = 0;
  if (row && regions[regionIndex]?.id !== row.region_id) {
    regionIndex = regions.findIndex((r) => r.id === row.region_id);
    reg = regions[regionIndex];
    populateTiles();
  }
  $("region").value = String(regionIndex);
  $("tile").value = String(idx);
  $("selection").textContent = row
    ? `${row.n_sites} sites · ${human(row.status)}${row.status === "ok" ? ` · CD ${fmt(row.cd_px, 5)} px` : ""}`
    : "No complete tiles";
  $("focus").disabled = !row;
  $("detail").disabled = !reg;
  $("outlines").disabled = row?.status !== "ok";
  for (const op of $("layer").options) if (tileLayer(op.value)) op.disabled = trace.mean === undefined;
  setLayer($("layer").value || "raw");
  if (recenter && row) fitTile();
  renderWalk();
}
function populateTiles() {
  const ids = tiles.flatMap((t, i) => (t.region_id === reg?.id ? [i] : []));
  $("tile").replaceChildren(
    ...ids.map((i) => new Option(`(${tiles[i].tile_i}, ${tiles[i].tile_j}) · ${human(tiles[i].status)}`, i)),
  );
  if (!ids.length) $("tile").add(new Option("None", -1));
  $("tile").disabled = !ids.length;
  return ids;
}
function chooseRegion(idx) {
  regionIndex = idx;
  reg = regions[idx] || null;
  const ids = populateTiles();
  chooseTile(ids.find((i) => tiles[i].status === "ok") ?? ids[0] ?? -1, false);
}
function changeStage(n) {
  stage = clamp(n, 0, 7);
  $("stage").value = String(stage);
  $("prev").disabled = stage === 0;
  $("next").disabled = stage === 7;
  $("sites").checked = stage === 1;
  $("outlines").checked = stage === 7 && row?.status === "ok";
  setLayer(
    trace.mean !== undefined && stage === 2
      ? "weighted"
      : trace.mean !== undefined && stage === 3
        ? "real"
        : "raw",
  );
  if ((stage === 2 || stage === 3) && row) fitTile();
  renderWalk();
}
function holeDetail() {
  if (!reg) return;
  const x = row?.x ?? cam.x,
    y = row?.y ?? cam.y,
    p = reg.pitch;
  let target = [x, y],
    distance = Infinity;
  visitSites(reg, [x - 3 * p, y - 3 * p, x + 3 * p, y + 3 * p], (a, b) => {
    if (Math.hypot(a - x, b - y) < distance) {
      target = [a + 0.5, b + 0.5];
      distance = Math.hypot(a - x, b - y);
    }
  });
  fitBox(target[0] - 2 * p, target[1] - 2 * p, 4 * p, 4 * p);
}
$("source").textContent =
  `${study.file} · Page ${study.page} · ${width.toLocaleString()} × ${height.toLocaleString()} px · ${study.bits}-bit grayscale${study.origin ? " · " + study.origin : ""}`;
$("yield").textContent = `${tiles.filter((t) => t.status === "ok").length} / ${tiles.length}`;
$("foot").textContent =
  `${tileSide} × ${tileSide} px tiles start at the image’s upper-left pixel boundary. Green borders mark qualified CDs, amber marks rejected tiles and white marks the selection. Unboxed margins and region fragments have no complete tile.`;
for (const [v, name] of Object.entries(layerNames)) $("layer").add(new Option(name, v));
regions.forEach((r, i) => $("region").add(new Option(`${r.id + 1} · ${r.shape}`, i)));
if (!regions.length) {
  $("region").add(new Option("None", 0));
  $("region").disabled = true;
}
steps.forEach((s, i) => $("stage").add(new Option(`${i + 1} / ${steps.length}   ${s}`, i)));
$("region").onchange = (e) => {
  chooseRegion(Number(e.target.value));
  if (reg) fitBox(reg.bounds[0], reg.bounds[1], reg.bounds[2] - reg.bounds[0], reg.bounds[3] - reg.bounds[1]);
};
$("tile").onchange = (e) => chooseTile(Number(e.target.value), true);
$("stage").onchange = (e) => changeStage(Number(e.target.value));
$("prev").onclick = () => changeStage(stage - 1);
$("next").onclick = () => changeStage(stage + 1);
$("layer").onchange = (e) => setLayer(e.target.value);
$("field").onclick = () => fitBox(0, 0, width, height);
$("focus").onclick = fitTile;
$("detail").onclick = holeDetail;
$("in").onclick = () => zoom(1.5);
$("out").onclick = () => zoom(1 / 1.5);
for (const id of ["boxes", "sites", "outlines", "stretch"]) $(id).onchange = requestDraw;
new ResizeObserver(requestDraw).observe(cv);
chooseRegion(
  Math.max(
    0,
    regions.findIndex((r) => tiles.some((t) => t.region_id === r.id && t.status === "ok")),
  ),
);
changeStage(0);
fitBox(0, 0, width, height);
