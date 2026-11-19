/* Offline ACTIS viewer. The camera is in native pixels of acquisition zero. */
(() => {
  "use strict";
  const $ = (selector, root = document) => root.querySelector(selector);
  const all = (selector) => Array.from(document.querySelectorAll(selector));
  const data = JSON.parse($("#study").textContent);
  const { frames, methods, sites } = data;
  const tones = ["#e8bb78", "#78d5d5"];
  const letters = ["A", "B"];
  const fmt = (value, digits = 2) =>
    Number.isFinite(value) ? value.toLocaleString("en-US", { maximumFractionDigits: digits }) : "—";
  const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
  function make(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  }
  function option(value, label) {
    return new Option(label, value);
  }
  const fieldCenter = { x: (data.width - 1) / 2, y: (data.height - 1) / 2 };
  const camera = { ...fieldCenter, scale: 1 };
  const regionPitch = new Map(data.regions.map((r) => [r.id, r.pitch]));
  const pitch = data.regions.length ? Math.min(...data.regions.map((r) => r.pitch)) : 10;
  const panes = [];
  let target = 0,
    combined = false,
    initialized = false,
    raf = 0;
  let selection = null,
    overviewMap = null;
  let currentView = "";
  const palette = [];
  const stops = [
    [104, 125, 228],
    [68, 199, 207],
    [242, 219, 134],
  ];
  for (let i = 0; i < 48; i++) {
    const p = (2 * i) / 47,
      a = Math.min(Math.floor(p), 1);
    const rgb = stops[a].map((v, k) => Math.round(v + (stops[a + 1][k] - v) * (p - a)));
    palette.push("rgb(" + rgb.join(",") + ")");
  }

  // Bucket reference sites once. Panning a detail view visits only nearby buckets.
  const cell = Math.max(16, Math.ceil(8 * pitch));
  const buckets = new Map();
  const key = (x, y) => x + "," + y;
  sites.forEach((s, i) => {
    const k = key(Math.floor(s[2] / cell), Math.floor(s[3] / cell));
    if (!buckets.has(k)) buckets.set(k, []);
    buckets.get(k).push(i);
  });
  function visit(box, action) {
    const [x0, y0, x1, y1] = box;
    for (
      let y = Math.max(-1, Math.floor(y0 / cell));
      y <= Math.min(Math.ceil(data.height / cell), Math.floor(y1 / cell));
      y++
    ) {
      for (
        let x = Math.max(-1, Math.floor(x0 / cell));
        x <= Math.min(Math.ceil(data.width / cell), Math.floor(x1 / cell));
        x++
      ) {
        const group = buckets.get(key(x, y));
        if (!group) continue;
        for (const i of group) {
          const s = sites[i];
          if (s[2] >= x0 && s[2] <= x1 && s[3] >= y0 && s[3] <= y1) action(i, s);
        }
      }
    }
  }
  function nearest(x, y, radius) {
    let best = null,
      distance = radius * radius;
    visit([x - radius, y - radius, x + radius, y + radius], (i, s) => {
      const d = (s[2] - x) ** 2 + (s[3] - y) ** 2;
      if (d <= distance) {
        best = i;
        distance = d;
      }
    });
    return best;
  }
  function selectPoint(x, y, i, region = null) {
    $("#rank-note").hidden = true;
    selection =
      i == null ? { x, y, i: null, region } : { x: sites[i][2], y: sites[i][3], i, region: sites[i][0] };
    inspect();
    schedule();
  }
  const imageCache = new Map();
  function assetImage(url) {
    if (!imageCache.has(url)) {
      const image = new Image();
      image.onload = () => {
        schedule();
        for (const p of panes) refreshNotice(p);
      };
      image.onerror = () => {
        image.failed = true;
        schedule();
        for (const p of panes) refreshNotice(p);
      };
      imageCache.set(url, image);
      image.src = url;
    }
    return imageCache.get(url);
  }
  const loadImage = (repeat) => assetImage(frames[repeat].image);

  const layers = new Map(),
    pending = new Map();
  function unpack(str, Type) {
    if (Uint8Array.fromBase64) return new Type(Uint8Array.fromBase64(str).buffer);
    const binary = atob(str),
      bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return new Type(bytes.buffer);
  }
  const layerKey = (repeat, method) => repeat + "-" + method;
  const tileKey = (region, i, j) => region + ":" + i + ":" + j;
  window.ACTIS_LAYER = (repeat, method, layer) => {
    const k = layerKey(repeat, method);
    if (methods[method].support === "hole") {
      layer.bySite = new Int32Array(sites.length).fill(-1);
      layer.rows.forEach((row, ri) => {
        layer.bySite[row[0]] = ri;
      });
      layer.offsets = unpack(layer.outlines[0], Uint32Array);
      layer.vertices = unpack(layer.outlines[1], Int16Array);
      delete layer.outlines;
    } else {
      layer.byTile = new Map(layer.rows.map((row) => [tileKey(...row.slice(0, 3)), row]));
    }
    layers.set(k, layer);
    // Bound retained measurement geometry after switching among long acquisitions.
    for (const old of layers.keys()) {
      if (layers.size <= 4) break;
      if (old !== k && !panes.some((p) => layerKey(p.repeat, p.method) === old)) layers.delete(old);
    }
  };
  function loadLayer(repeat, method) {
    const k = layerKey(repeat, method);
    if (layers.has(k)) return Promise.resolve(layers.get(k));
    if (pending.has(k)) return pending.get(k);
    const task = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = frames[repeat].layers[method];
      script.onload = () => {
        script.remove();
        if (layers.has(k)) resolve(layers.get(k));
        else reject(new Error("Measurement file could not be decoded."));
      };
      script.onerror = () => {
        script.remove();
        reject(new Error("Measurement file could not be opened."));
      };
      document.head.append(script);
    }).finally(() => pending.delete(k));
    pending.set(k, task);
    return task;
  }
  function methodSelect() {
    return [option("image", "Image only"), ...Object.entries(methods).map(([k, v]) => option(k, v.name))];
  }
  const initialMethods = ["contour", "model", "differential", "flux", "harmonic", "pooled", "raw"].filter(
    (method) => (frames[0].qualified ? frames[0].qualified[method] > 0 : sites.length > 0),
  );
  function createPane(idx) {
    const element = $("#pane-template").content.firstElementChild.cloneNode(true);
    $("#pane-grid").append(element);
    const p = {
      element,
      idx,
      field: $(".field", element),
      stage: $(".stage", element),
      repeat: 0,
      method: initialMethods[idx] || "image",
      layer: null,
      version: 0,
    };
    p.methodSelect = $(".method", element);
    p.frameSelect = $(".acq", element);
    p.methodSelect.append(...methodSelect());
    p.frameSelect.append(...frames.map((f, i) => option(i, String(i + 1).padStart(2, "0") + " · " + f.file)));
    for (const [control, label] of [
      [p.methodSelect, "measurement method"],
      [p.frameSelect, "acquisition"],
      [p.field, "registered image, click to inspect or use arrow keys to pan"],
    ]) {
      control.setAttribute("aria-label", "Pane " + letters[idx] + " " + label);
    }
    $(".pane-badge", element).textContent = letters[idx];
    p.methodSelect.addEventListener("change", () => {
      p.method = p.methodSelect.value;
      setTarget(idx);
      changePane(p);
    });
    p.frameSelect.addEventListener("change", () => {
      p.repeat = +p.frameSelect.value;
      setTarget(idx);
      changePane(p);
    });
    panes.push(p);
  }
  function setTarget(idx) {
    target = idx;
    all("[data-target]").forEach((b, i) => b.setAttribute("aria-pressed", String(i === idx)));
    acquisitionTags();
  }
  function acquisitionTags() {
    all(".acquisition").forEach((button, i) => {
      const assigned = panes.filter((p) => p.repeat === i);
      button.classList.toggle("assigned", assigned.length > 0);
      button.setAttribute("aria-pressed", String(panes[target].repeat === i));
      const tags = assigned.map((p) => make("i", letters[p.idx], "tone-" + letters[p.idx].toLowerCase()));
      $(".pane-tags", button).replaceChildren(...tags);
    });
  }
  function acquisitionList() {
    frames.forEach((f, i) => {
      const button = make("button", undefined, "acquisition");
      const img = make("img");
      img.src = f.thumb;
      img.alt = "";
      const caption = make("span");
      caption.append(
        make("strong", "Acquisition " + String(i + 1).padStart(2, "0")),
        make("small", fmt(f.mean, 0) + " DN"),
      );
      button.title = f.file + " · assign to the target pane";
      button.append(img, caption, make("span", "", "pane-tags"));
      button.onclick = () => {
        const p = panes[target];
        p.repeat = i;
        changePane(p);
      };
      $("#acquisitions").append(button);
    });
    all("[data-target]").forEach((button, i) => {
      button.onclick = () => setTarget(i);
    });
  }
  function refreshNotice(p) {
    const image = imageCache.get(frames[p.repeat].image);
    let msg = p.error || (image?.failed ? "Image file could not be opened." : "");
    if (!msg && p.method !== "image" && !p.layer) msg = "Loading measurements…";
    if (!msg && p.layer && !p.layer.valid)
      msg = p.layer.total ? "No qualified measurements" : "No measurements on this support";
    const node = $(".stage-msg", p.element);
    node.textContent = msg;
    node.dataset.message = msg;
  }
  async function changePane(p) {
    const version = ++p.version;
    for (const opt of p.methodSelect.options) {
      if (opt.value !== "image") {
        const suffix = frames[p.repeat].qualified?.[opt.value] === 0 ? " · no valid results" : "";
        opt.textContent = methods[opt.value].name + suffix;
      }
    }
    p.methodSelect.value = p.method;
    p.frameSelect.value = p.repeat;
    p.image = loadImage(p.repeat);
    p.error = "";
    p.layer = null;
    $("#rank-note").hidden = true;
    const method = methods[p.method];
    $(".support", p.element).textContent = method
      ? method.support === "hole"
        ? "Per hole"
        : "Per tile"
      : "Native image";
    acquisitionTags();
    $(".native-download", p.element).href = frames[p.repeat].source;
    refreshNotice(p);
    updateLegends();
    inspect();
    schedule();
    updateRegistration();
    if (method) {
      try {
        const layer = await loadLayer(p.repeat, p.method);
        if (p.version === version) p.layer = layer;
      } catch (error) {
        if (p.version === version) p.error = error.message;
      }
      if (p.version === version) {
        refreshNotice(p);
        updateLegends();
        inspect();
        schedule();
      }
    }
  }
  function updateRegistration() {
    const paired = panes[0].repeat !== panes[1].repeat;
    const alignment = $("#alignment-state");
    alignment.textContent = "Linked coordinates";
    if (paired && panes.some((p) => frames[p.repeat].ambiguous))
      alignment.textContent = "Registration unresolved";
    else if (paired && panes.some((p) => frames[p.repeat].periodicAmbiguity))
      alignment.textContent = "Aligned to nominal crop";
    const bad = panes.filter(
      (p) => frames[p.repeat].ambiguous || (paired && frames[p.repeat].periodicAmbiguity),
    );
    const note = $("#registration-note");
    note.hidden = !bad.length;
    note.textContent = bad
      .map(
        (p) =>
          "Pane " +
          letters[p.idx] +
          " registration: " +
          (frames[p.repeat].ambiguous
            ? frames[p.repeat].registration
            : "Whole-pitch alternatives exist. Alignment assumes the same nominal crop."),
      )
      .join(" · ");
  }

  let colorMode = "value",
    showMarks = true,
    showExcluded = false,
    pixelated = false;
  function byValue() {
    return !combined && colorMode !== "pane";
  }
  function limits(m) {
    return colorMode === "method" ? m.ownLimits : m.limits;
  }
  function updateLegends() {
    for (const p of panes) {
      const node = $(".metric-key", p.element),
        count = $(".coverage", p.element);
      const m = methods[p.method];
      node.replaceChildren();
      if (!m) {
        node.textContent = "Grayscale · shared DN stretch";
        count.textContent = fmt(frames[p.repeat].min, 0) + "–" + fmt(frames[p.repeat].max, 0) + " DN";
      } else {
        if (byValue()) {
          const [lo, hi] = limits(m);
          node.append(
            make("span", (m.support === "tile" ? "Tile " : "") + m.metric),
            make("span", fmt(lo, 1)),
            make("i", "", "gradient"),
            make("span", fmt(hi, 1) + " " + m.unit),
          );
          node.title =
            "2nd–98th percentiles across acquisitions" +
            (colorMode === "method" ? " for this method." : ". CD methods use the same range.");
        } else {
          node.append(make("i", "", "swatch"), make("span", m.metric + " · " + m.unit));
          node.title = "Annotation color identifies the comparison pane.";
        }
        count.textContent = p.layer
          ? fmt(p.layer.valid, 0) +
            "/" +
            fmt(p.layer.total, 0) +
            " " +
            (m.support === "hole" ? "holes" : "tiles")
          : "";
        count.title = "Qualified measurements / eligible measurements";
      }
      if (m?.support === "tile")
        node.title += " Rings mark reference sites and carry the enclosing tile estimate.";
      if (combined && p.idx === 1 && p.error) count.textContent = p.error;
    }
  }
  function context(canvas) {
    const w = canvas.clientWidth,
      h = canvas.clientHeight;
    if (!w || !h) return null;
    const ratio = window.devicePixelRatio || 1;
    const pw = Math.round(w * ratio),
      ph = Math.round(h * ratio);
    if (canvas.width !== pw || canvas.height !== ph) {
      canvas.width = pw;
      canvas.height = ph;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(pw / w, 0, 0, ph / h, 0, 0);
    return { ctx, w, h };
  }
  function viewAt(w, h) {
    const scale = camera.scale;
    return { scale, w, h, x: w / 2 - camera.x * scale, y: h / 2 - camera.y * scale };
  }
  function drawBase(ctx, p, view, opacity) {
    let img = p.image;
    if (!img || !img.complete || !img.naturalWidth || !opacity) return;
    const needed = view.scale * (window.devicePixelRatio || 1);
    const level = frames[p.repeat].levels?.filter((l) => l.scale >= needed).pop();
    if (level) {
      const reduced = assetImage(level.image);
      if (reduced.complete && reduced.naturalWidth) img = reduced;
    }
    const [dx, dy] = frames[p.repeat].shift;
    ctx.globalAlpha = opacity;
    ctx.imageSmoothingEnabled = !pixelated || view.scale < 1;
    ctx.imageSmoothingQuality = view.scale < 1 ? "high" : "low";
    ctx.drawImage(
      img,
      view.x - (dx + 0.5) * view.scale,
      view.y - (dy + 0.5) * view.scale,
      data.width * view.scale,
      data.height * view.scale,
    );
    ctx.globalAlpha = 1;
  }
  function mapped(value, m, idx) {
    if (!byValue()) return tones[idx];
    const [lo, hi] = limits(m);
    return palette[clamp(Math.round(((value - lo) / (hi - lo)) * 47), 0, 47)];
  }
  function bounds(v) {
    const margin = 2 * pitch;
    return [
      -v.x / v.scale - margin,
      -v.y / v.scale - margin,
      (v.w - v.x) / v.scale + margin,
      (v.h - v.y) / v.scale + margin,
    ];
  }
  function cross(ctx, x, y, r) {
    ctx.moveTo(x - r, y - r);
    ctx.lineTo(x + r, y + r);
    ctx.moveTo(x + r, y - r);
    ctx.lineTo(x - r, y + r);
  }
  function atTile(layer, region, x, y) {
    const i = Math.floor((x - data.bounds[0]) / data.tilePx),
      j = Math.floor((y - data.bounds[1]) / data.tilePx);
    return layer.byTile.get(tileKey(region, i, j));
  }
  function drawTiles(ctx, p, v, range) {
    const m = methods[p.method],
      side = data.tilePx * v.scale,
      layer = p.layer;
    // Each small ring is a reference site carrying its enclosing tile's value.
    visit(range, (_i, s) => {
      const row = atTile(layer, s[0], s[2], s[3]);
      if (!row || row[5] === null) return;
      const x = v.x + s[2] * v.scale,
        y = v.y + s[3] * v.scale;
      const radius = clamp((regionPitch.get(s[0]) || pitch) * v.scale * 0.13, 0.45, 3.5);
      ctx.strokeStyle = mapped(row[5], m, p.idx);
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.stroke();
    });
    for (const row of layer.rows) {
      if (row[5] === null && !showExcluded) continue;
      const x = v.x + row[3] * v.scale - side / 2,
        y = v.y + row[4] * v.scale - side / 2;
      if (x > v.w || y > v.h || x + side < 0 || y + side < 0) continue;
      ctx.strokeStyle = row[5] === null ? "#fa9298" : mapped(row[5], m, p.idx);
      ctx.setLineDash([5, 4]);
      ctx.strokeRect(x, y, side, side);
      ctx.setLineDash([]);
    }
  }
  function drawHoles(ctx, p, v, range) {
    const m = methods[p.method],
      layer = p.layer;
    const [dx, dy] = frames[p.repeat].shift;
    // One stroke per color group keeps the full-field view usable at high density.
    const paths = new Map();
    function path(color) {
      if (!paths.has(color)) paths.set(color, new Path2D());
      return paths.get(color);
    }
    visit(range, (siteId, s) => {
      const ri = layer.bySite[siteId];
      if (ri < 0) return;
      const row = layer.rows[ri],
        value = row[3];
      const x = v.x + (row[1] - dx) * v.scale,
        y = v.y + (row[2] - dy) * v.scale;
      const separation = (regionPitch.get(s[0]) || pitch) * v.scale;
      if (value === null) {
        if (showExcluded) cross(path("#ff9296"), x, y, clamp(separation * 0.13, 0.5, 3.4));
        return;
      }
      const line = path(mapped(value, m, p.idx));
      const start = layer.offsets[ri],
        end = layer.offsets[ri + 1];
      if (separation >= 7 && end > start) {
        const unit = v.scale / layer.outlineScale;
        for (let j = start; j < end; j++) {
          const xx = x + layer.vertices[2 * j] * unit,
            yy = y + layer.vertices[2 * j + 1] * unit;
          if (j === start) line.moveTo(xx, yy);
          else line.lineTo(xx, yy);
        }
        line.closePath();
      } else {
        const r = clamp(separation * 0.12, 0.38, 3.3);
        line.moveTo(x + r, y);
        line.arc(x, y, r, 0, Math.PI * 2);
      }
    });
    for (const [color, line] of paths) {
      ctx.strokeStyle = color;
      ctx.stroke(line);
    }
  }
  function annotate(ctx, p, v) {
    if (!p.layer || !showMarks) return;
    ctx.save();
    ctx.globalAlpha = 0.93;
    ctx.lineWidth = Math.min(combined && p.idx === 0 ? 2.1 : 1.15, Math.max(0.25, pitch * v.scale * 0.22));
    if (combined && p.idx === 1) ctx.setLineDash([3, 2.4]);
    if (methods[p.method].support === "tile") drawTiles(ctx, p, v, bounds(v));
    else drawHoles(ctx, p, v, bounds(v));
    ctx.restore();
  }
  function pin(ctx, v) {
    if (!selection) return;
    const x = v.x + selection.x * v.scale,
      y = v.y + selection.y * v.scale;
    const r = clamp(pitch * v.scale * 0.43, 7, 25),
      d = Math.max(4, r * 0.43);
    const line = new Path2D();
    for (const ix of [-1, 1])
      for (const iy of [-1, 1]) {
        line.moveTo(x + ix * (r - d), y + iy * r);
        line.lineTo(x + ix * r, y + iy * r);
        line.lineTo(x + ix * r, y + iy * (r - d));
      }
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#111a1a";
    ctx.stroke(line);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "#f3f6e7";
    ctx.stroke(line);
  }
  function drawPane(p) {
    const surface = context(p.field);
    if (!surface) return;
    const { ctx, w, h } = surface,
      v = viewAt(w, h);
    ctx.fillStyle = "#101315";
    ctx.fillRect(0, 0, w, h);
    drawBase(ctx, p, v, 1);
    if (combined) drawBase(ctx, panes[1], v, +$("#blend").value / 100);
    annotate(ctx, p, v);
    if (combined) annotate(ctx, panes[1], v);
    pin(ctx, v);
    const notice = $(".stage-msg", p.element);
    const issues = combined
      ? panes
          .filter((q) => $(".stage-msg", q.element).dataset.message)
          .map((q) => letters[q.idx] + " · " + $(".stage-msg", q.element).dataset.message)
      : [];
    notice.textContent = combined ? issues.join(" / ") : notice.dataset.message || "";
    const targetNm = (Math.min(74, w * 0.22) / v.scale) * data.pixelNm;
    const magnitude = 10 ** Math.floor(Math.log10(targetNm));
    const nm = [5, 2, 1].find((k) => k * magnitude <= targetNm) * magnitude;
    const ruler = $(".ruler", p.element);
    ruler.textContent = nm >= 1000 ? fmt(nm / 1000, 2) + " µm" : fmt(nm, 2) + " nm";
    ruler.style.setProperty("--bar", (nm / data.pixelNm) * v.scale + "px");
  }
  function drawOverview() {
    const surface = context($("#overview"));
    if (!surface) return;
    const { ctx, w, h } = surface;
    const scale = Math.min((w - 8) / data.width, (h - 8) / data.height);
    const x = (w - data.width * scale) / 2,
      y = (h - data.height * scale) / 2;
    overviewMap = { scale, x, y };
    ctx.fillStyle = "#101315";
    ctx.fillRect(0, 0, w, h);
    const image = assetImage(data.overview || frames[0].image);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    if (image.complete && image.naturalWidth)
      ctx.drawImage(image, x, y, data.width * scale, data.height * scale);
    const p = panes[0],
      vw = p.field.clientWidth / camera.scale,
      vh = p.field.clientHeight / camera.scale;
    const left = x + (camera.x + 0.5 - vw / 2) * scale,
      top = y + (camera.y + 0.5 - vh / 2) * scale;
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, data.width * scale, data.height * scale);
    ctx.clip();
    ctx.fillStyle = "#1013157a";
    ctx.fillRect(x, y, data.width * scale, Math.max(0, top - y));
    ctx.fillRect(x, top + vh * scale, data.width * scale, h);
    ctx.fillRect(x, top, Math.max(0, left - x), vh * scale);
    ctx.fillRect(left + vw * scale, top, w, vh * scale);
    const bx = Math.max(x, left),
      by = Math.max(y, top);
    ctx.strokeStyle = "#cee9a6";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(
      bx + 0.6,
      by + 0.6,
      Math.max(0, Math.min(x + data.width * scale, left + vw * scale) - bx - 1.2),
      Math.max(0, Math.min(y + data.height * scale, top + vh * scale) - by - 1.2),
    );
    ctx.restore();
  }
  function schedule() {
    if (raf) return;
    raf = requestAnimationFrame(() => {
      raf = 0;
      if (currentView !== "comparison") return;
      if (!initialized) initializeCamera();
      drawPane(panes[0]);
      if (!combined) drawPane(panes[1]);
      drawOverview();
      $("#zoom-level").textContent = fmt(camera.scale * 100, 0) + "%";
    });
  }
  function initializeCamera() {
    const w = panes[0].field.clientWidth,
      h = panes[0].field.clientHeight;
    if (!w || !h) return;
    initialized = true;
    let i = nearest(camera.x, camera.y, 5 * pitch);
    if (i === null && sites.length) {
      let distance = Infinity;
      sites.forEach((s, n) => {
        const d = (s[2] - camera.x) ** 2 + (s[3] - camera.y) ** 2;
        if (d < distance) {
          distance = d;
          i = n;
        }
      });
    }
    if (i !== null) {
      camera.x = sites[i][2];
      camera.y = sites[i][3];
      // Open on a readable patch. Fit and the navigator provide full-field context.
      camera.scale = clamp(Math.max(fitScale(), Math.min(w, h) / (15 * pitch)), 0.01, 64);
    } else camera.scale = fitScale();
    selectPoint(camera.x, camera.y, i);
  }
  function fitScale() {
    return Math.max(
      0.001,
      Math.min(
        ...panes
          .filter((p) => p.field.clientWidth > 0)
          .map((p) => Math.min(p.field.clientWidth / data.width, p.field.clientHeight / data.height)),
      ),
    );
  }
  function constrain() {
    camera.x = clamp(camera.x, -0.5, data.width - 0.5);
    camera.y = clamp(camera.y, -0.5, data.height - 0.5);
  }
  function fit() {
    Object.assign(camera, fieldCenter, { scale: fitScale() });
    schedule();
  }
  function zoom(scale, at = null) {
    const next = clamp(scale, Math.min(1, fitScale()) / 2, Math.max(64, fitScale()));
    if (at) {
      camera.x += at.x * (1 / camera.scale - 1 / next);
      camera.y += at.y * (1 / camera.scale - 1 / next);
    }
    camera.scale = next;
    constrain();
    schedule();
  }
  function point(event, canvas) {
    const r = canvas.getBoundingClientRect();
    return { x: event.clientX - r.left - r.width / 2, y: event.clientY - r.top - r.height / 2 };
  }
  function keyboard(event) {
    const step = (event.shiftKey ? 80 : 20) / camera.scale;
    const changes = {
      ArrowLeft: [-step, 0],
      ArrowRight: [step, 0],
      ArrowUp: [0, -step],
      ArrowDown: [0, step],
    };
    if (changes[event.key]) {
      const [x, y] = changes[event.key];
      camera.x += x;
      camera.y += y;
      constrain();
      schedule();
    } else if (["+", "="].includes(event.key)) zoom(camera.scale * 1.4);
    else if (event.key === "-") zoom(camera.scale / 1.4);
    else if (event.key.toLowerCase() === "f") fit();
    else if (event.key === "Enter") selectPoint(camera.x, camera.y, nearest(camera.x, camera.y, 1.2 * pitch));
    else return;
    event.preventDefault();
  }
  function bindField(p) {
    const canvas = p.field;
    let drag = null;
    canvas.addEventListener("keydown", keyboard);
    canvas.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        const dy = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 200 : 1);
        zoom(camera.scale * Math.exp(clamp(-dy * 0.002, -1, 1)), point(event, canvas));
      },
      { passive: false },
    );
    canvas.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || drag) return;
      if (!combined) setTarget(p.idx);
      canvas.focus({ preventScroll: true });
      canvas.setPointerCapture(event.pointerId);
      drag = {
        id: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        x: camera.x,
        y: camera.y,
        moved: false,
      };
    });
    canvas.addEventListener("pointermove", (event) => {
      const offset = point(event, canvas);
      const x = camera.x + offset.x / camera.scale,
        y = camera.y + offset.y / camera.scale;
      $("#position").textContent = "X " + fmt(x, 1) + "  Y " + fmt(y, 1) + " px";
      if (!drag || drag.id !== event.pointerId) return;
      const dx = event.clientX - drag.startX,
        dy = event.clientY - drag.startY;
      drag.moved ||= Math.hypot(dx, dy) > 3;
      if (drag.moved) {
        camera.x = drag.x - dx / camera.scale;
        camera.y = drag.y - dy / camera.scale;
        constrain();
        canvas.classList.add("dragging");
        schedule();
      }
    });
    canvas.addEventListener("pointerup", (event) => {
      if (!drag || drag.id !== event.pointerId) return;
      if (!drag.moved) {
        const at = point(event, canvas),
          x = camera.x + at.x / camera.scale,
          y = camera.y + at.y / camera.scale;
        const i = nearest(x, y, Math.max(pitch * 0.78, Math.min(4 / camera.scale, 2 * pitch)));
        selectPoint(x, y, i);
      }
      canvas.releasePointerCapture(event.pointerId);
    });
    canvas.addEventListener("lostpointercapture", () => {
      drag = null;
      canvas.classList.remove("dragging");
    });
    canvas.addEventListener("pointerleave", () => {
      $("#position").textContent = "";
    });
  }
  function navigate(event) {
    if (!overviewMap) return;
    const box = $("#overview").getBoundingClientRect(),
      m = overviewMap;
    camera.x = (event.clientX - box.left - m.x) / m.scale - 0.5;
    camera.y = (event.clientY - box.top - m.y) / m.scale - 0.5;
    constrain();
    schedule();
  }
  function reading(p) {
    const spec = methods[p.method];
    if (!spec)
      return {
        name: "Image only",
        value: frames[p.repeat].mean,
        unit: "DN",
        note: "Whole-acquisition mean",
        metric: "Frame mean",
        support: "frame",
      };
    const r = {
      name: spec.name,
      unit: spec.unit,
      value: null,
      note: "",
      metric: spec.metric,
      support: spec.support,
    };
    if (p.error || !p.layer) {
      r.note = p.error || "Loading measurements…";
      return r;
    }
    if (spec.support === "tile") {
      const { x, y, region } = selection;
      const row =
        region !== null
          ? atTile(p.layer, region, x, y)
          : p.layer.rows.find(
              (t) =>
                x >= t[3] - data.tilePx / 2 &&
                x < t[3] + data.tilePx / 2 &&
                y >= t[4] - data.tilePx / 2 &&
                y < t[4] + data.tilePx / 2,
            );
      if (row) {
        r.value = row[5];
        r.note =
          "Tile (" +
          row[1] +
          ", " +
          row[2] +
          ") · " +
          (r.value === null ? row[6].replaceAll("_", " ") : "shared by sites in tile");
      } else r.note = "Outside measured tiles";
      return r;
    }
    const ri = selection.i === null ? -1 : p.layer.bySite[selection.i];
    if (ri < 0) {
      r.note = sites.length ? "Click a measured site" : "No reference sites";
      return r;
    }
    const row = p.layer.rows[ri];
    r.value = row[3];
    if (r.value === null) {
      r.note = "Excluded · " + (row[4] || "unqualified").replaceAll("_", " ");
      return r;
    }
    const dims = row[5] !== null && row[6] !== null ? fmt(row[5], 1) + " × " + fmt(row[6], 1) + " nm" : "";
    const flags = row[4] && row[4] !== "ok" ? row[4].replaceAll("_", " ") : "";
    r.note = [dims, flags].filter(Boolean).join(" · ") || "Qualified site";
    r.detail = "Native center (" + fmt(row[1], 3) + ", " + fmt(row[2], 3) + ") px";
    if (row[7] !== null && row[8] !== null)
      r.detail += " · Held-out residual (" + fmt(row[7], 2) + ", " + fmt(row[8], 2) + ") nm";
    return r;
  }
  function inspect() {
    if (!selection || panes.length !== 2) return;
    const s = selection.i !== null ? sites[selection.i] : null;
    $("#site-label").textContent = s ? "Region " + s[0] + " / Site " + fmt(s[1], 0) : "Field position";
    $("#site-xy").textContent = "X " + fmt(selection.x, 2) + "  Y " + fmt(selection.y, 2) + " px";
    const pair = panes.map(reading),
      [a, b] = pair;
    const same = a.metric === b.metric && a.support === b.support && a.unit === b.unit;
    const registered =
      panes[0].repeat === panes[1].repeat ||
      !(frames[panes[0].repeat].ambiguous || frames[panes[1].repeat].ambiguous) ||
      a.support === "frame";
    $("#largest").hidden = !(
      same &&
      registered &&
      a.support !== "frame" &&
      panes.every((p) => p.layer?.valid) &&
      (panes[0].method !== panes[1].method || panes[0].repeat !== panes[1].repeat)
    );
    const delta = same && registered && a.value !== null && b.value !== null ? b.value - a.value : null;
    const readings = [
      ...pair,
      {
        name: "Difference · B − A",
        value: delta,
        unit: same ? a.unit : "",
        note: !same
          ? "Different quantities or supports"
          : !registered
            ? "Unresolved acquisition registration"
            : delta === null
              ? "Both panes need a valid value"
              : { hole: "Same reference site", tile: "Same reference tile", frame: "Acquisition means" }[
                  a.support
                ],
      },
    ];
    const grid = make("div", undefined, "inspection-grid");
    readings.forEach((r, i) => {
      const card = make("div", undefined, "reading"),
        label = make("h3");
      if (i < 2) label.append(make("b", letters[i], "tone-" + letters[i].toLowerCase()));
      label.append(r.name);
      const value = make("strong", (i === 2 && r.value > 0 ? "+" : "") + fmt(r.value, 2));
      if (r.value !== null) value.append(make("small", " " + r.unit));
      card.append(label, value, make("p", r.note));
      if (r.detail) card.append(make("p", r.detail, "coordinates"));
      grid.append(card);
    });
    $("#inspection").replaceChildren(grid);
  }
  function jump(x, y, site, region = null) {
    camera.x = x;
    camera.y = y;
    camera.scale = Math.max(
      camera.scale,
      Math.min(panes[0].field.clientWidth, panes[0].field.clientHeight) / (16 * pitch),
    );
    selectPoint(x, y, site, region);
  }
  function largestDifference() {
    const [a, b] = panes;
    let winner = null,
      largest = -1,
      matched = 0;
    function consider(v1, v2, x, y, i, region = null) {
      if (!Number.isFinite(v1) || !Number.isFinite(v2)) return;
      matched++;
      const magnitude = Math.abs(v2 - v1);
      if (magnitude > largest) {
        largest = magnitude;
        winner = { x, y, i, region };
      }
    }
    if (methods[a.method].support === "hole") {
      for (const row of a.layer.rows) {
        const i = row[0],
          j = b.layer.bySite[i];
        if (j >= 0) consider(row[3], b.layer.rows[j][3], sites[i][2], sites[i][3], i);
      }
    } else {
      for (const row of a.layer.rows) {
        const other = b.layer.byTile.get(tileKey(row[0], row[1], row[2]));
        if (other) consider(row[5], other[5], row[3], row[4], null, row[0]);
      }
    }
    const note = $("#rank-note");
    if (winner) jump(winner.x, winner.y, winner.i, winner.region);
    note.textContent = winner
      ? "Largest absolute difference across " +
        fmt(matched, 0) +
        " matched " +
        (methods[a.method].support === "hole" ? "holes." : "tiles.")
      : "These selections have no shared valid measurements.";
    note.hidden = false;
  }
  function setupSiteSearch() {
    $("#find").hidden = !sites.length;
    $("#region-id").append(...data.regions.map((r) => option(r.id, r.id)));
    $("#find").onclick = () => {
      const form = $("#site-search");
      form.hidden = !form.hidden;
      $("#find").setAttribute("aria-expanded", String(!form.hidden));
      if (!form.hidden) {
        if (selection?.i != null) $("#region-id").value = sites[selection.i][0];
        $("#site-id").focus();
      }
    };
    $("#site-search").onsubmit = (event) => {
      event.preventDefault();
      const i = sites.findIndex((s) => s[0] === +$("#region-id").value && s[1] === +$("#site-id").value);
      $("#search-note").textContent = i < 0 ? "Site is outside the measured field." : "";
      if (i >= 0) jump(sites[i][2], sites[i][3], i);
    };
    $("#largest").onclick = largestDifference;
  }
  function mode(value) {
    combined = value;
    $("#workbench").classList.toggle("overlay-mode", value);
    $("#split").setAttribute("aria-pressed", String(!value));
    $("#overlay").setAttribute("aria-pressed", String(value));
    $("#blend-control").hidden = !value;
    $(".color-control").hidden = value;
    updateLegends();
    schedule();
  }
  function route() {
    const name = location.hash.slice(1);
    currentView = ["precision", "run"].includes(name) ? name : "comparison";
    for (const id of ["comparison", "precision", "run"]) $("#" + id).hidden = currentView !== id;
    for (const link of all("[data-view]")) {
      if (link.dataset.view === currentView) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    }
    window.scrollTo(0, 0);
    document.title =
      "ACTIS · " +
      { comparison: "Measurement comparison", precision: "Measurement precision", run: "Run details" }[
        currentView
      ];
    schedule();
  }
  createPane(0);
  createPane(1);
  acquisitionList();
  setTarget(0);
  setupSiteSearch();
  for (const p of panes) {
    bindField(p);
    changePane(p);
  }
  $("#split").onclick = () => mode(false);
  $("#overlay").onclick = () => mode(true);
  $("#fit").onclick = $("#nav-fit").onclick = fit;
  $("#native").onclick = () => zoom(1);
  $("#zoom-in").onclick = () => zoom(camera.scale * Math.SQRT2);
  $("#zoom-out").onclick = () => zoom(camera.scale / Math.SQRT2);
  for (const id of ["marks", "excluded", "color-mode", "pixels"])
    $("#" + id).onchange = () => {
      showMarks = $("#marks").checked;
      showExcluded = $("#excluded").checked;
      colorMode = $("#color-mode").value;
      pixelated = $("#pixels").checked;
      updateLegends();
      schedule();
    };
  $("#blend").oninput = () => {
    $("#blend-value").textContent = $("#blend").value + "% B";
    schedule();
  };
  const overview = $("#overview");
  overview.onkeydown = keyboard;
  overview.onpointerdown = (e) => {
    overview.setPointerCapture(e.pointerId);
    navigate(e);
  };
  overview.onpointermove = (e) => {
    if (overview.hasPointerCapture(e.pointerId)) navigate(e);
  };
  const observer = new ResizeObserver(schedule);
  observer.observe(panes[0].stage);
  observer.observe(overview);
  window.addEventListener("hashchange", route);
  route();
})();
