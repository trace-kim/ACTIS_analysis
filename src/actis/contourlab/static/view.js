"use strict";
/* The image of the tab in front, with its layers: tiles of the image and of the mask, contours, the region
   and the pointer's readout. app.js owns the state (S, the tabs); this file draws it and turns the mouse
   into changes of it.

   Coordinates. The server's are pixel centres: the first pixel is at (0, 0). On screen that pixel covers
   the square from 0 to 1, so a point (x, y) is drawn at x + 0.5, y + 0.5. A tab's camera is the place of
   the image at the middle of the window (in those screen units) and the zoom in CSS pixels per image pixel. */

const TILE = 256;
const COLOUR = { contours: "#35d3f0", before: "#e866d2", pinned: "#a6e36c", picked: "#ffffff", region: "#ffd27a" };

const View = (() => {
  const canvas = $("view");
  const ctx = canvas.getContext("2d");
  const map = $("map");
  const mapCtx = map.getContext("2d");
  let vw = 1;
  let vh = 1;
  let dpr = 1;
  let pending = 0; // the animation frame asked for
  let frame = 0; // counts the frames drawn
  let pointer = null; // [x, y] in CSS pixels of the canvas; null: not over it
  let gesture = null; // a drag in progress
  let bare = false; // Space held: the image without its layers

  // --- Tiles ----------------------------------------------------------------------------------------------

  const tiles = new Map(); // key -> {state, values, w, h, canvas, variant, wanted}
  const waiting = [];
  let loading = 0;

  function tileOf(tab, layer, level, tx, ty, fetch = true) {
    const key = `${tab.id}/${layer.id}/${level}/${tx}/${ty}`;
    let tile = tiles.get(key);
    if (!tile) {
      if (!fetch) return null;
      tile = { key, state: "queued", wanted: frame, path: `/api/tabs/${tab.id}/tile/${layer.id}/${level}/${tx}/${ty}` };
      tiles.set(key, tile);
      waiting.push(tile);
      pump();
    }
    tile.wanted = frame;
    return tile.state === "ready" ? tile : null;
  }

  function pump() {
    while (loading < 5 && waiting.length) {
      const tile = waiting.pop(); // the newest first: what the window shows now
      if (tile.wanted < frame - 4) {
        tiles.delete(tile.key); // scrolled away before its turn came
        continue;
      }
      loading++;
      tile.state = "loading";
      load(tile)
        .catch(() => {
          tile.state = "failed";
        })
        .finally(() => {
          loading--;
          pump();
          draw();
        });
    }
  }

  async function load(tile) {
    const reply = await api.call(tile.path);
    const [kind, w, h] = reply.headers.get("X-Tile").split(" ");
    const buffer = await reply.arrayBuffer();
    const Kind = { uint8: Uint8Array, uint16: Uint16Array }[kind] || Float32Array;
    tile.values = new Kind(buffer);
    tile.w = Number(w);
    tile.h = Number(h);
    tile.whole = kind !== "float32";
    tile.state = "ready";
    if (tiles.size > 900) {
      const old = [...tiles.values()].filter((t) => t.state !== "loading").sort((a, b) => a.wanted - b.wanted);
      for (const t of old.slice(0, tiles.size - 700)) tiles.delete(t.key);
    }
  }

  /* A tile as a picture: grey between the two levels, or the mask's colour where there is mask. */
  function painted(tile, layer, black, white) {
    const variant = layer.mask ? "mask" : `${black}:${white}`;
    if (tile.canvas && tile.variant === variant) return tile.canvas;
    const picture = tile.canvas || document.createElement("canvas");
    picture.width = tile.w;
    picture.height = tile.h;
    const g = picture.getContext("2d");
    const image = g.createImageData(tile.w, tile.h);
    const out = new Uint32Array(image.data.buffer); // one pixel each: alpha, blue, green, red
    const values = tile.values;
    if (layer.mask) {
      for (let i = 0; i < values.length; i++) out[i] = (((values[i] * 0.45) | 0) << 24) | (0 << 16) | (192 << 8) | 240;
    } else {
      const gain = 255 / Math.max(white - black, 1e-30);
      for (let i = 0; i < values.length; i++) {
        let grey = (values[i] - black) * gain;
        grey = grey < 0 ? 0 : grey > 255 ? 255 : grey | 0;
        out[i] = 0xff000000 | (grey << 16) | (grey << 8) | grey;
      }
    }
    g.putImageData(image, 0, 0);
    tile.canvas = picture;
    tile.variant = variant;
    return picture;
  }

  function forget(tabId) {
    for (const key of tiles.keys()) if (key.startsWith(tabId + "/")) tiles.delete(key);
  }

  // --- Camera ---------------------------------------------------------------------------------------------

  function edges(tab) {
    const { x, y, z } = tab.cam;
    return { left: x - vw / (2 * z), top: y - vh / (2 * z), right: x + vw / (2 * z), bottom: y + vh / (2 * z) };
  }

  /* The place in the image under a point of the canvas (CSS pixels), in screen units (module comment). */
  function imageAt(tab, point) {
    return [tab.cam.x + (point[0] - vw / 2) / tab.cam.z, tab.cam.y + (point[1] - vh / 2) / tab.cam.z];
  }

  function fitZoom(tab) {
    return 0.96 * Math.min(vw / tab.info.width, vh / tab.info.height);
  }

  /* Show the whole image; of a long strip, its full width at the place of the region the workflow runs on. */
  function fit(tab) {
    measure();
    const whole = fitZoom(tab);
    const across = (0.96 * vw) / tab.info.width;
    if (whole < 0.3 * across) {
      const region = tab.region || [0, 0, tab.info.width, Math.min(tab.info.height, vh / across)];
      tab.cam = { x: tab.info.width / 2, y: (region[1] + region[3]) / 2, z: across };
    } else {
      tab.cam = { x: tab.info.width / 2, y: tab.info.height / 2, z: whole };
    }
    draw();
  }

  function zoomTo(tab, zoom, about = [vw / 2, vh / 2]) {
    const [x, y] = imageAt(tab, about);
    tab.cam.z = clamp(zoom, Math.min(fitZoom(tab), 1) * 0.5, 400);
    tab.cam.x = x - (about[0] - vw / 2) / tab.cam.z;
    tab.cam.y = y - (about[1] - vh / 2) / tab.cam.z;
    hold(tab);
    draw();
  }

  /* Keep some of the image in the window. */
  function hold(tab) {
    const marginX = vw / (2 * tab.cam.z) - 8 / tab.cam.z;
    const marginY = vh / (2 * tab.cam.z) - 8 / tab.cam.z;
    tab.cam.x = clamp(tab.cam.x, -marginX, tab.info.width + marginX);
    tab.cam.y = clamp(tab.cam.y, -marginY, tab.info.height + marginY);
  }

  function show(tab, box, margin = 1.25) {
    const [x0, y0, x1, y1] = box;
    tab.cam = {
      x: (x0 + x1) / 2,
      y: (y0 + y1) / 2,
      z: Math.min(vw / ((x1 - x0) * margin), vh / ((y1 - y0) * margin), 400),
    };
    draw();
  }

  /* What the window shows of the image, as a region of whole pixels [x0, y0, x1, y1]. */
  function shownRegion(tab) {
    const e = edges(tab);
    return [
      clamp(Math.floor(e.left), 0, tab.info.width),
      clamp(Math.floor(e.top), 0, tab.info.height),
      clamp(Math.ceil(e.right), 0, tab.info.width),
      clamp(Math.ceil(e.bottom), 0, tab.info.height),
    ];
  }

  // --- Drawing --------------------------------------------------------------------------------------------

  function measure() {
    dpr = window.devicePixelRatio || 1;
    vw = Math.max(1, canvas.clientWidth);
    vh = Math.max(1, canvas.clientHeight);
    if (canvas.width !== Math.round(vw * dpr) || canvas.height !== Math.round(vh * dpr)) {
      canvas.width = Math.round(vw * dpr);
      canvas.height = Math.round(vh * dpr);
    }
  }

  function draw() {
    if (!pending) pending = requestAnimationFrame(paint);
  }

  function paint() {
    pending = 0;
    frame++;
    measure();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#0a0d10";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const tab = S.active;
    $("empty").hidden = Boolean(tab);
    map.hidden = true;
    if (!tab) return;
    if (!tab.cam) fit(tab);
    const view = shown(tab); // {state, layers...} of what the result holds for the step shown
    const source = tab.info.source;
    const top = view.image; // the processed image, if that is what is shown
    if (!top || top.x || top.y || top.width / top.scale < tab.info.width || top.height / top.scale < tab.info.height) {
      // the image as loaded: all there is to show, or what lies around the region that was processed
      layer(tab, source, levelsOf(tab, source, !top), top ? 0.5 : 1);
    }
    // an image no module changed is shown with the grey levels of the whole image, as what lies around it
    if (top) layer(tab, top, levelsOf(tab, view.changed ? top : source, true), 1);
    if (!bare) {
      if (view.mask && S.show.mask) layer(tab, view.mask, null, 1);
      pages(tab);
      region(tab);
      if (S.show.pinned && tab.pinned) contours(tab, tab.pinned.set, COLOUR.pinned, 1.2, false);
      if (S.show.before && view.before) contours(tab, setOf(tab, view.before), COLOUR.before, 1.2, false);
      if (view.contours) {
        const set = setOf(tab, view.contours);
        if (S.show.contours) contours(tab, set, COLOUR.contours, 1.5, S.show.centres);
        else if (S.show.centres) contours(tab, set, COLOUR.contours, 0, true);
        if (set) mark(tab, set);
      }
    }
    if (gesture && gesture.kind === "region") rubber(tab);
    ruler(tab);
    overview(tab);
    $("zoom").textContent = tab.cam.z >= 1 ? `${trim(tab.cam.z, 3)}×` : `1:${trim(1 / tab.cam.z, 3)}`;
    readout(tab);
  }

  /* The grey levels shown as black and white: the tab's own if set by hand (for the image in front), else
     the layer's. */
  function levelsOf(tab, described, front) {
    if (front && !tab.levels.auto) return { black: tab.levels.black, white: tab.levels.white };
    return { black: described.black, white: described.white };
  }

  function layer(tab, described, levels, alpha) {
    const z = tab.cam.z * dpr; // device pixels per image pixel
    const each = z / described.scale; // device pixels per pixel of the layer
    const level = each >= 1 ? 0 : clamp(Math.floor(Math.log2(1 / each) + 1e-9), 0, described.levels - 1);
    const size = 2 ** level;
    const span = (TILE * size) / described.scale; // image pixels a tile covers
    const e = edges(tab);
    const wide = described.width / described.scale;
    const high = described.height / described.scale;
    const x0 = Math.max(0, Math.floor((e.left - described.x) / span));
    const y0 = Math.max(0, Math.floor((e.top - described.y) / span));
    const x1 = Math.min(Math.ceil(wide / span), Math.ceil((e.right - described.x) / span));
    const y1 = Math.min(Math.ceil(high / span), Math.ceil((e.bottom - described.y) / span));
    const black = levels ? levels.black : 0;
    const white = levels ? levels.white : 255;
    ctx.globalAlpha = alpha;
    // smaller than the screen's pixels: averaged; larger: each pixel a square, unless asked otherwise
    ctx.imageSmoothingEnabled = each * size < 0.999 || (S.show.smooth && !described.mask);
    ctx.imageSmoothingQuality = "high";
    tileOf(tab, described, described.levels - 1, 0, 0); // the whole layer, coarse: shown until the fine tiles come
    const place = (x, y) => [Math.round((x - e.left) * z), Math.round((y - e.top) * z)];
    for (let ty = y0; ty < y1; ty++) {
      for (let tx = x0; tx < x1; tx++) {
        const tile = tileOf(tab, described, level, tx, ty);
        const ix = described.x + tx * span;
        const iy = described.y + ty * span;
        if (tile) {
          const [dx, dy] = place(ix, iy);
          const [ex, ey] = place(ix + (tile.w * size) / described.scale, iy + (tile.h * size) / described.scale);
          ctx.drawImage(painted(tile, described, black, white), dx, dy, ex - dx, ey - dy);
          continue;
        }
        for (let up = level + 1; up < described.levels; up++) {
          // a coarser tile that is here already: its part under this tile
          const by = 2 ** (up - level);
          const parent = tileOf(tab, described, up, Math.floor(tx / by), Math.floor(ty / by), false);
          if (!parent) continue;
          const sx = (tx * TILE) / by - Math.floor(tx / by) * TILE;
          const sy = (ty * TILE) / by - Math.floor(ty / by) * TILE;
          const sw = Math.min(TILE / by, parent.w - sx);
          const sh = Math.min(TILE / by, parent.h - sy);
          if (sw <= 0 || sh <= 0) break;
          const [dx, dy] = place(ix, iy);
          const [ex, ey] = place(ix + (sw * by * size) / described.scale, iy + (sh * by * size) / described.scale);
          ctx.drawImage(painted(parent, described, black, white), sx, sy, sw, sh, dx, dy, ex - dx, ey - dy);
          break;
        }
      }
    }
    ctx.globalAlpha = 1;
  }

  /* The contours of a set that the window shows. Far out, where each is a few screen pixels, a dot each. */
  function contours(tab, set, colour, width, centres) {
    if (!set || !set.n) return;
    const z = tab.cam.z * dpr;
    const e = edges(tab);
    const seen = visible(set, e.left - 0.5, e.top - 0.5, e.right - 0.5, e.bottom - 0.5);
    const ox = (0.5 - e.left) * z;
    const oy = (0.5 - e.top) * z;
    const { xy, start, closed, centre, diameter } = set;
    ctx.fillStyle = ctx.strokeStyle = colour;
    const typical = set.typical * z; // a contour's size on screen
    if (seen.length > 20000 || typical < 5) {
      const side = clamp(typical * 0.6, 1, 3) * (dpr > 1 ? 1.2 : 1);
      ctx.globalAlpha = seen.length > 20000 ? 0.75 : 0.9;
      for (let k = 0; k < seen.length; k++) {
        const i = seen[k];
        ctx.fillRect(centre[2 * i] * z + ox - side / 2, centre[2 * i + 1] * z + oy - side / 2, side, side);
      }
      ctx.globalAlpha = 1;
      return;
    }
    if (width > 0) {
      const skip = typical < 14 ? 3 : typical < 30 ? 2 : 1; // fewer points while they are closer than a pixel
      ctx.beginPath();
      for (let k = 0; k < seen.length; k++) {
        const i = seen[k];
        const a = start[i];
        const b = start[i + 1];
        ctx.moveTo(xy[2 * a] * z + ox, xy[2 * a + 1] * z + oy);
        for (let p = a + skip; p < b; p += skip) ctx.lineTo(xy[2 * p] * z + ox, xy[2 * p + 1] * z + oy);
        if (closed[i]) ctx.closePath();
        else ctx.lineTo(xy[2 * b - 2] * z + ox, xy[2 * b - 1] * z + oy);
      }
      ctx.lineWidth = width * dpr;
      ctx.lineJoin = "round";
      ctx.stroke();
    }
    if (centres && typical >= 12) {
      const arm = clamp(typical * 0.09, 2.5, 7) * dpr;
      ctx.beginPath();
      for (let k = 0; k < seen.length; k++) {
        const i = seen[k];
        if (!closed[i] || Number.isNaN(diameter[i])) continue;
        const cx = centre[2 * i] * z + ox;
        const cy = centre[2 * i + 1] * z + oy;
        ctx.moveTo(cx - arm, cy);
        ctx.lineTo(cx + arm, cy);
        ctx.moveTo(cx, cy - arm);
        ctx.lineTo(cx, cy + arm);
      }
      ctx.lineWidth = 1.2 * dpr;
      ctx.stroke();
    }
  }

  /* The numbers of the contours of a set whose boxes reach into a rectangle (in the server's coordinates). */
  function visible(set, left, top, right, bottom) {
    const g = set.grid;
    const out = [];
    if (!g) return out;
    const cx0 = clamp(Math.floor((left - g.x) / g.cell), 0, g.cols - 1);
    const cy0 = clamp(Math.floor((top - g.y) / g.cell), 0, g.rows - 1);
    const cx1 = clamp(Math.floor((right - g.x) / g.cell), 0, g.cols - 1);
    const cy1 = clamp(Math.floor((bottom - g.y) / g.cell), 0, g.rows - 1);
    if (right < g.x || bottom < g.y || left > g.x + g.cols * g.cell || top > g.y + g.rows * g.cell) return out;
    const mark = ++g.round;
    for (let cy = cy0; cy <= cy1; cy++) {
      for (let cx = cx0; cx <= cx1; cx++) {
        const cell = cy * g.cols + cx;
        for (let k = g.first[cell]; k < g.first[cell + 1]; k++) {
          const i = g.items[k];
          if (g.seen[i] === mark) continue; // a contour lies in every cell its box touches
          g.seen[i] = mark;
          const b = 4 * i;
          if (set.box[b + 2] >= left && set.box[b] <= right && set.box[b + 3] >= top && set.box[b + 1] <= bottom) out.push(i);
        }
      }
    }
    return out;
  }

  /* The contour under a point of the image (server coordinates), or -1. */
  function contourAt(set, x, y) {
    if (!set || !set.n) return -1;
    let found = -1;
    let least = Infinity;
    for (const i of visible(set, x, y, x, y)) {
      if (!set.closed[i]) continue;
      let inside = false;
      const a = set.start[i];
      const b = set.start[i + 1];
      for (let p = a, q = b - 1; p < b; q = p++) {
        const yp = set.xy[2 * p + 1];
        const yq = set.xy[2 * q + 1];
        if (yp > y !== yq > y && x < ((set.xy[2 * q] - set.xy[2 * p]) * (y - yp)) / (yq - yp) + set.xy[2 * p]) inside = !inside;
      }
      const size = (set.box[4 * i + 2] - set.box[4 * i]) * (set.box[4 * i + 3] - set.box[4 * i + 1]);
      if (inside && size < least) {
        found = i; // of nested ones, the innermost
        least = size;
      }
    }
    return found;
  }

  /* The contour under the pointer, and the one picked. */
  function mark(tab, set) {
    const z = tab.cam.z * dpr;
    const e = edges(tab);
    const one = (i, colour, width) => {
      if (i < 0 || i >= set.n) return;
      ctx.beginPath();
      const a = set.start[i];
      const b = set.start[i + 1];
      for (let p = a; p < b; p++) {
        const X = (set.xy[2 * p] + 0.5 - e.left) * z;
        const Y = (set.xy[2 * p + 1] + 0.5 - e.top) * z;
        if (p === a) ctx.moveTo(X, Y);
        else ctx.lineTo(X, Y);
      }
      if (set.closed[i]) ctx.closePath();
      ctx.strokeStyle = colour;
      ctx.lineWidth = width * dpr;
      ctx.stroke();
    };
    if (tab.hover >= 0 && tab.hover !== tab.picked) one(tab.hover, "#ffffffaa", 1.5);
    if (tab.picked >= 0) one(tab.picked, COLOUR.picked, 2.4);
  }

  function box(tab, [x0, y0, x1, y1]) {
    const z = tab.cam.z * dpr;
    const e = edges(tab);
    return [(x0 - e.left) * z, (y0 - e.top) * z, (x1 - x0) * z, (y1 - y0) * z];
  }

  /* The region the workflow ran on, or will run on. */
  function region(tab) {
    const r = tab.region;
    if (!r) return;
    const [x, y, w, h] = box(tab, r);
    ctx.setLineDash([6 * dpr, 4 * dpr]);
    ctx.lineWidth = 1.2 * dpr;
    ctx.strokeStyle = COLOUR.region;
    ctx.strokeRect(Math.round(x) + 0.5, Math.round(y) + 0.5, Math.round(w), Math.round(h));
    ctx.setLineDash([]);
  }

  function rubber(tab) {
    const [x, y, w, h] = box(tab, normal(gesture.from, gesture.to));
    ctx.fillStyle = "#ffd27a22";
    ctx.fillRect(x, y, w, h);
    ctx.lineWidth = 1.5 * dpr;
    ctx.strokeStyle = COLOUR.region;
    ctx.strokeRect(x, y, w, h);
  }

  function normal(a, b) {
    return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])];
  }

  /* Where one page of a strip ends and the next begins, with the pages' numbers. */
  function pages(tab) {
    const list = tab.info.shown_pages;
    if (list.length < 2) return;
    const z = tab.cam.z * dpr;
    const e = edges(tab);
    const tall = tab.info.page_height;
    ctx.font = `${11 * dpr}px Consolas, monospace`;
    ctx.textBaseline = "top";
    const first = Math.max(0, Math.floor(e.top / tall));
    const last = Math.min(list.length - 1, Math.floor(e.bottom / tall));
    if ((last - first) * 14 * dpr > vh * dpr) return; // closer together than their labels
    for (let k = first; k <= last; k++) {
      const y = Math.round((k * tall - e.top) * z) + 0.5;
      if (k) {
        ctx.setLineDash([2 * dpr, 4 * dpr]);
        ctx.strokeStyle = "#9fb4c2";
        ctx.lineWidth = dpr;
        ctx.beginPath();
        ctx.moveTo(Math.max(0, -e.left * z), y);
        ctx.lineTo(Math.min(canvas.width, (tab.info.width - e.left) * z), y);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      const label = `page ${list[k] + 1}`;
      const x = Math.max(4 * dpr, -e.left * z + 4 * dpr);
      const wide = ctx.measureText(label).width + 8 * dpr;
      ctx.fillStyle = "#0a0d10cc";
      ctx.fillRect(x - 3 * dpr, Math.max(y, 0) + 3 * dpr, wide, 15 * dpr);
      ctx.fillStyle = "#cfe3ea";
      ctx.fillText(label, x + dpr, Math.max(y, 0) + 5 * dpr);
    }
  }

  /* A length of round size, in pixels and (with the pixel size) nanometres. */
  function ruler(tab) {
    const z = tab.cam.z;
    const unit = S.pixelNm > 0 ? S.pixelNm : 1; // nm per pixel, or pixels
    const wanted = (110 / z) * unit;
    const power = 10 ** Math.floor(Math.log10(wanted));
    const length = [1, 2, 5, 10].map((m) => m * power).reduce((a, b) => (Math.abs(b - wanted) < Math.abs(a - wanted) ? b : a));
    const pixels = length / unit;
    const wide = pixels * z * dpr;
    const x = 12 * dpr;
    const y = canvas.height - 14 * dpr;
    let text = S.pixelNm > 0 ? (length >= 1000 ? `${trim(length / 1000, 3)} µm` : `${trim(length, 3)} nm`) : "";
    text += `${text ? "  ·  " : ""}${trim(pixels, 3)} px`;
    ctx.font = `${11 * dpr}px Consolas, monospace`;
    ctx.textBaseline = "bottom";
    const back = Math.max(wide, ctx.measureText(text).width) + 12 * dpr;
    ctx.fillStyle = "#0a0d10b0";
    ctx.fillRect(x - 6 * dpr, y - 22 * dpr, back, 30 * dpr);
    ctx.fillStyle = "#e6edf1";
    ctx.fillRect(x, y, wide, 2 * dpr);
    ctx.fillRect(x, y - 4 * dpr, dpr, 6 * dpr);
    ctx.fillRect(x + wide - dpr, y - 4 * dpr, dpr, 6 * dpr);
    ctx.fillText(text, x, y - 5 * dpr);
  }

  /* The whole image, small, with the part the window shows: to find one's way in a long strip. */
  function overview(tab) {
    const e = edges(tab);
    const all = e.left <= 0 && e.top <= 0 && e.right >= tab.info.width && e.bottom >= tab.info.height;
    if (all) return;
    const source = tab.info.source;
    let scale = Math.min(150 / tab.info.width, Math.min(230, vh - 60) / tab.info.height);
    const wide = Math.max(16, Math.round(tab.info.width * scale));
    const high = Math.max(16, Math.round(tab.info.height * scale));
    map.hidden = false;
    if (map.width !== wide || map.height !== high) {
      map.width = wide;
      map.height = high;
      map.style.width = wide + "px";
      map.style.height = high + "px";
    }
    mapCtx.fillStyle = "#000";
    mapCtx.fillRect(0, 0, wide, high);
    const tile = tileOf(tab, source, source.levels - 1, 0, 0);
    if (tile) {
      mapCtx.imageSmoothingEnabled = true;
      mapCtx.drawImage(painted(tile, source, source.black, source.white), 0, 0, wide, high);
    }
    const sx = wide / tab.info.width;
    const sy = high / tab.info.height;
    if (tab.region) {
      mapCtx.strokeStyle = COLOUR.region;
      mapCtx.lineWidth = 1;
      const [x0, y0, x1, y1] = tab.region;
      mapCtx.strokeRect(x0 * sx + 0.5, y0 * sy + 0.5, Math.max(1, (x1 - x0) * sx - 1), Math.max(1, (y1 - y0) * sy - 1));
    }
    mapCtx.strokeStyle = "#ffffff";
    mapCtx.lineWidth = 1.5;
    const x = clamp(e.left, 0, tab.info.width) * sx;
    const y = clamp(e.top, 0, tab.info.height) * sy;
    const w = clamp(e.right, 0, tab.info.width) * sx - x;
    const hgt = clamp(e.bottom, 0, tab.info.height) * sy - y;
    mapCtx.strokeRect(x + 0.75, y + 0.75, Math.max(2, w - 1.5), Math.max(2, hgt - 1.5));
  }

  // --- The readout under the image ------------------------------------------------------------------------

  function valueAt(tab, described, x, y) {
    // the pixel of a layer under a place of the image; undefined: not loaded yet; null: outside it
    const u = Math.floor((x - described.x) * described.scale);
    const v = Math.floor((y - described.y) * described.scale);
    if (u < 0 || v < 0 || u >= described.width || v >= described.height) return null;
    const tile = tileOf(tab, described, 0, Math.floor(u / TILE), Math.floor(v / TILE));
    if (!tile) return undefined;
    const value = tile.values[(v % TILE) * tile.w + (u % TILE)];
    return tile.whole ? String(value) : trim(value, 5);
  }

  function readout(tab) {
    const line = $("readout");
    if (!pointer) {
      line.textContent = `${tab.info.width} × ${tab.info.height} px  ·  move over the image for its grey levels`;
      return;
    }
    const [sx, sy] = imageAt(tab, pointer);
    if (sx < 0 || sy < 0 || sx >= tab.info.width || sy >= tab.info.height) {
      line.textContent = "outside the image";
      return;
    }
    const x = sx - 0.5; // the server's coordinates: pixel centres at whole numbers
    const y = sy - 0.5;
    const parts = [`x ${x.toFixed(2)}   y ${y.toFixed(2)}`];
    const original = valueAt(tab, tab.info.source, sx, sy);
    parts.push(`${original === undefined ? "…" : original} DN`);
    const view = shown(tab);
    if (view.image && view.changed) {
      const made = valueAt(tab, view.image, sx, sy);
      if (made !== null) parts.push(`processed ${made === undefined ? "…" : made}`);
    }
    if (view.mask) {
      const m = valueAt(tab, view.mask, sx, sy);
      if (m !== null && m !== undefined) parts.push(Number(m) ? "in the mask" : "not in the mask");
    }
    if (tab.info.shown_pages.length > 1) {
      const k = Math.floor(sy / tab.info.page_height);
      parts.push(`page ${tab.info.shown_pages[k] + 1}, row ${Math.floor(sy - k * tab.info.page_height)}`);
    }
    const set = view.contours ? setOf(tab, view.contours) : null;
    if (set && tab.hover >= 0 && tab.hover < set.n) {
      const d = set.diameter[tab.hover];
      parts.push(`contour ${tab.hover + 1}: ⌀ ${length(d)}`);
    }
    line.textContent = parts.join("   ·   ");
  }

  // --- Mouse and keys -------------------------------------------------------------------------------------

  function at(event) {
    const b = canvas.getBoundingClientRect();
    return [event.clientX - b.left, event.clientY - b.top];
  }

  function hover(tab) {
    if (!pointer || gesture) return;
    const view = shown(tab);
    const set = view.contours && S.show.contours ? setOf(tab, view.contours) : null;
    let found = -1;
    if (set && set.typical * tab.cam.z >= 5) {
      const [sx, sy] = imageAt(tab, pointer);
      found = contourAt(set, sx - 0.5, sy - 0.5);
    }
    if (found !== tab.hover) tab.hover = found;
  }

  canvas.addEventListener(
    "wheel",
    (event) => {
      const tab = S.active;
      if (!tab) return;
      event.preventDefault();
      pointer = at(event);
      zoomTo(tab, tab.cam.z * Math.exp(-clamp(event.deltaY, -240, 240) * 0.0035), pointer);
      camMoved(tab);
    },
    { passive: false },
  );

  canvas.addEventListener("pointerdown", (event) => {
    const tab = S.active;
    if (!tab || event.button > 1) return;
    canvas.focus();
    canvas.setPointerCapture(event.pointerId);
    pointer = at(event);
    if (event.shiftKey && event.button === 0) {
      const from = imageAt(tab, pointer);
      gesture = { kind: "region", from, to: from };
    } else {
      gesture = { kind: "move", start: pointer, cam: { ...tab.cam }, moved: false };
      canvas.classList.add("moving");
    }
  });

  canvas.addEventListener("pointermove", (event) => {
    const tab = S.active;
    pointer = at(event);
    if (!tab) return;
    if (gesture && gesture.kind === "move") {
      const dx = pointer[0] - gesture.start[0];
      const dy = pointer[1] - gesture.start[1];
      if (Math.hypot(dx, dy) > 3) gesture.moved = true;
      if (gesture.moved) {
        tab.cam.x = gesture.cam.x - dx / tab.cam.z;
        tab.cam.y = gesture.cam.y - dy / tab.cam.z;
        hold(tab);
      }
    } else if (gesture && gesture.kind === "region") {
      gesture.to = imageAt(tab, pointer);
    } else {
      hover(tab);
    }
    draw();
  });

  function release(event) {
    const tab = S.active;
    const done = gesture;
    gesture = null;
    canvas.classList.remove("moving");
    if (!tab || !done) return;
    if (done.kind === "region") {
      const [x0, y0, x1, y1] = normal(done.from, done.to).map(Math.round);
      if (x1 - x0 >= 8 && y1 - y0 >= 8) regionDrawn(tab, [x0, y0, x1, y1]);
    } else if (done.moved) {
      camMoved(tab);
    } else if (event.type === "pointerup") {
      hover(tab);
      contourPicked(tab, tab.hover);
    }
    draw();
  }
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);
  canvas.addEventListener("pointerleave", () => {
    pointer = null;
    if (S.active) S.active.hover = -1;
    draw();
  });
  canvas.addEventListener("dblclick", (event) => {
    const tab = S.active;
    if (!tab) return;
    zoomTo(tab, tab.cam.z * (event.altKey ? 0.5 : 2), at(event));
    camMoved(tab);
  });

  function navigate(event) {
    const tab = S.active;
    if (!tab) return;
    const b = map.getBoundingClientRect();
    tab.cam.x = ((event.clientX - b.left) / b.width) * tab.info.width;
    tab.cam.y = ((event.clientY - b.top) / b.height) * tab.info.height;
    hold(tab);
    camMoved(tab);
    draw();
  }
  map.addEventListener("pointerdown", (event) => {
    map.setPointerCapture(event.pointerId);
    navigate(event);
  });
  map.addEventListener("pointermove", (event) => {
    if (event.buttons & 1) navigate(event);
  });

  new ResizeObserver(draw).observe($("viewport"));
  window.matchMedia("(resolution: 1dppx)").addEventListener?.("change", draw);

  return {
    draw,
    fit,
    show,
    hold,
    forget,
    shownRegion,
    measure,
    zoomBy(tab, factor) {
      zoomTo(tab, tab.cam.z * factor);
    },
    oneToOne(tab) {
      zoomTo(tab, 1 / (window.devicePixelRatio || 1));
    },
    pan(tab, dx, dy) {
      tab.cam.x += (dx * vw * 0.2) / tab.cam.z;
      tab.cam.y += (dy * vh * 0.2) / tab.cam.z;
      hold(tab);
      draw();
    },
    bare(on) {
      if (bare !== on) {
        bare = on;
        draw();
      }
    },
    picture() {
      return new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
    },
  };
})();

/* The contours of a buffer of the server (ContourSet.payload), with a grid to find those in a rectangle. */
function decodeContours(buffer) {
  const [n, points] = new Uint32Array(buffer, 0, 2);
  let at = 8;
  const start = new Uint32Array(buffer, at, n + 1);
  at += 4 * (n + 1);
  const closed = new Uint8Array(buffer, at, n);
  at += Math.ceil(n / 4) * 4;
  const xy = new Float32Array(buffer, at, 2 * points);
  at += 8 * points;
  const centre = new Float32Array(buffer, at, 2 * n);
  at += 8 * n;
  const diameter = new Float32Array(buffer, at, n);
  at += 4 * n;
  const box = new Float32Array(buffer, at, 4 * n);
  const set = { n, points, start, closed, xy, centre, diameter, box, grid: null, typical: 0 };
  if (!n) return set;
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  const sizes = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    x0 = Math.min(x0, box[4 * i]);
    y0 = Math.min(y0, box[4 * i + 1]);
    x1 = Math.max(x1, box[4 * i + 2]);
    y1 = Math.max(y1, box[4 * i + 3]);
    sizes[i] = Math.max(box[4 * i + 2] - box[4 * i], box[4 * i + 3] - box[4 * i + 1]);
  }
  set.typical = sizes.slice().sort()[n >> 1]; // the median extent: how large a contour is
  const cell = clamp(set.typical * 4, 16, 512);
  const cols = Math.max(1, Math.ceil((x1 - x0 + 1e-3) / cell));
  const rows = Math.max(1, Math.ceil((y1 - y0 + 1e-3) / cell));
  const span = (i) => [
    clamp(Math.floor((box[4 * i] - x0) / cell), 0, cols - 1),
    clamp(Math.floor((box[4 * i + 1] - y0) / cell), 0, rows - 1),
    clamp(Math.floor((box[4 * i + 2] - x0) / cell), 0, cols - 1),
    clamp(Math.floor((box[4 * i + 3] - y0) / cell), 0, rows - 1),
  ];
  const first = new Uint32Array(cols * rows + 1);
  for (let i = 0; i < n; i++) {
    const [a, b, c, d] = span(i);
    for (let cy = b; cy <= d; cy++) for (let cx = a; cx <= c; cx++) first[cy * cols + cx + 1]++;
  }
  for (let k = 0; k < cols * rows; k++) first[k + 1] += first[k];
  const items = new Uint32Array(first[cols * rows]);
  const next = first.slice(0, cols * rows);
  for (let i = 0; i < n; i++) {
    const [a, b, c, d] = span(i);
    for (let cy = b; cy <= d; cy++) for (let cx = a; cx <= c; cx++) items[next[cy * cols + cx]++] = i;
  }
  set.grid = { x: x0, y: y0, cell, cols, rows, first, items, seen: new Uint32Array(n), round: 0 };
  return set;
}
