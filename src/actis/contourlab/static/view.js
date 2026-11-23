"use strict";
/* The image of the tab in front. app.js owns the state (S, the tabs); this file draws it and turns the
   mouse into changes of it.

   Three canvases lie over each other, so that each is drawn only when what it shows changes:
     image   the grey levels (and the mask), as tiles the server cuts at halving resolutions;
     lines   the contours, drawn by the graphics card (gl.js), or by the canvas where there is none;
     marks   what follows the pointer: the contour under it, the one picked, the region being drawn.
   Moving the pointer over the image redraws the marks only; the numbers under the pointer are text.

   Coordinates. The server's are pixel centres: the first pixel is at (0, 0). On screen that pixel covers
   the square from 0 to 1, so a point (x, y) is drawn at x + 0.5, y + 0.5. A tab's camera is the place of
   the image at the middle of the window (in those screen units) and the zoom in CSS pixels per image pixel. */

const TILE = 256;
const INK = { main: "#d95926", reference: "#199e70", before: "#3987e5", picked: "#ffffff", region: "#f0efec" };
const MAIN_INKS = { orange: "#d95926", blue: "#3987e5", aqua: "#199e70", yellow: "#c98500", magenta: "#d55181", white: "#f0efec" };
/* Colour scales for a measure. Two-sided: blue below the middle, red above, near white at it, so that with
   their dark casing the contours show on dark and on bright areas alike. One-sided: one hue, lighter = more. */
const SCALES = {
  diverging: { name: "diverging", stops: [[42, 120, 214], [109, 167, 236], [183, 211, 246], [240, 239, 236], [236, 184, 181], [231, 128, 127], [227, 73, 72]] },
  sequential: { name: "sequential", stops: [[24, 79, 149], [37, 106, 191], [57, 135, 229], [109, 167, 236], [158, 197, 244], [205, 226, 251]] },
};

function rgb(hex) {
  return [1, 3, 5].map((k) => parseInt(hex.slice(k, k + 2), 16) / 255);
}

const View = (() => {
  const image = $("image");
  const ctx = image.getContext("2d");
  const marks = $("marks");
  const mk = marks.getContext("2d");
  const map = $("map");
  const mapCtx = map.getContext("2d");
  let lines = $("lines");
  let renderer = new URLSearchParams(location.search).get("draw") === "canvas" ? null : GLContours.create(lines);
  let flat = null; // the canvas context that draws the contours where the card does not
  if (!renderer) {
    const plain = lines.cloneNode(); // a canvas that was asked for WebGL gives no other context
    lines.replaceWith(plain);
    lines = plain;
    flat = lines.getContext("2d");
  }
  let vw = 1;
  let vh = 1;
  let dpr = 1;
  let pending = 0;
  let dirty = { scene: true, marks: true };
  let frame = 0;
  let pointer = null; // [x, y] in CSS pixels of the view; null: not over it
  let gesture = null; // a drag in progress
  let bare = false; // Space held: the image without its layers
  let rested = ""; // the camera of the last frame, and the timer of the full picture after it:
  let resting = 0; // for drawing the contours without the graphics card

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
          readout();
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
    const data = g.createImageData(tile.w, tile.h);
    const out = new Uint32Array(data.data.buffer); // one pixel each: alpha, blue, green, red
    const values = tile.values;
    if (layer.mask) {
      for (let i = 0; i < values.length; i++) out[i] = (((values[i] * 0.42) | 0) << 24) | (233 << 16) | (133 << 8) | 144;
    } else {
      const gain = 255 / Math.max(white - black, 1e-30);
      for (let i = 0; i < values.length; i++) {
        let grey = (values[i] - black) * gain;
        grey = grey < 0 ? 0 : grey > 255 ? 255 : grey | 0;
        out[i] = 0xff000000 | (grey << 16) | (grey << 8) | grey;
      }
    }
    g.putImageData(data, 0, 0);
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

  /* The place in the image under a point of the view (CSS pixels), in screen units (module comment). */
  function imageAt(tab, point) {
    return [tab.cam.x + (point[0] - vw / 2) / tab.cam.z, tab.cam.y + (point[1] - vh / 2) / tab.cam.z];
  }

  function fitZoom(tab) {
    return 0.94 * Math.min(vw / tab.info.width, vh / tab.info.height);
  }

  /* Show the whole image; of a long strip, its full width at the place of the region the workflow runs on. */
  function fit(tab) {
    measure();
    const whole = fitZoom(tab);
    const across = (0.94 * vw) / tab.info.width;
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
    const box = $("viewport");
    vw = Math.max(1, box.clientWidth);
    vh = Math.max(1, box.clientHeight);
    for (const canvas of [image, lines, marks]) {
      if (canvas.width !== Math.round(vw * dpr) || canvas.height !== Math.round(vh * dpr)) {
        canvas.width = Math.round(vw * dpr);
        canvas.height = Math.round(vh * dpr);
        dirty.scene = dirty.marks = true;
      }
    }
  }

  /* Draw again: everything, or (marksOnly) what follows the pointer. */
  function draw(marksOnly = false) {
    if (marksOnly) dirty.marks = true;
    else dirty.scene = dirty.marks = true;
    if (!pending) pending = requestAnimationFrame(paint);
  }

  function paint() {
    pending = 0;
    frame++;
    measure();
    const tab = S.active;
    $("empty").hidden = Boolean(tab);
    document.body.classList.toggle("no-image", !tab);
    if (tab && !tab.cam) fit(tab);
    const scene = dirty.scene;
    dirty = { scene: false, marks: false };
    if (scene) {
      paintImage(tab);
      paintContours(tab);
      paintMap(tab);
      paintScale(tab);
    }
    paintMarks(tab);
  }

  /* The grey levels shown as black and white: the tab's own if set by hand (for the image in front), else
     the layer's. */
  function levelsOf(tab, described, front) {
    if (front && !tab.levels.auto) return { black: tab.levels.black, white: tab.levels.white };
    return { black: described.black, white: described.white };
  }

  function paintImage(tab) {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#0d0d0d";
    ctx.fillRect(0, 0, image.width, image.height);
    if (!tab) return;
    const view = shown(tab);
    const source = tab.info.source;
    const top = view.image; // the processed image, if that is what is shown
    if (!top || top.x || top.y || top.width / top.scale < tab.info.width || top.height / top.scale < tab.info.height) {
      // the image as loaded: all there is to show, or what lies around the region that was processed
      layer(tab, source, levelsOf(tab, source, !top), top ? 0.55 : 1);
    }
    // an image no module changed is shown with the grey levels of the whole image, as what lies around it
    if (top) layer(tab, top, levelsOf(tab, view.changed ? top : source, true), 1);
    if (!bare && view.mask && S.show.mask) layer(tab, view.mask, null, 1);
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

  /* The sets of contours the window shows, the one in front last: [set, how it looks]. */
  function setsShown(tab) {
    const out = [];
    if (!tab || bare) return out;
    const view = shown(tab);
    const width = S.look.width * dpr;
    const one = (set, hex, share) => ({ width: Math.max(1, width * share), colour: rgb(hex), values: null, low: 0, high: 1, scale: SCALES.diverging });
    if (S.show.reference && tab.pinned) out.push([tab.pinned.set, one(tab.pinned.set, INK.reference, 0.8)]);
    if (S.show.before && view.before) {
      const set = setOf(tab, view.before);
      if (set) out.push([set, one(set, INK.before, 0.8)]);
    }
    if (S.show.contours && view.contours) {
      const set = setOf(tab, view.contours);
      if (set) {
        const main = one(set, MAIN_INKS[S.look.colour] || INK.main, 1);
        const by = colouring(set); // app.js: the measure the contours are coloured by, if any
        if (by) Object.assign(main, by);
        out.push([set, main]);
      }
    }
    return out;
  }

  function paintContours(tab) {
    const e = tab ? edges(tab) : null;
    if (renderer) {
      renderer.begin(lines.width, lines.height);
      if (!tab) return;
      const view = { left: e.left, top: e.top, zoom: tab.cam.z * dpr, width: lines.width, height: lines.height };
      for (const [set, look] of setsShown(tab)) renderer.draw(set, view, look);
      return;
    }
    flat.setTransform(1, 0, 0, 1, 0, 0);
    flat.clearRect(0, 0, lines.width, lines.height);
    if (!tab) return;
    // while the camera moves, a quick picture; the full one once it has rested
    const place = `${tab.id} ${tab.cam.x} ${tab.cam.y} ${tab.cam.z}`;
    const quick = place !== rested;
    rested = place;
    clearTimeout(resting);
    if (quick) resting = setTimeout(() => draw(), 140);
    for (const [set, look] of setsShown(tab)) plainContours(tab, set, look, quick);
  }

  /* Without the graphics card: outlines while few are in view, else a disc each. Never more work than a
     frame can take: that many outlines as one path is what made the first lab slow. */
  function plainContours(tab, set, look, quick) {
    if (!set.n) return;
    const z = tab.cam.z * dpr;
    const e = edges(tab);
    const seen = visible(set, e.left - 0.5, e.top - 0.5, e.right - 0.5, e.bottom - 0.5);
    const ox = (0.5 - e.left) * z;
    const oy = (0.5 - e.top) * z;
    const { xy, start, closed, centre, diameter } = set;
    const typical = set.typical * z;
    const groups = 8;
    const group = new Uint8Array(seen.length);
    const inks = [];
    if (look.values) {
      for (let k = 0; k < seen.length; k++) {
        const value = look.values[seen[k]];
        group[k] = Number.isNaN(value) ? groups : clamp(Math.floor(((value - look.low) / (look.high - look.low)) * groups), 0, groups - 1);
      }
      for (let g = 0; g < groups; g++) inks.push(`rgb(${GLContours.colourAt(look.scale.stops, (g + 0.5) / groups).slice(0, 3).join(",")})`);
    }
    inks[groups] = `rgb(${look.colour.map((c) => Math.round(c * 255)).join(",")})`;
    if (!look.values) group.fill(groups);
    // an outline for a contour large enough on screen, while few enough are; else a disc of its size
    const { box } = set;
    const drawn = new Uint8Array(seen.length);
    let budget = quick ? 300 : 2500;
    for (let k = 0; k < seen.length && budget > 0; k++) {
      const i = seen[k];
      const size = Math.max(box[4 * i + 2] - box[4 * i], box[4 * i + 3] - box[4 * i + 1]) * z;
      if (size >= GLContours.OUTLINE_FROM) {
        drawn[k] = 1;
        budget--;
      }
    }
    const skip = typical < 30 ? 2 : 1; // fewer points while they are closer than a pixel
    for (let g = 0; g <= groups; g++) {
      for (const outlines of [0, 1]) {
        flat.beginPath();
        let any = false;
        for (let k = 0; k < seen.length; k++) {
          if (group[k] !== g || drawn[k] !== outlines) continue;
          any = true;
          const i = seen[k];
          if (outlines) {
            const a = start[i];
            const b = start[i + 1];
            flat.moveTo(xy[2 * a] * z + ox, xy[2 * a + 1] * z + oy);
            for (let p = a + 1; p < b; p += skip) flat.lineTo(xy[2 * p] * z + ox, xy[2 * p + 1] * z + oy);
            if (closed[i]) flat.closePath();
          } else {
            const radius = Math.max(1.2, ((Number.isNaN(diameter[i]) ? typical / z : diameter[i]) * z) / 2);
            const cx = centre[2 * i] * z + ox;
            const cy = centre[2 * i + 1] * z + oy;
            if (radius < 2.5) flat.rect(cx - radius, cy - radius, 2 * radius, 2 * radius);
            else {
              flat.moveTo(cx + radius, cy);
              flat.arc(cx, cy, radius, 0, 2 * Math.PI);
            }
          }
        }
        if (!any) continue;
        if (outlines) {
          flat.lineJoin = "round";
          flat.strokeStyle = "rgba(10,10,10,0.9)";
          flat.lineWidth = look.width + 2 * Math.max(1, 0.5 * look.width);
          flat.stroke();
          flat.strokeStyle = inks[g];
          flat.lineWidth = look.width;
          flat.stroke();
        } else {
          flat.fillStyle = inks[g];
          flat.fill();
        }
      }
    }
  }

  /* The numbers of the contours of a set whose boxes reach into a rectangle (in the server's coordinates). */
  function visible(set, left, top, right, bottom) {
    const g = set.grid;
    const out = [];
    if (!g) return out;
    if (right < g.x || bottom < g.y || left > g.x + g.cols * g.cell || top > g.y + g.rows * g.cell) return out;
    const cx0 = clamp(Math.floor((left - g.x) / g.cell), 0, g.cols - 1);
    const cy0 = clamp(Math.floor((top - g.y) / g.cell), 0, g.rows - 1);
    const cx1 = clamp(Math.floor((right - g.x) / g.cell), 0, g.cols - 1);
    const cy1 = clamp(Math.floor((bottom - g.y) / g.cell), 0, g.rows - 1);
    const round = ++g.round;
    for (let cy = cy0; cy <= cy1; cy++) {
      for (let cx = cx0; cx <= cx1; cx++) {
        const cell = cy * g.cols + cx;
        for (let k = g.first[cell]; k < g.first[cell + 1]; k++) {
          const i = g.items[k];
          if (g.seen[i] === round) continue; // a contour lies in every cell its box touches
          g.seen[i] = round;
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

  // --- What follows the pointer ---------------------------------------------------------------------------

  function paintMarks(tab) {
    mk.setTransform(1, 0, 0, 1, 0, 0);
    mk.clearRect(0, 0, marks.width, marks.height);
    if (!tab) return;
    const z = tab.cam.z * dpr;
    const e = edges(tab);
    if (!bare) {
      pages(tab, z, e);
      if (tab.region) {
        const [x, y, w, h] = boxOn(tab.region, z, e);
        mk.setLineDash([7 * dpr, 5 * dpr]);
        mk.lineWidth = 1.5 * dpr;
        mk.strokeStyle = "#0d0d0dcc";
        mk.strokeRect(Math.round(x) + 0.5, Math.round(y) + 0.5, Math.round(w), Math.round(h));
        mk.lineDashOffset = 6 * dpr;
        mk.strokeStyle = INK.region;
        mk.strokeRect(Math.round(x) + 0.5, Math.round(y) + 0.5, Math.round(w), Math.round(h));
        mk.setLineDash([]);
        mk.lineDashOffset = 0;
      }
      const view = shown(tab);
      const set = view.contours ? tab.sets.get(view.contours)?.set : null;
      if (set) {
        if (S.show.centres && S.show.contours) centres(tab, set, z, e);
        if (tab.hover >= 0 && tab.hover !== tab.picked) outline(set, tab.hover, z, e, 2.5, "#ffffffcc");
        if (tab.picked >= 0) outline(set, tab.picked, z, e, 3.5, INK.picked);
      }
    }
    if (gesture && gesture.kind === "region") {
      const [x, y, w, h] = boxOn(normal(gesture.from, gesture.to), z, e);
      mk.fillStyle = "#f0efec22";
      mk.fillRect(x, y, w, h);
      mk.lineWidth = 1.5 * dpr;
      mk.strokeStyle = INK.region;
      mk.strokeRect(x, y, w, h);
    }
  }

  function outline(set, i, z, e, width, colour) {
    if (i < 0 || i >= set.n) return;
    mk.beginPath();
    const a = set.start[i];
    const b = set.start[i + 1];
    for (let p = a; p < b; p++) {
      const X = (set.xy[2 * p] + 0.5 - e.left) * z;
      const Y = (set.xy[2 * p + 1] + 0.5 - e.top) * z;
      if (p === a) mk.moveTo(X, Y);
      else mk.lineTo(X, Y);
    }
    if (set.closed[i]) mk.closePath();
    mk.lineJoin = "round";
    mk.strokeStyle = "#0d0d0d";
    mk.lineWidth = (width + 2.5) * dpr;
    mk.stroke();
    mk.strokeStyle = colour;
    mk.lineWidth = width * dpr;
    mk.stroke();
  }

  /* A cross at the centre of every closed contour, while they are large and few enough to tell apart. */
  function centres(tab, set, z, e) {
    if (set.typical * z < 14) return;
    const seen = visible(set, e.left - 0.5, e.top - 0.5, e.right - 0.5, e.bottom - 0.5);
    if (seen.length > 6000) return;
    const arm = clamp(set.typical * z * 0.09, 3, 8);
    mk.beginPath();
    for (const i of seen) {
      if (!set.closed[i] || Number.isNaN(set.diameter[i])) continue;
      const cx = (set.centre[2 * i] + 0.5 - e.left) * z;
      const cy = (set.centre[2 * i + 1] + 0.5 - e.top) * z;
      mk.moveTo(cx - arm, cy);
      mk.lineTo(cx + arm, cy);
      mk.moveTo(cx, cy - arm);
      mk.lineTo(cx, cy + arm);
    }
    mk.strokeStyle = "#0d0d0d";
    mk.lineWidth = 3.5 * dpr;
    mk.stroke();
    mk.strokeStyle = "#f0efec";
    mk.lineWidth = 1.5 * dpr;
    mk.stroke();
  }

  function boxOn([x0, y0, x1, y1], z, e) {
    return [(x0 - e.left) * z, (y0 - e.top) * z, (x1 - x0) * z, (y1 - y0) * z];
  }

  function normal(a, b) {
    return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])];
  }

  /* Where one page of a strip ends and the next begins, with the pages' numbers. */
  function pages(tab, z, e) {
    const list = tab.info.shown_pages;
    if (list.length < 2) return;
    const tall = tab.info.page_height;
    mk.font = `600 ${13 * dpr}px "Segoe UI", system-ui, sans-serif`;
    mk.textBaseline = "top";
    const first = Math.max(0, Math.floor(e.top / tall));
    const last = Math.min(list.length - 1, Math.floor(e.bottom / tall));
    if ((last - first) * 20 > vh) return; // closer together than their labels
    for (let k = first; k <= last; k++) {
      const y = Math.round((k * tall - e.top) * z) + 0.5;
      if (k) {
        mk.setLineDash([3 * dpr, 5 * dpr]);
        mk.strokeStyle = "#c3c2b7";
        mk.lineWidth = dpr;
        mk.beginPath();
        mk.moveTo(Math.max(0, -e.left * z), y);
        mk.lineTo(Math.min(marks.width, (tab.info.width - e.left) * z), y);
        mk.stroke();
        mk.setLineDash([]);
      }
      const label = `Page ${list[k] + 1}`;
      const x = Math.max(8 * dpr, -e.left * z + 8 * dpr);
      const wide = mk.measureText(label).width + 12 * dpr;
      mk.fillStyle = "#0d0d0dd9";
      mk.fillRect(x - 4 * dpr, Math.max(y, 0) + 5 * dpr, wide, 21 * dpr);
      mk.fillStyle = "#f0efec";
      mk.fillText(label, x + 2 * dpr, Math.max(y, 0) + 9 * dpr);
    }
  }

  // --- The scale, the zoom and the small map --------------------------------------------------------------

  function paintScale(tab) {
    $("scale").hidden = !tab;
    if (!tab) return;
    const z = tab.cam.z;
    const unit = S.pixelNm > 0 ? S.pixelNm : 1; // nm per pixel, or pixels
    const wanted = (120 / z) * unit;
    const power = 10 ** Math.floor(Math.log10(wanted));
    const length = [1, 2, 5, 10].map((m) => m * power).reduce((a, b) => (Math.abs(b - wanted) < Math.abs(a - wanted) ? b : a));
    const pixels = length / unit;
    $("scale-bar").style.width = `${pixels * z}px`;
    const nm = S.pixelNm > 0 ? (length >= 1000 ? `${trim(length / 1000, 3)} µm` : `${trim(length, 3)} nm`) : "";
    $("scale-text").textContent = `${nm}${nm ? " · " : ""}${trim(pixels, 3)} px`;
    const percent = z * dpr * 100;
    $("zoom").textContent = percent >= 100 ? `${trim(percent, 3)} %` : `${trim(percent, 2)} %`;
  }

  /* The whole image, small, with the part the window shows: to find one's way in a long strip. */
  function paintMap(tab) {
    const e = tab ? edges(tab) : null;
    const all = !tab || (e.left <= 0 && e.top <= 0 && e.right >= tab.info.width && e.bottom >= tab.info.height);
    map.hidden = all;
    if (all) return;
    const source = tab.info.source;
    const scale = Math.min(190 / tab.info.width, Math.min(260, vh - 150) / tab.info.height);
    const wide = Math.max(20, Math.round(tab.info.width * scale));
    const high = Math.max(20, Math.round(tab.info.height * scale));
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
      // a picture of its own: the tile's is repainted whenever the display levels change
      if (!tile.small) {
        tile.small = document.createElement("canvas");
        tile.small.width = tile.w;
        tile.small.height = tile.h;
        const g = tile.small.getContext("2d");
        const data = g.createImageData(tile.w, tile.h);
        const gain = 255 / Math.max(source.white - source.black, 1e-30);
        for (let i = 0; i < tile.values.length; i++) {
          const grey = clamp((tile.values[i] - source.black) * gain, 0, 255);
          data.data[4 * i] = data.data[4 * i + 1] = data.data[4 * i + 2] = grey;
          data.data[4 * i + 3] = 255;
        }
        g.putImageData(data, 0, 0);
      }
      mapCtx.drawImage(tile.small, 0, 0, wide, high);
    }
    const sx = wide / tab.info.width;
    const sy = high / tab.info.height;
    const x = clamp(e.left, 0, tab.info.width) * sx;
    const y = clamp(e.top, 0, tab.info.height) * sy;
    const w = clamp(e.right, 0, tab.info.width) * sx - x;
    const hgt = clamp(e.bottom, 0, tab.info.height) * sy - y;
    mapCtx.fillStyle = "#0d0d0d88";
    mapCtx.fillRect(0, 0, wide, y);
    mapCtx.fillRect(0, y + hgt, wide, high - y - hgt);
    mapCtx.fillRect(0, y, x, hgt);
    mapCtx.fillRect(x + w, y, wide - x - w, hgt);
    mapCtx.strokeStyle = "#ffffff";
    mapCtx.lineWidth = 2;
    mapCtx.strokeRect(x + 1, y + 1, Math.max(3, w - 2), Math.max(3, hgt - 2));
  }

  // --- The numbers under the pointer ----------------------------------------------------------------------

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

  function readout() {
    const tab = S.active;
    const box = $("readout");
    box.hidden = !tab;
    if (!tab || !tab.cam) return;
    if (!pointer) return put(box, h("span", { class: "dim", text: "Point at the image for its grey levels" }));
    const [sx, sy] = imageAt(tab, pointer);
    if (sx < 0 || sy < 0 || sx >= tab.info.width || sy >= tab.info.height) return put(box, h("span", { class: "dim", text: "Outside the image" }));
    const part = (name, value) => h("span", { class: "pair" }, h("span", { class: "dim", text: name }), ` ${value}`);
    const parts = [part("x", (sx - 0.5).toFixed(1)), part("y", (sy - 0.5).toFixed(1))];
    const original = valueAt(tab, tab.info.source, sx, sy);
    parts.push(part("grey level", original === undefined ? "…" : original));
    const view = shown(tab);
    if (view.image && view.changed) {
      const made = valueAt(tab, view.image, sx, sy);
      if (made !== null) parts.push(part("processed", made === undefined ? "…" : made));
    }
    if (tab.info.shown_pages.length > 1) {
      const k = clamp(Math.floor(sy / tab.info.page_height), 0, tab.info.shown_pages.length - 1);
      parts.push(part("page", `${tab.info.shown_pages[k] + 1}, row ${Math.floor(sy - k * tab.info.page_height)}`));
    }
    const set = view.contours ? tab.sets.get(view.contours)?.set : null;
    if (set && tab.hover >= 0 && tab.hover < set.n) parts.push(part("contour", `⌀ ${lengthText(set.diameter[tab.hover])}`));
    put(box, ...parts);
  }

  // --- Mouse and keys -------------------------------------------------------------------------------------

  const stage = $("viewport");

  function at(event) {
    const b = stage.getBoundingClientRect();
    return [event.clientX - b.left, event.clientY - b.top];
  }

  /* The contour under the pointer; true if it is another than before. */
  function hover(tab) {
    if (!pointer || gesture) return false;
    const view = shown(tab);
    const set = view.contours && S.show.contours ? tab.sets.get(view.contours)?.set : null;
    let found = -1;
    if (set && set.typical * tab.cam.z * dpr >= 5) {
      const [sx, sy] = imageAt(tab, pointer);
      found = contourAt(set, sx - 0.5, sy - 0.5);
    }
    const changed = found !== tab.hover;
    tab.hover = found;
    return changed;
  }

  marks.addEventListener(
    "wheel",
    (event) => {
      const tab = S.active;
      if (!tab) return;
      event.preventDefault();
      pointer = at(event);
      const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 200 : 1);
      zoomTo(tab, tab.cam.z * Math.exp(-clamp(delta, -240, 240) * 0.0035), pointer);
    },
    { passive: false },
  );

  marks.addEventListener("pointerdown", (event) => {
    const tab = S.active;
    if (!tab || event.button > 1) return;
    marks.focus({ preventScroll: true });
    marks.setPointerCapture(event.pointerId);
    pointer = at(event);
    if (event.shiftKey && event.button === 0) {
      const from = imageAt(tab, pointer);
      gesture = { kind: "region", from, to: from };
    } else {
      gesture = { kind: "move", start: pointer, cam: { ...tab.cam }, moved: false };
    }
  });

  let told = 0; // the animation frame that will write the numbers under the pointer

  marks.addEventListener("pointermove", (event) => {
    const tab = S.active;
    pointer = at(event);
    if (!tab || !tab.cam) return;
    if (gesture && gesture.kind === "move") {
      const dx = pointer[0] - gesture.start[0];
      const dy = pointer[1] - gesture.start[1];
      if (Math.hypot(dx, dy) > 3 && !gesture.moved) {
        gesture.moved = true;
        marks.classList.add("moving");
      }
      if (gesture.moved) {
        tab.cam.x = gesture.cam.x - dx / tab.cam.z;
        tab.cam.y = gesture.cam.y - dy / tab.cam.z;
        hold(tab);
        draw();
      }
    } else if (gesture && gesture.kind === "region") {
      gesture.to = imageAt(tab, pointer);
      draw(true);
    } else if (hover(tab)) {
      draw(true); // only what follows the pointer: the image and the contours stay as they are
    }
    if (!told) {
      told = requestAnimationFrame(() => {
        told = 0;
        readout();
      });
    }
  });

  function release(event) {
    const tab = S.active;
    const done = gesture;
    gesture = null;
    marks.classList.remove("moving");
    if (!tab || !done) return;
    if (done.kind === "region") {
      const [x0, y0, x1, y1] = normal(done.from, done.to).map(Math.round);
      if (x1 - x0 >= 8 && y1 - y0 >= 8) regionDrawn(tab, [x0, y0, x1, y1]);
    } else if (!done.moved && event.type === "pointerup") {
      hover(tab);
      contourPicked(tab, tab.hover);
    }
    draw(true);
  }
  marks.addEventListener("pointerup", release);
  marks.addEventListener("pointercancel", release);
  marks.addEventListener("pointerleave", () => {
    pointer = null;
    if (S.active && S.active.hover >= 0) {
      S.active.hover = -1;
      draw(true);
    }
    readout();
  });
  marks.addEventListener("dblclick", (event) => {
    const tab = S.active;
    if (tab) zoomTo(tab, tab.cam.z * (event.altKey ? 0.5 : 2), at(event));
  });

  function navigate(event) {
    const tab = S.active;
    if (!tab) return;
    const b = map.getBoundingClientRect();
    tab.cam.x = ((event.clientX - b.left) / b.width) * tab.info.width;
    tab.cam.y = ((event.clientY - b.top) / b.height) * tab.info.height;
    hold(tab);
    draw();
  }
  map.addEventListener("pointerdown", (event) => {
    map.setPointerCapture(event.pointerId);
    navigate(event);
  });
  map.addEventListener("pointermove", (event) => {
    if (event.buttons & 1) navigate(event);
  });

  new ResizeObserver(() => draw()).observe(stage);
  window.matchMedia("(resolution: 1dppx)").addEventListener?.("change", () => draw());

  return {
    draw,
    fit,
    show,
    hold,
    forget,
    shownRegion,
    measure,
    readout,
    usesCard: Boolean(renderer),
    dropSet(set) {
      if (renderer && set) renderer.release(set);
    },
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
    /* The view as a picture: the three canvases over each other. */
    picture() {
      dirty.scene = dirty.marks = true;
      if (pending) cancelAnimationFrame(pending);
      paint(); // now: what the card drew is gone once the browser has shown it
      const whole = document.createElement("canvas");
      whole.width = image.width;
      whole.height = image.height;
      const g = whole.getContext("2d");
      for (const canvas of [image, lines, marks]) g.drawImage(canvas, 0, 0);
      return new Promise((resolve) => whole.toBlob(resolve, "image/png"));
    },
  };
})();

/* The contours of a buffer of the server (ContourSet.payload), with a grid to find those in a rectangle.
   keys: the measures that follow the contours in the buffer, in their order. */
function decodeContours(buffer, keys) {
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
  at += 16 * n;
  const values = {};
  for (const key of keys) {
    values[key] = new Float32Array(buffer, at, n);
    at += 4 * n;
  }
  const set = { n, points, start, closed, xy, centre, diameter, box, values, grid: null, typical: 0, bounds: [0, 0, 0, 0], summaries: {}, gpu: null };
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
  set.bounds = [x0, y0, x1, y1];
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
