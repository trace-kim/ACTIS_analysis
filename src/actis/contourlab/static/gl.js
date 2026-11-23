"use strict";
/* Contours drawn by the graphics card.

   A dense page has sixty thousand contours of fifty points each. Stroking them as one canvas path takes up
   to 1.7 s a frame, because the path is built and rasterised again for every frame. Here every set of
   contours is uploaded once, in chunks of the image, and a frame only tells the card where the camera is:
   moving or zooming costs the same whether a hundred contours are in view or a million.

   What is drawn depends on how large a contour is on screen: its outline (a coloured line with a dark
   casing, so that it shows on dark and on bright areas alike), or, once it is only a few screen pixels
   wide, a disc at its centre of the contour's own size, which far out merges into a map of the image.
   The colour is one colour, or each contour's value of a measure looked up in a colour scale. */

const GLContours = (() => {
  const CHUNK = 512; // image pixels per side of a chunk
  const DATA_WIDTH = 2048; // contours per row of the texture that holds their values
  const GONE = 1e30; // marks the break between two contours, and a value that is not there
  const OUTLINE_FROM = 7; // device pixels: contours at least this large are drawn as outlines

  const LINE_VS = `#version 300 es
  layout(location = 0) in vec3 a_from;   // x, y in the chunk; the contour's number
  layout(location = 1) in vec3 a_to;
  uniform vec2 u_origin;                 // the chunk's origin, device pixels
  uniform float u_zoom;                  // device pixels per image pixel
  uniform vec2 u_view;                   // the canvas, device pixels
  uniform float u_half;                  // half the width drawn
  uniform float u_outlineFrom;
  uniform vec2 u_range;                  // low value, 1 / (high - low)
  uniform highp sampler2D u_data;        // per contour: value, size in image pixels
  out float v_across;
  out float v_value;
  void main() {
    int corner = gl_VertexID;
    float along = float(corner >> 1);
    float side = float(corner & 1) * 2.0 - 1.0;
    int number = int(a_from.z);
    vec4 data = texelFetch(u_data, ivec2(number % ${DATA_WIDTH}, number / ${DATA_WIDTH}), 0);
    if (a_from.x > 1e20 || a_to.x > 1e20 || data.g * u_zoom < u_outlineFrom) {
      gl_Position = vec4(2.0, 2.0, 2.0, 1.0);    // outside: nothing is drawn
      return;
    }
    vec2 from = u_origin + a_from.xy * u_zoom;
    vec2 to = u_origin + a_to.xy * u_zoom;
    vec2 way = to - from;
    float long_ = length(way);
    way = long_ > 1e-6 ? way / long_ : vec2(1.0, 0.0);
    vec2 across = vec2(-way.y, way.x);
    float reach = u_half + 1.0;                   // a pixel more, for the soft edge
    vec2 at = mix(from, to, along) + way * (along * 2.0 - 1.0) * u_half * 0.6 + across * side * reach;
    v_across = side * reach;
    v_value = data.r > 1e20 ? -1.0 : clamp((data.r - u_range.x) * u_range.y, 0.0, 1.0);
    gl_Position = vec4(at.x / u_view.x * 2.0 - 1.0, 1.0 - at.y / u_view.y * 2.0, 0.0, 1.0);
  }`;

  const LINE_FS = `#version 300 es
  precision highp float;
  in float v_across;
  in float v_value;
  uniform float u_half;
  uniform vec4 u_colour;
  uniform float u_scaled;                // 1: the colour scale; 0: the one colour
  uniform sampler2D u_scale;
  out vec4 colour;
  void main() {
    float cover = clamp(u_half + 0.5 - abs(v_across), 0.0, 1.0);
    vec4 ink = (u_scaled > 0.5 && v_value >= 0.0) ? texture(u_scale, vec2(v_value, 0.5)) : u_colour;
    colour = vec4(ink.rgb, 1.0) * ink.a * cover;
  }`;

  const DOT_VS = `#version 300 es
  layout(location = 0) in vec4 a_dot;    // centre x, y in the chunk; radius; the contour's number
  uniform vec2 u_origin;
  uniform float u_zoom;
  uniform vec2 u_view;
  uniform float u_outlineFrom;
  uniform float u_least;                 // smallest radius drawn, device pixels
  uniform float u_extra;                 // casing around the disc, device pixels
  uniform vec2 u_range;
  uniform highp sampler2D u_data;
  out vec2 v_at;
  out float v_radius;
  out float v_value;
  void main() {
    int number = int(a_dot.w);
    vec4 data = texelFetch(u_data, ivec2(number % ${DATA_WIDTH}, number / ${DATA_WIDTH}), 0);
    if (data.g * u_zoom >= u_outlineFrom) {
      gl_Position = vec4(2.0, 2.0, 2.0, 1.0);    // large enough for its outline
      return;
    }
    vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1)) * 2.0 - 1.0;
    float radius = max(a_dot.z * u_zoom, u_least);
    float reach = radius + u_extra + 1.0;
    vec2 at = u_origin + a_dot.xy * u_zoom + corner * reach;
    v_at = corner * reach;
    v_radius = radius;
    v_value = data.r > 1e20 ? -1.0 : clamp((data.r - u_range.x) * u_range.y, 0.0, 1.0);
    gl_Position = vec4(at.x / u_view.x * 2.0 - 1.0, 1.0 - at.y / u_view.y * 2.0, 0.0, 1.0);
  }`;

  const DOT_FS = `#version 300 es
  precision highp float;
  in vec2 v_at;
  in float v_radius;
  in float v_value;
  uniform vec4 u_colour;
  uniform float u_scaled;
  uniform float u_extra;
  uniform float u_ring;                  // 0: a disc; else the width of a ring, device pixels
  uniform sampler2D u_scale;
  out vec4 colour;
  void main() {
    float away = length(v_at);
    float outer = clamp(v_radius + u_extra + 0.5 - away, 0.0, 1.0);
    float inner = clamp(v_radius + 0.5 - away, 0.0, 1.0);
    if (u_ring > 0.0) {
      // a ring: casing, colour, casing, and the image through its middle
      float hole = clamp(v_radius - u_ring - u_extra + 0.5 - away, 0.0, 1.0);
      float core = inner * (1.0 - clamp(v_radius - u_ring + 0.5 - away, 0.0, 1.0));
      outer *= 1.0 - hole;
      inner = core;
    }
    vec4 ink = (u_scaled > 0.5 && v_value >= 0.0) ? texture(u_scale, vec2(v_value, 0.5)) : u_colour;
    vec3 shade = mix(vec3(0.04), ink.rgb, inner);
    float cover = max(inner * ink.a, (outer - inner) * 0.85);
    colour = vec4(shade, 1.0) * cover;
  }`;

  function program(gl, vertex, fragment) {
    const made = gl.createProgram();
    for (const [kind, source] of [[gl.VERTEX_SHADER, vertex], [gl.FRAGMENT_SHADER, fragment]]) {
      const shader = gl.createShader(kind);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      gl.attachShader(made, shader);
    }
    gl.linkProgram(made);
    if (!gl.getProgramParameter(made, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(made));
    const uniforms = {};
    const count = gl.getProgramParameter(made, gl.ACTIVE_UNIFORMS);
    for (let k = 0; k < count; k++) {
      const name = gl.getActiveUniform(made, k).name;
      uniforms[name] = gl.getUniformLocation(made, name);
    }
    return { made, uniforms };
  }

  /* A renderer on a canvas, or null where the browser has no WebGL 2. */
  function create(canvas) {
    const gl = canvas.getContext("webgl2", { alpha: true, premultipliedAlpha: true, antialias: false, depth: false, stencil: false });
    if (!gl) return null;
    let lines;
    let dots;
    try {
      lines = program(gl, LINE_VS, LINE_FS);
      dots = program(gl, DOT_VS, DOT_FS);
    } catch (error) {
      console.warn("The contours are drawn without the graphics card:", error.message);
      return null;
    }
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    const scales = new Map(); // name -> texture of a colour scale
    const uploaded = new Set(); // the sets that hold buffers here
    let lost = false;
    canvas.addEventListener("webglcontextlost", (event) => {
      event.preventDefault();
      lost = true;
    });

    function scaleTexture(name, stops) {
      let texture = scales.get(name);
      if (texture) return texture;
      const pixels = new Uint8Array(256 * 4);
      for (let k = 0; k < 256; k++) pixels.set(colourAt(stops, k / 255), 4 * k);
      texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
      for (const [key, value] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) {
        gl.texParameteri(gl.TEXTURE_2D, key, value);
      }
      scales.set(name, texture);
      return texture;
    }

    /* The buffers of a set: its points and its centres, chunk by chunk, and the texture of its values. */
    function upload(set) {
      const { n, start, closed, xy, centre, diameter, box } = set;
      const across = Math.max(1, Math.ceil((set.bounds[2] + 1) / CHUNK));
      const down = Math.max(1, Math.ceil((set.bounds[3] + 1) / CHUNK));
      const where = new Uint32Array(n);
      const points = new Uint32Array(across * down);
      const members = new Uint32Array(across * down);
      for (let i = 0; i < n; i++) {
        const cx = clamp(Math.floor(centre[2 * i] / CHUNK), 0, across - 1);
        const cy = clamp(Math.floor(centre[2 * i + 1] / CHUNK), 0, down - 1);
        where[i] = cy * across + cx;
        points[where[i]] += start[i + 1] - start[i] + (closed[i] ? 2 : 1); // its first point again, and the break
        members[where[i]]++;
      }
      const chunks = [];
      for (let c = 0; c < across * down; c++) {
        if (!members[c]) continue;
        chunks[c] = {
          x: (c % across) * CHUNK,
          y: Math.floor(c / across) * CHUNK,
          line: new Float32Array(3 * points[c]),
          dot: new Float32Array(4 * members[c]),
          used: 0,
          dots: 0,
          box: [Infinity, Infinity, -Infinity, -Infinity],
          least: Infinity,
          most: 0,
        };
      }
      const data = new Float32Array(DATA_WIDTH * Math.max(1, Math.ceil(n / DATA_WIDTH)) * 2);
      for (let i = 0; i < n; i++) {
        const chunk = chunks[where[i]];
        const line = chunk.line;
        let at = chunk.used;
        for (let p = start[i]; p < start[i + 1]; p++) {
          line[at++] = xy[2 * p] + 0.5 - chunk.x; // the place of a pixel's centre on screen
          line[at++] = xy[2 * p + 1] + 0.5 - chunk.y;
          line[at++] = i;
        }
        if (closed[i]) {
          line[at++] = xy[2 * start[i]] + 0.5 - chunk.x;
          line[at++] = xy[2 * start[i] + 1] + 0.5 - chunk.y;
          line[at++] = i;
        }
        line[at++] = GONE;
        line[at++] = GONE;
        line[at++] = i;
        chunk.used = at;
        const size = Math.max(box[4 * i + 2] - box[4 * i], box[4 * i + 3] - box[4 * i + 1]);
        const radius = Number.isNaN(diameter[i]) ? size / 2 : diameter[i] / 2;
        chunk.dot.set([centre[2 * i] + 0.5 - chunk.x, centre[2 * i + 1] + 0.5 - chunk.y, radius, i], 4 * chunk.dots++);
        chunk.box[0] = Math.min(chunk.box[0], box[4 * i]);
        chunk.box[1] = Math.min(chunk.box[1], box[4 * i + 1]);
        chunk.box[2] = Math.max(chunk.box[2], box[4 * i + 2]);
        chunk.box[3] = Math.max(chunk.box[3], box[4 * i + 3]);
        chunk.least = Math.min(chunk.least, size);
        chunk.most = Math.max(chunk.most, size);
        data[2 * i] = GONE;
        data[2 * i + 1] = size;
      }
      const made = [];
      for (const chunk of chunks) {
        if (!chunk) continue;
        const lineArray = gl.createVertexArray();
        const lineBuffer = gl.createBuffer();
        gl.bindVertexArray(lineArray);
        gl.bindBuffer(gl.ARRAY_BUFFER, lineBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, chunk.line, gl.STATIC_DRAW);
        // every pair of neighbouring points is a segment: the same buffer, read twice, one point apart
        for (const [slot, offset] of [[0, 0], [1, 12]]) {
          gl.enableVertexAttribArray(slot);
          gl.vertexAttribPointer(slot, 3, gl.FLOAT, false, 12, offset);
          gl.vertexAttribDivisor(slot, 1);
        }
        const dotArray = gl.createVertexArray();
        const dotBuffer = gl.createBuffer();
        gl.bindVertexArray(dotArray);
        gl.bindBuffer(gl.ARRAY_BUFFER, dotBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, chunk.dot, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 4, gl.FLOAT, false, 16, 0);
        gl.vertexAttribDivisor(0, 1);
        made.push({
          x: chunk.x, y: chunk.y, box: chunk.box, least: chunk.least, most: chunk.most,
          lineArray, lineBuffer, segments: chunk.line.length / 3 - 1, dotArray, dotBuffer, dots: chunk.dots,
        });
      }
      gl.bindVertexArray(null);
      const texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG32F, DATA_WIDTH, data.length / 2 / DATA_WIDTH, 0, gl.RG, gl.FLOAT, data);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      set.gpu = { chunks: made, texture, data, values: null };
      uploaded.add(set);
      if (uploaded.size > 6) {
        // the sets of results long gone: the oldest go first
        for (const old of [...uploaded].sort((a, b) => a.gpu.drawn - b.gpu.drawn).slice(0, uploaded.size - 6)) release(old);
      }
    }

    function release(set) {
      if (!set.gpu) return;
      for (const chunk of set.gpu.chunks) {
        gl.deleteVertexArray(chunk.lineArray);
        gl.deleteBuffer(chunk.lineBuffer);
        gl.deleteVertexArray(chunk.dotArray);
        gl.deleteBuffer(chunk.dotBuffer);
      }
      gl.deleteTexture(set.gpu.texture);
      set.gpu = null;
      uploaded.delete(set);
    }

    /* The values the contours are coloured by (one per contour; null: none), into the set's texture. */
    function setValues(set, values) {
      const gpu = set.gpu;
      if (gpu.values === values) return;
      gpu.values = values;
      for (let i = 0; i < set.n; i++) gpu.data[2 * i] = values && !Number.isNaN(values[i]) ? values[i] : GONE;
      gl.bindTexture(gl.TEXTURE_2D, gpu.texture);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, DATA_WIDTH, gpu.data.length / 2 / DATA_WIDTH, gl.RG, gl.FLOAT, gpu.data);
    }

    let frame = 0;

    function begin(width, height) {
      frame++;
      gl.viewport(0, 0, width, height);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }

    /* Draw a set. view: {left, top (image, screen units), zoom (device pixels per image pixel), width,
       height (device pixels)}. look: {width (device pixels), colour [r, g, b] 0..1, values, low, high,
       scale: {name, stops}, outlines: false to draw the centres only}. */
    function draw(set, view, look) {
      if (lost || !set.n) return;
      if (!set.gpu) upload(set);
      const gpu = set.gpu;
      gpu.drawn = frame;
      const scaled = Boolean(look.values);
      setValues(set, look.values || null);
      const right = view.left + view.width / view.zoom;
      const bottom = view.top + view.height / view.zoom;
      const seen = gpu.chunks.filter((c) => c.box[2] + 2 >= view.left && c.box[0] - 1 <= right && c.box[3] + 2 >= view.top && c.box[1] - 1 <= bottom);
      if (!seen.length) return;
      const outlineFrom = look.outlines === false ? Infinity : OUTLINE_FROM;
      const casing = Math.max(1, 0.5 * look.width);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, gpu.texture);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, scaleTexture(look.scale.name, look.scale.stops));
      const common = (p) => {
        gl.useProgram(p.made);
        gl.uniform1f(p.uniforms.u_zoom, view.zoom);
        gl.uniform2f(p.uniforms.u_view, view.width, view.height);
        gl.uniform1f(p.uniforms.u_outlineFrom, outlineFrom);
        gl.uniform2f(p.uniforms.u_range, look.low, 1 / Math.max(look.high - look.low, 1e-12));
        gl.uniform1i(p.uniforms.u_data, 0);
        gl.uniform1i(p.uniforms.u_scale, 1);
      };
      const origin = (p, chunk) => gl.uniform2f(p.uniforms.u_origin, (chunk.x - view.left) * view.zoom, (chunk.y - view.top) * view.zoom);
      // centres first, so that outlines lie over them
      const small = seen.filter((c) => c.least * view.zoom < outlineFrom);
      if (small.length) {
        common(dots);
        gl.uniform4f(dots.uniforms.u_colour, ...look.colour, 1);
        gl.uniform1f(dots.uniforms.u_scaled, scaled ? 1 : 0);
        for (const chunk of small) {
          // a ring while the image can be seen through it, a disc once it is a speck
          const typical = Math.sqrt(chunk.least * chunk.most) * view.zoom;
          gl.uniform1f(dots.uniforms.u_ring, typical >= 4.5 ? Math.max(1.25, 0.6 * look.width) : 0);
          gl.uniform1f(dots.uniforms.u_extra, typical >= 4.5 ? 1 : typical >= 2.5 ? 0.75 : 0);
          gl.uniform1f(dots.uniforms.u_least, typical >= 2.5 ? 1.5 : 1);
          origin(dots, chunk);
          gl.bindVertexArray(chunk.dotArray);
          gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, chunk.dots);
        }
      }
      const large = seen.filter((c) => c.most * view.zoom >= outlineFrom);
      if (large.length) {
        common(lines);
        // all casings, then all colours: no segment's casing lies over its neighbour's colour
        for (const pass of [0, 1]) {
          gl.uniform1f(lines.uniforms.u_half, pass ? look.width / 2 : look.width / 2 + casing);
          gl.uniform4f(lines.uniforms.u_colour, ...(pass ? look.colour : [0.04, 0.04, 0.04]), pass ? 1 : 0.9);
          gl.uniform1f(lines.uniforms.u_scaled, pass && scaled ? 1 : 0);
          for (const chunk of large) {
            origin(lines, chunk);
            gl.bindVertexArray(chunk.lineArray);
            gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, chunk.segments);
          }
        }
      }
      gl.bindVertexArray(null);
    }

    return { begin, draw, release, canvas, get lost() { return lost; } };
  }

  /* The colour of a scale at a place 0..1: [r, g, b, a] 0..255, between its stops. */
  function colourAt(stops, place) {
    const at = clamp(place, 0, 1) * (stops.length - 1);
    const k = Math.min(Math.floor(at), stops.length - 2);
    const share = at - k;
    return [0, 1, 2].map((c) => Math.round(stops[k][c] + share * (stops[k + 1][c] - stops[k][c]))).concat(255);
  }

  return { create, colourAt, OUTLINE_FROM };
})();
