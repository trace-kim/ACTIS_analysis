"use strict";
/* The contour lab's window: the tabs, the workflow and its options, and when the workflow runs.
   view.js draws the image of the tab in front. The server (session.py) holds the images and runs the
   workflows; this page holds the workflows themselves and what is shown. */

const KEY = document.body.dataset.key;
const STORE = "actis-contour-lab";
const $ = (id) => document.getElementById(id);
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const clone = (value) => JSON.parse(JSON.stringify(value));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

const S = {
  meta: null,
  modules: new Map(), // by name
  tabs: [],
  active: null, // the tab in front
  shared: null, // the workflow of every tab that has none of its own
  auto: true,
  sync: true,
  pixelNm: 21,
  show: { of: "result", image: "processed", contours: true, before: false, centres: false, mask: false, pinned: true, smooth: false },
  folder: "",
  folders: [], // used lately, for the Open dialog
  request: (Math.floor(performance.timeOrigin) % 1e6) * 1000, // numbers the runs asked for
};

// --- Small things ---------------------------------------------------------------------------------------------

function h(tag, props = {}, ...kids) {
  const element = document.createElement(tag);
  for (const [name, value] of Object.entries(props)) {
    if (value === false || value === null || value === undefined) continue;
    if (name === "class") element.className = value;
    else if (name === "text") element.textContent = value;
    else if (name.startsWith("on")) element.addEventListener(name.slice(2), value);
    else if (name in element && name !== "list") element[name] = value;
    else element.setAttribute(name, value === true ? "" : value);
  }
  for (const kid of kids.flat()) if (kid !== null && kid !== undefined && kid !== false) element.append(kid);
  return element;
}

/* A number with no more digits than say something. */
function trim(value, digits = 4) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "–";
  return String(Number(Number(value).toPrecision(digits)));
}

function count(value) {
  return Number(value).toLocaleString("en-US");
}

/* A length in pixels, and in nanometres if the pixel size is known. */
function length(pixels, digits = 4) {
  if (pixels === null || pixels === undefined || !Number.isFinite(Number(pixels))) return "–";
  const nm = S.pixelNm > 0 ? ` = ${trim(pixels * S.pixelNm, digits)} nm` : "";
  return `${trim(pixels, digits)} px${nm}`;
}

function toast(message, kind = "info", ms = 7000) {
  const note = h("div", { class: `toast ${kind}`, text: message });
  $("toasts").append(note);
  setTimeout(() => note.remove(), ms);
}

const api = {
  async call(path, options = {}) {
    const reply = await fetch(path, { ...options, headers: { ...(options.headers || {}), "X-Lab-Key": KEY } });
    if (!reply.ok) {
      let message = `${reply.status} ${reply.statusText}`;
      try {
        message = (await reply.json()).error || message;
      } catch {
        /* not JSON: the status says it */
      }
      throw new Error(message);
    }
    return reply;
  },
  async get(path, signal) {
    return (await this.call(path, { signal })).json();
  },
  async post(path, data) {
    return (await this.call(path, { method: "POST", body: JSON.stringify(data ?? {}) })).json();
  },
};

function download(blob, name) {
  const link = h("a", { href: URL.createObjectURL(blob), download: name });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(link.href), 4000);
}

function remember() {
  try {
    const { shared, auto, sync, pixelNm, show, folder, folders } = S;
    localStorage.setItem(STORE, JSON.stringify({ shared, auto, sync, pixelNm, show, folder, folders }));
    const own = {};
    const regions = {};
    for (const tab of S.tabs) {
      if (tab.own) own[tab.id] = tab.own;
      regions[tab.id] = tab.region;
    }
    localStorage.setItem(`${STORE}:${KEY}`, JSON.stringify({ own, regions, active: S.active && S.active.id }));
  } catch {
    /* no storage: nothing is kept between visits */
  }
}

function recalled(name) {
  try {
    return JSON.parse(localStorage.getItem(name)) || {};
  } catch {
    return {};
  }
}

// --- Workflows ------------------------------------------------------------------------------------------------

function defaults(moduleName) {
  const options = {};
  for (const option of S.modules.get(moduleName).options) options[option.name] = option.default;
  return options;
}

/* Steps as a workflow file or an example gives them, with every option of their modules (those the file does
   not give at their defaults). An Error says what is wrong. */
function completed(steps) {
  if (!Array.isArray(steps)) throw new Error("the workflow has no list of steps");
  return steps.map((entry, k) => {
    const module = entry && S.modules.get(entry.module);
    if (!module) throw new Error(`step ${k + 1}: there is no module ${JSON.stringify(entry && entry.module)}`);
    const options = defaults(module.name);
    for (const [name, value] of Object.entries(entry.options || {})) {
      if (!(name in options)) throw new Error(`step ${k + 1}: ${module.name} has no option ${JSON.stringify(name)}`);
      options[name] = value;
    }
    return { module: module.name, enabled: entry.enabled !== false, options };
  });
}

/* A workflow: its name (of the example or file it came from), its steps, the row of the step selected, and
   its steps as they were when it was opened or saved (to tell whether it was changed since). */
function workflow(name, steps, origin = null) {
  const made = completed(steps);
  return { name, steps: made, origin: origin || JSON.stringify(made), selected: Math.max(0, made.length - 1) };
}

function changedSince(wf) {
  return JSON.stringify(wf.steps) !== wf.origin;
}

/* The workflow a tab runs: its own, or the one of all tabs. */
function flow(tab = S.active) {
  return (tab && tab.own) || S.shared;
}

function title(steps, k) {
  return `${k + 1}. ${S.modules.get(steps[k].module).title}`;
}

/* The row to put a new step of a module at, `selected` being the row of the step selected (-1: none): right
   after that step if it is a module of the same kind; else after the last step of the new module's kind,
   which keeps the image modules before the mask modules before the contours modules (a contours module added
   while the threshold is selected goes after the tracing, which makes what it needs; a blur added while the
   last step is selected goes before the threshold). A list not in that order already: after the step
   selected, as it is. */
function placeFor(steps, moduleName, selected) {
  const order = (name) => S.meta.stages.indexOf(S.modules.get(name).stage);
  const stages = steps.map((s) => order(s.module));
  if (stages.some((stage, k) => k && stage < stages[k - 1])) return selected + 1;
  const stage = order(moduleName);
  if (selected >= 0 && stages[selected] === stage) return selected + 1;
  return stages.filter((other) => other <= stage).length;
}

/* The workflow was changed: show it, and run it once the changes pause (Auto process). */
function edited(redraw = true) {
  remember();
  if (redraw) renderFlow();
  else {
    renderFlowName();
    renderSteps();
  }
  for (const tab of S.tabs) if (!tab.own || tab === S.active) tab.stale = true;
  schedule();
}

function replaceFlow(name, steps) {
  const made = workflow(name, steps);
  if (S.active && S.active.own) S.active.own = made;
  else S.shared = made;
  remember();
  renderFlow();
  schedule(0);
}

function mayReplace() {
  const wf = flow();
  return !changedSince(wf) || !wf.steps.length || confirm("Replace the workflow? Its changes are not saved.");
}

function addStep(moduleName) {
  const wf = flow();
  const at = placeFor(wf.steps, moduleName, wf.steps.length ? wf.selected : -1);
  const options = defaults(moduleName);
  if ("level" in options && S.active) {
    // a grey level that divides this image, rather than one that suits none
    const { black, white } = S.active.info.source;
    options.level = Number(trim((black + white) / 2, 3));
  }
  wf.steps.splice(at, 0, { module: moduleName, enabled: true, options });
  wf.selected = at;
  edited();
}

function removeStep() {
  const wf = flow();
  if (!wf.steps.length) return;
  wf.steps.splice(wf.selected, 1);
  wf.selected = clamp(wf.selected, 0, Math.max(0, wf.steps.length - 1));
  edited();
}

function moveStep(by) {
  const wf = flow();
  const to = wf.selected + by;
  if (to < 0 || to >= wf.steps.length) return;
  wf.steps.splice(to, 0, wf.steps.splice(wf.selected, 1)[0]);
  wf.selected = to;
  edited();
}

function saveFlow() {
  const wf = flow();
  const data = { ...S.meta.format, steps: wf.steps };
  const name = (wf.name || "workflow").replace(/[\\/:*?"<>|]+/g, " ").replace(/\s+/g, " ").trim() || "workflow";
  download(new Blob([JSON.stringify(data, null, 2) + "\n"], { type: "application/json" }), `${name}.json`);
  wf.origin = JSON.stringify(wf.steps);
  remember();
  renderFlowName();
}

function openFlow(text, fileName) {
  try {
    const data = JSON.parse(text);
    if (!data || data.format !== S.meta.format.format) throw new Error("not a contour workflow file");
    if (!Number.isInteger(data.version) || data.version > S.meta.format.version) {
      throw new Error(`a workflow file of version ${data.version}: this program reads up to ${S.meta.format.version}`);
    }
    completed(data.steps);
    if (!mayReplace()) return;
    replaceFlow(fileName.replace(/\.json$/i, ""), data.steps);
  } catch (error) {
    toast(`${fileName} could not be read: ${error.message}`, "error");
  }
}

// --- Menus ----------------------------------------------------------------------------------------------------

function openMenu(anchor, items) {
  const menu = $("menu");
  menu.replaceChildren(
    ...items.map((item) =>
      item.heading
        ? h("div", { class: "heading", text: item.heading })
        : h(
            "button",
            {
              type: "button",
              onclick: () => {
                closeMenu();
                item.action();
              },
            },
            h("b", { text: item.title }),
            item.text ? h("span", { text: item.text }) : null,
          ),
    ),
  );
  menu.hidden = false;
  const box = anchor.getBoundingClientRect();
  const wide = menu.offsetWidth;
  const high = menu.offsetHeight;
  menu.style.left = `${clamp(box.left, 8, window.innerWidth - wide - 8)}px`;
  menu.style.top = `${box.bottom + high + 8 > window.innerHeight ? Math.max(8, box.top - high - 4) : box.bottom + 4}px`;
  menu.scrollTop = 0;
  const first = menu.querySelector("button");
  if (first) first.focus();
}

function closeMenu() {
  $("menu").hidden = true;
}

function firstSentence(text) {
  const end = text.search(/[.:](\s|$)/);
  return end < 0 ? text : text.slice(0, end + 1).replace(/:$/, ".");
}

// --- The workflow panel ---------------------------------------------------------------------------------------

function renderFlow() {
  renderFlowName();
  renderSteps();
  renderOptions();
}

function renderFlowName() {
  const wf = flow();
  $("flow-name").replaceChildren(wf.name || "Workflow", changedSince(wf) ? h("span", { class: "changed", text: "  (changed)" }) : "");
  const own = Boolean(S.active && S.active.own);
  for (const button of $("scope").children) {
    button.classList.toggle("on", (button.dataset.value === "own") === own);
    button.disabled = !S.active;
  }
  document.title = [S.active && S.active.info.name, wf.name, "ACTIS contour lab"].filter(Boolean).join(" – ");
}

/* How many steps, from the first, are those the result shown ran. */
function alike(tab) {
  if (!tab || !tab.result || !tab.ranSteps) return 0;
  const steps = flow(tab).steps;
  let k = 0;
  while (k < steps.length && k < tab.ranSteps.length && same(steps[k], tab.ranSteps[k])) k++;
  return k;
}

function renderSteps() {
  const wf = flow();
  const tab = S.active;
  const list = $("steps");
  const result = tab && tab.result;
  const matching = alike(tab);
  list.replaceChildren(
    ...wf.steps.map((step, k) => {
      const module = S.modules.get(step.module);
      const ran = result && k < matching && k < result.ran;
      const failed = result && result.failed_step === k + 1 && k < matching; // and still the step that failed
      let note = "";
      let time = "";
      if (failed) note = result.error.replace(/^Step \d+ \([^)]*\) failed: /, "");
      else if (ran) {
        note = result.steps[k].note || "";
        time = `${result.steps[k].seconds.toFixed(2)} s`;
      }
      const options = Object.entries(step.options)
        .map(([name, value]) => `${name} = ${value}`)
        .join(", ");
      return h(
        "li",
        {
          class: ["step", k === wf.selected && "selected", !step.enabled && "off", failed && "failed"].filter(Boolean).join(" "),
          title: `${module.help}${options ? `\n\nOptions: ${options}` : ""}${note ? `\n\nFound: ${note}` : ""}`,
          onclick: (event) => {
            if (event.target.tagName === "INPUT") return;
            wf.selected = k;
            renderSteps();
            renderOptions();
            if (S.show.of === "step") shownChanged();
          },
        },
        h("input", {
          type: "checkbox",
          checked: step.enabled,
          title: "Run this step; unticked, the steps after it get what the steps before it made",
          onchange: (event) => {
            step.enabled = event.target.checked;
            edited(false);
          },
        }),
        h("span", { class: "title", text: title(wf.steps, k) }),
        h("span", { class: "note", text: step.enabled || failed ? note : "off" }),
        h("span", { class: "time", text: time }),
      );
    }),
  );
  if (!wf.steps.length) {
    list.append(h("li", { class: "none", text: "No steps: add one, or choose a workflow to start from under Examples." }));
  }
  $("step-remove").disabled = !wf.steps.length;
  $("step-up").disabled = wf.selected <= 0;
  $("step-down").disabled = wf.selected >= wf.steps.length - 1;
  const chosen = list.children[wf.selected];
  if (chosen && chosen.scrollIntoView) chosen.scrollIntoView({ block: "nearest" });
}

function renderOptions() {
  const wf = flow();
  const box = $("options");
  const step = wf.steps[wf.selected];
  if (!step) {
    box.replaceChildren(h("p", { class: "dim", text: "The options of the step selected appear here, each with what it does." }));
    return;
  }
  const module = S.modules.get(step.module);
  const rows = module.options.map((option) => {
    const row = h("div", { class: "option" });
    const mark = () => row.classList.toggle("changed", step.options[option.name] !== option.default);
    const set = (value) => {
      if (step.options[option.name] === value) return;
      step.options[option.name] = value;
      mark();
      edited(false);
    };
    let control;
    if (option.kind === "flag") {
      control = h("input", { type: "checkbox", checked: step.options[option.name], onchange: (e) => set(e.target.checked) });
    } else if (option.kind === "choice") {
      control = h(
        "select",
        { onchange: (e) => set(e.target.value) },
        option.choices.map((choice) => h("option", { value: choice, text: choice, selected: choice === step.options[option.name] })),
      );
    } else {
      control = h("input", { type: "number", min: option.low, max: option.high, step: option.step, value: step.options[option.name] });
      control.addEventListener("input", () => {
        const value = control.valueAsNumber;
        const fits = Number.isFinite(value) && value >= option.low && value <= option.high && (option.kind !== "whole" || Number.isInteger(value));
        control.classList.toggle("invalid", !fits);
        if (fits) set(value);
      });
      control.addEventListener("change", () => {
        // left with something it cannot be: back to what it is
        if (control.classList.contains("invalid")) {
          control.value = step.options[option.name];
          control.classList.remove("invalid");
        }
      });
    }
    const id = `option-${option.name}`;
    control.id = id;
    const range = option.kind === "number" || option.kind === "whole" ? `  (${trim(option.low, 6)} to ${trim(option.high, 6)})` : "";
    row.append(
      h("label", { class: "name", for: id, text: option.name, title: `Default: ${option.default}${range}` }),
      h("span", { class: "control" }, control, option.unit ? h("span", { class: "unit", text: option.unit }) : null),
      h("div", { class: "help", text: option.help }),
    );
    mark();
    return row;
  });
  box.replaceChildren(
    h("h3", {}, title(wf.steps, wf.selected), h("span", { class: "stage-name", text: `   ${module.stage.toLowerCase()} module` })),
    h("p", { class: "about", text: module.help }),
    ...rows,
    module.options.length
      ? h(
          "div",
          { class: "defaults" },
          h("button", {
            type: "button",
            text: "Defaults",
            title: "Set the options of this step back to the module's defaults",
            onclick: () => {
              if (same(step.options, defaults(step.module))) return;
              step.options = defaults(step.module);
              edited();
            },
          }),
        )
      : null,
  );
  box.scrollTop = 0;
}

// --- Running the workflow -------------------------------------------------------------------------------------

let timer = 0;

/* What a tab would run now: to tell whether what it shows is that. */
function wanted(tab) {
  return JSON.stringify([flow(tab).steps, tab.region, tab.info.source.id]);
}

function fresh(tab) {
  return Boolean(tab.result) && tab.ranKey === wanted(tab);
}

function schedule(ms = 250) {
  clearTimeout(timer);
  renderBusy();
  if (S.auto && S.active) timer = setTimeout(() => process(), ms);
}

async function process(tab = S.active) {
  clearTimeout(timer);
  timer = 0;
  if (!tab) return;
  const request = ++S.request;
  tab.sent.set(request, { key: wanted(tab), steps: clone(flow(tab).steps) });
  for (const old of tab.sent.keys()) if (tab.sent.size > 30) tab.sent.delete(old);
  tab.busy = true;
  tab.refused = null;
  renderBusy();
  if (!tab.result && tab === S.active) renderSummary();
  try {
    await api.post(`/api/tabs/${tab.id}/run`, { steps: flow(tab).steps, region: tab.region, request });
  } catch (error) {
    tab.busy = false;
    tab.refused = error.message; // not run at all: an option out of range, a region too large
    renderBusy();
    renderSummary();
  }
}

let watching = null;

/* Follow the tab in front: every result the server publishes for it, as soon as it does. */
async function watch(tab) {
  if (watching) watching.abort();
  if (!tab) return;
  const mine = (watching = new AbortController());
  let after = -1;
  let failures = 0;
  while (!mine.signal.aborted && S.tabs.includes(tab)) {
    let state;
    try {
      state = await api.get(`/api/tabs/${tab.id}/result?after=${after}&timeout=25`, mine.signal);
      failures = 0;
      notice(null);
    } catch (error) {
      if (mine.signal.aborted) return;
      if (++failures > 2) notice("The lab does not answer: is it still running? (Start it again and reload this page.)");
      await new Promise((resolve) => setTimeout(resolve, 1500));
      continue;
    }
    if (mine.signal.aborted || state.closed) return;
    const news = state.serial !== after;
    after = state.serial;
    tab.busy = state.busy;
    if (!news) {
      /* nothing new: as it is */
    } else if (state.result) {
      adopt(tab, state.result);
    } else if (tab.result) {
      // the server let go of it (to make room for other tabs): run it again
      tab.result = null;
      tab.ranKey = null;
      shownChanged();
    }
    if (!state.busy && !fresh(tab) && S.auto && !tab.refused && !timer && !tab.failedKey) process(tab);
    renderBusy();
  }
}

function adopt(tab, result) {
  const sent = tab.sent.get(result.request);
  tab.result = result;
  tab.ranKey = sent ? sent.key : null;
  tab.ranSteps = sent ? sent.steps : null;
  tab.failedKey = sent && result.error ? sent.key : null;
  tab.stale = false;
  for (const id of tab.sets.keys()) if (!(id in result.contours)) tab.sets.delete(id);
  tab.picked = -1;
  tab.hover = -1;
  tab.detail = null;
  if (tab === S.active) {
    renderSteps();
    shownChanged();
  }
  renderTabs();
}

function renderBusy() {
  const tab = S.active;
  const waits = tab && !tab.busy && !timer && !fresh(tab) && !tab.refused;
  $("busy").textContent = !tab ? "" : tab.busy || timer ? "Processing…" : waits ? "Changed: press Process (F5)" : "";
  $("process").classList.toggle("waiting", Boolean(waits));
  $("process").disabled = !tab;
  renderTabs();
}

// --- What is shown --------------------------------------------------------------------------------------------

/* What the tab's result holds for the step shown (view.js draws it). index -1: there is no result. */
function shown(tab) {
  const result = tab.result;
  if (!result || !result.states.length) return { index: -1 };
  const last = result.states.length - 1;
  const wf = flow(tab);
  const index = S.show.of === "step" && wf.steps.length ? Math.min(wf.selected + 1, last) : last;
  const state = result.states[index];
  return {
    index,
    state,
    image: S.show.image === "processed" ? result.layers[state.image] : null,
    changed: state.image !== result.states[0].image,
    mask: state.mask ? result.layers[state.mask] : null,
    contours: state.contours,
    before: state.before,
  };
}

/* The contours of a set of the tab's result, once they are here (they are fetched when first asked for). */
function setOf(tab, id) {
  let entry = tab.sets.get(id);
  if (!entry) {
    entry = { set: null };
    tab.sets.set(id, entry);
    api
      .call(`/api/tabs/${tab.id}/contours/${id}`)
      .then((reply) => reply.arrayBuffer())
      .then((buffer) => {
        entry.set = decodeContours(buffer);
        View.draw();
      })
      .catch(() => tab.sets.delete(id)); // of a result that is gone meanwhile
  }
  return entry.set;
}

function shownChanged() {
  renderSummary();
  renderLayers();
  View.draw();
}

function renderLayers() {
  const tab = S.active;
  const view = tab ? shown(tab) : { index: -1 };
  const there = {
    contours: Boolean(view.contours),
    before: Boolean(view.before),
    centres: Boolean(view.contours),
    mask: Boolean(view.mask),
    pinned: Boolean(tab && tab.pinned),
  };
  for (const box of document.querySelectorAll("[data-layer]")) {
    box.checked = S.show[box.dataset.layer];
    box.parentElement.classList.toggle("off", !there[box.dataset.layer]);
  }
  for (const [id, value] of [["view-of", S.show.of], ["image-of", S.show.image]]) {
    for (const button of $(id).children) button.classList.toggle("on", button.dataset.value === value);
  }
  $("pin").classList.toggle("on", Boolean(tab && tab.pinned));
  $("pin").textContent = tab && tab.pinned ? "Unpin" : "Pin";
  $("pin").disabled = !tab || (!tab.pinned && !view.contours);
  $("smooth").checked = S.show.smooth;
  const levels = tab ? tab.levels : { auto: true };
  $("levels-auto").checked = levels.auto;
  $("levels-auto").disabled = !tab;
  for (const [id, name] of [["level-black", "black"], ["level-white", "white"]]) {
    $(id).disabled = !tab || levels.auto;
    if (tab && document.activeElement !== $(id)) $(id).value = trim(levels.auto ? front(tab)[name] : levels[name], 5);
  }
}

/* The layer whose grey levels are shown: the image a module changed, else the image as loaded. */
function front(tab) {
  const view = shown(tab);
  return view.image && view.changed ? view.image : tab.info.source;
}

function renderRegion() {
  const tab = S.active;
  $("region-group").hidden = !tab;
  if (!tab) return;
  const r = tab.region;
  $("region-text").textContent = r
    ? `x ${r[0]}–${r[2]}, y ${r[1]}–${r[3]}  (${r[2] - r[0]} × ${r[3] - r[1]} px)`
    : `whole image  (${tab.info.width} × ${tab.info.height} px)`;
  const tooLarge = tab.info.width * tab.info.height > tab.info.max_region_pixels;
  $("region-whole").disabled = !r || tooLarge;
  $("region-whole").title = tooLarge
    ? `The image is larger than a workflow runs on at once (${trim(tab.info.max_region_pixels / 1e6, 3)} million pixels): choose a region`
    : "Run the workflow on the whole image";
}

function setRegion(tab, region) {
  if (region) {
    const [x0, y0, x1, y1] = region;
    region = [clamp(x0, 0, tab.info.width), clamp(y0, 0, tab.info.height), clamp(x1, 0, tab.info.width), clamp(y1, 0, tab.info.height)];
    if (region[2] - region[0] < 8 || region[3] - region[1] < 8) return toast("A region needs at least 8 × 8 pixels.", "error");
    if ((region[2] - region[0]) * (region[3] - region[1]) > tab.info.max_region_pixels) {
      return toast(
        `That is ${trim(((region[2] - region[0]) * (region[3] - region[1])) / 1e6, 3)} million pixels: a workflow runs on at most ` +
          `${trim(tab.info.max_region_pixels / 1e6, 3)} million at once. Zoom in, or draw a smaller region with Shift + drag.`,
        "error",
      );
    }
    if (region[0] === 0 && region[1] === 0 && region[2] === tab.info.width && region[3] === tab.info.height) region = null;
  }
  tab.region = region;
  tab.refused = null;
  tab.failedKey = null;
  remember();
  renderRegion();
  View.draw();
  schedule(0);
}

// view.js calls these three.
function regionDrawn(tab, region) {
  setRegion(tab, region);
}

function camMoved() {
  /* nothing to keep: a tab's camera is copied to the next tab when that comes to the front */
}

async function contourPicked(tab, index) {
  tab.picked = index;
  tab.detail = null;
  renderSummary();
  const view = shown(tab);
  if (index < 0 || !view.contours) return;
  try {
    const detail = await api.get(`/api/tabs/${tab.id}/contours/${view.contours}/${index}`);
    if (tab.picked === index) {
      tab.detail = detail;
      renderSummary();
    }
  } catch {
    /* of a result that is gone meanwhile */
  }
}

// --- The summary under the image ------------------------------------------------------------------------------

function renderSummary() {
  const tab = S.active;
  const box = $("summary-text");
  const histogram = $("histogram");
  const g = histogram.getContext("2d");
  g.clearRect(0, 0, histogram.width, histogram.height);
  for (const button of document.querySelectorAll("[data-export]")) button.disabled = true;
  $("snapshot").disabled = !tab;
  if (!tab) return box.replaceChildren();
  const lines = [];
  if (tab.refused) lines.push(h("div", { class: "failed", text: `Not run: ${tab.refused}` }));
  const result = tab.result;
  const view = shown(tab);
  if (!result) {
    lines.push(h("div", { class: "dim", text: tab.busy ? "Processing…" : tab.refused ? "" : "Not processed yet." }));
    return box.replaceChildren(...lines);
  }
  if (result.error) lines.push(h("div", { class: "failed", text: `${result.error}.` }), h("span", { class: "dim", text: "Shown: " }));
  const steps = tab.ranSteps || flow(tab).steps;
  const where =
    view.index === 0
      ? "in the image as loaded"
      : `after step ${view.index} (${steps[view.index - 1] ? S.modules.get(steps[view.index - 1].module).title : "?"})`;
  const stats = view.contours ? result.contours[view.contours] : null;
  const r = result.region;
  const area = `${r[2] - r[0]} × ${r[3] - r[1]} px`;
  if (!stats) {
    lines.push(h("span", {}, h("b", { text: "No contours" }), ` ${where}: ${area}, ${result.seconds.toFixed(2)} s.`));
  } else {
    const left = Object.entries(stats.left_out).map(([why, n]) => `${count(n)} ${why}`);
    lines.push(
      h(
        "span",
        {},
        h("b", { text: `${count(stats.count)} contours` }),
        ` ${where}: ${count(stats.closed)} closed, ${count(stats.open)} open`,
        left.length ? `; left out: ${left.join(", ")}` : "",
        `. ${area}, ${result.seconds.toFixed(2)} s.`,
      ),
    );
    if (stats.diameter) {
      const d = stats.diameter;
      const three = d.median ? ` (3σ = ${trim((300 * d.robust_sigma) / d.median, 3)} %)` : "";
      lines.push(
        h(
          "div",
          {},
          "Diameter of equal area: median ",
          h("b", { text: length(d.median) }),
          `, robust σ ${length(d.robust_sigma, 3)}${three}; width × height ${trim(stats.width.median)} × ${trim(stats.height.median)} px` +
            (S.pixelNm > 0 ? ` = ${trim(stats.width.median * S.pixelNm)} × ${trim(stats.height.median * S.pixelNm)} nm.` : "."),
        ),
      );
      drawHistogram(g, histogram, stats);
    }
    for (const button of document.querySelectorAll("[data-export]")) button.disabled = false;
  }
  if (tab.pinned) {
    const median = tab.pinned.median ? `, median ⌀ ${length(tab.pinned.median)}` : "";
    lines.push(h("div", { class: "dim", text: `Pinned: ${count(tab.pinned.set.n)} contours of “${tab.pinned.label}”${median}.` }));
  }
  if (tab.detail) {
    const d = tab.detail;
    lines.push(
      h(
        "div",
        { class: "picked" },
        `Contour ${d.index + 1}: centre (${trim(d.x, 6)}, ${trim(d.y, 6)})` +
          (d.closed
            ? `, ⌀ ${length(d.diameter)}, area ${trim(d.area)} px², ${trim(d.width)} × ${trim(d.height)} px across, axes ` +
              `${trim(d.major)} × ${trim(d.minor)} px at ${trim(d.angle, 3)}°`
            : ", open") +
          `, ${trim(d.length)} px long, ${d.points} points.`,
      ),
    );
  }
  box.replaceChildren(...lines);
}

function drawHistogram(g, canvas, stats) {
  const hist = stats.histogram;
  if (!hist) return;
  const wide = canvas.width;
  const high = canvas.height;
  const top = Math.max(...hist.counts, 1);
  const base = high - 16;
  const n = hist.counts.length;
  const low = hist.edges[0];
  const span = hist.edges[n] - low || 1;
  g.fillStyle = "#35d3f0";
  hist.counts.forEach((value, k) => {
    const tall = Math.round((value / top) * (base - 4));
    g.fillRect(Math.round((k * wide) / n), base - tall, Math.max(1, Math.round(wide / n) - 1), tall);
  });
  const x = ((stats.diameter.median - low) / span) * wide;
  g.fillStyle = "#ffffff";
  g.fillRect(Math.round(x), 0, 1, base);
  g.fillStyle = "#93a3ae";
  g.font = "10px Consolas, monospace";
  g.textBaseline = "bottom";
  g.textAlign = "left";
  g.fillText(trim(low, 3), 0, high);
  g.textAlign = "right";
  g.fillText(`${trim(hist.edges[n], 3)} px`, wide, high);
  g.textAlign = "center";
  g.fillText(`⌀ ${trim(stats.diameter.median, 4)}`, clamp(x, 34, wide - 60), high);
}

function notice(text, calm = false) {
  const box = $("notice");
  box.hidden = !text;
  box.textContent = text || "";
  box.classList.toggle("calm", calm);
}

// --- Tabs and their images ------------------------------------------------------------------------------------

/* A region kept from an earlier visit, if it still fits the image; else the one the server proposes. */
function keptRegion(info, kept) {
  if (!kept.regions || !(info.id in kept.regions)) return info.region;
  const region = kept.regions[info.id];
  if (region === null) return info.width * info.height <= info.max_region_pixels ? null : info.region;
  if (!Array.isArray(region) || region.length !== 4) return info.region;
  const [x0, y0, x1, y1] = region;
  const fits = x0 >= 0 && y0 >= 0 && x1 <= info.width && y1 <= info.height && x1 - x0 >= 8 && y1 - y0 >= 8;
  return fits && (x1 - x0) * (y1 - y0) <= info.max_region_pixels ? region : info.region;
}

function makeTab(info, kept = {}) {
  return {
    id: info.id,
    info,
    own: (kept.own && kept.own[info.id]) || null,
    region: keptRegion(info, kept),
    cam: null,
    levels: { auto: true, black: info.source.black, white: info.source.white },
    result: null,
    ranKey: null,
    ranSteps: null,
    failedKey: null,
    refused: null,
    busy: false,
    stale: false,
    sent: new Map(), // the runs asked for, by number: what they were of
    sets: new Map(), // the contours fetched, by id
    pinned: null,
    picked: -1,
    hover: -1,
    detail: null,
  };
}

function sub(info) {
  if (info.pages < 2) return "";
  const v = info.view;
  return v.stack ? `pages ${v.first + 1}–${v.first + v.count}` : `page ${v.page + 1}/${info.pages}`;
}

function renderTabs() {
  const bar = $("tabs");
  bar.replaceChildren(
    ...S.tabs.map((tab) =>
      h(
        "button",
        {
          type: "button",
          class: ["tab", tab === S.active && "active", tab.busy && "busy", tab.result && tab.result.error && "failed"].filter(Boolean).join(" "),
          title: `${tab.info.path || tab.info.name}${tab.own ? "\nThis tab has a workflow of its own." : ""}`,
          onclick: () => activate(tab),
          onauxclick: (event) => {
            if (event.button === 1) closeTab(tab);
          },
        },
        h("span", { class: "dot" }),
        h("span", { class: "name", text: tab.info.name }),
        sub(tab.info) ? h("span", { class: "sub", text: sub(tab.info) }) : null,
        tab.own ? h("span", { class: "own", text: "◆", title: "This tab has a workflow of its own" }) : null,
        h("span", {
          class: "close",
          role: "button",
          title: "Close this image",
          text: "×",
          onclick: (event) => {
            event.stopPropagation();
            closeTab(tab);
          },
        }),
      ),
    ),
    h("button", { type: "button", class: "tab-add", text: "+", title: "Open more images (Ctrl+O)", onclick: openDialog }),
  );
}

function activate(tab) {
  const before = S.active;
  if (before === tab) return;
  if (tab && before && S.sync && before.cam && before.info.width === tab.info.width && before.info.height === tab.info.height) {
    tab.cam = { ...before.cam }; // the same place: switching back and forth compares the two
  }
  S.active = tab || null;
  remember();
  renderTabs();
  renderSource();
  renderRegion();
  renderFlow();
  shownChanged();
  renderBusy();
  watch(S.active);
}

async function closeTab(tab) {
  const at = S.tabs.indexOf(tab);
  if (at < 0) return;
  S.tabs.splice(at, 1);
  View.forget(tab.id);
  if (S.active === tab) activate(S.tabs[Math.min(at, S.tabs.length - 1)] || null);
  else renderTabs();
  remember();
  try {
    await api.post(`/api/tabs/${tab.id}/close`);
  } catch {
    /* closed already */
  }
}

function addTabs(reply) {
  const tabs = reply.tabs.map((info) => makeTab(info));
  S.tabs.push(...tabs);
  for (const error of reply.errors || []) toast(error, "error");
  if (tabs.length) activate(tabs[0]);
  else renderTabs();
  return tabs;
}

async function openPaths(paths) {
  if (!paths.length) return;
  try {
    addTabs(await api.post("/api/open", { paths }));
  } catch (error) {
    toast(error.message, "error");
  }
}

async function upload(file) {
  try {
    const reply = await api.call(`/api/upload?name=${encodeURIComponent(file.name)}`, { method: "POST", body: file });
    addTabs(await reply.json());
  } catch (error) {
    toast(error.message.includes(file.name) ? error.message : `${file.name}: ${error.message}`, "error");
  }
}

/* The server's new account of a tab's image: another page, or pages stacked. */
function reloaded(tab, info) {
  const resized = info.width !== tab.info.width || info.height !== tab.info.height;
  tab.info = info;
  tab.result = null;
  tab.ranKey = tab.ranSteps = tab.failedKey = tab.refused = null;
  tab.sets.clear();
  tab.picked = tab.hover = -1;
  tab.detail = null;
  tab.levels = { auto: true, black: info.source.black, white: info.source.white };
  View.forget(tab.id);
  if (resized) {
    tab.region = info.region;
    tab.cam = null;
    tab.pinned = null;
  }
  remember();
  renderTabs();
  renderSource();
  renderRegion();
  renderSteps();
  shownChanged();
  schedule(0);
}

let viewing = Promise.resolve();

function setView(tab, change) {
  // one after the other: the arrows can be clicked faster than a page loads
  viewing = viewing.then(async () => {
    try {
      reloaded(tab, await api.post(`/api/tabs/${tab.id}/view`, { ...tab.info.view, ...change }));
    } catch (error) {
      toast(error.message, "error");
      renderSource();
    }
  });
}

function stepPage(by) {
  const tab = S.active;
  if (!tab || tab.info.pages < 2) return;
  const v = tab.info.view;
  if (v.stack) {
    const first = clamp(v.first + by * v.count, 0, tab.info.pages - v.count);
    if (first !== v.first) setView(tab, { first });
  } else {
    const page = clamp(v.page + by, 0, tab.info.pages - 1);
    if (page !== v.page) setView(tab, { page });
  }
}

function renderSource() {
  const tab = S.active;
  const bar = $("source-bar");
  if (!tab) return bar.replaceChildren();
  const info = tab.info;
  const v = info.view;
  const bits = { uint8: "8 bit", uint16: "16 bit", float32: "float" }[info.dtype] || info.dtype;
  const parts = [
    h(
      "span",
      { class: "group" },
      h("b", { text: info.name }),
      h("span", { class: "dim mono", text: `${info.width} × ${info.height} px · ${bits} · ${trim(info.min, 6)}–${trim(info.max, 6)} DN` }),
      info.notes.length ? h("span", { class: "dim", text: info.notes.join("; ") }) : null,
    ),
  ];
  if (info.pages > 1) {
    const number = (value, max, title, change, min = 1) =>
      h("input", {
        type: "number",
        min,
        max,
        step: 1,
        value,
        title,
        onchange: (event) => {
          const chosen = Math.round(event.target.valueAsNumber);
          if (Number.isFinite(chosen) && chosen >= min && chosen <= max) change(chosen);
          else renderSource();
        },
      });
    const first = v.stack ? v.first : v.page;
    parts.push(
      h(
        "span",
        { class: "group" },
        h("span", { class: "label", text: v.stack ? "From page" : "Page" }),
        h("button", { type: "button", text: "◂", title: "The page before (,)", disabled: first <= 0, onclick: () => stepPage(-1) }),
        number(first + 1, info.pages, "Page of the file, from 1", (page) => setView(tab, v.stack ? { first: Math.min(page - 1, info.pages - v.count) } : { page: page - 1 })),
        h("button", {
          type: "button",
          text: "▸",
          title: "The page after (.)",
          disabled: first + (v.stack ? v.count : 1) >= info.pages,
          onclick: () => stepPage(1),
        }),
        h("span", { class: "dim", text: `of ${count(info.pages)}` }),
      ),
      h(
        "span",
        { class: "group" },
        h(
          "label",
          { class: "check", title: "Join pages below each other into one long strip, as the scan recorded them" },
          h("input", {
            type: "checkbox",
            checked: v.stack,
            onchange: (event) =>
              setView(
                tab,
                event.target.checked
                  ? { stack: true, first: Math.min(v.page, info.pages - 1), count: Math.min(8, info.pages - v.page) }
                  : { stack: false, page: v.first },
              ),
          }),
          "Stack",
        ),
        v.stack
          ? [
              number(v.count, info.pages - v.first, "How many pages are stacked", (pages) => setView(tab, { count: pages })),
              h("span", { class: "dim", text: "pages" }),
              h(
                "label",
                { class: "check", title: "The last of the pages at the top: a stripe scanned upwards stores its pages in that order" },
                h("input", { type: "checkbox", checked: v.reverse, onchange: (e) => setView(tab, { reverse: e.target.checked }) }),
                "reverse order",
              ),
              h(
                "label",
                { class: "check", title: "Every page upside down: as a stripe scanned upwards stores them" },
                h("input", { type: "checkbox", checked: v.flip, onchange: (e) => setView(tab, { flip: e.target.checked }) }),
                "flip pages",
              ),
            ]
          : null,
      ),
    );
  }
  parts.push(
    h("span", { class: "grow" }),
    h("button", {
      type: "button",
      text: "Duplicate tab",
      title: "The same image in a second tab: give one of the two a workflow of its own (this tab) to compare two workflows on it",
      onclick: async () => {
        try {
          const copy = makeTab(await api.post(`/api/tabs/${tab.id}/duplicate`));
          copy.region = tab.region;
          copy.cam = tab.cam && { ...tab.cam };
          copy.own = clone(flow(tab));
          S.tabs.splice(S.tabs.indexOf(tab) + 1, 0, copy);
          activate(copy);
        } catch (error) {
          toast(error.message, "error");
        }
      },
    }),
  );
  bar.replaceChildren(...parts);
}

// --- The Open dialog ------------------------------------------------------------------------------------------

const picker = { listing: null, chosen: new Set(), last: -1 };

function openDialog() {
  $("open-dialog").showModal();
  browseTo(S.folder || "");
}

async function browseTo(folder) {
  const files = $("files");
  try {
    const listing = await api.get(`/api/browse?folder=${encodeURIComponent(folder)}`);
    picker.listing = listing;
    picker.chosen.clear();
    picker.last = -1;
    S.folder = listing.folder;
    $("folder").value = listing.folder;
    $("folder-up").disabled = !listing.parent;
    renderPlaces();
    renderFiles();
    remember();
  } catch (error) {
    if (/\.(png|tiff?)$/i.test(folder.trim())) {
      // the path of an image, typed or pasted: open it
      $("open-dialog").close();
      return openPaths([folder.trim()]);
    }
    files.replaceChildren(h("div", { class: "failed", text: error.message }));
  }
}

function renderPlaces() {
  const places = [...(picker.listing.drives || []), ...S.folders.filter((folder) => folder !== picker.listing.folder)];
  $("places").replaceChildren(
    h("button", { type: "button", text: "Home", onclick: () => browseTo("") }),
    ...places.slice(0, 14).map((place) => h("button", { type: "button", text: place.length > 34 ? `…${place.slice(-32)}` : place, title: place, onclick: () => browseTo(place) })),
  );
}

function size(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 2 ** 20) return `${trim(bytes / 1024, 3)} KB`;
  if (bytes < 2 ** 30) return `${trim(bytes / 2 ** 20, 3)} MB`;
  return `${trim(bytes / 2 ** 30, 3)} GB`;
}

function renderFiles() {
  const { entries, folder } = picker.listing;
  const join = (name) => folder.replace(/[\\/]$/, "") + (folder.includes("\\") ? "\\" : "/") + name;
  const rows = entries.map((entry, k) =>
    h(
      "div",
      {
        class: ["file", entry.folder && "folder", picker.chosen.has(k) && "chosen"].filter(Boolean).join(" "),
        role: "option",
        onclick: (event) => {
          if (entry.folder) return;
          if (event.shiftKey && picker.last >= 0) {
            for (let i = Math.min(k, picker.last); i <= Math.max(k, picker.last); i++) if (!entries[i].folder) picker.chosen.add(i);
          } else if (event.ctrlKey || event.metaKey) {
            if (picker.chosen.has(k)) picker.chosen.delete(k);
            else picker.chosen.add(k);
          } else {
            picker.chosen.clear();
            picker.chosen.add(k);
          }
          picker.last = k;
          renderFiles();
        },
        ondblclick: () => {
          if (entry.folder) return browseTo(join(entry.name));
          picker.chosen.clear();
          picker.chosen.add(k);
          openChosen();
        },
      },
      h("span", { text: entry.folder ? "▸" : "" }),
      h("span", { class: "name", text: entry.name }),
      h("span", { class: "size", text: entry.folder ? "" : size(entry.size) }),
      h("span", { class: "when", text: entry.folder ? "" : new Date(entry.modified * 1000).toLocaleString() }),
    ),
  );
  const scroll = $("files").scrollTop;
  $("files").replaceChildren(...(rows.length ? rows : [h("div", { class: "none", text: "No folders and no PNG or TIFF images here." })]));
  $("files").scrollTop = scroll;
  const n = picker.chosen.size;
  $("chosen").textContent = n ? `${n} image${n > 1 ? "s" : ""} chosen` : "Click an image; Ctrl or Shift + click for several. Double-click opens.";
  $("open-ok").disabled = !n;
  picker.paths = () => [...picker.chosen].sort((a, b) => a - b).map((k) => join(entries[k].name));
}

function openChosen() {
  const paths = picker.paths ? picker.paths() : [];
  if (!paths.length) return;
  S.folders = [S.folder, ...S.folders.filter((folder) => folder !== S.folder)].slice(0, 8);
  $("open-dialog").close();
  openPaths(paths);
}

// --- Exports --------------------------------------------------------------------------------------------------

async function exportContours(kind) {
  const tab = S.active;
  const view = tab ? shown(tab) : {};
  if (!view.contours) return;
  try {
    const reply = await api.call(`/api/tabs/${tab.id}/export/${view.contours}/${kind}?pixel_nm=${S.pixelNm || 0}`);
    const named = /filename="([^"]+)"/.exec(reply.headers.get("Content-Disposition") || "");
    download(await reply.blob(), named ? named[1] : `contours.${kind}`);
  } catch (error) {
    toast(error.message, "error");
  }
}

// --- Buttons, keys and files dropped --------------------------------------------------------------------------

function togglePin() {
  const tab = S.active;
  if (!tab) return;
  if (tab.pinned) tab.pinned = null;
  else {
    const view = shown(tab);
    const set = view.contours && setOf(tab, view.contours);
    if (!set) return;
    const stats = tab.result.contours[view.contours];
    tab.pinned = { set, label: flow(tab).name || "workflow", median: stats && stats.diameter && stats.diameter.median };
    S.show.pinned = true;
  }
  shownChanged();
}

function toggle(layer) {
  S.show[layer] = !S.show[layer];
  remember();
  shownChanged();
}

function wire() {
  $("open-images").onclick = $("open-images-2").onclick = openDialog;
  $("help").onclick = () => $("help-dialog").showModal();
  $("pixel-nm").oninput = (event) => {
    const value = event.target.valueAsNumber;
    S.pixelNm = Number.isFinite(value) && value > 0 ? value : 0;
    remember();
    shownChanged();
  };
  $("sync-view").onchange = (event) => {
    S.sync = event.target.checked;
    remember();
  };
  for (const [id, name] of [["view-of", "of"], ["image-of", "image"]]) {
    for (const button of $(id).children) {
      button.onclick = () => {
        S.show[name] = button.dataset.value;
        remember();
        shownChanged();
      };
    }
  }
  for (const box of document.querySelectorAll("[data-layer]")) {
    box.onchange = () => {
      S.show[box.dataset.layer] = box.checked;
      remember();
      shownChanged();
    };
  }
  $("pin").onclick = togglePin;
  $("smooth").onchange = (event) => {
    S.show.smooth = event.target.checked;
    remember();
    View.draw();
  };
  $("zoom-fit").onclick = () => S.active && View.fit(S.active);
  $("zoom-one").onclick = () => S.active && View.oneToOne(S.active);
  $("zoom-in").onclick = () => S.active && View.zoomBy(S.active, 1.6);
  $("zoom-out").onclick = () => S.active && View.zoomBy(S.active, 1 / 1.6);
  $("region-whole").onclick = () => S.active && setRegion(S.active, null);
  $("region-view").onclick = () => S.active && setRegion(S.active, View.shownRegion(S.active));
  $("levels-auto").onchange = (event) => {
    const tab = S.active;
    if (!tab) return;
    const { black, white } = front(tab);
    tab.levels = { auto: event.target.checked, black, white };
    shownChanged();
  };
  for (const [id, name] of [["level-black", "black"], ["level-white", "white"]]) {
    $(id).oninput = (event) => {
      const tab = S.active;
      const value = event.target.valueAsNumber;
      if (!tab || !Number.isFinite(value)) return;
      tab.levels[name] = value;
      if (tab.levels.white <= tab.levels.black) return; // not a range yet: wait for the other
      View.draw();
    };
  }
  for (const button of document.querySelectorAll("[data-export]")) button.onclick = () => exportContours(button.dataset.export);
  $("snapshot").onclick = async () => {
    const tab = S.active;
    if (tab) download(await View.picture(), `${tab.info.name.replace(/\.[^.]+$/, "")}-snapshot.png`);
  };

  // the workflow
  $("examples").onclick = (event) =>
    openMenu(
      event.currentTarget,
      S.meta.presets.map((preset) => ({
        title: preset.name,
        text: preset.help,
        action: () => mayReplace() && replaceFlow(preset.name, preset.steps),
      })),
    );
  $("step-add").onclick = (event) =>
    openMenu(
      event.currentTarget,
      S.meta.stages.flatMap((stage) => [
        { heading: `${stage} modules` },
        ...S.meta.modules
          .filter((module) => module.stage === stage)
          .map((module) => ({ title: module.title, text: firstSentence(module.help), action: () => addStep(module.name) })),
      ]),
    );
  $("step-remove").onclick = removeStep;
  $("step-up").onclick = () => moveStep(-1);
  $("step-down").onclick = () => moveStep(1);
  $("flow-save").onclick = saveFlow;
  $("flow-open").onclick = () => $("flow-file").click();
  $("flow-file").onchange = async (event) => {
    const file = event.target.files[0];
    event.target.value = "";
    if (file) openFlow(await file.text(), file.name);
  };
  $("scope").onclick = (event) => {
    const tab = S.active;
    const value = event.target.dataset.value;
    if (!tab || !value) return;
    if (value === "own" && !tab.own) {
      tab.own = clone(S.shared);
      remember();
      renderFlow();
      renderTabs();
    } else if (value === "all" && tab.own) {
      openMenu(event.target, [
        {
          title: "Use this workflow in all tabs",
          text: "This tab's workflow becomes the one every tab runs.",
          action: () => {
            S.shared = tab.own;
            tab.own = null;
            remember();
            renderFlow();
            renderTabs();
            schedule(0);
          },
        },
        {
          title: "Go back to the workflow of the other tabs",
          text: "This tab's own workflow is dropped.",
          action: () => {
            tab.own = null;
            remember();
            renderFlow();
            renderTabs();
            schedule(0);
          },
        },
      ]);
    }
  };
  $("auto").onchange = (event) => {
    S.auto = event.target.checked;
    remember();
    if (S.auto && S.active && !fresh(S.active)) process();
    else {
      clearTimeout(timer);
      timer = 0;
      renderBusy();
    }
  };
  $("process").onclick = () => {
    if (S.active) {
      S.active.failedKey = null;
      process();
    }
  };

  // the Open dialog
  $("folder-go").onclick = () => browseTo($("folder").value);
  $("folder").onkeydown = (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      browseTo($("folder").value);
    }
  };
  $("folder-up").onclick = () => picker.listing && picker.listing.parent && browseTo(picker.listing.parent);
  $("open-cancel").onclick = () => $("open-dialog").close();
  $("open-ok").onclick = openChosen;
  $("files").onkeydown = (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      openChosen();
    }
  };

  document.addEventListener("pointerdown", (event) => {
    if (!$("menu").hidden && !$("menu").contains(event.target) && !event.target.closest("#examples, #step-add, #scope")) closeMenu();
  });
  window.addEventListener("blur", closeMenu);

  // files dropped anywhere on the window
  let depth = 0;
  const files = (event) => event.dataTransfer && [...event.dataTransfer.types].includes("Files");
  window.addEventListener("dragenter", (event) => {
    if (!files(event)) return;
    event.preventDefault();
    depth++;
    document.body.classList.add("dropping");
  });
  window.addEventListener("dragover", (event) => files(event) && event.preventDefault());
  window.addEventListener("dragleave", () => {
    depth = Math.max(0, depth - 1);
    if (!depth) document.body.classList.remove("dropping");
  });
  window.addEventListener("drop", async (event) => {
    if (!files(event)) return;
    event.preventDefault();
    depth = 0;
    document.body.classList.remove("dropping");
    for (const file of event.dataTransfer.files) {
      if (/\.json$/i.test(file.name)) openFlow(await file.text(), file.name);
      else await upload(file);
    }
  });

  window.addEventListener("keydown", keyDown);
  window.addEventListener("keyup", (event) => {
    if (event.code === "Space") View.bare(false);
  });
  window.addEventListener("beforeunload", remember);
}

function keyDown(event) {
  const typing = /^(INPUT|SELECT|TEXTAREA)$/.test(event.target.tagName) && event.target.type !== "checkbox";
  const modal = document.querySelector("dialog[open]");
  const command = event.ctrlKey || event.metaKey;
  if (event.key === "Escape") {
    if (!$("menu").hidden) return closeMenu();
    if (!modal && S.active && S.active.picked >= 0) contourPicked(S.active, -1).then(View.draw);
    return;
  }
  if (modal) return;
  if (event.key === "F5") {
    event.preventDefault();
    return $("process").click();
  }
  if (event.key === "F1") {
    event.preventDefault();
    return $("help").click();
  }
  if (command && event.key.toLowerCase() === "o") {
    event.preventDefault();
    return openDialog();
  }
  if (command && event.key.toLowerCase() === "s") {
    event.preventDefault();
    return saveFlow();
  }
  if (command && event.key === "Delete") return removeStep();
  if (event.altKey && event.key === "ArrowUp") return event.preventDefault(), moveStep(-1);
  if (event.altKey && event.key === "ArrowDown") return event.preventDefault(), moveStep(1);
  if (typing || command || event.altKey) return;
  const tab = S.active;
  const at = S.tabs.indexOf(tab);
  const keys = {
    "[": () => at > 0 && activate(S.tabs[at - 1]),
    "]": () => at >= 0 && at < S.tabs.length - 1 && activate(S.tabs[at + 1]),
    ",": () => stepPage(-1),
    ".": () => stepPage(1),
    0: () => tab && View.fit(tab),
    1: () => tab && View.oneToOne(tab),
    "+": () => tab && View.zoomBy(tab, 1.6),
    "=": () => tab && View.zoomBy(tab, 1.6),
    "-": () => tab && View.zoomBy(tab, 1 / 1.6),
    o: () => {
      S.show.image = S.show.image === "processed" ? "original" : "processed";
      remember();
      shownChanged();
    },
    c: () => toggle("contours"),
    m: () => toggle("mask"),
    p: togglePin,
    ArrowLeft: () => tab && View.pan(tab, -1, 0),
    ArrowRight: () => tab && View.pan(tab, 1, 0),
    ArrowUp: () => tab && View.pan(tab, 0, -1),
    ArrowDown: () => tab && View.pan(tab, 0, 1),
  };
  if (event.code === "Space" && event.target.tagName !== "BUTTON") {
    event.preventDefault();
    return View.bare(true);
  }
  const act = keys[event.key.length === 1 ? event.key.toLowerCase() : event.key];
  if (act && !(event.key.startsWith("Arrow") && event.target.id !== "view" && event.target !== document.body)) {
    event.preventDefault();
    act();
  }
}

// --- Start ----------------------------------------------------------------------------------------------------

async function start() {
  try {
    S.meta = await api.get("/api/meta");
  } catch (error) {
    return notice(`The lab does not answer (${error.message}). Start it again and open the address it prints.`);
  }
  for (const module of S.meta.modules) S.modules.set(module.name, module);
  const kept = recalled(STORE);
  for (const name of ["auto", "sync", "pixelNm", "folder", "folders"]) if (name in kept) S[name] = kept[name];
  Object.assign(S.show, kept.show || {});
  const first = S.meta.presets.find((preset) => preset.name === S.meta.start);
  S.shared = workflow(first.name, first.steps);
  if (kept.shared) {
    try {
      S.shared = { ...workflow(kept.shared.name, kept.shared.steps, kept.shared.origin), selected: kept.shared.selected || 0 };
    } catch {
      /* of another version of the modules: the workflow to start from instead */
    }
  }
  for (const name of Object.keys(localStorage)) if (name.startsWith(`${STORE}:`) && name !== `${STORE}:${KEY}`) localStorage.removeItem(name);
  const here = recalled(`${STORE}:${KEY}`);
  for (const wf of Object.values(here.own || {})) {
    try {
      wf.steps = completed(wf.steps);
      wf.origin = wf.origin || JSON.stringify(wf.steps);
    } catch {
      here.own = {};
    }
  }
  $("auto").checked = S.auto;
  $("sync-view").checked = S.sync;
  $("pixel-nm").value = S.pixelNm || "";
  wire();
  const session = await api.get("/api/session");
  S.tabs = session.tabs.map((info) => makeTab(info, here));
  renderFlow();
  renderLayers();
  renderRegion();
  renderSummary();
  renderBusy();
  activate(S.tabs.find((tab) => tab.id === here.active) || S.tabs[0] || null);
  if (!S.tabs.length) View.draw();
}

window.addEventListener("DOMContentLoaded", start);
