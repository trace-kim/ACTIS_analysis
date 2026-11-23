"use strict";
/* The contour lab's window. Left: the method (a list of steps) and the settings of the step selected.
   Middle: the image (view.js draws it). Right: every number the method gives, in one place.
   The server (session.py) holds the images and runs the methods; this page holds the methods themselves
   and what is shown. */

const KEY = document.body.dataset.key;
const STORE = "actis-contour-lab-2";
const $ = (id) => document.getElementById(id);
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const clone = (value) => JSON.parse(JSON.stringify(value));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

const S = {
  meta: null,
  modules: new Map(), // by name
  tabs: [],
  active: null, // the tab in front
  shared: null, // the method of every tab that has none of its own
  auto: true,
  sync: true,
  pixelNm: 21,
  show: { of: "result", image: "processed", contours: true, before: false, centres: false, mask: false, reference: true, smooth: false },
  // how the contours look: coloured by a measure or in one colour, how wide, and what the colours span
  look: { byMeasure: true, measure: "diameter", width: 2.5, colour: "orange", range: {} },
  told: {}, // the explanations opened, by "module.option"
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

/* Make these the children of an element: nodes and text, lists of them, and nothing for null or false. */
function put(element, ...kids) {
  element.replaceChildren(...kids.flat(3).filter((kid) => kid !== null && kid !== undefined && kid !== false));
}

/* A number with no more digits than say something. */
function trim(value, digits = 4) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "–";
  return String(Number(Number(value).toPrecision(digits)));
}

function count(value) {
  return Number(value).toLocaleString("en-US");
}

function nm(pixels, digits = 4) {
  return `${trim(pixels * S.pixelNm, digits)} nm`;
}

/* A length in pixels, and in nanometres if the pixel size is known. */
function lengthText(pixels, digits = 4) {
  if (pixels === null || pixels === undefined || !Number.isFinite(Number(pixels))) return "–";
  return `${trim(pixels, digits)} px${S.pixelNm > 0 ? ` · ${nm(pixels, digits)}` : ""}`;
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
    const { shared, auto, sync, pixelNm, show, look, folder, folders } = S;
    localStorage.setItem(STORE, JSON.stringify({ shared, auto, sync, pixelNm, show, look, folder, folders }));
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

// --- Methods: lists of steps ----------------------------------------------------------------------------------

function defaults(moduleName) {
  const options = {};
  for (const option of S.modules.get(moduleName).options) options[option.name] = option.default;
  return options;
}

/* Steps as a file or an example gives them, with every option of their modules (those the file does not give
   at their defaults). An Error says what is wrong. */
function completed(steps) {
  if (!Array.isArray(steps)) throw new Error("the method has no list of steps");
  return steps.map((entry, k) => {
    const module = entry && S.modules.get(entry.module);
    if (!module) throw new Error(`step ${k + 1}: there is no step called ${JSON.stringify(entry && entry.module)}`);
    const options = defaults(module.name);
    for (const [name, value] of Object.entries(entry.options || {})) {
      if (!(name in options)) throw new Error(`step ${k + 1}: ${module.title} has no setting ${JSON.stringify(name)}`);
      options[name] = value;
    }
    return { module: module.name, enabled: entry.enabled !== false, options };
  });
}

/* A method: its name (of the example or file it came from), its steps, the row of the step selected, and its
   steps as they were when it was chosen or saved (to tell whether it was changed since). */
function workflow(name, steps, origin = null) {
  const made = completed(steps);
  return { name, steps: made, origin: origin || JSON.stringify(made), selected: Math.max(0, made.length - 1) };
}

function changedSince(wf) {
  return JSON.stringify(wf.steps) !== wf.origin;
}

/* The method a tab runs: its own, or the one of all tabs. */
function flow(tab = S.active) {
  return (tab && tab.own) || S.shared;
}

/* The row to put a new step at, `selected` being the row of the step selected (-1: none): right after that
   step if it is of the same kind; else after the last step of the new step's kind, which keeps the image
   steps before the mask steps before the contour steps. A list not in that order already: after the step
   selected, as it is. */
function placeFor(steps, moduleName, selected) {
  const order = (name) => S.meta.stages.indexOf(S.modules.get(name).stage);
  const stages = steps.map((s) => order(s.module));
  if (stages.some((stage, k) => k && stage < stages[k - 1])) return selected + 1;
  const stage = order(moduleName);
  if (selected >= 0 && stages[selected] === stage) return selected + 1;
  return stages.filter((other) => other <= stage).length;
}

/* The method was changed: show it, and run it once the changes pause. */
function edited(redraw = true) {
  remember();
  renderMethod();
  renderSteps();
  if (redraw) renderSettings();
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
  return !changedSince(wf) || !wf.steps.length || confirm("Replace the method? Its changes are not saved.");
}

function addStep(moduleName) {
  const wf = flow();
  const at = placeFor(wf.steps, moduleName, wf.steps.length ? wf.selected : -1);
  const options = defaults(moduleName);
  if ("level" in options && moduleName === "fixed_threshold" && S.active) {
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
  const name = (wf.name || "method").replace(/[\\/:*?"<>|]+/g, " ").replace(/\s+/g, " ").trim() || "method";
  download(new Blob([JSON.stringify(data, null, 2) + "\n"], { type: "application/json" }), `${name}.json`);
  wf.origin = JSON.stringify(wf.steps);
  remember();
  renderMethod();
}

function openFlow(text, fileName) {
  try {
    const data = JSON.parse(text);
    if (!data || data.format !== S.meta.format.format) throw new Error("not a method file of the contour lab");
    if (!Number.isInteger(data.version) || data.version > S.meta.format.version) {
      throw new Error(`a file of version ${data.version}: this program reads up to ${S.meta.format.version}`);
    }
    completed(data.steps);
    if (!mayReplace()) return;
    replaceFlow(fileName.replace(/\.json$/i, ""), data.steps);
  } catch (error) {
    toast(`${fileName} could not be read: ${error.message}`, "error");
  }
}

// --- Menus and pop-overs --------------------------------------------------------------------------------------

function place(box, anchor) {
  box.hidden = false;
  const at = anchor.getBoundingClientRect();
  const wide = box.offsetWidth;
  const high = box.offsetHeight;
  box.style.left = `${clamp(at.left, 8, window.innerWidth - wide - 8)}px`;
  box.style.top = `${at.bottom + high + 8 > window.innerHeight ? Math.max(8, at.top - high - 6) : at.bottom + 6}px`;
}

function openMenu(anchor, items) {
  const menu = $("menu");
  put(menu, 
    ...items.filter(Boolean).map((item) =>
      h(
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
  place(menu, anchor);
  menu.scrollTop = 0;
  const first = menu.querySelector("button");
  if (first) first.focus();
}

function closeMenu() {
  $("menu").hidden = true;
}

// --- The method panel -----------------------------------------------------------------------------------------

function renderFlow() {
  renderMethod();
  renderSteps();
  renderSettings();
}

function presetNamed(name) {
  return S.meta.presets.find((preset) => preset.name === name);
}

function renderMethod() {
  const wf = flow();
  const changed = changedSince(wf);
  put($("flow-name"), wf.name || "Method", changed ? h("span", { class: "edited", text: "edited" }) : "");
  const known = presetNamed(wf.name);
  $("flow-lead").textContent = known ? `Best for: ${known.best.charAt(0).toLowerCase()}${known.best.slice(1)}.` : "A method of your own.";
  const tab = S.active;
  $("scope-note").textContent = tab && tab.own ? "This tab has a method of its own; the other tabs keep theirs." : "";
  document.title = [tab && tab.info.name, wf.name, "Contour lab"].filter(Boolean).join(" – ");
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
  put(list, 
    ...wf.steps.map((step, k) => {
      const module = S.modules.get(step.module);
      const ran = result && k < matching && k < result.ran;
      const failed = result && result.failed_step === k + 1 && k < matching; // and still the step that failed
      let note = module.stage === "Image" ? "changes the image" : module.stage === "Mask" ? "marks the pattern" : "works on the contours";
      let time = "";
      if (!step.enabled) note = "off: skipped";
      else if (failed) note = result.error.replace(/^Step \d+ \([^)]*\) failed: /, "");
      else if (ran) {
        note = result.steps[k].note || "done";
        time = ` · ${result.steps[k].seconds.toFixed(2)} s`;
      }
      return h(
        "li",
        {
          class: ["step", k === wf.selected && "selected", !step.enabled && "off", failed && "failed", !ran && !failed && "stale"].filter(Boolean).join(" "),
          onclick: (event) => {
            if (event.target.closest(".switch")) return;
            wf.selected = k;
            renderSteps();
            renderSettings();
            $("settings").scrollIntoView({ block: "nearest", behavior: "smooth" });
            if (S.show.of === "step") shownChanged();
          },
        },
        h("span", { class: "num", text: String(k + 1) }),
        h("div", { class: "about" }, h("div", { class: "title", text: module.title }), h("div", { class: "note" }, note, time ? h("span", { class: "time", text: time }) : null)),
        h(
          "label",
          { class: "switch", title: "Run this step. Off: it is skipped, and the steps after it get what the steps before it made." },
          h("input", {
            type: "checkbox",
            checked: step.enabled,
            "aria-label": `Run ${module.title}`,
            onchange: (event) => {
              step.enabled = event.target.checked;
              edited(false);
            },
          }),
          h("span"),
        ),
      );
    }),
  );
  if (!wf.steps.length) list.append(h("li", { class: "none", text: "No steps yet. Add one, or choose a method." }));
}

/* The control of one setting of a step. */
function optionRow(step, module, option) {
  const id = `option-${option.name}`;
  const key = `${module.name}.${option.name}`;
  const row = h("div", { class: "opt" });
  const mark = () => row.classList.toggle("changed", step.options[option.name] !== option.default);
  const set = (value) => {
    if (step.options[option.name] === value) return;
    step.options[option.name] = value;
    mark();
    edited(false);
  };
  let body;
  if (option.kind === "flag") {
    const state = h("span", { class: "dim", text: step.options[option.name] ? "On" : "Off" });
    body = h(
      "label",
      { class: "switch" },
      h("input", {
        type: "checkbox",
        id,
        checked: step.options[option.name],
        onchange: (event) => {
          state.textContent = event.target.checked ? "On" : "Off";
          set(event.target.checked);
        },
      }),
      h("span"),
      state,
    );
  } else if (option.kind === "choice") {
    const buttons = option.choices.map((choice) =>
      h("button", {
        type: "button",
        class: choice === step.options[option.name] ? "on" : "",
        text: choice.charAt(0).toUpperCase() + choice.slice(1),
        onclick: () => {
          set(choice);
          renderSettings();
        },
      }),
    );
    body = h("div", { class: "choices", id, role: "group" }, buttons);
  } else {
    const whole = option.kind === "whole";
    const value = step.options[option.name];
    const number = h("input", { type: "number", id, min: option.low, max: option.high, step: option.step, value });
    // a slider over the values in use; the box beside it takes any value the setting allows
    let most = option.default > 0 ? option.default * 4 : option.low + 100 * option.step;
    most = Math.min(option.high, Math.max(most, value, option.low + 4 * option.step));
    const usable = (most - option.low) / option.step <= 4000 && most > option.low;
    const slider = usable ? h("input", { type: "range", min: option.low, max: most, step: option.step, value, "aria-label": option.label }) : null;
    const taken = (entered, from) => {
      const fits = Number.isFinite(entered) && entered >= option.low && entered <= option.high && (!whole || Number.isInteger(entered));
      number.classList.toggle("invalid", !fits);
      if (!fits) return;
      if (from !== number) number.value = entered;
      if (slider && from !== slider) slider.value = entered;
      set(entered);
    };
    number.addEventListener("input", () => taken(number.valueAsNumber, number));
    number.addEventListener("change", () => {
      // left with something it cannot be: back to what it is
      if (number.classList.contains("invalid")) {
        number.value = step.options[option.name];
        number.classList.remove("invalid");
      }
    });
    if (slider) slider.addEventListener("input", () => taken(slider.valueAsNumber, slider));
    body = h("div", { class: "opt-body" }, slider, number, option.unit ? h("span", { class: "unit", text: option.unit }) : null);
  }
  const help = h("p", { class: "opt-help", text: option.help, hidden: !S.told[key] });
  const what = h("button", {
    type: "button",
    class: "what",
    text: "?",
    title: "What this setting does",
    "aria-expanded": String(Boolean(S.told[key])),
    onclick: () => {
      S.told[key] = !S.told[key];
      help.hidden = !S.told[key];
      what.setAttribute("aria-expanded", String(S.told[key]));
    },
  });
  row.append(h("div", { class: "opt-head" }, h("label", { for: id, text: option.label }), what), body, help);
  mark();
  return row;
}

function renderSettings() {
  const wf = flow();
  const box = $("settings");
  const step = wf.steps[wf.selected];
  if (!step) {
    put(box, h("p", { class: "hint", text: "Click a step to see and change its settings." }));
    return;
  }
  const module = S.modules.get(step.module);
  const first = module.options.filter((option) => !option.advanced);
  const more = module.options.filter((option) => option.advanced);
  const changedMore = more.filter((option) => step.options[option.name] !== option.default).length;
  const open = box.querySelector("details.more")?.open && box.dataset.module === step.module;
  box.dataset.module = step.module;
  put(box, 
    h(
      "div",
      { class: "block-head" },
      h("h3", { text: `${wf.selected + 1} · ${module.title}` }),
      h(
        "div",
        { class: "tools" },
        h("button", { type: "button", class: "small", text: "↑", title: "Run this step one step earlier (Alt+↑)", disabled: wf.selected <= 0, onclick: () => moveStep(-1) }),
        h("button", { type: "button", class: "small", text: "↓", title: "Run this step one step later (Alt+↓)", disabled: wf.selected >= wf.steps.length - 1, onclick: () => moveStep(1) }),
        h("button", { type: "button", class: "small", text: "Remove", title: "Take this step out of the method (Ctrl+Delete)", onclick: removeStep }),
      ),
    ),
    h("p", { class: "lead", text: module.summary }),
    module.help.length > module.summary.length + 2 ? h("details", {}, h("summary", { text: "More about this step" }), h("p", { text: module.help.slice(module.summary.length).trim() })) : null,
    first.length ? h("div", { class: "opts" }, first.map((option) => optionRow(step, module, option))) : null,
    more.length
      ? h(
          "details",
          { class: "more", open: Boolean(open) },
          h("summary", { text: `${first.length ? "More settings" : "Settings"} (${more.length})${changedMore ? ` · ${changedMore} changed` : ""}` }),
          h("div", { class: "opts" }, more.map((option) => optionRow(step, module, option))),
        )
      : null,
    !module.options.length ? h("p", { class: "hint", text: "This step has no settings." }) : null,
    module.options.length && !same(step.options, defaults(step.module))
      ? h("div", {}, h("button", {
          type: "button",
          class: "small",
          text: "Back to this step's defaults",
          onclick: () => {
            step.options = defaults(step.module);
            edited();
          },
        }))
      : null,
  );
}

// --- Running the method ---------------------------------------------------------------------------------------

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
  timer = 0;
  if (S.auto && S.active) timer = setTimeout(() => process(), ms);
  renderBusy();
}

async function process(tab = S.active) {
  clearTimeout(timer);
  timer = 0;
  if (!tab) return;
  const request = ++S.request;
  tab.sent.set(request, { key: wanted(tab), steps: clone(flow(tab).steps), name: flow(tab).name });
  for (const old of tab.sent.keys()) if (tab.sent.size > 30) tab.sent.delete(old);
  tab.busy = true;
  tab.refused = null;
  renderBusy();
  if (!tab.result && tab === S.active) renderResults();
  try {
    await api.post(`/api/tabs/${tab.id}/run`, { steps: flow(tab).steps, region: tab.region, request });
  } catch (error) {
    tab.busy = false;
    tab.refused = error.message; // not run at all: a setting out of range, a part too large
    renderBusy();
    renderResults();
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
      if (++failures > 2) notice("The lab does not answer. Is it still running? Start it again and reload this page.");
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
  tab.ranName = sent ? sent.name : "";
  tab.failedKey = sent && result.error ? sent.key : null;
  tab.stale = false;
  for (const [id, entry] of tab.sets) {
    if (id in result.contours) continue;
    if (!tab.pinned || tab.pinned.set !== entry.set) View.dropSet(entry.set);
    tab.sets.delete(id);
  }
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
  const line = $("busy");
  line.classList.toggle("waiting", Boolean(waits));
  line.textContent = !tab ? "Open an image to run the method on." : tab.busy || timer ? "Running…" : waits ? "Changed since the last run. Press Run (F5)." : "";
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
        entry.set = decodeContours(buffer, S.meta.measures.map((m) => m.key));
        if (tab === S.active) {
          renderResults();
          View.draw();
        }
      })
      .catch(() => tab.sets.delete(id)); // of a result that is gone meanwhile
  }
  return entry.set;
}

function shownChanged() {
  renderResults();
  renderLayers();
  View.draw();
  View.readout();
}

function renderLayers() {
  const tab = S.active;
  const view = tab ? shown(tab) : { index: -1 };
  const there = {
    contours: Boolean(view.contours),
    before: Boolean(view.before),
    centres: Boolean(view.contours),
    mask: Boolean(view.mask),
    reference: Boolean(tab && tab.pinned),
  };
  for (const box of document.querySelectorAll("[data-layer]")) {
    const name = box.dataset.layer;
    box.checked = S.show[name];
    box.parentElement.classList.toggle("on", S.show[name] && there[name]);
    // a layer that is not there is not offered, but for the two a method always has
    box.parentElement.classList.toggle("none", !there[name] && !["contours", "mask"].includes(name));
  }
  for (const [id, value] of [["view-of", S.show.of], ["image-of", S.show.image]]) {
    for (const button of $(id).children) button.classList.toggle("on", button.dataset.value === value);
  }
  const wf = flow();
  const selected = wf.steps[wf.selected];
  $("view-of").children[1].textContent = selected ? `After step ${wf.selected + 1}` : "Selected step";
  document.documentElement.style.setProperty("--main", MAIN_INKS[S.look.colour] || INK.main);
}

function renderDisplay() {
  const tab = S.active;
  $("line-width").value = S.look.width;
  $("width-out").textContent = `${trim(S.look.width, 2)} px`;
  put($("swatches"), 
    ...Object.entries(MAIN_INKS).map(([name, hex]) =>
      h("button", {
        type: "button",
        class: name === S.look.colour ? "on" : "",
        style: `background:${hex}`,
        title: name.charAt(0).toUpperCase() + name.slice(1),
        "aria-label": name,
        onclick: () => {
          S.look.colour = name;
          remember();
          renderDisplay();
          renderLayers();
          renderResults();
          View.draw();
        },
      }),
    ),
  );
  $("smooth").checked = S.show.smooth;
  $("sync-view").checked = S.sync;
  const levels = tab ? tab.levels : { auto: true };
  $("levels-auto").checked = levels.auto;
  $("levels-auto").disabled = !tab;
  for (const [id, name] of [["level-black", "black"], ["level-white", "white"]]) {
    $(id).disabled = !tab || levels.auto;
    if (tab && document.activeElement !== $(id)) $(id).value = trim(levels.auto ? front(tab)[name] : levels[name], 5);
  }
}

/* The layer whose grey levels are shown: the image a step changed, else the image as loaded. */
function front(tab) {
  const view = shown(tab);
  return view.image && view.changed ? view.image : tab.info.source;
}

// --- The part of the image the method runs on -----------------------------------------------------------------

function renderRegion() {
  const tab = S.active;
  const select = $("run-on");
  select.disabled = !tab;
  if (!tab) return;
  const r = tab.region;
  const tooLarge = tab.info.width * tab.info.height > tab.info.max_region_pixels;
  select.options[0].disabled = tooLarge;
  select.options[0].textContent = tooLarge ? "the whole image (too large to run at once)" : `the whole image (${count(tab.info.width)} × ${count(tab.info.height)} px)`;
  select.options[2].textContent = r ? `a part: ${count(r[2] - r[0])} × ${count(r[3] - r[1])} px at ${r[0]}, ${r[1]}` : "a box drawn on the image (Shift + drag)";
  select.value = r ? "box" : "whole";
}

function setRegion(tab, region) {
  if (region) {
    const [x0, y0, x1, y1] = region;
    region = [clamp(x0, 0, tab.info.width), clamp(y0, 0, tab.info.height), clamp(x1, 0, tab.info.width), clamp(y1, 0, tab.info.height)];
    const pixels = (region[2] - region[0]) * (region[3] - region[1]);
    if (region[2] - region[0] < 8 || region[3] - region[1] < 8) {
      renderRegion();
      return toast("The part needs at least 8 × 8 pixels.", "error");
    }
    if (pixels > tab.info.max_region_pixels) {
      renderRegion();
      return toast(
        `That is ${trim(pixels / 1e6, 3)} million pixels; a method runs on at most ${trim(tab.info.max_region_pixels / 1e6, 3)} million at once. ` +
          "Zoom in, or draw a smaller box with Shift + drag.",
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

// view.js calls these two.
function regionDrawn(tab, region) {
  setRegion(tab, region);
}

async function contourPicked(tab, index) {
  tab.picked = index;
  tab.detail = null;
  renderPicked();
  const view = shown(tab);
  if (index < 0 || !view.contours) return;
  try {
    const detail = await api.get(`/api/tabs/${tab.id}/contours/${view.contours}/${index}`);
    if (tab.picked === index) {
      tab.detail = detail;
      renderPicked();
    }
  } catch {
    /* of a result that is gone meanwhile */
  }
}

// --- The results: every number in one place -------------------------------------------------------------------

function measureOf(key) {
  return S.meta.measures.find((m) => m.key === key) || S.meta.measures[0];
}

/* The numbers of one measure over a set's contours: how many have it, its median, its robust sigma (1.4826
   times the median distance from the median), the values 2 % and 98 % of the contours lie below, and all of
   them in rising order. Made once per set and measure. */
function summary(set, key) {
  if (set.summaries[key]) return set.summaries[key];
  const sorted = set.values[key].filter((value) => !Number.isNaN(value)).sort();
  const n = sorted.length;
  const at = (share) => (n ? sorted[clamp(Math.round(share * (n - 1)), 0, n - 1)] : NaN);
  const median = at(0.5);
  const away = Float32Array.from(sorted, (value) => Math.abs(value - median)).sort();
  const made = { count: n, sorted, median, sigma: n ? 1.4826 * away[n >> 1] : NaN, low: at(0.02), high: at(0.98), least: at(0), most: at(1) };
  set.summaries[key] = made;
  return made;
}

/* A value of a measure in words: [what it is, the same in another unit or ""]. */
function told(kind, value, digits = 4) {
  if (!Number.isFinite(value)) return ["–", ""];
  if (kind === "length") return [`${trim(value, digits)} px`, S.pixelNm > 0 ? nm(value, digits) : ""];
  if (kind === "percent") {
    const shown = Number(value.toFixed(Math.abs(value) < 10 ? 2 : 1)); // no "-0.00014"
    return [`${shown > 0 ? "+" : ""}${shown} %`, ""];
  }
  if (kind === "level") return [trim(value, 5), "grey levels"];
  return [trim(value, digits), ""];
}

const SPANS = { length: [1, 2, 5, 10, 20], level: [2, 5, 10, 20, 50], percent: [1, 2, 5, 10, 20], ratio: [1.1, 1.25, 1.5, 2] };

/* What the two ends of the colour scale stand for, for a measure of a set: {low, high, scale}. */
function colourSpan(set, key) {
  const kind = measureOf(key).kind;
  const sum = summary(set, key);
  const chosen = S.look.range[kind] || "auto";
  if (kind === "ratio") return { low: 1, high: chosen === "auto" ? Math.max(sum.high, 1.05) : Number(chosen), scale: SCALES.sequential };
  const middle = kind === "percent" ? 0 : sum.median;
  let half = Math.max(sum.high - middle, middle - sum.low, 1e-9);
  if (chosen !== "auto") half = kind === "percent" ? Number(chosen) : (Number(chosen) / 100) * Math.abs(middle);
  return { low: middle - half, high: middle + half, scale: SCALES.diverging };
}

/* view.js asks: what the contours of a set are coloured by, or null for one colour. */
function colouring(set) {
  const key = S.look.measure;
  if (!S.look.byMeasure || !set.values[key] || !summary(set, key).count) return null;
  return { values: set.values[key], ...colourSpan(set, key) };
}

function renderResults() {
  const tab = S.active;
  const head = $("headline");
  for (const id of ["measure-block", "picked-block", "reference-block", "export-block"]) $(id).hidden = true;
  if (!tab) return put(head, h("p", { class: "hint", text: "The numbers of the contours appear here." }));
  const result = tab.result;
  const lines = [];
  if (tab.refused) lines.push(h("div", { class: "bad", text: `Not run: ${tab.refused}` }));
  if (!result) {
    lines.push(h("div", { text: tab.busy ? "Running…" : tab.refused ? "" : "Not run yet." }));
    return put(head, h("div", { class: "facts" }, lines));
  }
  if (result.error) lines.push(h("div", { class: "bad", text: `${result.error}.` }));
  const view = shown(tab);
  const stats = view.contours ? result.contours[view.contours] : null;
  const steps = tab.ranSteps || flow(tab).steps;
  const stepName = view.index > 0 && steps[view.index - 1] ? S.modules.get(steps[view.index - 1].module).title : "";
  const r = result.region;
  const part = r[0] === 0 && r[1] === 0 && r[2] === tab.info.width && r[3] === tab.info.height ? "the whole image" : `a part of ${count(r[2] - r[0])} × ${count(r[3] - r[1])} px`;
  const ran = `${view.index < result.states.length - 1 || S.show.of === "step" ? `After step ${view.index}${stepName ? `, ${stepName}` : ""} · ` : ""}${part} · ${result.seconds.toFixed(2)} s`;
  if (!stats) {
    put(head, h("div", { class: "count" }, "No contours", h("small", { text: view.index === 0 ? "yet" : "at this step" })), h("div", { class: "facts" }, lines, h("div", { text: ran })));
  } else {
    lines.push(h("div", { text: `${count(stats.closed)} closed · ${count(stats.open)} open` }), h("div", { text: ran }));
    const left = Object.entries(stats.left_out);
    put(head, 
      h("div", { class: "count" }, count(stats.count), h("small", { text: stats.count === 1 ? "contour" : "contours" })),
      h("div", { class: "facts" }, lines),
      left.length ? h("div", { class: "left-out", title: "Candidates the method left out, and why" }, left.map(([why, n]) => h("span", { text: `${count(n)} ${why}` }))) : null,
    );
  }
  const set = view.contours ? setOf(tab, view.contours) : null;
  renderMeasure(set);
  renderPicked();
  renderReference(tab, set, stats);
  $("export-block").hidden = !stats;
  for (const button of document.querySelectorAll("[data-export]")) button.disabled = !stats;
}

let chart = null; // what the histogram shows, for its tooltip

function renderMeasure(set) {
  const block = $("measure-block");
  block.hidden = !set || !set.n;
  chart = null;
  if (block.hidden) return;
  const measure = measureOf(S.look.measure);
  const select = $("measure");
  if (select.options.length !== S.meta.measures.length) put(select, ...S.meta.measures.map((m) => h("option", { value: m.key, text: m.label })));
  select.value = measure.key;
  $("measure-help").textContent = measure.help + ".";
  const sum = summary(set, measure.key);
  const kind = measure.kind;
  const tiles = $("tiles");
  if (!sum.count) {
    put(tiles, h("p", { class: "hint", text: "No closed contours to measure." }));
    $("chart").hidden = true;
    return;
  }
  $("chart").hidden = false;
  const [typical, typicalToo] = told(kind, sum.median);
  const [spread, spreadToo] = told(kind === "percent" ? "percent" : kind, sum.sigma, 3);
  const relative = kind === "length" || kind === "level" ? `3σ = ${trim((300 * sum.sigma) / Math.abs(sum.median), 3)} %` : spreadToo;
  put(tiles, 
    h("div", { class: "tile", title: "The median: half of the contours lie below it" }, h("div", { class: "k", text: "Typical" }), h("div", { class: "v", text: typical }), h("div", { class: "s", text: typicalToo || "median" })),
    h(
      "div",
      { class: "tile", title: "Robust sigma: 1.4826 times the median distance from the median, which a few odd contours do not inflate" },
      h("div", { class: "k", text: "Spread" }),
      h("div", { class: "v", text: spread.replace(/^\+/, "") }),
      h("div", { class: "s", text: relative || "robust sigma" }),
    ),
    h(
      "div",
      { class: "tile", title: "96 of 100 contours lie between these two values" },
      h("div", { class: "k", text: "Most lie in" }),
      h("div", { class: "v", text: `${trim(sum.low, 3)} – ${trim(sum.high, 3)}` }),
      h("div", { class: "s", text: kind === "length" ? "px" : kind === "percent" ? "%" : kind === "level" ? "grey levels" : "" }),
    ),
  );
  $("colour-on").checked = S.look.byMeasure;
  const range = $("range");
  const options = [["auto", kind === "ratio" ? "from 1 to where most lie" : "where most contours lie"]];
  for (const span of SPANS[kind]) {
    options.push([String(span), kind === "ratio" ? `from 1 to ${span}` : kind === "percent" ? `±${span} %` : `±${span} % around the typical value`]);
  }
  put(range, ...options.map(([value, text]) => h("option", { value, text })));
  range.value = S.look.range[kind] || "auto";
  $("range-row").hidden = !S.look.byMeasure;
  drawHistogram(set, measure, sum);
}

function drawHistogram(set, measure, sum) {
  const canvas = $("histogram");
  const ratio = window.devicePixelRatio || 1;
  const wide = canvas.clientWidth || 320;
  const high = 150;
  canvas.width = Math.round(wide * ratio);
  canvas.height = Math.round(high * ratio);
  const g = canvas.getContext("2d");
  g.setTransform(ratio, 0, 0, ratio, 0, 0);
  g.clearRect(0, 0, wide, high);
  // the middle 99 % of the values, with a margin; what lies beyond is counted into the outermost bars
  const n = sum.count;
  const q = (share) => sum.sorted[clamp(Math.round(share * (n - 1)), 0, n - 1)];
  let low = q(0.005);
  let top = q(0.995);
  const margin = Math.max((top - low) * 0.12, Math.abs(sum.median) * 1e-4, 1e-9);
  low -= margin;
  top += margin;
  if (sum.least >= 0) low = Math.max(low, 0); // a length is not drawn below nothing
  const bins = clamp(Math.round(Math.sqrt(n)), 8, 44);
  const counts = new Uint32Array(bins);
  for (const value of sum.sorted) counts[clamp(Math.floor(((value - low) / (top - low)) * bins), 0, bins - 1)]++;
  const most = Math.max(...counts, 1);
  const span = S.look.byMeasure ? colourSpan(set, measure.key) : null;
  const base = high - 1;
  const each = wide / bins;
  for (let k = 0; k < bins; k++) {
    const tall = Math.max(counts[k] ? 2 : 0, Math.round((counts[k] / most) * (base - 8)));
    if (span) {
      const value = low + ((k + 0.5) / bins) * (top - low);
      const [r, gr, b] = GLContours.colourAt(span.scale.stops, (value - span.low) / (span.high - span.low));
      g.fillStyle = `rgb(${r},${gr},${b})`;
    } else {
      g.fillStyle = MAIN_INKS[S.look.colour] || INK.main;
    }
    g.beginPath();
    g.roundRect(k * each + 1, base - tall, Math.max(1, each - 2), tall, [Math.min(3, each / 3), Math.min(3, each / 3), 0, 0]);
    g.fill();
  }
  g.fillStyle = "#55544f";
  g.fillRect(0, base, wide, 1);
  // the typical value
  const at = ((sum.median - low) / (top - low)) * wide;
  g.fillStyle = "#0d0d0d";
  g.fillRect(Math.round(at) - 2, 0, 4, base);
  g.fillStyle = "#f0efec";
  g.fillRect(Math.round(at) - 1, 0, 2, base);
  const unit = measure.kind === "length" ? " px" : measure.kind === "percent" ? " %" : "";
  put($("axis"), h("span", { text: trim(low, 3) + unit }), h("span", { class: "mid", text: `typical ${trim(sum.median, 4)}${unit}` }), h("span", { text: trim(top, 3) + unit }));
  chart = { low, top, bins, counts, unit, wide };
}

function chartTip(event) {
  const tip = $("chart-tip");
  if (!chart || !event) return (tip.hidden = true);
  const box = $("histogram").getBoundingClientRect();
  const k = clamp(Math.floor(((event.clientX - box.left) / box.width) * chart.bins), 0, chart.bins - 1);
  const from = chart.low + (k / chart.bins) * (chart.top - chart.low);
  const to = chart.low + ((k + 1) / chart.bins) * (chart.top - chart.low);
  tip.textContent = `${trim(from, 4)} to ${trim(to, 4)}${chart.unit}: ${count(chart.counts[k])} contours`;
  tip.hidden = false;
  tip.style.left = `${clamp(event.clientX - box.left, 90, box.width - 90)}px`;
}

function renderPicked() {
  const tab = S.active;
  const block = $("picked-block");
  block.hidden = !tab || tab.picked < 0;
  if (block.hidden) return;
  const d = tab.detail;
  const rows = [];
  const row = (name, value) => rows.push(h("dt", { text: name }), h("dd", { text: value }));
  if (d) {
    row("Centre", `x ${trim(d.x, 6)}, y ${trim(d.y, 6)}`);
    if (d.closed) {
      row("Diameter", lengthText(d.diameter));
      row("Width × height", `${trim(d.width)} × ${trim(d.height)} px${S.pixelNm > 0 ? ` · ${trim(d.width * S.pixelNm)} × ${trim(d.height * S.pixelNm)} nm` : ""}`);
      row("Long × short axis", `${trim(d.major)} × ${trim(d.minor)} px, long axis at ${trim(d.angle, 3)}°`);
      row("Area", `${trim(d.area)} px²`);
      if (d.deviation !== null) row("Versus neighbours", told("percent", d.deviation)[0]);
    } else {
      row("Kind", "open: it runs out of the image");
    }
    if (d.brightness !== null) row("Brightness", `${trim(d.brightness, 5)} grey levels at its centre`);
    row("Outline", `${trim(d.length)} px long, ${count(d.points)} points`);
  }
  put(block, 
    h(
      "div",
      { class: "block-head" },
      h("h3", { text: `Contour ${count(tab.picked + 1)}` }),
      h("button", { type: "button", class: "quiet small", text: "Clear", title: "Let the contour go (Esc)", onclick: () => contourPicked(tab, -1).then(() => View.draw(true)) }),
    ),
    d ? h("dl", { class: "numbers" }, rows) : h("p", { class: "hint", text: "…" }),
  );
}

function renderReference(tab, set, stats) {
  const block = $("reference-block");
  block.hidden = !(tab.pinned || (set && set.n));
  if (block.hidden) return;
  const pin = $("pin");
  const text = $("reference-text");
  if (!tab.pinned) {
    text.textContent = "Keep these contours on screen in green while you change the method or choose another, to see what changes.";
    pin.textContent = "Keep these as a reference";
    return;
  }
  const kept = tab.pinned;
  let words = `Reference: ${count(kept.set.n)} contours of “${kept.label}”, typical diameter ${lengthText(kept.median)}.`;
  if (set && set !== kept.set && set.n && Number.isFinite(kept.median)) {
    const now = summary(set, "diameter").median;
    if (Number.isFinite(now)) words += ` Now: ${count(set.n)} contours, ${told("percent", 100 * (now / kept.median - 1))[0]} in typical diameter.`;
  }
  text.textContent = words;
  pin.textContent = "Let the reference go";
  void stats;
}

function notice(text) {
  const box = $("notice");
  box.hidden = !text;
  box.textContent = text || "";
}

// --- Tabs and their images ------------------------------------------------------------------------------------

/* A part kept from an earlier visit, if it still fits the image; else the one the server proposes. */
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
    ranName: "",
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
  return v.stack ? `pages ${v.first + 1}–${v.first + v.count}` : `page ${v.page + 1}`;
}

function renderTabs() {
  const bar = $("tabs");
  put(bar, 
    ...S.tabs.map((tab) =>
      h(
        "button",
        {
          type: "button",
          class: ["tab", tab === S.active && "active", tab.busy && "busy", tab.result && tab.result.error && "failed"].filter(Boolean).join(" "),
          title: `${tab.info.path || tab.info.name}${tab.own ? "\nThis tab has a method of its own." : ""}`,
          onclick: () => activate(tab),
          onauxclick: (event) => {
            if (event.button === 1) closeTab(tab);
          },
        },
        h("span", { class: "dot" }),
        h("span", { class: "name", text: tab.info.name }),
        sub(tab.info) ? h("span", { class: "sub", text: sub(tab.info) }) : null,
        tab.own ? h("span", { class: "sub", text: "own method" }) : null,
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
    h("button", { type: "button", class: "tab-add", text: S.tabs.length ? "+ Open" : "Open images…", title: "Open images, each in a tab of its own (Ctrl+O)", onclick: openDialog }),
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
  renderPages();
  renderRegion();
  renderFlow();
  renderDisplay();
  shownChanged();
  renderBusy();
  watch(S.active);
}

async function closeTab(tab) {
  const at = S.tabs.indexOf(tab);
  if (at < 0) return;
  S.tabs.splice(at, 1);
  View.forget(tab.id);
  for (const entry of tab.sets.values()) View.dropSet(entry.set);
  if (tab.pinned) View.dropSet(tab.pinned.set);
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

async function duplicateTab() {
  const tab = S.active;
  if (!tab) return;
  try {
    const copy = makeTab(await api.post(`/api/tabs/${tab.id}/duplicate`));
    copy.region = tab.region;
    copy.cam = tab.cam && { ...tab.cam };
    copy.own = clone(flow(tab));
    S.tabs.splice(S.tabs.indexOf(tab) + 1, 0, copy);
    activate(copy);
    toast("The copy has a method of its own: choose another for it, and switch between the two tabs to compare.");
  } catch (error) {
    toast(error.message, "error");
  }
}

/* The server's new account of a tab's image: another page, or pages stacked. */
function reloaded(tab, info) {
  const resized = info.width !== tab.info.width || info.height !== tab.info.height;
  tab.info = info;
  tab.result = null;
  tab.ranKey = tab.ranSteps = tab.failedKey = tab.refused = null;
  for (const entry of tab.sets.values()) if (!tab.pinned || tab.pinned.set !== entry.set) View.dropSet(entry.set);
  tab.sets.clear();
  tab.picked = tab.hover = -1;
  tab.detail = null;
  tab.levels = { auto: true, black: info.source.black, white: info.source.white };
  View.forget(tab.id);
  if (resized) {
    tab.region = info.region;
    tab.cam = null;
    if (tab.pinned) View.dropSet(tab.pinned.set);
    tab.pinned = null;
  }
  remember();
  renderTabs();
  renderPages();
  renderRegion();
  renderSteps();
  renderDisplay();
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
      renderPages();
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

/* The bar under the image of a file with several pages: which page, or which pages stacked. */
function renderPages() {
  const tab = S.active;
  const bar = $("pagebar");
  bar.hidden = !tab || tab.info.pages < 2;
  if (bar.hidden) return put(bar);
  const info = tab.info;
  const v = info.view;
  const first = v.stack ? v.first : v.page;
  const last = info.pages - (v.stack ? v.count : 1);
  const go = (page) => setView(tab, v.stack ? { first: clamp(page, 0, last) } : { page: clamp(page, 0, last) });
  const number = (value, max, label, change) =>
    h("input", {
      type: "number",
      min: 1,
      max,
      step: 1,
      value,
      "aria-label": label,
      onchange: (event) => {
        const chosen = Math.round(event.target.valueAsNumber);
        if (Number.isFinite(chosen) && chosen >= 1 && chosen <= max) change(chosen);
        else renderPages();
      },
    });
  put(bar, 
    h("button", { type: "button", text: "◂", title: "The page before (,)", disabled: first <= 0, onclick: () => stepPage(-1) }),
    h("span", { class: "dim", text: v.stack ? "From page" : "Page" }),
    number(first + 1, last + 1, "Page", (page) => go(page - 1)),
    h("span", { class: "dim", text: `of ${count(info.pages)}` }),
    h("button", { type: "button", text: "▸", title: "The page after (.)", disabled: first >= last, onclick: () => stepPage(1) }),
    v.stack ? null : h("input", { type: "range", min: 0, max: Math.max(last, 0), step: 1, value: first, "aria-label": "Page", title: "Drag to another page", onchange: (event) => go(event.target.valueAsNumber) }),
    h("span", { class: "sep" }),
    h(
      "label",
      { class: "switch", title: "Join pages below each other into one long strip, as the scan recorded them" },
      h("input", {
        type: "checkbox",
        checked: v.stack,
        onchange: (event) => setView(tab, event.target.checked ? { stack: true, first: Math.min(v.page, info.pages - 1), count: Math.min(4, info.pages - v.page) } : { stack: false, page: v.first }),
      }),
      h("span"),
      "Stack",
    ),
    v.stack ? [number(v.count, info.pages - v.first, "Pages stacked", (pages) => setView(tab, { count: pages })), h("span", { class: "dim", text: "pages" })] : null,
    v.stack
      ? h("button", {
          type: "button",
          text: "Order ▾",
          title: "For stripes scanned upwards, whose pages are stored last first and upside down",
          onclick: (event) =>
            openMenu(event.currentTarget, [
              { title: v.reverse ? "✓ Last page at the top" : "Last page at the top", text: "A stripe scanned upwards stores its pages in that order.", action: () => setView(tab, { reverse: !v.reverse }) },
              { title: v.flip ? "✓ Every page upside down" : "Every page upside down", text: "As a stripe scanned upwards stores them.", action: () => setView(tab, { flip: !v.flip }) },
            ]),
        })
      : null,
  );
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
    put(files, h("div", { class: "failed", text: error.message }));
  }
}

function renderPlaces() {
  const places = [...(picker.listing.drives || []), ...S.folders.filter((folder) => folder !== picker.listing.folder)];
  put($("places"), 
    h("button", { type: "button", text: "Home", onclick: () => browseTo("") }),
    ...places.slice(0, 12).map((spot) => h("button", { type: "button", text: spot.length > 34 ? `…${spot.slice(-32)}` : spot, title: spot, onclick: () => browseTo(spot) })),
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
          if (entry.folder) return browseTo(join(entry.name));
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
          if (entry.folder) return;
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
  put($("files"), ...(rows.length ? rows : [h("div", { class: "none", text: "No folders and no PNG or TIFF images here." })]));
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

// --- Choosing a method, adding a step -------------------------------------------------------------------------

function openMethods() {
  const current = flow().name;
  put($("methods"), 
    ...S.meta.groups.flatMap((group) => [
      h("div", { class: "group-name", text: group }),
      h(
        "div",
        { class: "cards" },
        S.meta.presets
          .filter((preset) => preset.group === group)
          .map((preset) =>
            h(
              "button",
              {
                type: "button",
                class: `card${preset.name === current ? " current" : ""}`,
                onclick: () => {
                  if (!mayReplace()) return;
                  $("methods-dialog").close();
                  replaceFlow(preset.name, preset.steps);
                },
              },
              h("b", { text: preset.name }),
              h("span", { class: "text", text: preset.help }),
              h("dl", {}, h("dt", { text: "Best for" }), h("dd", { text: preset.best }), h("dt", { text: "Watch" }), h("dd", { text: preset.watch })),
              h("span", { class: "flow" }, preset.steps.flatMap((step, k) => [k ? "→" : null, h("span", { text: S.modules.get(step.module).title })])),
            ),
          ),
      ),
    ]),
  );
  $("methods-dialog").showModal();
}

const STAGE_WORDS = { Image: "Change the image", Mask: "Mark the pattern", Contours: "Make and change contours" };

function openModules() {
  put($("modules"), 
    ...S.meta.stages.flatMap((stage) => [
      h("div", { class: "group-name", text: STAGE_WORDS[stage] || stage }),
      h(
        "div",
        { class: "cards" },
        S.meta.modules
          .filter((module) => module.stage === stage)
          .map((module) =>
            h(
              "button",
              {
                type: "button",
                class: "card module",
                onclick: () => {
                  $("steps-dialog").close();
                  addStep(module.name);
                },
              },
              h("b", { text: module.title }),
              h("span", { class: "text", text: module.summary }),
            ),
          ),
      ),
    ]),
  );
  $("steps-dialog").showModal();
}

// --- Saving results -------------------------------------------------------------------------------------------

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
  if (tab.pinned) {
    const live = [...tab.sets.values()].some((entry) => entry.set === tab.pinned.set);
    if (!live) View.dropSet(tab.pinned.set);
    tab.pinned = null;
  } else {
    const view = shown(tab);
    const set = view.contours && setOf(tab, view.contours);
    if (!set) return;
    tab.pinned = { set, label: tab.ranName || flow(tab).name || "method", median: summary(set, "diameter").median };
    S.show.reference = true;
  }
  shownChanged();
}

function toggle(layer) {
  S.show[layer] = !S.show[layer];
  remember();
  shownChanged();
}

function wire() {
  $("open-first").onclick = openDialog;
  $("help").onclick = () => $("help-dialog").showModal();
  $("pixel-nm").oninput = (event) => {
    const value = event.target.valueAsNumber;
    S.pixelNm = Number.isFinite(value) && value > 0 ? value : 0;
    remember();
    shownChanged();
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
  $("display").onclick = (event) => {
    const pop = $("display-pop");
    if (!pop.hidden) return (pop.hidden = true);
    renderDisplay();
    place(pop, event.currentTarget);
  };
  $("line-width").oninput = (event) => {
    S.look.width = event.target.valueAsNumber;
    $("width-out").textContent = `${trim(S.look.width, 2)} px`;
    remember();
    View.draw();
  };
  $("smooth").onchange = (event) => {
    S.show.smooth = event.target.checked;
    remember();
    View.draw();
  };
  $("sync-view").onchange = (event) => {
    S.sync = event.target.checked;
    remember();
  };
  $("levels-auto").onchange = (event) => {
    const tab = S.active;
    if (!tab) return;
    const { black, white } = front(tab);
    tab.levels = { auto: event.target.checked, black, white };
    renderDisplay();
    View.draw();
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
  $("zoom-fit").onclick = () => S.active && View.fit(S.active);
  $("zoom-one").onclick = () => S.active && View.oneToOne(S.active);
  $("zoom-in").onclick = () => S.active && View.zoomBy(S.active, 1.6);
  $("zoom-out").onclick = () => S.active && View.zoomBy(S.active, 1 / 1.6);

  // the results
  $("measure").onchange = (event) => {
    S.look.measure = event.target.value;
    remember();
    renderResults();
    View.draw();
  };
  $("colour-on").onchange = (event) => {
    S.look.byMeasure = event.target.checked;
    remember();
    renderResults();
    View.draw();
  };
  $("range").onchange = (event) => {
    S.look.range[measureOf(S.look.measure).kind] = event.target.value;
    remember();
    renderResults();
    View.draw();
  };
  $("histogram").onpointermove = chartTip;
  $("histogram").onpointerleave = () => chartTip(null);
  $("pin").onclick = togglePin;
  for (const button of document.querySelectorAll("[data-export]")) button.onclick = () => exportContours(button.dataset.export);
  $("snapshot").onclick = async () => {
    const tab = S.active;
    if (tab) download(await View.picture(), `${tab.info.name.replace(/\.[^.]+$/, "")}-contours.png`);
  };

  // the method
  $("choose-method").onclick = openMethods;
  $("step-add").onclick = openModules;
  $("flow-more").onclick = (event) => {
    const tab = S.active;
    const wf = flow();
    const known = presetNamed(wf.name);
    openMenu(event.currentTarget, [
      { title: "Open a method file…", text: "A method saved here before. Or drop the file on the window.", action: () => $("flow-file").click() },
      { title: "Save the method to a file…", text: "To open it here another day, or to run it from Python without this window (Ctrl+S).", action: saveFlow },
      known && changedSince(wf) ? { title: `Back to “${wf.name}” as it came`, text: "Undo every change to its steps and settings.", action: () => replaceFlow(known.name, known.steps) } : null,
      tab && !tab.own
        ? {
            title: "Give this tab a method of its own",
            text: "Changes then apply to this image only; the other tabs keep the method they have.",
            action: () => {
              tab.own = clone(S.shared);
              remember();
              renderFlow();
              renderTabs();
            },
          }
        : null,
      tab && tab.own
        ? {
            title: "Use this method in every tab",
            text: "This tab's method becomes the one all tabs run.",
            action: () => {
              S.shared = tab.own;
              tab.own = null;
              remember();
              renderFlow();
              renderTabs();
              schedule(0);
            },
          }
        : null,
      tab && tab.own
        ? {
            title: "Go back to the method of the other tabs",
            text: "This tab's own method is dropped.",
            action: () => {
              tab.own = null;
              remember();
              renderFlow();
              renderTabs();
              schedule(0);
            },
          }
        : null,
      tab ? { title: "Duplicate this tab", text: "The same image in a second tab with a method of its own: to compare two methods on one image.", action: duplicateTab } : null,
    ]);
  };
  $("flow-file").onchange = async (event) => {
    const file = event.target.files[0];
    event.target.value = "";
    if (file) openFlow(await file.text(), file.name);
  };
  $("run-on").onchange = (event) => {
    const tab = S.active;
    if (!tab) return;
    if (event.target.value === "whole") setRegion(tab, null);
    else if (event.target.value === "view") setRegion(tab, View.shownRegion(tab));
    else {
      renderRegion();
      toast("Hold Shift and drag on the image to draw the box the method runs on.");
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
    if (!$("menu").hidden && !$("menu").contains(event.target) && !event.target.closest("#flow-more, .pagebar")) closeMenu();
    if (!$("display-pop").hidden && !$("display-pop").contains(event.target) && !event.target.closest("#display")) $("display-pop").hidden = true;
  });
  window.addEventListener("blur", closeMenu);
  window.addEventListener("resize", () => S.active && renderResults());

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
    if (!$("display-pop").hidden) return ($("display-pop").hidden = true);
    if (!modal && S.active && S.active.picked >= 0) contourPicked(S.active, -1).then(() => View.draw(true));
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
  if (act && !(event.key.startsWith("Arrow") && event.target.id !== "marks" && event.target !== document.body)) {
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
  Object.assign(S.look, kept.look || {});
  if (!measureOf(S.look.measure) || measureOf(S.look.measure).key !== S.look.measure) S.look.measure = S.meta.measures[0].key;
  if (!S.look.range || typeof S.look.range !== "object") S.look.range = {};
  const first = presetNamed(S.meta.start);
  S.shared = workflow(first.name, first.steps);
  if (kept.shared) {
    try {
      S.shared = { ...workflow(kept.shared.name, kept.shared.steps, kept.shared.origin), selected: kept.shared.selected || 0 };
    } catch {
      /* of another version of the steps: the method to start from instead */
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
  $("pixel-nm").value = S.pixelNm || "";
  wire();
  const session = await api.get("/api/session");
  S.tabs = session.tabs.map((info) => makeTab(info, here));
  renderFlow();
  renderLayers();
  renderDisplay();
  renderRegion();
  renderResults();
  renderBusy();
  renderPages();
  activate(S.tabs.find((tab) => tab.id === here.active) || S.tabs[0] || null);
  if (!S.tabs.length) View.draw();
}

window.addEventListener("DOMContentLoaded", start);
