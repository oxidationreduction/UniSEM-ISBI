/* Cached-prediction browser: no network inference, uploads, or dependencies. */
"use strict";

const $ = (selector) => document.querySelector(selector);
const assets = "assets/";
const state = { data: null, caseId: null, left: null, right: null, split: 50, opacity: 100, zoom: 1, x: 0, y: 0, drag: null, filmPlaying: false };

function element(tag, value, className) {
  const node = document.createElement(tag);
  if (value !== undefined) node.textContent = value;
  if (className) node.className = className;
  return node;
}

function metricCell(row, key, ranks) {
  const cell = element("td");
  const value = row[key].toFixed(4);
  cell.append(element(value === ranks[0] ? "strong" : value === ranks[1] ? "u" : "span", value));
  if (row[key + "_sd"] !== undefined) cell.append(element("span", "± " + row[key + "_sd"].toFixed(4), "sd"));
  return cell;
}

function createTable(target, headings, rows, fields, options = {}) {
  const table = element("table");
  const caption = element("caption", options.caption || "Experimental results");
  caption.className = "sr-only";
  table.append(caption);
  const header = element("tr");
  headings.forEach((heading) => { const th = element("th", heading); th.scope = "col"; header.append(th); });
  const thead = element("thead"); thead.append(header); table.append(thead);
  const tbody = element("tbody");
  const ranked = {};
  for (const field of fields) {
    if (["voi", "voi_merge", "arand"].includes(field)) ranked[field] = [...new Set(rows.map((row) => row[field].toFixed(4)))].sort((a, b) => Number(a) - Number(b));
  }
  rows.forEach((row) => {
    const tr = element("tr");
    if (options.highlight && options.highlight(row)) tr.className = "highlight-row";
    fields.forEach((field) => {
      let cell;
      if (ranked[field]) cell = metricCell(row, field, ranked[field]);
      else if (["stage1", "stage2", "source", "residual", "fusion", "early"].includes(field)) {
        cell = element("td", row[field] ? "✓" : "—", "check-cell");
        cell.setAttribute("aria-label", row[field] ? "Yes" : "No");
      } else cell = element("td", field === "parameters_m" ? row[field].toFixed(2) : String(row[field]));
      tr.append(cell);
    });
    tbody.append(tr);
  });
  table.append(tbody);
  target.replaceChildren(table);
}

function renderEvidence(data) {
  function benchmark(name) {
    createTable($("#baseline-table"), ["Method", "VOIsum ↓", "VOIm ↓", "ARAND ↓"], data.baseline[name], ["name", "voi", "voi_merge", "arand"], {caption: name === "emneuron" ? "EMNeuron external comparisons" : "WaferEM external comparisons", highlight: (row) => row.name.startsWith("UniSEM")});
    $("#baseline-note").textContent = name === "emneuron" ? "EMNeuron · 1,003 slices: mean within each volume, then equal weight over 8 volumes. Source UniSEM uses the frozen source checkpoint." : "WaferEM · means over the fixed 125-slice test volume. UniSEM LightFT and FullFT each summarize 3 seeds; external methods retain their recorded checkpoints.";
    document.querySelectorAll("[data-benchmark]").forEach((button) => { const active = button.dataset.benchmark === name; button.classList.toggle("active", active); button.setAttribute("aria-pressed", String(active)); });
  }
  benchmark("emneuron");
  document.querySelectorAll("[data-benchmark]").forEach((button) => button.addEventListener("click", () => benchmark(button.dataset.benchmark)));
  createTable($("#structure-table"), ["Residual", "Fusion", "Early", "VOIsum ↓", "ARAND ↓"], data.structure, ["residual", "fusion", "early", "voi", "arand"], {caption: "Dense structure configurations", highlight: (row) => row.early});
  createTable($("#stages-table"), ["Stage 1", "Stage 2", "VOIsum ↓", "ARAND ↓"], data.stages, ["stage1", "stage2", "voi", "arand"], {caption: "Neural training stages", highlight: (row) => row.stage1 && row.stage2});
  createTable($("#annotation-table"), ["Stage 1", "Slices", "VOIsum ↓", "ARAND ↓"], data.annotation, ["source", "slices", "voi", "arand"], {caption: "Target annotation budget", highlight: (row) => row.source && row.slices === 160});
  createTable($("#parameters-table"), ["Updated components", "M", "VOIsum ↓", "ARAND ↓"], data.parameters, ["name", "parameters_m", "voi", "arand"], {caption: "Trainable neural parameters", highlight: (row) => row.lightft});
}

function currentCase() { return state.data.cases.find((item) => item.id === state.caseId); }

function caseShape(item) {
  return item.crop_shape || item.shape || [item.crop_yxyx[2] - item.crop_yxyx[0], item.crop_yxyx[3] - item.crop_yxyx[1]];
}

function availableMethods(item) {
  return (item.methods || Object.keys(item.images)).filter((key) => item.images[key] && state.data.methods[key]);
}

function caseDefaults(item) {
  const keys = availableMethods(item);
  return item.defaults || {left: keys.includes("sam625") ? "sam625" : keys[0], right: keys.includes("source160") ? "source160" : keys[keys.length - 1]};
}

function renderCase() {
  const item = currentCase();
  const methods = state.data.methods;
  const keys = availableMethods(item);
  const defaults = caseDefaults(item);
  const [height, width] = caseShape(item);
  for (const side of ["left", "right"]) {
    if (!keys.includes(state[side])) state[side] = defaults[side];
    const key = state[side];
    const select = $("#" + side + "-select");
    select.replaceChildren(...keys.map((method) => {
      const option = element("option", methods[method]);
      option.value = method;
      return option;
    }));
    const image = $("#" + side + "-image");
    image.src = assets + item.images[key];
    image.alt = methods[key] + ", " + item.title + ", slice " + item.z;
    image.width = width; image.height = height;
    $("#" + side + "-label").textContent = methods[key];
    select.value = key;
    const original = $("#" + side + "-original");
    const download = $("#" + side + "-download");
    original.href = download.href = image.src;
    original.textContent = side[0].toUpperCase() + side.slice(1) + ": " + methods[key] + " · " + width + " × " + height + " PNG ↗";
    original.setAttribute("aria-label", "Open full-resolution " + side + " image: " + methods[key]);
    download.download = item.images[key].split("/").pop();
    download.setAttribute("aria-label", "Download full-resolution " + side + " image: " + methods[key]);
  }
  document.querySelectorAll(".raw-base").forEach((image) => {
    image.src = assets + item.images.raw;
    image.width = width; image.height = height;
  });
  $("#compare-viewport").style.aspectRatio = width + " / " + height;
  $("#case-kicker").textContent = "CASE " + (state.data.cases.indexOf(item) + 1).toString().padStart(2, "0") + " / SLICE " + item.z;
  $("#case-title").textContent = item.title;
  $("#case-description").textContent = item.interpretation;
  $("#case-dataset").textContent = item.dataset || state.data.dataset;
  $("#case-volume").textContent = item.volume || state.data.volume;
  const seed = item.seed !== undefined ? item.seed : state.data.seed;
  $("#case-seed-row").hidden = seed === null || seed === undefined;
  $("#case-seed").textContent = seed ?? "";
  $("#crop-coordinates").textContent = "[" + item.crop_yxyx.join(", ") + ") · " + width + " × " + height + " px";
  $("#selection-note").textContent = item.selection_note || state.data.selection.slices + " " + state.data.selection.crops;
  const fullShape = item.full_slice_shape || state.data.full_slice_shape;
  const isFullSlice = fullShape[0] === height && fullShape[1] === width && item.crop_yxyx[0] === 0 && item.crop_yxyx[1] === 0;
  $("#case-metric-note").textContent = isFullSlice
    ? "The values below describe this entire " + width + " × " + height + " slice."
    : "The values below describe the full " + fullShape[1] + " × " + fullShape[0] + " slice, not this " + width + " × " + height + " crop.";
  renderWholeSliceMetrics();
  updateView();
}

function renderWholeSliceMetrics() {
  const item = currentCase();
  const rows = [];
  for (const side of ["left", "right"]) {
    const key = state[side];
    const metrics = item.whole_slice_metrics[key];
    if (metrics) rows.push({name: (side === "left" ? "L · " : "R · ") + state.data.methods[key], voi: metrics.VOI_sum, arand: metrics.ARAND});
  }
  if (rows.length) createTable($("#case-metrics"), ["Selected prediction", "VOIsum ↓", "ARAND ↓"], rows, ["name", "voi", "arand"], {caption: "Recorded whole-slice metrics"});
  else $("#case-metrics").replaceChildren(element("p", "Choose a prediction to see its existing whole-slice metrics.", "fine-print"));
}

function updateView() {
  const viewport = $("#compare-viewport");
  viewport.classList.toggle("zoomed", state.zoom > 1);
  const maxPanX = (state.zoom - 1) * viewport.clientWidth / 2;
  const maxPanY = (state.zoom - 1) * viewport.clientHeight / 2;
  state.x = Math.max(-maxPanX, Math.min(maxPanX, state.x));
  state.y = Math.max(-maxPanY, Math.min(maxPanY, state.y));
  viewport.style.setProperty("--split", state.split + "%");
  viewport.style.setProperty("--zoom", state.zoom);
  viewport.style.setProperty("--pan-x", state.x + "px");
  viewport.style.setProperty("--pan-y", state.y + "px");
  for (const side of ["left", "right"]) $("#" + side + "-image").style.opacity = ["raw", "gt"].includes(state[side]) ? 1 : state.opacity / 100;
  $("#opacity-value").textContent = state.opacity + "%";
  $("#zoom-value").textContent = state.zoom.toFixed(2).replace(/\.?0+$/, "") + "×";
  const bar = $("#scale-bar span");
  const item = currentCase();
  const width = caseShape(item)[1];
  const hasPhysicalScale = Number.isFinite(item.xy_nm) && item.xy_nm > 0;
  const lengthPixels = hasPhysicalScale ? 500 / item.xy_nm : 100;
  const barWidth = lengthPixels / width * viewport.clientWidth * state.zoom;
  $("#scale-bar small").textContent = hasPhysicalScale ? "500 nm" : "100 px";
  bar.style.width = barWidth + "px";
  $("#scale-bar").hidden = barWidth > viewport.clientWidth * 0.65;
}

function resetView() {
  state.split = 50; state.opacity = 100; state.zoom = 1; state.x = 0; state.y = 0;
  $("#wipe-range").value = "50"; $("#opacity-range").value = "100"; $("#zoom-range").value = "1";
  updateView();
}

function initViewer(data) {
  state.data = data;
  state.caseId = data.default_case_id || data.cases[0].id;
  const caseSelect = $("#case-select");
  caseSelect.replaceChildren(...data.cases.map((item) => {
    const [height, width] = caseShape(item);
    const dataset = (item.dataset || data.dataset).split(",")[0];
    const label = dataset + (item.volume ? " " + item.volume : "") + " · z" + item.z + " · " + width + " px";
    const option = element("option", label);
    option.value = item.id;
    return option;
  }));
  caseSelect.value = state.caseId;
  Object.assign(state, caseDefaults(currentCase()));
  for (const side of ["left", "right"]) {
    const select = $("#" + side + "-select");
    select.addEventListener("change", () => { state[side] = select.value; renderCase(); });
  }
  $("#case-select").addEventListener("change", (event) => {
    state.caseId = event.target.value;
    Object.assign(state, caseDefaults(currentCase()));
    resetView(); renderCase();
  });
  $("#wipe-range").addEventListener("input", (event) => { state.split = Number(event.target.value); updateView(); });
  $("#opacity-range").addEventListener("input", (event) => { state.opacity = Number(event.target.value); updateView(); });
  $("#zoom-range").addEventListener("input", (event) => { state.zoom = Number(event.target.value); updateView(); });
  $("#reset-view").addEventListener("click", resetView);
  document.querySelectorAll("[data-left-reference]").forEach((button) => button.addEventListener("click", () => { state.left = button.dataset.leftReference; renderCase(); }));
  const viewport = $("#compare-viewport");
  viewport.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    const split = event.target.closest("#split-handle");
    if (!split && state.zoom === 1) return;
    state.drag = {kind: split ? "split" : "pan", pointerId: event.pointerId, startX: event.clientX, startY: event.clientY, x: state.x, y: state.y};
    viewport.setPointerCapture(event.pointerId);
    viewport.classList.add("dragging");
    event.preventDefault();
  });
  viewport.addEventListener("pointermove", (event) => {
    if (!state.drag || state.drag.pointerId !== event.pointerId) return;
    if (state.drag.kind === "split") {
      const rect = viewport.getBoundingClientRect();
      state.split = Math.max(0, Math.min(100, (event.clientX - rect.left) / rect.width * 100));
      $("#wipe-range").value = String(state.split);
    } else { state.x = state.drag.x + event.clientX - state.drag.startX; state.y = state.drag.y + event.clientY - state.drag.startY; }
    updateView();
  });
  function stopDrag() { state.drag = null; viewport.classList.remove("dragging"); }
  viewport.addEventListener("pointerup", stopDrag); viewport.addEventListener("pointercancel", stopDrag); viewport.addEventListener("lostpointercapture", stopDrag);
  $("#split-handle").addEventListener("keydown", (event) => {
    if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
      state.split = event.key === "Home" ? 0 : event.key === "End" ? 100 : Math.max(0, Math.min(100, state.split + (event.key === "ArrowLeft" ? -5 : 5)));
      $("#wipe-range").value = String(state.split); updateView(); event.preventDefault();
    }
  });
  viewport.addEventListener("keydown", (event) => {
    if (event.target !== viewport) return;
    if (event.key === "Escape") { resetView(); return; }
    if (state.zoom > 1 && ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) {
      state.x += event.key === "ArrowLeft" ? 25 : event.key === "ArrowRight" ? -25 : 0;
      state.y += event.key === "ArrowUp" ? 25 : event.key === "ArrowDown" ? -25 : 0;
      updateView(); event.preventDefault();
    }
  });
  window.addEventListener("resize", updateView);
  renderCase();
}

function setFilm(playing) {
  const image = $("#comparison-film");
  state.filmPlaying = playing;
  image.src = playing ? image.dataset.animation : image.dataset.poster;
  $("#film-toggle").textContent = playing ? "Pause comparison Ⅱ" : "Play comparison ▶";
  $("#film-toggle").setAttribute("aria-pressed", String(playing));
}

$("#film-toggle").addEventListener("click", () => setFilm(!state.filmPlaying));
const motionPreference = window.matchMedia("(prefers-reduced-motion: reduce)");
motionPreference.addEventListener("change", () => { if (motionPreference.matches) setFilm(false); });
document.addEventListener("visibilitychange", () => { if (document.hidden) setFilm(false); });


function initEvidenceTabs() {
  const tabs = [...document.querySelectorAll("[data-evidence]")];
  function selectTab(index, moveFocus = false) {
    tabs.forEach((tab, position) => {
      const active = position === index;
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
      document.getElementById(tab.getAttribute("aria-controls")).hidden = !active;
    });
    if (moveFocus) tabs[index].focus();
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => selectTab(index));
    tab.addEventListener("keydown", (event) => {
      let next;
      if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
      else if (event.key === "ArrowLeft") next = (index + tabs.length - 1) % tabs.length;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = tabs.length - 1;
      else return;
      event.preventDefault();
      selectTab(next, true);
    });
  });
}
initEvidenceTabs();

async function start() {
  try {
    const [resultsResponse, casesResponse] = await Promise.all([fetch("data/results.json"), fetch(assets + "cases.json")]);
    if (!resultsResponse.ok || !casesResponse.ok) throw new Error("Missing frozen data assets");
    const [results, cases] = await Promise.all([resultsResponse.json(), casesResponse.json()]);
    renderEvidence(results); initViewer(cases);
  } catch (error) {
    const notice = element("p", "The frozen data could not be loaded. Serve this directory with a local HTTP server; opening index.html directly may block the viewer. The static examples and full-resolution PNG links remain available on this page.", "notice");
    $("#explore .section-shell").prepend(notice);
    console.error(error);
  }
}
start();
