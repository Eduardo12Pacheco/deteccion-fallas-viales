/**
 * Entry point: wires upload/drop, polling, the canvas overlay, filters,
 * sorting, the video timeline, session history and keyboard shortcuts.
 */

import { createAnalysis, fetchAnalysis, fetchModel } from "./api.js";
import {
  CLASS_ORDER,
  classColor,
  className,
  clear,
  el,
  formatCount,
  formatPercent,
  formatScore,
} from "./format.js";
import { wireDownloads } from "./report.js";
import {
  classCounts,
  createStore,
  frameDetections,
  unionArea,
  visibleDetections,
} from "./state.js";
import { initTheme } from "./theme.js";
import { createVideoController } from "./video.js";
import { configureViewer, drawOverlay, hitTest } from "./viewer.js";

const POLL_INTERVAL_MS = 500;
const MAX_TABLE_ROWS = 500;
const MAX_TIMELINE_FRAMES = 240;

const store = createStore();
const dom = {
  status: document.getElementById("status"),
  modelName: document.getElementById("model-name"),
  modelDevice: document.getElementById("model-device"),
  themeToggle: document.getElementById("theme-toggle"),
  dropSection: document.getElementById("drop-section"),
  dropzone: document.getElementById("dropzone"),
  browse: document.getElementById("browse-button"),
  fileInput: document.getElementById("file-input"),
  progressPanel: document.getElementById("progress-panel"),
  progressBar: document.getElementById("progress-bar"),
  progressFill: document.getElementById("progress-fill"),
  progressMessage: document.getElementById("progress-message"),
  progressValue: document.getElementById("progress-value"),
  uploadError: document.getElementById("upload-error"),
  uploadErrorMessage: document.getElementById("upload-error-message"),
  resultView: document.getElementById("result-view"),
  resultFilename: document.getElementById("result-filename"),
  resultMeta: document.getElementById("result-meta"),
  imageViewer: document.getElementById("image-viewer"),
  videoViewer: document.getElementById("video-viewer"),
  canvas: document.getElementById("overlay-canvas"),
  tooltip: document.getElementById("canvas-tooltip"),
  viewer: document.getElementById("viewer"),
  videoControls: document.getElementById("video-controls"),
  timeline: document.getElementById("density-timeline"),
  framePosition: document.getElementById("frame-position"),
  emptyState: document.getElementById("empty-state"),
  emptyTitle: document.getElementById("empty-title"),
  emptyText: document.getElementById("empty-text"),
  tiles: {
    total: document.getElementById("tile-total"),
    max: document.getElementById("tile-max"),
    coverage: document.getElementById("tile-coverage"),
    extra: document.getElementById("tile-extra"),
    extraLabel: document.getElementById("tile-extra-label"),
  },
  summaryScope: document.getElementById("summary-scope"),
  legend: document.getElementById("class-legend"),
  slider: document.getElementById("confidence-slider"),
  sliderValue: document.getElementById("confidence-value"),
  tableBody: document.getElementById("detections-body"),
  tableCount: document.getElementById("table-count"),
  downloadPng: document.getElementById("download-png"),
  downloadJson: document.getElementById("download-json"),
  downloadHtml: document.getElementById("download-html"),
  reset: document.getElementById("reset-button"),
  historySection: document.getElementById("history-section"),
  historyList: document.getElementById("history-list"),
};

let pollTimer = null;
let uploadToken = 0;
let videoController = null;
let overlayRows = [];
let visibleRows = [];

initTheme();
loadModel();
bindUpload();
bindShortcuts();
bindStaticEvents();
renderHistory();

function setStatus(message) {
  dom.status.textContent = message;
}

function hideError() {
  dom.uploadError.hidden = true;
  dom.uploadErrorMessage.textContent = "";
}

function showError(message) {
  dom.uploadError.hidden = false;
  dom.uploadErrorMessage.textContent = message;
  setStatus("Upload failed.");
}

function setBusy(busy) {
  dom.dropzone.classList.toggle("is-busy", busy);
  dom.dropzone.setAttribute("aria-busy", busy ? "true" : "false");
  dom.browse.disabled = busy;
}

function setProgress(percent, message, indeterminate = false) {
  dom.progressPanel.hidden = false;
  const bounded = Math.max(0, Math.min(100, Math.round(percent)));
  dom.progressBar.classList.toggle("is-indeterminate", indeterminate);
  dom.progressBar.setAttribute("aria-valuenow", String(bounded));
  dom.progressFill.style.width = `${indeterminate ? 35 : bounded}%`;
  dom.progressValue.textContent = `${bounded}%`;
  dom.progressMessage.textContent = message;
}

function hideProgress() {
  dom.progressPanel.hidden = true;
  dom.progressBar.classList.remove("is-indeterminate");
}

async function loadModel() {
  try {
    const model = await fetchModel();
    store.setModel(model);
    dom.modelName.textContent = model.id;
    dom.modelDevice.textContent = model.device;
    dom.slider.value = String(store.state.threshold);
    dom.sliderValue.textContent = Number(store.state.threshold).toFixed(2);
    setStatus(`Model ${model.id} ready on ${model.device}.`);
  } catch (error) {
    dom.modelName.textContent = "model unavailable";
    dom.modelDevice.textContent = "—";
    setStatus(error.message || "The model is not available. Run rdv models pull.");
  }
}

function bindStaticEvents() {
  dom.reset.addEventListener("click", resetView);
  dom.downloadPng.addEventListener("click", () => {
    /* Replaced by wireDownloads for each result. */
  });
}

function openPicker() {
  dom.fileInput.click();
}

function bindUpload() {
  dom.dropzone.addEventListener("click", (event) => {
    if (event.target === dom.browse || dom.browse.contains(event.target)) {
      return;
    }
    openPicker();
  });
  dom.dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openPicker();
    }
  });
  dom.browse.addEventListener("click", (event) => {
    event.stopPropagation();
    openPicker();
  });
  dom.fileInput.addEventListener("change", () => {
    const [file] = dom.fileInput.files;
    dom.fileInput.value = "";
    if (file) {
      handleFile(file);
    }
  });
  for (const name of ["dragenter", "dragover"]) {
    dom.dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dom.dropzone.classList.add("is-dragover");
    });
  }
  for (const name of ["dragleave", "dragend"]) {
    dom.dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dom.dropzone.classList.remove("is-dragover");
    });
  }
  dom.dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dom.dropzone.classList.remove("is-dragover");
    const file = event.dataTransfer?.files?.[0];
    if (file) {
      handleFile(file);
    }
  });
}

function stopPolling() {
  if (pollTimer !== null) {
    window.clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function handleFile(file) {
  stopPolling();
  uploadToken += 1;
  const token = uploadToken;
  hideError();
  setBusy(true);
  dom.resultView.hidden = true;
  dom.dropzone.hidden = true;
  setProgress(0, "Uploading…", true);
  setStatus(`Uploading ${file.name}…`);
  const objectUrl = URL.createObjectURL(file);
  try {
    const created = await createAnalysis(file);
    if (token !== uploadToken) {
      return;
    }
    setStatus(`Analyzing ${file.name}…`);
    setProgress(35, "Analysis queued", false);
    pollTimer = window.setInterval(() => {
      pollAnalysis(created.id, file, objectUrl, token);
    }, POLL_INTERVAL_MS);
    pollAnalysis(created.id, file, objectUrl, token);
  } catch (error) {
    if (token !== uploadToken) {
      return;
    }
    setBusy(false);
    dom.dropzone.hidden = false;
    hideProgress();
    showError(error.message || "The upload could not be analyzed.");
  }
}

async function pollAnalysis(id, file, objectUrl, token) {
  try {
    const payload = await fetchAnalysis(id);
    if (token !== uploadToken) {
      return;
    }
    const fraction = payload.progress?.fraction;
    if (payload.status === "queued" || payload.status === "running") {
      if (typeof fraction === "number") {
        setProgress(15 + fraction * 80, payload.progress.message || "Analyzing…");
      } else {
        setProgress(20, payload.progress?.message || "Analyzing…", true);
      }
      return;
    }
    stopPolling();
    if (payload.status === "error") {
      setBusy(false);
      dom.dropzone.hidden = false;
      hideProgress();
      showError(payload.error || "Analysis failed.");
      return;
    }
    setProgress(100, "Analysis complete");
    store.setResult(payload, file, objectUrl);
    await presentResult();
    setBusy(false);
    const total = store.state.document?.summary?.counts
      ? Object.values(store.state.document.summary.counts).reduce((sum, value) => sum + value, 0)
      : visibleRows.length;
    setStatus(
      total > 0
        ? `Analysis complete · ${formatCount(total)} detection${total === 1 ? "" : "s"}.`
        : "Analysis complete · no damage detected.",
    );
  } catch (error) {
    if (token !== uploadToken) {
      return;
    }
    stopPolling();
    setBusy(false);
    dom.dropzone.hidden = false;
    hideProgress();
    showError(error.message || "The analysis could not be completed.");
  }
}

function teardownViewer() {
  if (videoController) {
    videoController.destroy();
    videoController = null;
  }
  dom.videoViewer.pause();
  dom.videoViewer.removeAttribute("src");
  dom.videoViewer.hidden = true;
  dom.imageViewer.hidden = true;
  dom.videoControls.hidden = true;
  clear(dom.timeline);
}

function resetView() {
  stopPolling();
  uploadToken += 1;
  teardownViewer();
  store.reset();
  visibleRows = [];
  overlayRows = [];
  hideError();
  hideProgress();
  setBusy(false);
  dom.dropzone.hidden = false;
  dom.resultView.hidden = true;
  dom.emptyState.hidden = true;
  detachImage();
  document.getElementById("result-filename").textContent = "—";
  setStatus("Ready.");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function detachImage() {
  dom.imageViewer.removeAttribute("src");
  dom.canvas.width = 0;
  dom.canvas.height = 0;
}

async function presentResult() {
  const state = store.state;
  const document_ = state.document;
  if (!document_) {
    return;
  }
  teardownViewer();
  dom.resultView.hidden = false;
  dom.resultView.dataset.analysisId = state.analysisId;
  dom.emptyState.hidden = true;
  dom.resultFilename.textContent = state.filename;
  dom.resultMeta.textContent = `${state.mediaKind} · ${document_.source.width}×${document_.source.height} · ${document_.model.id}`;
  configureViewer(dom.viewer, document_);
  dom.slider.value = String(state.threshold);
  dom.sliderValue.textContent = Number(state.threshold).toFixed(2);

  if (state.mediaKind === "video") {
    dom.imageViewer.hidden = true;
    dom.videoViewer.hidden = false;
    dom.videoControls.hidden = false;
    dom.videoViewer.src = state.objectUrl;
    dom.videoViewer.load();
    videoController = createVideoController({
      video: dom.videoViewer,
      document: document_,
      timeline: dom.timeline,
      position: dom.framePosition,
      onChange: (framePosition) => store.setCurrentFrame(framePosition),
    });
    pruneTimeline();
  } else {
    dom.videoControls.hidden = true;
    dom.videoViewer.hidden = true;
    dom.imageViewer.hidden = false;
    dom.imageViewer.src = state.objectUrl;
    try {
      await dom.imageViewer.decode();
    } catch {
      /* onload path already renders; a decode error shows boxes only. */
    }
  }

  buildLegend();
  const rowsNow = visibleDetections(state);
  wireDownloads({
    id: state.analysisId,
    filename: state.filename,
    kind: state.mediaKind,
    jsonLink: dom.downloadJson,
    htmlLink: dom.downloadHtml,
    pngButton: dom.downloadPng,
    getSource: () => (state.mediaKind === "video" ? dom.videoViewer : dom.imageViewer),
    getDocument: () => store.state.document,
    getRows: () => overlayRows,
  });
  renderFilters(rowsNow);
  renderHistory();
}

function pruneTimeline() {
  const bars = dom.timeline.querySelectorAll(".timeline-bar");
  if (bars.length <= MAX_TIMELINE_FRAMES) {
    return;
  }
  bars.forEach((bar, index) => {
    if (index % Math.ceil(bars.length / MAX_TIMELINE_FRAMES) !== 0) {
      bar.hidden = true;
    }
  });
}

function buildLegend() {
  clear(dom.legend);
  for (const code of CLASS_ORDER) {
    const button = el("button", {
      type: "button",
      className: "legend-toggle",
      "data-class": code,
      "data-testid": `legend-${code}`,
      "aria-pressed": "true",
      title: className(code),
    });
    const swatch = el("span", { className: "swatch", "aria-hidden": "true" });
    swatch.style.setProperty("--swatch", classColor(code));
    button.append(
      swatch,
      el("span", { className: "legend-name", text: `${code} · ${className(code)}` }),
      el("span", { className: "legend-count", "data-count": code, text: "0" }),
    );
    button.addEventListener("click", () => store.toggleClass(code));
    dom.legend.append(el("li", {}, [button]));
  }
}

function renderFilters(rows) {
  const state = store.state;
  const document_ = state.document;
  if (!document_) {
    return;
  }
  visibleRows = rows ?? visibleDetections(state);
  const counts = classCounts(visibleRows);
  const thresholdCounts = classCounts(
    document_.frames.flatMap((frame) =>
      frame.detections.filter((detection) => Number(detection.score) >= state.threshold),
    ),
  );

  for (const button of dom.legend.querySelectorAll(".legend-toggle")) {
    const code = button.dataset.class;
    button.setAttribute("aria-pressed", state.classes[code] ? "true" : "false");
    const badge = button.querySelector(`[data-count="${code}"]`);
    if (badge) {
      badge.textContent = String(thresholdCounts[code] ?? 0);
    }
  }

  dom.tiles.total.textContent = formatCount(visibleRows.length);
  const maximum = visibleRows.reduce((best, row) => Math.max(best, Number(row.score)), 0);
  dom.tiles.max.textContent = formatScore(maximum);
  if (state.mediaKind === "video") {
    const summary = state.summary || {};
    dom.tiles.coverage.textContent = formatPercent(summary.coverage_ratio ?? 0);
    const unique = summary.unique_instances || {};
    dom.tiles.extra.textContent = formatCount(
      Object.values(unique).reduce((sum, value) => sum + Number(value || 0), 0),
    );
    dom.tiles.extraLabel.textContent = "Unique instances";
    dom.summaryScope.textContent = `${summary.frames_analyzed ?? document_.frames.length} frames · ${summary.frames_with_damage ?? 0} with damage`;
  } else {
    const area = Number(document_.source.width) * Number(document_.source.height);
    const coverage = area > 0 ? unionArea(visibleRows.map((row) => row.box.map(Number))) / area : 0;
    dom.tiles.coverage.textContent = formatPercent(coverage);
    const present = Object.values(counts).filter((count) => count > 0).length;
    dom.tiles.extra.textContent = formatCount(present);
    dom.tiles.extraLabel.textContent = "Classes present";
    dom.summaryScope.textContent = `1 frame · threshold ${Number(state.threshold).toFixed(2)}`;
  }

  renderTable();
  refreshSelection();
  updateEmptyState();
}

function renderTable() {
  clear(dom.tableBody);
  const shown = visibleRows.slice(0, MAX_TABLE_ROWS);
  const fragment = document.createDocumentFragment();
  for (const row of shown) {
    const tr = el(
      "tr",
      {
        "data-testid": "detection-row",
        "data-class": row.class,
        "data-frame": row.frameIndex,
        "data-index": row.detectionIndex,
        "data-frame-position": row.framePosition,
        tabindex: "0",
        title: `${className(row.class)} · ${formatScore(row.score)}`,
      },
      [
        el("td", { text: row.frameIndex + 1 }),
        el("td", { text: row.class }),
        el("td", { className: "numeric", text: formatScore(row.score) }),
        el("td", {
          className: "mono",
          text: row.box.map((value) => Number(value).toFixed(0)).join(", "),
        }),
        el("td", {
          text: row.track_id === null || row.track_id === undefined ? "—" : String(row.track_id),
        }),
      ],
    );
    tr.addEventListener("click", () => selectRow(row));
    tr.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectRow(row);
      }
    });
    fragment.append(tr);
  }
  dom.tableBody.append(fragment);
  dom.tableCount.textContent =
    visibleRows.length > shown.length
      ? `${formatCount(shown.length)} of ${formatCount(visibleRows.length)} rows`
      : `${formatCount(visibleRows.length)} row${visibleRows.length === 1 ? "" : "s"}`;
  updateSortIndicators();
}

function updateSortIndicators() {
  const sort = store.state.sort;
  for (const button of document.querySelectorAll("th button[data-sort]")) {
    const header = button.closest("th");
    if (button.dataset.sort === sort.key) {
      header.setAttribute("aria-sort", sort.direction === "asc" ? "ascending" : "descending");
      button.dataset.direction = sort.direction;
    } else {
      header.removeAttribute("aria-sort");
      delete button.dataset.direction;
    }
  }
}

function selectRow(row) {
  store.select({ framePosition: row.framePosition, index: row.detectionIndex });
  if (store.state.mediaKind === "video" && videoController) {
    videoController.seekTo(row.framePosition, { updateVideo: true });
  }
  const selected = dom.tableBody.querySelector("tr.is-selected");
  if (selected) {
    selected.scrollIntoView({ block: "nearest" });
  }
}

function refreshSelection() {
  const state = store.state;
  const selection = state.selection;
  for (const tr of dom.tableBody.querySelectorAll("tr")) {
    const matches =
      selection &&
      Number(tr.dataset.framePosition) === selection.framePosition &&
      Number(tr.dataset.index) === selection.index;
    tr.classList.toggle("is-selected", Boolean(matches));
  }
  refreshOverlay(state);
}

function currentOverlayRows() {
  const state = store.state;
  if (!state.document) {
    return [];
  }
  if (state.mediaKind === "video" && videoController) {
    return frameDetections(state, videoController.current);
  }
  return visibleRows;
}

function refreshOverlay(state) {
  const rows = currentOverlayRows();
  overlayRows = rows;
  drawOverlay(dom.canvas, state.document, rows, state.selection);
}

function updateEmptyState() {
  const total =
    store.state.document?.frames?.reduce((sum, frame) => sum + frame.detections.length, 0) ?? 0;
  if (visibleRows.length > 0) {
    dom.emptyState.hidden = true;
    return;
  }
  dom.emptyState.hidden = false;
  if (total === 0) {
    dom.emptyTitle.textContent = "No damage detected";
    dom.emptyText.textContent =
      "The model found nothing above the detection threshold. Try a closer, well-lit photo of the road surface.";
  } else {
    dom.emptyTitle.textContent = "No detections match the filters";
    dom.emptyText.textContent =
      "Lower the confidence threshold or re-enable a class to see the detections again.";
  }
}

function renderHistory() {
  const history = store.state.history;
  dom.historySection.hidden = history.length === 0;
  clear(dom.historyList);
  for (const entry of history) {
    const button = el(
      "button",
      {
        type: "button",
        className: "history-button",
        "aria-current": entry.id === store.state.analysisId ? "true" : "false",
        title: entry.filename,
      },
      [el("span", { text: `${entry.mediaKind === "video" ? "▶" : "▧"} ${entry.filename}` })],
    );
    button.addEventListener("click", () => {
      store.restore(entry);
      presentResult();
      dom.resultView.scrollIntoView({ block: "start", behavior: "smooth" });
    });
    dom.historyList.append(el("li", {}, [button]));
  }
}

function bindShortcuts() {
  document.addEventListener("keydown", (event) => {
    const target = event.target;
    if (
      target instanceof HTMLElement &&
      (target.matches("input, textarea, select") || target.isContentEditable)
    ) {
      return;
    }
    if (event.metaKey || event.ctrlKey || event.altKey) {
      return;
    }
    if (event.key === "o") {
      event.preventDefault();
      openPicker();
    } else if (event.key === "t") {
      event.preventDefault();
      dom.themeToggle.click();
    } else if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
      if (store.state.mediaKind === "video" && videoController) {
        event.preventDefault();
        videoController.step(event.key === "ArrowRight" ? 1 : -1);
      }
    }
  });
}

dom.slider.addEventListener("input", () => {
  const value = Number(dom.slider.value);
  dom.sliderValue.textContent = value.toFixed(2);
  store.setThreshold(value);
});

for (const button of document.querySelectorAll("th button[data-sort]")) {
  button.addEventListener("click", () => store.setSort(button.dataset.sort));
}

dom.canvas.addEventListener("pointermove", (event) => {
  if (overlayRows.length === 0) {
    dom.tooltip.hidden = true;
    return;
  }
  const row = hitTest(event, dom.canvas, overlayRows);
  if (!row) {
    dom.tooltip.hidden = true;
    return;
  }
  const rect = dom.viewer.getBoundingClientRect();
  dom.tooltip.hidden = false;
  dom.tooltip.textContent = `${row.class} · ${className(row.class)} · ${formatScore(row.score)}`;
  dom.tooltip.style.left = `${event.clientX - rect.left}px`;
  dom.tooltip.style.top = `${event.clientY - rect.top}px`;
});

dom.canvas.addEventListener("pointerleave", () => {
  dom.tooltip.hidden = true;
});

dom.canvas.addEventListener("click", (event) => {
  const row = hitTest(event, dom.canvas, overlayRows);
  if (row) {
    const flattened = visibleRows.find(
      (candidate) =>
        candidate.framePosition === row.framePosition &&
        candidate.detectionIndex === row.detectionIndex,
    );
    if (flattened) {
      selectRow(flattened);
    }
  }
});

store.subscribe((_, patch) => {
  if (!store.state.document) {
    return;
  }
  if (patch.type === "filters" || patch.type === "sort") {
    renderFilters();
  } else if (patch.type === "selection" || patch.type === "frame") {
    refreshSelection();
  }
});
