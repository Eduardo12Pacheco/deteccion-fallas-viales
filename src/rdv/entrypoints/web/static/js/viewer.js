/**
 * Canvas overlay rendering, hover hit-testing and annotated PNG export.
 *
 * Boxes are stored in source-image coordinates, so the overlay canvas keeps
 * the source resolution and CSS scales it over the image/video. This keeps the
 * drawn geometry identical to the JSON contract and to exported PNGs.
 */

import { classColor } from "./format.js";

function context(canvas) {
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Canvas 2D is not available in this browser");
  }
  return ctx;
}

/** Match the viewer aspect ratio to the detection document source. */
export function configureViewer(viewer, document) {
  const width = Number(document?.source?.width);
  const height = Number(document?.source?.height);
  if (width > 0 && height > 0) {
    viewer.style.setProperty("--aspect", `${width} / ${height}`);
  }
}

function drawBoxes(ctx, rows, options = {}) {
  const selected = options.selection;
  for (const row of rows) {
    const [x1, y1, x2, y2] = row.box.map(Number);
    const color = classColor(row.class);
    ctx.save();
    ctx.globalAlpha = 0.16;
    ctx.fillStyle = color;
    ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
    ctx.globalAlpha = 1;
    const isSelected =
      selected && selected.framePosition === row.framePosition && selected.index === row.index;
    ctx.lineWidth = Math.max(2, Math.round((options.width || 0) / 360));
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    if (isSelected) {
      ctx.lineWidth += 2;
      ctx.strokeStyle = "#ffffff";
      ctx.setLineDash([8, 5]);
      ctx.strokeRect(x1 - 2, y1 - 2, x2 - x1 + 4, y2 - y1 + 4);
    }
    ctx.restore();
  }
}

/** Draw the filtered detections on the overlay canvas. */
export function drawOverlay(canvas, document, rows, selection) {
  const width = Number(document?.source?.width);
  const height = Number(document?.source?.height);
  if (!(width > 0 && height > 0)) {
    return;
  }
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = context(canvas);
  ctx.clearRect(0, 0, width, height);
  drawBoxes(ctx, rows, { selection, width });
}

/** Convert a pointer event to source-image coordinates. */
export function pointerPosition(event, canvas) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };
}

/** Return the topmost detection under the pointer, if any. */
export function hitTest(event, canvas, rows) {
  const point = pointerPosition(event, canvas);
  for (let index = rows.length - 1; index >= 0; index -= 1) {
    const [x1, y1, x2, y2] = rows[index].box.map(Number);
    if (point.x >= x1 && point.x <= x2 && point.y >= y1 && point.y <= y2) {
      return rows[index];
    }
  }
  return null;
}

/** Draw a source frame plus the filtered boxes into an export canvas. */
export function renderExportCanvas(source, document, rows, options = {}) {
  const width = Number(document.source.width);
  const height = Number(document.source.height);
  const canvas = window.document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = context(canvas);
  ctx.fillStyle = "#05080c";
  ctx.fillRect(0, 0, width, height);
  if (source) {
    try {
      if (source.tagName === "VIDEO") {
        if (source.readyState >= 2) {
          ctx.drawImage(source, 0, 0, width, height);
        }
      } else if (source.complete && source.naturalWidth > 0) {
        ctx.drawImage(source, 0, 0, width, height);
      }
    } catch {
      /* A codec the browser cannot decode still exports the boxes. */
    }
  }
  drawBoxes(ctx, rows, { selection: options.selection, width });
  return canvas;
}

/** Trigger a browser download for a blob. */
export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = window.document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.rel = "noopener";
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 2000);
}

/** Export the current annotated frame at source resolution. */
export function downloadAnnotatedPng(source, document, rows, filename) {
  const canvas = renderExportCanvas(source, document, rows);
  canvas.toBlob((blob) => {
    if (blob) {
      downloadBlob(blob, filename);
    }
  }, "image/png");
}
