/**
 * Download actions for one analysis: annotated PNG, detections JSON and the
 * self-contained HTML report.
 */

import { reportHtmlUrl, reportJsonUrl } from "./api.js";
import { downloadAnnotatedPng } from "./viewer.js";

/** Basename without directories or a trailing extension. */
export function fileStem(filename) {
  const name = String(filename || "upload")
    .replaceAll("\\", "/")
    .split("/")
    .pop();
  const dot = name.lastIndexOf(".");
  return (dot > 0 ? name.slice(0, dot) : name) || "upload";
}

export function artifactNames(filename, kind) {
  const stem = fileStem(filename);
  return {
    png: kind === "video" ? `${stem}-frame.annotated.png` : `${stem}.annotated.png`,
    json: `${stem}.detections.json`,
    html: `${stem}.report.html`,
  };
}

/**
 * Point the report links at one analysis and wire the PNG button.
 *
 * Args:
 *   id: analysis id.
 *   filename: display filename used to name downloads.
 *   kind: "image" or "video".
 *   jsonLink / htmlLink: anchor elements.
 *   pngButton: button element.
 *   source: current <img> or <video> element.
 *   document: detections document.
 *   rows: currently filtered detections.
 */
export function wireDownloads({
  id,
  filename,
  kind,
  jsonLink,
  htmlLink,
  pngButton,
  getSource,
  getDocument,
  getRows,
}) {
  const names = artifactNames(filename, kind);
  jsonLink.href = reportJsonUrl(id);
  jsonLink.download = names.json;
  jsonLink.setAttribute("aria-label", `Download detections JSON for ${filename}`);
  htmlLink.href = reportHtmlUrl(id);
  htmlLink.download = names.html;
  htmlLink.setAttribute("aria-label", `Download HTML report for ${filename}`);

  pngButton.disabled = false;
  pngButton.textContent = kind === "video" ? "Download annotated frame" : "Download annotated PNG";
  pngButton.onclick = () => {
    const document = getDocument();
    if (!document) {
      return;
    }
    downloadAnnotatedPng(getSource(), document, getRows(), names.png);
  };
}
