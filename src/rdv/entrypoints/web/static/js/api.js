/**
 * Thin fetch wrappers for the local analysis API.
 *
 * Every call is same-origin; the server enforces loopback Host/Origin checks.
 */

async function readJson(response) {
  let body = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (!response.ok) {
    const detail = body && typeof body.detail === "string" ? body.detail : null;
    const error = new Error(detail || `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return body;
}

/** Fetch model identity and the recommended score threshold. */
export async function fetchModel() {
  return readJson(await fetch("/api/model", { headers: { Accept: "application/json" } }));
}

/** Upload one file and queue an analysis. */
export async function createAnalysis(file, analysisFps) {
  const form = new FormData();
  form.append("file", file, file.name || "upload");
  if (analysisFps) {
    form.append("analysis_fps", String(analysisFps));
  }
  const response = await fetch("/api/analyses", { method: "POST", body: form });
  return readJson(response);
}

/** Fetch the current state and, once finished, the inline result. */
export async function fetchAnalysis(id) {
  return readJson(
    await fetch(`/api/analyses/${encodeURIComponent(id)}`, {
      headers: { Accept: "application/json" },
    }),
  );
}

/** Absolute URL of the detections JSON artifact. */
export function reportJsonUrl(id) {
  return `/api/analyses/${encodeURIComponent(id)}/report.json`;
}

/** Absolute URL of the self-contained HTML artifact. */
export function reportHtmlUrl(id) {
  return `/api/analyses/${encodeURIComponent(id)}/report.html`;
}

/** Delete a finished analysis and its temporary files. */
export async function deleteAnalysis(id) {
  const response = await fetch(`/api/analyses/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!response.ok && response.status !== 404) {
    await readJson(response);
  }
}
