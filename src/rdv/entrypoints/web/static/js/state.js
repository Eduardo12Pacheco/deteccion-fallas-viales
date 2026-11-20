/**
 * Viewer state: the current document, client-side filters, sorting, selection
 * and the in-memory session history.
 */

import { CLASS_ORDER } from "./format.js";

export function createStore() {
  const state = {
    model: null,
    document: null,
    summary: null,
    analysisId: null,
    filename: null,
    mediaKind: null,
    mediaFile: null,
    objectUrl: null,
    threshold: 0,
    classes: Object.fromEntries(CLASS_ORDER.map((code) => [code, true])),
    sort: { key: "score", direction: "desc" },
    selection: null,
    currentFrame: 0,
    history: [],
  };
  const listeners = new Set();

  function notify(patch) {
    for (const listener of listeners) {
      listener(state, patch);
    }
  }

  return {
    state,
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    setModel(model) {
      state.model = model;
      const recommended = Number(model?.score_threshold);
      state.threshold = Number.isFinite(recommended) ? recommended : 0;
      notify({ type: "model" });
    },
    setResult(payload, file, objectUrl) {
      const document = payload.result.document;
      state.document = document;
      state.summary = document.summary;
      state.analysisId = payload.id;
      state.filename = payload.filename || file?.name || "upload";
      state.mediaKind = payload.kind;
      state.mediaFile = file ?? null;
      state.objectUrl = objectUrl ?? null;
      state.selection = null;
      state.currentFrame = 0;
      state.threshold = Number(document.params?.score_threshold ?? state.threshold);
      for (const code of CLASS_ORDER) {
        state.classes[code] = true;
      }
      state.history = state.history.filter((entry) => entry.id !== payload.id);
      state.history.unshift({
        id: payload.id,
        filename: state.filename,
        mediaKind: payload.kind,
        document,
        summary: state.summary,
        objectUrl,
        file,
      });
      state.history = state.history.slice(0, 8);
      notify({ type: "result" });
    },
    restore(entry) {
      state.document = entry.document;
      state.summary = entry.summary;
      state.analysisId = entry.id;
      state.filename = entry.filename;
      state.mediaKind = entry.mediaKind;
      state.mediaFile = entry.file;
      state.objectUrl = entry.objectUrl;
      state.selection = null;
      state.currentFrame = 0;
      state.threshold = Number(entry.document.params?.score_threshold ?? state.threshold);
      for (const code of CLASS_ORDER) {
        state.classes[code] = true;
      }
      notify({ type: "result" });
    },
    setThreshold(value) {
      state.threshold = Number(value);
      notify({ type: "filters" });
    },
    toggleClass(code) {
      state.classes[code] = !state.classes[code];
      notify({ type: "filters" });
    },
    setSort(key) {
      if (state.sort.key === key) {
        state.sort = {
          key,
          direction: state.sort.direction === "asc" ? "desc" : "asc",
        };
      } else {
        state.sort = { key, direction: key === "score" ? "desc" : "asc" };
      }
      notify({ type: "sort" });
    },
    select(selection) {
      state.selection = selection;
      notify({ type: "selection" });
    },
    setCurrentFrame(framePosition) {
      state.currentFrame = framePosition;
      notify({ type: "frame" });
    },
    reset() {
      state.document = null;
      state.summary = null;
      state.analysisId = null;
      state.filename = null;
      state.mediaKind = null;
      state.mediaFile = null;
      state.objectUrl = null;
      state.selection = null;
      state.currentFrame = 0;
      notify({ type: "reset" });
    },
  };
}

/** Flatten document detections and apply the class/confidence filters. */
export function visibleDetections(state) {
  const document = state.document;
  if (!document) {
    return [];
  }
  const rows = [];
  document.frames.forEach((frame, framePosition) => {
    frame.detections.forEach((detection, index) => {
      if (!state.classes[detection.class]) {
        return;
      }
      if (Number(detection.score) < state.threshold) {
        return;
      }
      rows.push({
        ...detection,
        frameIndex: frame.index,
        framePosition,
        t: frame.t,
        detectionIndex: index,
      });
    });
  });
  return sortRows(rows, state.sort);
}

/** Sort flattened rows without mutating the input. */
export function sortRows(rows, sort) {
  const factor = sort.direction === "asc" ? 1 : -1;
  const classRank = Object.fromEntries(CLASS_ORDER.map((code, index) => [code, index]));
  return rows.slice().sort((left, right) => {
    let delta = 0;
    if (sort.key === "score") {
      delta = Number(left.score) - Number(right.score);
    } else if (sort.key === "class") {
      delta = (classRank[left.class] ?? 99) - (classRank[right.class] ?? 99);
    } else if (sort.key === "frame") {
      delta = left.frameIndex - right.frameIndex;
    } else if (sort.key === "box") {
      delta = Number(left.box[0]) - Number(right.box[0]);
    }
    if (delta === 0) {
      delta = left.frameIndex - right.frameIndex;
    }
    if (delta === 0) {
      delta = left.detectionIndex - right.detectionIndex;
    }
    return delta * factor;
  });
}

/** Detections of one frame that pass the current filters. */
export function frameDetections(state, framePosition) {
  const frame = state.document?.frames?.[framePosition];
  if (!frame) {
    return [];
  }
  return frame.detections.filter(
    (detection) => state.classes[detection.class] && Number(detection.score) >= state.threshold,
  );
}

/** Per-class detection counts for the filtered rows. */
export function classCounts(rows) {
  const counts = Object.fromEntries(CLASS_ORDER.map((code) => [code, 0]));
  for (const row of rows) {
    counts[row.class] = (counts[row.class] ?? 0) + 1;
  }
  return counts;
}

/** Exact union area of rectangles, used for the descriptive coverage tile. */
export function unionArea(rectangles) {
  if (rectangles.length === 0) {
    return 0;
  }
  const xs = [...new Set(rectangles.flatMap((box) => [box[0], box[2]]))].sort(
    (left, right) => left - right,
  );
  let total = 0;
  for (let index = 0; index < xs.length - 1; index += 1) {
    const left = xs[index];
    const right = xs[index + 1];
    if (right <= left) {
      continue;
    }
    const intervals = rectangles
      .filter((box) => box[0] <= left && box[2] >= right)
      .map((box) => [box[1], box[3]])
      .sort((first, second) => first[0] - second[0]);
    let covered = 0;
    let currentStart = null;
    let currentEnd = null;
    for (const [start, end] of intervals) {
      if (currentStart === null) {
        currentStart = start;
        currentEnd = end;
      } else if (start <= currentEnd) {
        currentEnd = Math.max(currentEnd, end);
      } else {
        covered += currentEnd - currentStart;
        currentStart = start;
        currentEnd = end;
      }
    }
    if (currentStart !== null) {
      covered += currentEnd - currentStart;
    }
    total += covered * (right - left);
  }
  return total;
}
