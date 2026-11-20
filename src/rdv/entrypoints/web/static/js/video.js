/**
 * Video overlay controller: density timeline, timeline seeking and
 * frame-by-frame stepping over the analyzed frames.
 *
 * The browser plays the user's own file; no transcoding happens. If the
 * browser cannot decode the container the detection overlay and timeline still
 * work because geometry comes from the JSON document, not from the video.
 */

import { el } from "./format.js";

function frameTime(frame, fps) {
  if (typeof frame.t === "number" && Number.isFinite(frame.t)) {
    return frame.t;
  }
  return fps > 0 ? frame.index / fps : 0;
}

export function createVideoController({ video, document, timeline, position, onChange }) {
  const frames = document.frames;
  const fps = Number(document.source.fps) || 0;
  let current = 0;
  let bars = [];
  let userSeeking = false;

  function renderTimeline() {
    timeline.replaceChildren();
    const maximum = Math.max(1, ...frames.map((frame) => frame.detections.length));
    bars = frames.map((frame, index) => {
      const height = Math.max(8, Math.round((frame.detections.length / maximum) * 100));
      const button = el("button", {
        type: "button",
        className: "timeline-bar",
        "data-testid": "timeline-bar",
        "aria-label": `Frame ${index + 1}: ${frame.detections.length} detections`,
        title: `Frame ${index + 1}: ${frame.detections.length} detections`,
      });
      button.style.setProperty("--height", `${height}%`);
      button.addEventListener("click", () => seekTo(index, { updateVideo: true }));
      return button;
    });
    timeline.append(...bars);
    markCurrent();
  }

  function markCurrent() {
    bars.forEach((bar, index) => {
      if (index === current) {
        bar.setAttribute("aria-current", "true");
      } else {
        bar.removeAttribute("aria-current");
      }
    });
    const frame = frames[current];
    const currentTime = frameTime(frame, fps);
    let text = `Frame ${current + 1} of ${frames.length}`;
    if (fps > 0) {
      text += ` · ${currentTime.toFixed(2)} s`;
    }
    if (frame.detections.length === 0) {
      text += " · no detections";
    }
    position.textContent = text;
  }

  function emit() {
    markCurrent();
    if (typeof onChange === "function") {
      onChange(current);
    }
  }

  function seekTo(index, options = {}) {
    if (index < 0 || index >= frames.length) {
      return;
    }
    current = index;
    if (options.updateVideo && video) {
      const time = frameTime(frames[current], fps);
      try {
        if (Number.isFinite(time) && Math.abs((video.currentTime || 0) - time) > 0.001) {
          userSeeking = true;
          video.currentTime = time;
        }
      } catch {
        /* Not seekable (codec unsupported); geometry still updates. */
      }
    }
    emit();
  }

  function step(delta) {
    seekTo(Math.min(frames.length - 1, Math.max(0, current + delta)), { updateVideo: true });
  }

  function followPlayback() {
    if (!video || userSeeking) {
      userSeeking = false;
      return;
    }
    const time = Number(video.currentTime) || 0;
    let nearest = 0;
    for (let index = 0; index < frames.length; index += 1) {
      if (frameTime(frames[index], fps) <= time + 0.001) {
        nearest = index;
      } else {
        break;
      }
    }
    if (nearest !== current) {
      current = nearest;
      emit();
    }
  }

  renderTimeline();
  emit();

  if (video) {
    video.addEventListener("timeupdate", followPlayback);
    video.addEventListener("seeked", followPlayback);
  }

  return {
    step,
    seekTo,
    get current() {
      return current;
    },
    get currentFrame() {
      return frames[current];
    },
    destroy() {
      if (video) {
        video.removeEventListener("timeupdate", followPlayback);
        video.removeEventListener("seeked", followPlayback);
      }
    },
  };
}
