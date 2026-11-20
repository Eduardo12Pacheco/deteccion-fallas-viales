# 0011 - Video analysis semantics, tracker limits and atomic outputs

- **Status:** Accepted
- **Date:** 2026-11-20
- **Deciders:** project owner (plan decisions D20, D21 and spec S6)
- **Task:** T7

## Context

Spec S6 requires `rdv detect <video>` to write an annotated MP4 with the same
frame count as the source, a detections document with per-analyzed-frame
results plus a summary, and to fail cleanly on truncated or corrupt input.
Detecting every frame is slow on CPU and the tracker cannot know how the camera
moved, so the pipeline needs explicit, documented semantics instead of
best-effort behavior.

## Decision

- **Analysis stride.** Only every `max(1, round(source_fps / analysis_fps))`-th
  source frame runs inference (`--analysis-fps`, default 5). The annotated MP4
  still contains every source frame; frames that were not analyzed repeat the
  detections of the most recent analyzed frame. This keeps the output timeline
  aligned with the input video while paying inference only for sampled frames.
- **Tracker.** `IouTracker` runs on analyzed frames only, per class, with
  greedy IoU matching (`iou_threshold=0.3`), `max_age=5` analyzed frames and
  `min_hits=2`. A "unique instance" is a track that reached `min_hits`; its id
  is stable and recorded in the detections document (`track_id`).
- **No ego-motion compensation.** The tracker matches boxes in image
  coordinates. Moving cameras, pans and zoom change box positions between
  analyzed frames, so unique-instance counts are approximate and are labeled as
  such in the documentation and reports. They are never presented as traffic
  counts or severity measurements.
- **Coverage.** The summary coverage ratio is the arithmetic mean of the exact
  per-analyzed-frame box-union coverage (`D21`), a descriptive statistic in
  `[0, 1]`, never a severity score.
- **`--max-frames N`.** Caps the number of source frames read. The detections
  document records `params.max_frames`, the summary flag `truncated` and the
  Markdown summary says the run stopped early, so a partial analysis is never
  mistaken for a complete one.
- **Atomic outputs.** The MP4 is encoded to `<stem>.annotated.part.mp4` and
  renamed only after a complete, successful stream; the detections JSON and the
  summaries are written afterwards. A truncated/corrupt input raises
  `MediaError` (exit 5) and leaves no artifact under its final name.
- **Frame-count guard.** When the container declares a frame count, a stream
  that ends early is an error instead of a silently short analysis. Containers
  without a frame count are analyzed to end-of-stream.
- **Reports.** `rdv detect` writes `<stem>.annotated.mp4`,
  `<stem>.detections.json`, `<stem>.summary.json` and `<stem>.summary.md` per
  video, so several videos in one invocation cannot overwrite each other.

## Consequences

- CPU inference cost scales with `analysis_fps`, not with the source frame
  rate, while output videos remain frame-accurate.
- Very short clips may report zero unique instances when fewer than two
  analyzed frames contain the object (`min_hits=2`); the raw detections are
  still reported.
- Detections on skipped frames are a visual aid, not independent predictions;
  the JSON only lists analyzed frames so metrics and audits are unambiguous.
- `rdv benchmark` measures video throughput as analyzed frames per second on
  synthetic 320x240 frames, independently of container decoding.
