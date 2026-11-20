# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Documentation

- **readme:** Add the English and Spanish project pages with generated metrics, architecture diagram and demo media
- **adr:** Record the detector choice (0001) and index decisions 0001-0011
- **architecture:** Document layers, ports, patterns and the optional heavy-dependency split
- **contributing/security:** Finalize the development, testing and vulnerability-reporting guides

## [0.1.0] - 2026-11-20

### Bug Fixes

- **data:** Extract labelled train trees and keep nested reads exact
- **cli:** Restrict evaluate splits to the documented val, test and all
- **training:** Load the pinned config and record run metadata
- **models:** Resolve private release assets through the GitHub API
- **models:** Match local artifacts to the manifest by sha256

### Build

- **models:** Publish the pinned ONNX artifact
- Run every CI job locally with make ci-local
- Declare packaging as a test dependency

### CI/CD

- Pin setup-uv to its published release tag
- **types:** Install the train and cpu extras before mypy
- Add matrix, coverage, build and installer smoke jobs
- **security:** Scan secrets, dependencies and runtime licenses
- Make the installer smoke portable across runners
- **release:** Build, attest and publish tag-driven releases

### Documentation

- **data:** Document sources, licensing, splits, and measured statistics
- **data:** Clarify that Norway is sampled, not downloaded
- **evaluation:** Document metrics, threshold selection and artifacts
- **training:** Document the fine-tuning, resume and export pipeline
- **model:** Publish the model card and measured training report
- **reports:** Add the v1 evaluation, benchmark and gallery artifacts
- **reports:** Refresh the benchmark and document calibration
- Add contribution templates and third-party license inventory
- **release:** Document the release order, rollback and verification

### Features

- Scaffold project, CLI, and quality gates
- **domain:** Add validated boxes, detections, and annotations
- **domain:** Decode DETR outputs into clipped pixel detections
- **domain:** Add deterministic seeded block splits
- **domain:** Add damage coverage and summary value objects
- **domain:** Add greedy IoU tracker and unique-instance counts
- **domain:** Add COCO-style matching and average precision
- **domain:** Add PR curves, confusion matrix, and error taxonomy
- **domain:** Add COCO evaluation orchestrator and country grouping
- **config:** Add typed settings and structured logging
- **application:** Add dataset repository and progress ports
- **datasets:** Add VOC parser and RDD2022 repository
- **datasets:** Add ranged archive reader and seeded split preparation
- **cli:** Add data download, verify, prepare, and stats commands
- **fixtures:** Generate deterministic mini dataset and packaged samples
- **inference:** Add ONNX detector, preprocessing, and provider strategy
- **models:** Add pinned manifest and resumable model store CLI
- **cli:** Add image detection with rendered overlays and reports
- **cli:** Add doctor and samples commands with reporting tests
- **media:** Add OpenCV video source and atomic MP4 sink
- **video:** Add lazy staged frame pipeline
- **video:** Add detect-video use case with tracker semantics
- **reporting:** Render video summaries and truncation state
- **cli:** Detect videos and mixed image/video inputs
- **benchmark:** Measure latency, gates and video throughput
- **cli:** Add benchmark command and GPU capability probe
- **evaluation:** Add evaluation use case and deterministic fakes
- **reporting:** Render evaluation metrics, plots and error gallery
- **cli:** Add evaluate command with reports and OOD metrics
- **training:** Add typed pinned training configuration
- **training:** Add torchvision augmentation pipeline
- **training:** Add EMA and the auto micro-batch ladder
- **training:** Add torch dataset over VOC with HF labels
- **training:** Add resumable RT-DETRv2 trainer with benchmark mode
- **training:** Export RT-DETR to the single-file ONNX contract
- **inference:** Add optional torch checkpoint detector
- **training:** Add train and export use cases with lazy composition
- **cli:** Add train and export commands with benchmark checkpoint
- **reporting:** Render honest README metrics from committed reports
- **models:** Allow optional GitHub auth for private release assets
- **models:** Publish the v1 RDD2022 checkpoint in the manifest
- **web:** Add the analysis job manager
- **web:** Enforce loopback access and streamed upload limits
- **web:** Render a self-contained analysis report
- **web:** Expose the local analysis API and app factory
- **cli:** Add rdv serve and document the local web stack
- **web:** Replace the placeholder with the viewer shell
- **web:** Add formatting, API and theme modules
- **web:** Add viewer state, filters and session history
- **web:** Render the canvas overlay and video timeline
- **web:** Add download actions for analysis artifacts
- **web:** Wire the interactive analysis viewer
- **scripts:** Capture README media from the live app
- **desktop:** Add the portable bundle launcher
- **packaging:** Build PyInstaller onedir desktop bundles
- **scripts:** Smoke-test built desktop bundles
- **release:** Verify tag, version and pinned model before release
- **release:** Smoke-test released detections against sample expectations

### Refactoring

- **inference:** Support image-only bundles without OpenCV

### Styling

- **web:** Add base layout primitives and motion preferences
- **web:** Style the upload flow and canvas viewer
- **web:** Style the results panel, timeline and empty states

### Testing

- Make CLI help assertions terminal-width independent
- **domain:** Cover empty decode arrays and geometry guards
- **metrics:** Verify engine against pycocotools and record ADR
- **datasets:** Cover manifest and decode failures
- **models:** Assert the manifest flag through command behavior
- **video:** Cover analysis stride, tracking and atomic outputs
- **evaluation:** Assert the gallery covers every non-empty error category
- **training:** Cover CPU smoke, resume and validation checkpoints
- **cli:** Assert train and export options from the command model
- **realmodel:** Verify packaged samples and ONNX parity
- **realmodel:** Match tied query order on the eight-image parity gate
- **web:** Decode the annotated PNG and probe the docs route
- **web:** Pin accessible design tokens
- **web:** Cover the image flow in a real browser
- **web:** Cover video, error, keyboard, theme and responsive UI
- **desktop:** Cover port selection, health wait and Ctrl+C
- **quality:** Guard licenses, network policy and docs links


