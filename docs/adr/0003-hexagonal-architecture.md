# 0003 - Hexagonal architecture and its enforcement

- **Status:** Accepted
- **Date:** 2026-11-20
- **Deciders:** project owner (plan decisions D5, D9, S16)
- **Task:** T4

## Context

The project combines a pure-pixel domain (boxes, decoding, splits, metrics),
several I/O adapters (VOC datasets, ONNX/PyTorch inference, media, reports) and
three entrypoints (Typer CLI, local web app, desktop launcher). Without a hard
rule the domain slowly imports Pillow, ONNX Runtime or FastAPI "just this once",
which makes it impossible to test the expensive logic without those stacks and
blocks the torch-free base install required by spec S2.

The plan mandates hexagonal architecture: `entrypoints -> composition ->
adapters -> application -> domain`, with the domain limited to the standard
library plus NumPy and adapters forbidden from importing each other (D9).

## Decision

Adopt ports and adapters with an explicit composition root and enforce the
dependency direction mechanically:

- **Domain** (`rdv.domain`): immutable value objects and pure algorithms, only
  stdlib and NumPy. No I/O, no logging, no settings.
- **Application** (`rdv.application`): use cases and `typing.Protocol` ports in
  `ports.py`. It imports only the domain. Ports describe capabilities, not
  implementations: the first ones are `DatasetRepository` (metadata iteration
  plus lazy `load_image`) and `ProgressSink` (observer for `ProgressEvent`
  values in `events.py`). `NullProgressSink` is the Null Object used when no
  observer is attached.
- **Adapters** (`rdv.adapters.*`): driven implementations of the ports. They
  may import third-party libraries and the domain/application, but not one
  another; only `rdv.adapters.training` may import torch/transformers/scipy.
  `Rdd2022VocRepository` and the VOC parser live in `rdv.adapters.datasets`.
- **Composition** (`rdv.composition`): the only module that knows concrete
  adapters; it builds them from typed `rdv.config.Settings`.
- **Entrypoints** (`rdv.entrypoints.*`): CLI, web and desktop. They translate
  user input into use-case calls and render results. They never instantiate a
  driven adapter directly.

The contracts are enforced by `import-linter` (`layers`, `forbidden`,
`independence` in `pyproject.toml`) and by an AST purity test that walks
`src/rdv/domain` and rejects any import outside stdlib + NumPy.

## Consequences

- Domain tests run in milliseconds and cannot accidentally hit the network,
  the filesystem, or a GPU; the metrics engine can be compared with
  `pycocotools` in an isolated environment.
- The base install stays torch-free (S2) because the only torch import path is
  `adapters.training`, loaded lazily by the composition root.
- Adding a data source or a model backend means adding an adapter that
  satisfies a protocol; application code does not change. The repository and
  progress observer added here are the first two such seams.
- The composition root is the one place allowed to grow as adapters appear in
  later tasks; keeping it adapter-only is itself covered by the layer
  contract.
- Protocol classes are structural: adapters do not inherit from them, so a
  typo is caught by `mypy --strict` at the call site instead of at import time.
  Tests assert the behavior through the protocol-shaped public API.
