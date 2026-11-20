#!/usr/bin/env python3
"""Build the committed tiny detector fixture (decision D15d).

The tiny model follows the production contract (S4) but returns constant,
deterministic answers: two confident queries, one per class D00/D40, and low
scores everywhere else. The graph still consumes ``pixel_values`` through a
reduction that multiplies it by zero, so ONNX Runtime cannot prune the input
and the export contract is exercised end to end.

Run from the repository root::

    uv run python scripts/build_fixture_model.py

Re-running must produce byte-identical bytes; a unit test regenerates it into a
temporary directory and compares.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "tests" / "fixtures" / "models" / "tiny-detector.onnx"

INPUT_NAME = "pixel_values"
NUM_QUERIES = 300
NUM_CLASSES = 4
INPUT_SIZE = 640
LOW_LOGIT = -10.0
# (query index, damage-class id, pre-sigmoid score, normalized cxcywh box)
SCRIPTED_QUERIES: tuple[tuple[int, int, float, tuple[float, float, float, float]], ...] = (
    (0, 0, 5.0, (0.5, 0.5, 0.2, 0.2)),
    (1, 3, 3.0, (0.25, 0.25, 0.1, 0.1)),
)


def _constant(name: str, values: list[float], shape: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, TensorProto.FLOAT, shape, values)


def build_tiny_detector() -> onnx.ModelProto:
    """Return the deterministic tiny detector graph (opset 17)."""
    logits = [LOW_LOGIT] * (NUM_QUERIES * NUM_CLASSES)
    boxes = [0.0] * (NUM_QUERIES * 4)
    for query, class_id, score, box in SCRIPTED_QUERIES:
        logits[query * NUM_CLASSES + class_id] = score
        boxes[query * 4 : query * 4 + 4] = box

    tensors = [
        _constant("logits_base", logits, [1, NUM_QUERIES, NUM_CLASSES]),
        _constant("boxes_base", boxes, [1, NUM_QUERIES, 4]),
        _constant("zero_scale", [0.0], [1]),
    ]
    nodes = [
        helper.make_node(
            "ReduceMean",
            [INPUT_NAME],
            ["input_mean"],
            axes=[0, 1, 2, 3],
            keepdims=0,
            name="consume_input",
        ),
        helper.make_node("Mul", ["input_mean", "zero_scale"], ["dependency"], name="zero_input"),
        helper.make_node("Add", ["logits_base", "dependency"], ["logits"], name="add_logits"),
        helper.make_node("Add", ["boxes_base", "dependency"], ["pred_boxes"], name="add_boxes"),
    ]
    graph = helper.make_graph(
        nodes,
        "rdv_tiny_detector",
        [
            helper.make_tensor_value_info(
                INPUT_NAME, TensorProto.FLOAT, [1, 3, INPUT_SIZE, INPUT_SIZE]
            )
        ],
        [
            helper.make_tensor_value_info(
                "logits", TensorProto.FLOAT, [1, NUM_QUERIES, NUM_CLASSES]
            ),
            helper.make_tensor_value_info("pred_boxes", TensorProto.FLOAT, [1, NUM_QUERIES, 4]),
        ],
        initializer=tensors,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="rdv-fixture-model",
        producer_version="1",
        doc_string="Deterministic tiny RT-DETR-shaped model for tests (not a real detector).",
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def serialize_tiny_detector() -> bytes:
    """Serialize the tiny detector deterministically."""
    return build_tiny_detector().SerializeToString()


def main(argv: list[str] | None = None) -> int:
    """Write the fixture file and print the path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(serialize_tiny_detector())
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
