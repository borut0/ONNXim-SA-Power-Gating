#!/usr/bin/env python3
"""Strictly audit an ONNX graph against ONNXim's OperationFactory coverage.

The audit intentionally rejects every operator that OperationFactory would
silently turn into a zero-work Dummy, except the two explicitly handled
pass-through operators below.  A passing audit is necessary, but not
sufficient: the model must still initialize and complete in ONNXim.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


MODELED_OPERATORS = {
    "Conv",
    "FusedConv",
    "Gemm",
    "FusedGemm",
    "MatMul",
    "MaxPool",
    "GlobalAveragePool",
    "AdaptiveAveragePool",
    "AveragePool",
    "Flatten",
    "Attention",
    "EmbedLayerNormalization",
    "SkipLayerNormalization",
    "BiasGelu",
    "FastGelu",
}

EXPLICIT_PASS_THROUGH_OPERATORS = {"Cast", "ReorderOutput"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    try:
        import onnx
    except ImportError as error:
        raise SystemExit(
            "Python package 'onnx' is required; run this inside the ONNXim "
            "Docker environment."
        ) from error

    model = onnx.load(str(args.model), load_external_data=False)
    operator_counts = Counter(node.op_type for node in model.graph.node)
    grouped_convolution_count = 0
    for node in model.graph.node:
        if node.op_type not in {"Conv", "FusedConv"}:
            continue
        group = next(
            (attribute.i for attribute in node.attribute if attribute.name == "group"),
            1,
        )
        if group != 1:
            grouped_convolution_count += 1
    modeled = {
        operator: count
        for operator, count in sorted(operator_counts.items())
        if operator in MODELED_OPERATORS
    }
    pass_through = {
        operator: count
        for operator, count in sorted(operator_counts.items())
        if operator in EXPLICIT_PASS_THROUGH_OPERATORS
    }
    unmodeled = {
        operator: count
        for operator, count in sorted(operator_counts.items())
        if operator not in MODELED_OPERATORS
        and operator not in EXPLICIT_PASS_THROUGH_OPERATORS
    }

    output = {
        "name": args.name or args.model.stem,
        "path": str(args.model),
        "node_count": sum(operator_counts.values()),
        "operator_counts": dict(sorted(operator_counts.items())),
        "modeled_operator_counts": modeled,
        "explicit_pass_through_operator_counts": pass_through,
        "unmodeled_operator_counts": unmodeled,
        "unsupported_grouped_convolution_count": grouped_convolution_count,
        "strict_operator_audit_pass": (
            not unmodeled and grouped_convolution_count == 0
        ),
        "acceptance_rule": (
            "Reject if unmodeled_operator_counts is non-empty. OperationFactory "
            "would otherwise silently represent those nodes as zero-work Dummy "
            "operations. Also reject grouped/depthwise Conv because the current "
            "weight-stationary ConvWS source marks that path unsupported and it "
            "fails runtime SRAM allocation validation. A pass still requires "
            "ONNXim initialization/completion and mapping-capacity validation."
        ),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
