#!/usr/bin/env python3
"""Validate and summarize compact per-core SA semantic JSONL traces."""

import argparse
import json
from collections import Counter
from pathlib import Path


def load_trace(path):
    records = []
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not records or records[0].get("event") != "TRACE_START":
        raise ValueError(f"{path}: missing TRACE_START")
    if records[-1].get("event") != "TRACE_END":
        raise ValueError(f"{path}: missing TRACE_END")
    core_ids = {int(record["core_id"]) for record in records}
    if len(core_ids) != 1:
        raise ValueError(f"{path}: mixed core ids")
    return core_ids.pop(), records


def summarize_core(path):
    core_id, records = load_trace(path)
    event_counts = Counter(record["event"] for record in records)
    issue_counts = Counter()
    stall_occurrences = Counter()
    tile_switches = 0
    sa_intervals = []
    current_sa = None
    power_transitions = Counter()
    sa_issue_sequence = []

    for record in records:
        kind = record["event"]
        if kind == "INSTRUCTION_ISSUED":
            issue_counts[record["opcode"]] += 1
            tile_switches += int(bool(record.get("tile_switch", False)))
            if record["opcode"] in ("GEMM", "GEMM_PRELOAD"):
                sa_issue_sequence.append({
                    "tile_id": int(record["tile_id"]),
                    "opcode": record["opcode"],
                    "weight_block_known": bool(
                        record.get("weight_block_known", False)
                    ),
                    "weight_block_id": int(
                        record.get("weight_block_id", 0)
                    ),
                })
        elif kind == "INSTRUCTION_STALL":
            occurrences = int(record["occurrences"])
            if occurrences <= 0:
                raise ValueError(f"{path}: nonpositive stall run")
            if int(record["end_cycle"]) - int(record["start_cycle"]) + 1 != occurrences:
                raise ValueError(f"{path}: non-contiguous stall run")
            stall_occurrences[record["reason"]] += occurrences
        elif kind == "SA_START":
            if current_sa is not None:
                raise ValueError(f"{path}: nested SA_START")
            current_sa = record
        elif kind == "SA_FINISH":
            if current_sa is None:
                raise ValueError(f"{path}: SA_FINISH without SA_START")
            if int(record["cycle"]) < int(current_sa["cycle"]):
                raise ValueError(f"{path}: negative SA interval")
            sa_intervals.append(int(record["cycle"]) - int(current_sa["cycle"]))
            current_sa = None
        elif kind == "POWER_STATE":
            power_transitions[f"{record['from']}->{record['to']}"] += 1
    if current_sa is not None:
        raise ValueError(f"{path}: unterminated SA interval")

    known_weight_sa_issues = sum(
        issue["weight_block_known"] for issue in sa_issue_sequence
    )
    consecutive_sa_issue_pairs = max(0, len(sa_issue_sequence) - 1)
    same_weight_successors = 0
    cross_tile_pairs = 0
    cross_tile_known_weight_pairs = 0
    cross_tile_unknown_weight_pairs = 0
    cross_tile_same_weight_pairs = 0
    cross_tile_different_weight_pairs = 0
    cross_tile_same_weight_gemm_reuse = 0
    cross_tile_same_weight_preload_reload = 0

    for previous, current in zip(
            sa_issue_sequence, sa_issue_sequence[1:]):
        both_weights_known = (
            previous["weight_block_known"]
            and current["weight_block_known"]
        )
        same_weight = (
            both_weights_known
            and previous["weight_block_id"] == current["weight_block_id"]
        )
        same_weight_successors += int(same_weight)

        if previous["tile_id"] == current["tile_id"]:
            continue
        cross_tile_pairs += 1
        if not both_weights_known:
            cross_tile_unknown_weight_pairs += 1
            continue

        cross_tile_known_weight_pairs += 1
        if not same_weight:
            cross_tile_different_weight_pairs += 1
            continue

        cross_tile_same_weight_pairs += 1
        if current["opcode"] == "GEMM":
            # The next tile consumes the exact weight block without issuing a
            # new preload.  This is observed cross-tile state reuse.
            cross_tile_same_weight_gemm_reuse += 1
        elif current["opcode"] == "GEMM_PRELOAD":
            # The exact same block is requested, but ONNXim explicitly reloads
            # it.  This is a potential redundant-preload opportunity, not
            # observed reuse by the baseline execution.
            cross_tile_same_weight_preload_reload += 1

    if (cross_tile_pairs != (
            cross_tile_known_weight_pairs
            + cross_tile_unknown_weight_pairs)):
        raise ValueError(f"{path}: cross-tile weight coverage mismatch")
    if (cross_tile_known_weight_pairs != (
            cross_tile_same_weight_pairs
            + cross_tile_different_weight_pairs)):
        raise ValueError(f"{path}: cross-tile weight identity mismatch")
    if cross_tile_same_weight_pairs != (
            cross_tile_same_weight_gemm_reuse
            + cross_tile_same_weight_preload_reload):
        raise ValueError(f"{path}: cross-tile same-weight opcode mismatch")
    return core_id, {
        "path": str(path),
        "trace_end_cycle": int(records[-1]["cycle"]),
        "event_counts": dict(sorted(event_counts.items())),
        "instruction_issue_counts": dict(sorted(issue_counts.items())),
        "stall_attempts_by_reason": dict(sorted(stall_occurrences.items())),
        "tile_switches": tile_switches,
        "sa_interval_count": len(sa_intervals),
        "sa_active_cycles_from_intervals": sum(sa_intervals),
        "known_weight_sa_issues": known_weight_sa_issues,
        "consecutive_sa_issue_pairs": consecutive_sa_issue_pairs,
        "same_weight_consecutive_sa_issues": same_weight_successors,
        "cross_tile_consecutive_sa_issue_pairs": cross_tile_pairs,
        "cross_tile_known_weight_pairs": cross_tile_known_weight_pairs,
        "cross_tile_unknown_weight_pairs": cross_tile_unknown_weight_pairs,
        "cross_tile_same_weight_pairs": cross_tile_same_weight_pairs,
        "cross_tile_different_weight_pairs": cross_tile_different_weight_pairs,
        "cross_tile_same_weight_gemm_reuse_issues": (
            cross_tile_same_weight_gemm_reuse
        ),
        "cross_tile_same_weight_preload_reload_issues": (
            cross_tile_same_weight_preload_reload
        ),
        "power_transitions": dict(sorted(power_transitions.items())),
    }


def merge_counts(per_core, field):
    merged = Counter()
    for record in per_core.values():
        merged.update(record[field])
    return dict(sorted(merged.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    args = parser.parse_args()
    per_core = {}
    for path in args.traces:
        core_id, summary = summarize_core(path)
        if core_id in per_core:
            raise ValueError(f"duplicate trace for core {core_id}")
        per_core[core_id] = summary
    result = {
        "validation_pass": True,
        "core_count": len(per_core),
        "aggregate": {
            "event_counts": merge_counts(per_core, "event_counts"),
            "instruction_issue_counts": merge_counts(
                per_core, "instruction_issue_counts"
            ),
            "stall_attempts_by_reason": merge_counts(
                per_core, "stall_attempts_by_reason"
            ),
            "tile_switches": sum(r["tile_switches"] for r in per_core.values()),
            "sa_interval_count": sum(
                r["sa_interval_count"] for r in per_core.values()
            ),
            "sa_active_cycles_from_intervals": sum(
                r["sa_active_cycles_from_intervals"] for r in per_core.values()
            ),
            "known_weight_sa_issues": sum(
                r["known_weight_sa_issues"] for r in per_core.values()
            ),
            "consecutive_sa_issue_pairs": sum(
                r["consecutive_sa_issue_pairs"]
                for r in per_core.values()
            ),
            "same_weight_consecutive_sa_issues": sum(
                r["same_weight_consecutive_sa_issues"]
                for r in per_core.values()
            ),
            "cross_tile_consecutive_sa_issue_pairs": sum(
                r["cross_tile_consecutive_sa_issue_pairs"]
                for r in per_core.values()
            ),
            "cross_tile_known_weight_pairs": sum(
                r["cross_tile_known_weight_pairs"]
                for r in per_core.values()
            ),
            "cross_tile_unknown_weight_pairs": sum(
                r["cross_tile_unknown_weight_pairs"]
                for r in per_core.values()
            ),
            "cross_tile_same_weight_pairs": sum(
                r["cross_tile_same_weight_pairs"]
                for r in per_core.values()
            ),
            "cross_tile_different_weight_pairs": sum(
                r["cross_tile_different_weight_pairs"]
                for r in per_core.values()
            ),
            "cross_tile_same_weight_gemm_reuse_issues": sum(
                r["cross_tile_same_weight_gemm_reuse_issues"]
                for r in per_core.values()
            ),
            "cross_tile_same_weight_preload_reload_issues": sum(
                r["cross_tile_same_weight_preload_reload_issues"]
                for r in per_core.values()
            ),
            "power_transitions": merge_counts(per_core, "power_transitions"),
        },
        "cores": {str(core): value for core, value in sorted(per_core.items())},
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
