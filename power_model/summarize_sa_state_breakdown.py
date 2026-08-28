#!/usr/bin/env python3
"""Summarize aggregate-only SA_STATE_BREAKDOWN records from an ONNXim log."""

import argparse
import json
from collections import Counter
from pathlib import Path


PREFIX = "SA_STATE_BREAKDOWN "


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    args = parser.parse_args()

    records = []
    malformed = 0
    with args.log.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            marker = line.find(PREFIX)
            if marker < 0:
                continue
            try:
                records.append(json.loads(line[marker + len(PREFIX):]))
            except json.JSONDecodeError:
                malformed += 1

    by_core = {}
    aggregate_counts = Counter()
    total_cycles = 0
    validation_errors = []
    for record in records:
        core_id = record.get("core_id")
        if core_id in by_core:
            validation_errors.append(f"duplicate core_id {core_id}")
            continue
        by_core[core_id] = record

        counts = record.get("cycle_counts", {})
        core_cycles = int(record.get("core_cycles", 0))
        counted_cycles = sum(int(value) for value in counts.values())
        if counted_cycles != core_cycles or not record.get("accounting_exact"):
            validation_errors.append(
                f"core {core_id}: counted={counted_cycles}, core={core_cycles}"
            )
        total_cycles += core_cycles
        aggregate_counts.update({key: int(value) for key, value in counts.items()})

    active_cycles = sum(
        value for key, value in aggregate_counts.items() if key.startswith("ACTIVE_")
    )
    idle_cycles = sum(
        value for key, value in aggregate_counts.items() if key.startswith("IDLE_")
    )

    def percent(value: int) -> float:
        return 0.0 if total_cycles == 0 else 100.0 * value / total_cycles

    output = {
        "records": len(records),
        "malformed": malformed,
        "core_count": len(by_core),
        "validation_pass": malformed == 0 and not validation_errors,
        "validation_errors": validation_errors,
        "aggregate": {
            "total_core_cycles": total_cycles,
            "active_cycles": active_cycles,
            "idle_cycles": idle_cycles,
            "active_percent": percent(active_cycles),
            "idle_percent": percent(idle_cycles),
            "cycle_counts": dict(sorted(aggregate_counts.items())),
            "percent_of_total_core_cycles": {
                key: percent(value)
                for key, value in sorted(aggregate_counts.items())
            },
        },
        "per_core": {
            str(core_id): {
                "core_cycles": record.get("core_cycles"),
                "accounting_exact": record.get("accounting_exact"),
                "cycle_counts": record.get("cycle_counts"),
                "percent_of_core_cycles": record.get(
                    "percent_of_core_cycles"
                ),
            }
            for core_id, record in sorted(by_core.items())
        },
        "scope_note": (
            "Architectural SA activity/stall states only. Final IDLE_AWAKE versus "
            "POWER_GATED cycles must be taken from the equivalent synthesized-controller "
            "replay, because power state is not part of baseline ONNXim scheduling."
        ),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
