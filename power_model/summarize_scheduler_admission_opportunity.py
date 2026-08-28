#!/usr/bin/env python3
"""Aggregate measurement-only scheduler/admission opportunity records."""

import argparse
import json
import re
from pathlib import Path


PREFIX = "SA_SCHEDULER_ADMISSION_OPPORTUNITY"


def percentage(value, total):
    return 0.0 if total == 0 else 100.0 * value / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    args = parser.parse_args()

    pattern = re.compile(
        rf"(?:^|\] \[info\] ){PREFIX} (\{{.*\}})$"
    )
    records = []
    malformed_lines = []
    with args.log.open("r", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            match = pattern.search(line.rstrip())
            if not match:
                continue
            try:
                records.append(json.loads(match.group(1)))
            except json.JSONDecodeError:
                malformed_lines.append(line_number)

    cycle_fields = (
        "fully_drained_scheduler_stall_cycles",
        "causal_bound_available_cycles",
        "entry_qualified_cycles",
        "oracle_cycles_in_long_intervals",
        "causal_qualified_run_cycles",
    )
    count_fields = (
        "scheduler_stall_interval_count",
        "certified_interval_count",
        "oracle_intervals_at_least_entry_required",
        "causal_qualified_run_count",
        "certificate_violations",
        "right_censored_intervals",
    )
    max_fields = (
        "maximum_causal_bound",
        "longest_scheduler_stall_interval",
        "longest_causal_qualified_run",
    )

    core_cycles = sum(record["core_cycles"] for record in records)
    aggregate = {field: sum(record[field] for record in records)
                 for field in cycle_fields + count_fields}
    aggregate.update({field: max((record[field] for record in records),
                                 default=0)
                      for field in max_fields})
    aggregate["core_cycles"] = core_cycles
    for field in cycle_fields:
        aggregate[field.replace("_cycles", "_percent_of_core_cycles")] = (
            percentage(aggregate[field], core_cycles)
        )

    entry_required_values = {
        record["entry_required_bound"] for record in records
    }
    validation = {
        "records_present": bool(records),
        "unique_core_ids": len({record["core_id"] for record in records})
                           == len(records),
        "measurement_only": bool(records)
                            and all(record["measurement_only"]
                                    for record in records),
        "consistent_entry_required_bound": len(entry_required_values) == 1,
        "no_malformed_records": not malformed_lines,
        "no_certificate_violations":
            aggregate["certificate_violations"] == 0,
    }

    result = {
        "source_log": str(args.log),
        "record_count": len(records),
        "malformed_line_numbers": malformed_lines,
        "validation": validation,
        "validation_pass": bool(validation) and all(validation.values()),
        "entry_required_bound": (
            next(iter(entry_required_values))
            if len(entry_required_values) == 1 else None
        ),
        "scheduler_admission_opportunity_exists":
            aggregate["causal_qualified_run_count"] > 0
            and aggregate["causal_qualified_run_cycles"] > 0
            and aggregate["certificate_violations"] == 0,
        "aggregate": aggregate,
        "per_core": sorted(records, key=lambda record: record["core_id"]),
        "interpretation": {
            "causal_result":
                "entry-qualified cycles/runs use only information visible "
                "at that cycle and include break-even, wake guard, and the "
                "three-cycle certificate pipeline requirement",
            "oracle_result":
                "oracle long intervals use completed interval duration and "
                "are a retrospective upper bound, not an implementable result",
            "scope":
                "this audit measures fully drained simple-scheduler admission "
                "gaps only; it does not change the existing controller or "
                "compute net energy",
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
