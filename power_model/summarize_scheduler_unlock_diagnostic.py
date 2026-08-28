#!/usr/bin/env python3
"""Aggregate the measurement-only cross-core scheduler-unlock audit."""

import argparse
import json
import re
from pathlib import Path


PREFIX = "SA_SCHEDULER_UNLOCK_DIAGNOSTIC"


def percentage(value, total):
    return 0.0 if total == 0 else 100.0 * value / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    args = parser.parse_args()

    pattern = re.compile(rf"(?:^|\] \[info\] ){PREFIX} (\{{.*\}})$")
    records = []
    malformed = []
    with args.log.open("r", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            match = pattern.search(line.rstrip())
            if not match:
                continue
            try:
                records.append(json.loads(match.group(1)))
            except json.JSONDecodeError:
                malformed.append(line_number)

    sum_fields = (
        "core_cycles",
        "context_cycles",
        "issue_only_bound_available_cycles",
        "enhanced_bound_available_cycles",
        "issue_only_entry_qualified_cycles",
        "enhanced_entry_qualified_cycles",
        "enhanced_bound_improved_cycles",
        "compute_bound_binding_cycles",
        "vector_bound_binding_cycles",
        "memory_injection_bound_binding_cycles",
        "issue_only_certificate_violations",
        "enhanced_certificate_violations",
    )
    max_fields = (
        "maximum_issue_only_bound",
        "maximum_enhanced_bound",
        "maximum_committed_compute_bound",
        "maximum_committed_vector_bound",
        "maximum_future_movin_injection_bound",
    )
    shadow_sum_fields = (
        "entry_attempts",
        "expired_before_entry",
        "episode_count",
        "sleep_cycles",
        "early_context_end_violations",
        "censored_episodes",
    )

    aggregate = {
        field: sum(record[field] for record in records)
        for field in sum_fields
    }
    aggregate.update(
        {
            field: max((record[field] for record in records), default=0)
            for field in max_fields
        }
    )
    for name in ("issue_only_shadow", "enhanced_shadow"):
        aggregate[name] = {
            field: sum(record[name][field] for record in records)
            for field in shadow_sum_fields
        }
        aggregate[name]["longest_episode"] = max(
            (record[name]["longest_episode"] for record in records),
            default=0,
        )

    core_cycles = aggregate["core_cycles"]
    for field in (
        "context_cycles",
        "issue_only_bound_available_cycles",
        "enhanced_bound_available_cycles",
        "issue_only_entry_qualified_cycles",
        "enhanced_entry_qualified_cycles",
        "enhanced_bound_improved_cycles",
    ):
        aggregate[field.replace("_cycles", "_percent_of_core_cycles")] = (
            percentage(aggregate[field], core_cycles)
        )

    issue_sleep = aggregate["issue_only_shadow"]["sleep_cycles"]
    enhanced_sleep = aggregate["enhanced_shadow"]["sleep_cycles"]
    aggregate["issue_only_shadow_sleep_percent"] = percentage(
        issue_sleep, core_cycles
    )
    aggregate["enhanced_shadow_sleep_percent"] = percentage(
        enhanced_sleep, core_cycles
    )
    aggregate["incremental_shadow_sleep_cycles"] = (
        enhanced_sleep - issue_sleep
    )
    aggregate["incremental_shadow_sleep_percentage_points"] = percentage(
        enhanced_sleep - issue_sleep, core_cycles
    )

    violations = (
        aggregate["issue_only_certificate_violations"]
        + aggregate["enhanced_certificate_violations"]
        + aggregate["issue_only_shadow"]["early_context_end_violations"]
        + aggregate["enhanced_shadow"]["early_context_end_violations"]
    )
    validation = {
        "records_present": bool(records),
        "unique_core_ids": len({record["core_id"] for record in records})
        == len(records),
        "measurement_only": bool(records)
        and all(record["measurement_only"] for record in records),
        "control_path_unmodified": bool(records)
        and all(not record["control_path_modified"] for record in records),
        "no_malformed_records": not malformed,
        "no_certificate_or_shadow_violations": violations == 0,
    }
    validation_pass = bool(validation) and all(validation.values())
    gain = aggregate["incremental_shadow_sleep_percentage_points"]
    if not validation_pass:
        recommendation = "INVALID: investigate certificate violations"
    elif gain >= 5.0:
        recommendation = "GO: strong incremental scheduler-unlock opportunity"
    elif gain >= 2.0:
        recommendation = "GO: material opportunity; prototype only this extension"
    elif gain > 0.0:
        recommendation = "STOP: measurable but too small for RTL implementation"
    else:
        recommendation = "STOP: no incremental opportunity"

    result = {
        "source_log": str(args.log),
        "record_count": len(records),
        "malformed_line_numbers": malformed,
        "validation": validation,
        "validation_pass": validation_pass,
        "aggregate": aggregate,
        "recommendation": recommendation,
        "decision_rule": {
            "go_threshold_percentage_points": 2.0,
            "strong_threshold_percentage_points": 5.0,
            "metric": "incremental shadow OFF cycles / total core cycles",
        },
        "interpretation": {
            "causal": (
                "all shadow entries use only bounds visible at entry; the "
                "wake frontier is frozen and includes entry/wake margins"
            ),
            "scope": (
                "scheduler-admission gaps while the local core and SA are "
                "fully drained; this is not an energy result"
            ),
            "next_step": (
                "only a GO result justifies changing RTL/control; otherwise "
                "retain the current controller and report the measured limit"
            ),
        },
        "per_core": sorted(records, key=lambda record: record["core_id"]),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
