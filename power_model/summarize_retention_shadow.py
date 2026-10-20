#!/usr/bin/env python3
"""Summarize the final per-core SA retention-shadow records from an ONNXim log."""

import argparse
import json
from pathlib import Path


PREFIX = "SA_RETENTION_SHADOW "


def integer_histogram(record, field, core_id):
    raw_histogram = record.get(field)
    if not isinstance(raw_histogram, dict):
        raise ValueError(f"core {core_id}: missing {field}")
    histogram = {}
    for raw_value, raw_count in raw_histogram.items():
        value = int(raw_value)
        count = int(raw_count)
        if value <= 0 or count <= 0:
            raise ValueError(f"core {core_id}: invalid {field} entry")
        histogram[value] = histogram.get(value, 0) + count
    return histogram


def load_latest_records(path: Path):
    records = {}
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            marker = raw_line.find(PREFIX)
            if marker < 0:
                continue
            payload = raw_line[marker + len(PREFIX):].strip()
            try:
                record = json.loads(payload)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"malformed retention record at {path}:{line_number}: "
                    f"{error}"
                ) from error
            records[int(record["core_id"])] = record
    if not records:
        raise ValueError(f"no {PREFIX.strip()} records found in {path}")
    return records


def require_valid_record(core_id, record):
    if record.get("version") != 2:
        raise ValueError(f"core {core_id}: unsupported record version")
    if record.get("measurement_only") is not True:
        raise ValueError(f"core {core_id}: record is not measurement-only")
    if record.get("control_path_modified") is not False:
        raise ValueError(f"core {core_id}: simulator control path was modified")
    if record.get("zero_wake_latency_assumption") is not True:
        raise ValueError(f"core {core_id}: unexpected wake-latency assumption")
    if record.get("ordinary_gemm_retention_allowed") is not True:
        raise ValueError(f"core {core_id}: ordinary GEMM retention is disabled")
    if record.get("two_resident_candidate_requirement") is not False:
        raise ValueError(f"core {core_id}: non-retentive residency rule remains")
    if int(record.get("causal_deadline_violations", -1)) != 0:
        raise ValueError(f"core {core_id}: causal deadline violation observed")

    cycles = int(record["core_cycles"])
    idle_cycles = int(record["sa_idle_cycles"])
    retention_cycles = int(record["retention_cycles"])
    episodes = int(record["episode_count"])
    classified = (
        int(record["state_reuse_episode_count"])
        + int(record["preload_only_episode_count"])
        + int(record["unknown_future_episode_count"])
    )
    if cycles < 0 or not 0 <= retention_cycles <= idle_cycles <= cycles:
        raise ValueError(f"core {core_id}: invalid retention-cycle accounting")
    if episodes != classified:
        raise ValueError(f"core {core_id}: episode classification is not exact")

    duration_histogram = integer_histogram(
        record, "episode_duration_histogram", core_id
    )
    if sum(duration_histogram.values()) != episodes:
        raise ValueError(f"core {core_id}: duration histogram episode mismatch")
    if sum(
        duration * count
        for duration, count in duration_histogram.items()
    ) != retention_cycles:
        raise ValueError(f"core {core_id}: duration histogram cycle mismatch")

    entry_histogram = integer_histogram(
        record, "entry_bound_histogram", core_id
    )
    if sum(entry_histogram.values()) != episodes:
        raise ValueError(f"core {core_id}: entry-bound histogram mismatch")

    raw_sweep = record.get("entry_threshold_sweep")
    if not isinstance(raw_sweep, dict) or not raw_sweep:
        raise ValueError(f"core {core_id}: missing entry-threshold sweep")
    thresholds = sorted(int(value) for value in raw_sweep)
    if thresholds[0] != 1:
        raise ValueError(f"core {core_id}: threshold sweep must begin at one")

    previous_episodes = episodes
    previous_cycles = retention_cycles
    for threshold in thresholds:
        entry = raw_sweep[str(threshold)]
        qualified_episodes = int(entry["entry_qualified_episode_count"])
        qualified_cycles = int(entry["observed_retention_cycles"])
        early_exits = int(entry["episodes_shorter_than_threshold"])
        censored = int(entry["right_censored_episodes"])
        if not 0 <= qualified_episodes <= previous_episodes:
            raise ValueError(
                f"core {core_id}: non-monotonic threshold episode count"
            )
        if not 0 <= qualified_cycles <= previous_cycles:
            raise ValueError(
                f"core {core_id}: non-monotonic threshold cycle count"
            )
        if not 0 <= early_exits <= qualified_episodes:
            raise ValueError(f"core {core_id}: invalid early-exit count")
        if not 0 <= censored <= min(
            qualified_episodes, int(record["right_censored_episodes"])
        ):
            raise ValueError(f"core {core_id}: invalid censored count")
        previous_episodes = qualified_episodes
        previous_cycles = qualified_cycles

    threshold_one = raw_sweep["1"]
    if int(threshold_one["entry_qualified_episode_count"]) != episodes:
        raise ValueError(f"core {core_id}: threshold-one episode mismatch")
    if int(threshold_one["observed_retention_cycles"]) != retention_cycles:
        raise ValueError(f"core {core_id}: threshold-one cycle mismatch")

    return thresholds, duration_histogram, entry_histogram


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument(
        "--include-histograms",
        action="store_true",
        help="include exact aggregate and per-core histogram maps",
    )
    args = parser.parse_args()

    records = load_latest_records(args.log)
    thresholds = None
    aggregate_duration_histogram = {}
    aggregate_entry_histogram = {}
    for core_id, record in sorted(records.items()):
        (core_thresholds,
         duration_histogram,
         entry_histogram) = require_valid_record(core_id, record)
        if thresholds is None:
            thresholds = core_thresholds
        elif thresholds != core_thresholds:
            raise ValueError("per-core threshold sweeps do not match")
        for duration, count in duration_histogram.items():
            aggregate_duration_histogram[duration] = (
                aggregate_duration_histogram.get(duration, 0) + count
            )
        for bound, count in entry_histogram.items():
            aggregate_entry_histogram[bound] = (
                aggregate_entry_histogram.get(bound, 0) + count
            )

    def total(field):
        return sum(int(record[field]) for record in records.values())

    core_cycles = total("core_cycles")
    idle_cycles = total("sa_idle_cycles")
    retention_cycles = total("retention_cycles")
    episode_count = total("episode_count")

    entry_threshold_sweep = {}
    observed_duration_survival = {}
    for threshold in thresholds:
        qualified_episodes = sum(
            int(record["entry_threshold_sweep"][str(threshold)][
                "entry_qualified_episode_count"
            ])
            for record in records.values()
        )
        qualified_cycles = sum(
            int(record["entry_threshold_sweep"][str(threshold)][
                "observed_retention_cycles"
            ])
            for record in records.values()
        )
        early_exits = sum(
            int(record["entry_threshold_sweep"][str(threshold)][
                "episodes_shorter_than_threshold"
            ])
            for record in records.values()
        )
        censored = sum(
            int(record["entry_threshold_sweep"][str(threshold)][
                "right_censored_episodes"
            ])
            for record in records.values()
        )
        entry_threshold_sweep[str(threshold)] = {
            "entry_qualified_episode_count": qualified_episodes,
            "observed_retention_cycles": qualified_cycles,
            "average_observed_episode_cycles": (
                qualified_cycles / qualified_episodes
                if qualified_episodes else 0.0
            ),
            "observed_retention_percent_of_sa_idle": (
                100.0 * qualified_cycles / idle_cycles
                if idle_cycles else 0.0
            ),
            "observed_retention_percent_of_core_cycles": (
                100.0 * qualified_cycles / core_cycles
                if core_cycles else 0.0
            ),
            "episodes_shorter_than_threshold": early_exits,
            "early_exit_percent": (
                100.0 * early_exits / qualified_episodes
                if qualified_episodes else 0.0
            ),
            "right_censored_episodes": censored,
        }

        duration_episodes = sum(
            count
            for duration, count in aggregate_duration_histogram.items()
            if duration >= threshold
        )
        duration_cycles = sum(
            duration * count
            for duration, count in aggregate_duration_histogram.items()
            if duration >= threshold
        )
        observed_duration_survival[str(threshold)] = {
            "episode_count_at_least_duration": duration_episodes,
            "cycles_in_episodes_at_least_duration": duration_cycles,
            "percent_of_retention_cycles": (
                100.0 * duration_cycles / retention_cycles
                if retention_cycles else 0.0
            ),
        }
    result = {
        "validation_pass": True,
        "core_count": len(records),
        "aggregate": {
            "core_cycles": core_cycles,
            "sa_idle_cycles": idle_cycles,
            "sa_idle_percent": (
                100.0 * idle_cycles / core_cycles
                if core_cycles else 0.0
            ),
            "retention_cycles": retention_cycles,
            "retention_percent": (
                100.0 * retention_cycles / core_cycles
                if core_cycles else 0.0
            ),
            "retention_percent_of_sa_idle": (
                100.0 * retention_cycles / idle_cycles
                if idle_cycles else 0.0
            ),
            "episode_count": episode_count,
            "average_episode_cycles": (
                retention_cycles / episode_count
                if episode_count else 0.0
            ),
            "longest_episode_cycles": max(
                int(record["longest_episode"])
                for record in records.values()
            ),
            "state_reuse_episode_count": total(
                "state_reuse_episode_count"
            ),
            "preload_only_episode_count": total(
                "preload_only_episode_count"
            ),
            "unknown_future_episode_count": total(
                "unknown_future_episode_count"
            ),
            "immediate_wake_events": total("immediate_wake_events"),
            "right_censored_episodes": total(
                "right_censored_episodes"
            ),
            "causal_deadline_violations": total(
                "causal_deadline_violations"
            ),
            "episode_duration_histogram": {
                str(duration): count
                for duration, count in sorted(
                    aggregate_duration_histogram.items()
                )
            },
            "entry_bound_histogram": {
                str(bound): count
                for bound, count in sorted(
                    aggregate_entry_histogram.items()
                )
            },
            "entry_threshold_sweep": entry_threshold_sweep,
            "observed_duration_survival": observed_duration_survival,
        },
        "cores": {
            str(core_id): record
            for core_id, record in sorted(records.items())
        },
        "method": {
            "kind": "measurement-only causal SA retention shadow",
            "architectural_timing_modified": False,
            "wake_latency_cycles": 0,
            "break_even_filter_applied": False,
            "entry_pipeline_applied": False,
            "ordinary_gemm_retention_allowed": True,
            "two_resident_candidate_requirement": False,
            "minimum_causal_bound_cycles": 1,
            "energy_savings_known": False,
            "threshold_sweep_semantics": (
                "episodes selected by the causal bound available at entry; "
                "this is not an oracle duration filter"
            ),
            "duration_survival_semantics": (
                "post-observation episode-duration distribution; do not "
                "interpret it as an implementable prediction policy"
            ),
            "purpose": (
                "opportunity screening before retained-cell and "
                "wake-transition characterization"
            ),
        },
    }
    if not args.include_histograms:
        result["aggregate"].pop("episode_duration_histogram")
        result["aggregate"].pop("entry_bound_histogram")
        for record in result["cores"].values():
            record.pop("episode_duration_histogram", None)
            record.pop("entry_bound_histogram", None)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
