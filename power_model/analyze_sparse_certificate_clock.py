#!/usr/bin/env python3
"""Measure whether sparse certificate evaluation is worth an RTL redesign.

This is an offline feasibility analysis.  It does not alter the predictor and
does not estimate power.  It uses the existing 44-column controller traces and
the exact replay sleep episodes to answer two narrower questions:

1. Could the heavy three-stage certificate pipeline be clocked only after an
   event that can expose a new/larger certificate, plus certificate expiry?
2. Would those sparse triggers reproduce the observed sleep-entry cycles?

Response-count decreases and dequeue-ticket progress cannot increase either
safe lower bound.  They therefore do not trigger a new sample.  Count
increases, semantic descriptor writes, eligibility rises, and other inputs
that can increase the current frontier do trigger a sample.  A trigger is
considered capable of reproducing an observed entry only when it occurs three
cycles before that entry, matching the registered RTL pipeline.

The reported clock-cycle reduction is an activity proxy.  Physical power must
still be obtained from synthesized RTL and workload-annotated gate activity.
"""

import argparse
import bisect
import concurrent.futures
import json
import multiprocessing
import os
import re
from pathlib import Path


REQUIRED_COLUMNS = [
    "cycle", "final", "sa_active", "compute_pipeline_empty",
    "compute_front_future_valid", "compute_front_remaining",
    "ex_inst_queue_empty", "ex_front_is_sa",
    "scheduler_core_tile_waiting", "global_issue_drain_bound",
    "core_queue_tickets_exact", "last_dequeued_ticket",
    "tile0_we", "tile0_count_we", "tile0_present",
    "tile0_candidate_found", "tile0_candidate_preload",
    "tile0_sources_supported", "tile0_prefix_count",
    "tile0_remaining_count", "tile0_operand_used",
    "tile0_operand_allocated", "tile0_operand_valid",
    "tile0_operand_pending_memory", "tile0_operand0_tail_ticket",
    "tile0_operand1_tail_ticket", "tile0_operand0_pending_count",
    "tile0_operand1_pending_count", "tile1_we", "tile1_count_we",
    "tile1_present", "tile1_candidate_found",
    "tile1_candidate_preload", "tile1_sources_supported",
    "tile1_prefix_count", "tile1_remaining_count",
    "tile1_operand_used", "tile1_operand_allocated",
    "tile1_operand_valid", "tile1_operand_pending_memory",
    "tile1_operand0_tail_ticket", "tile1_operand1_tail_ticket",
    "tile1_operand0_pending_count", "tile1_operand1_pending_count",
]

CORE_RE = re.compile(r"core(\d+)")
EPISODE_RE = re.compile(
    r'^CONTROLLER_SLEEP_EPISODE (\{.*"start_cycle".*\})$')


def core_id(path: Path) -> int:
    match = CORE_RE.search(path.name)
    if not match:
        raise ValueError(f"cannot determine core id from {path}")
    return int(match.group(1))


def load_episodes(path: Path):
    episodes = []
    with path.open(errors="replace") as stream:
        for line in stream:
            match = EPISODE_RE.match(line.rstrip("\n"))
            if not match:
                continue
            record = json.loads(match.group(1))
            start = int(record["start_cycle"])
            end = int(record["end_cycle"])
            duration = int(record["duration_cycles"])
            if end - start != duration or duration <= 0:
                raise ValueError(f"malformed sleep episode in {path}: {record}")
            if episodes and start < episodes[-1][1]:
                raise ValueError(f"overlapping sleep episodes in {path}")
            episodes.append((start, end))
    if not episodes:
        raise ValueError(f"no sleep episodes found in {path}")
    return episodes


def is_eligible(values, index):
    return (
        values[index["sa_active"]] == 0
        and values[index["compute_pipeline_empty"]] != 0
        and values[index["ex_inst_queue_empty"]] != 0
        and values[index["core_queue_tickets_exact"]] != 0
        and values[index["tile0_present"]] != 0
        and values[index["tile0_candidate_found"]] != 0
        and values[index["tile0_candidate_preload"]] != 0
        and values[index["tile1_present"]] != 0
        and values[index["tile1_candidate_found"]] != 0
        and values[index["tile1_candidate_preload"]] != 0
    )


def count_direction(old, new, index, tile):
    if not new[index[f"tile{tile}_count_we"]]:
        return None
    deltas = [
        new[index[f"tile{tile}_operand0_pending_count"]]
        - old[index[f"tile{tile}_operand0_pending_count"]],
        new[index[f"tile{tile}_operand1_pending_count"]]
        - old[index[f"tile{tile}_operand1_pending_count"]],
    ]
    has_increase = any(delta > 0 for delta in deltas)
    has_decrease = any(delta < 0 for delta in deltas)
    if has_increase and has_decrease:
        return "mixed"
    if has_increase:
        return "increase_only"
    if has_decrease:
        return "decrease_only"
    return "no_delta"


def frontier_can_increase(old, new, index):
    # Removing an older visible contender or increasing a remaining bound can
    # increase the safe frontier.  Decreases are monotonic progress and are
    # deliberately ignored.
    if (old[index["compute_front_future_valid"]]
            and not new[index["compute_front_future_valid"]]):
        return True
    if (new[index["compute_front_remaining"]]
            > old[index["compute_front_remaining"]]):
        return True
    if old[index["ex_front_is_sa"]] and not new[index["ex_front_is_sa"]]:
        return True
    if (old[index["scheduler_core_tile_waiting"]]
            and not new[index["scheduler_core_tile_waiting"]]):
        return True
    if (new[index["global_issue_drain_bound"]]
            > old[index["global_issue_drain_bound"]]):
        return True
    return False


def overlap(intervals, start, end, cursor):
    while cursor < len(intervals) and intervals[cursor][1] <= start:
        cursor += 1
    total = 0
    scan = cursor
    while scan < len(intervals) and intervals[scan][0] < end:
        total += max(0, min(end, intervals[scan][1])
                     - max(start, intervals[scan][0]))
        if intervals[scan][1] >= end:
            break
        scan += 1
    return total, cursor


def contains(intervals, cycle):
    position = bisect.bisect_right(intervals, (cycle, float("inf"))) - 1
    return position >= 0 and intervals[position][0] <= cycle < intervals[position][1]


def merge_window_length(starts, width):
    if not starts:
        return 0
    ordered = sorted(starts)
    total = 0
    begin = ordered[0]
    end = begin + width
    for start in ordered[1:]:
        next_end = start + width
        if start <= end:
            end = max(end, next_end)
        else:
            total += end - begin
            begin, end = start, next_end
    return total + end - begin


def analyze_one(trace_path: str, episode_path: str):
    trace = Path(trace_path)
    episode_log = Path(episode_path)
    episodes = load_episodes(episode_log)
    sleep_starts = {start: end for start, end in episodes}
    sleep_ends_plus_one = sorted(end + 1 for _, end in episodes)
    sleep_expiry_cursor = 0
    sleep_cycles = sum(end - start for start, end in episodes)

    trigger_cycles = set()
    trigger_reasons = {
        "eligibility_rise_or_write_release": 0,
        "semantic_write": 0,
        "pending_count_increase": 0,
        "frontier_increase": 0,
        "sleep_expiry": 0,
    }
    count_directions = {
        "decrease_only": 0,
        "increase_only": 0,
        "mixed": 0,
        "no_delta": 0,
    }
    semantic_write_cycles = 0
    eligible_cycles = 0
    eligible_awake_cycles = 0
    total_cycles = 0
    records = 0
    sleep_cursor = 0

    with trace.open() as stream:
        header = stream.readline().rstrip("\r\n").split(",")
        if header != REQUIRED_COLUMNS:
            raise ValueError(f"{trace}: expected exact 44-column trace schema")
        index = {name: position for position, name in enumerate(header)}

        first_line = stream.readline()
        if not first_line:
            raise ValueError(f"{trace}: empty trace")
        old = [int(field) for field in first_line.rstrip("\r\n").split(",")]
        if len(old) != len(header) or old[index["cycle"]] != 0:
            raise ValueError(f"{trace}: malformed cycle-0 record")
        old_cycle = 0
        old_eligible = is_eligible(old, index)
        old_tile_write = bool(old[index["tile0_we"]] or old[index["tile1_we"]])
        records = 1

        for line_number, line in enumerate(stream, start=3):
            new = [int(field) for field in line.rstrip("\r\n").split(",")]
            if len(new) != len(header):
                raise ValueError(f"{trace}:{line_number}: malformed record")
            cycle = new[index["cycle"]]
            if cycle < old_cycle:
                raise ValueError(f"{trace}:{line_number}: non-monotonic cycle")

            duration = cycle - old_cycle
            if old_eligible:
                eligible_cycles += duration
                sleeping, sleep_cursor = overlap(
                    episodes, old_cycle, cycle, sleep_cursor)
                eligible_awake_cycles += duration - sleeping

                # Expiry can occur inside a run-length encoded trace segment.
                # The small countdown domain supplies this event; only accept
                # it when the unchanged architectural inputs are eligible.
                while (sleep_expiry_cursor < len(sleep_ends_plus_one)
                       and sleep_ends_plus_one[sleep_expiry_cursor] < cycle):
                    expiry = sleep_ends_plus_one[sleep_expiry_cursor]
                    if expiry > old_cycle and not old_tile_write:
                        trigger_cycles.add(expiry)
                        trigger_reasons["sleep_expiry"] += 1
                    sleep_expiry_cursor += 1
            total_cycles = cycle

            if new[index["final"]]:
                records += 1
                break

            new_eligible = is_eligible(new, index)
            tile_write = bool(new[index["tile0_we"]]
                              or new[index["tile1_we"]])
            if tile_write:
                semantic_write_cycles += 1
                trigger_reasons["semantic_write"] += 1

            increase = False
            for tile in (0, 1):
                direction = count_direction(old, new, index, tile)
                if direction is not None:
                    count_directions[direction] += 1
                    increase |= direction in ("increase_only", "mixed")

            sleeping_at_cycle = contains(episodes, cycle)
            if new_eligible and not tile_write and not sleeping_at_cycle:
                reasons = []
                if not old_eligible or old_tile_write:
                    reasons.append("eligibility_rise_or_write_release")
                if increase:
                    reasons.append("pending_count_increase")
                if frontier_can_increase(old, new, index):
                    reasons.append("frontier_increase")
                expiry_position = bisect.bisect_left(sleep_ends_plus_one, cycle)
                if (expiry_position < len(sleep_ends_plus_one)
                        and sleep_ends_plus_one[expiry_position] == cycle):
                    reasons.append("sleep_expiry")
                if reasons:
                    trigger_cycles.add(cycle)
                    for reason in reasons:
                        trigger_reasons[reason] += 1

            old = new
            old_cycle = cycle
            old_eligible = new_eligible
            old_tile_write = tile_write
            records += 1
        else:
            raise ValueError(f"{trace}: missing final record")

    preserved = []
    missed = []
    offset_histogram = {str(offset): 0 for offset in range(2, 6)}
    for start, end in episodes:
        for offset in range(2, 6):
            if start - offset in trigger_cycles:
                offset_histogram[str(offset)] += 1
        if start - 3 in trigger_cycles:
            preserved.append((start, end))
        else:
            missed.append((start, end))

    preserved_cycles = sum(end - start for start, end in preserved)
    sparse_heavy_clock_cycles = merge_window_length(trigger_cycles, 4)
    # The existing evaluation clock is on throughout eligible awake time and
    # throughout sleep.  This lower bound omits short pipeline-flush tails and
    # write-only ineligible edges, making the reduction estimate conservative.
    current_heavy_clock_lower_bound = eligible_awake_cycles + sleep_cycles
    heavy_reduction = (
        1.0 - sparse_heavy_clock_cycles / current_heavy_clock_lower_bound
        if current_heavy_clock_lower_bound else 0.0
    )

    total_count_events = sum(count_directions.values())
    monotonic_decrease_fraction = (
        count_directions["decrease_only"] / total_count_events
        if total_count_events else 0.0
    )
    return {
        "core": core_id(trace),
        "trace": str(trace),
        "episode_log": str(episode_log),
        "records": records,
        "cycles": total_cycles,
        "sleep_episodes": len(episodes),
        "sleep_cycles": sleep_cycles,
        "eligible_cycles": eligible_cycles,
        "eligible_awake_cycles": eligible_awake_cycles,
        "semantic_write_cycles": semantic_write_cycles,
        "count_update_events": total_count_events,
        "count_update_directions": count_directions,
        "monotonic_decrease_fraction": monotonic_decrease_fraction,
        "sparse_trigger_count": len(trigger_cycles),
        "sparse_trigger_reasons": trigger_reasons,
        "entry_trigger_offset_histogram": offset_histogram,
        "preserved_sleep_episodes_at_offset3": len(preserved),
        "missed_sleep_episodes_at_offset3": len(missed),
        "preserved_sleep_cycles_at_offset3": preserved_cycles,
        "preserved_sleep_cycle_fraction": (
            preserved_cycles / sleep_cycles if sleep_cycles else 0.0),
        "current_heavy_clock_cycles_lower_bound":
            current_heavy_clock_lower_bound,
        "sparse_heavy_clock_cycles_upper_bound": sparse_heavy_clock_cycles,
        "heavy_clock_cycle_reduction_potential": heavy_reduction,
    }


def aggregate(results):
    summed = {}
    keys = [
        "records", "cycles", "sleep_episodes", "sleep_cycles",
        "eligible_cycles", "eligible_awake_cycles",
        "semantic_write_cycles", "count_update_events",
        "sparse_trigger_count", "preserved_sleep_episodes_at_offset3",
        "missed_sleep_episodes_at_offset3",
        "preserved_sleep_cycles_at_offset3",
        "current_heavy_clock_cycles_lower_bound",
        "sparse_heavy_clock_cycles_upper_bound",
    ]
    for key in keys:
        summed[key] = sum(result[key] for result in results)
    directions = {
        key: sum(result["count_update_directions"][key] for result in results)
        for key in ("decrease_only", "increase_only", "mixed", "no_delta")
    }
    summed["count_update_directions"] = directions
    summed["monotonic_decrease_fraction"] = (
        directions["decrease_only"] / summed["count_update_events"]
        if summed["count_update_events"] else 0.0)
    summed["preserved_sleep_cycle_fraction"] = (
        summed["preserved_sleep_cycles_at_offset3"] / summed["sleep_cycles"]
        if summed["sleep_cycles"] else 0.0)
    summed["heavy_clock_cycle_reduction_potential"] = (
        1.0 - summed["sparse_heavy_clock_cycles_upper_bound"]
        / summed["current_heavy_clock_cycles_lower_bound"]
        if summed["current_heavy_clock_cycles_lower_bound"] else 0.0)
    return summed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", nargs="+", required=True, type=Path)
    parser.add_argument("--episode-logs", nargs="+", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    traces = {core_id(path): path for path in args.traces}
    logs = {core_id(path): path for path in args.episode_logs}
    if traces.keys() != logs.keys():
        raise SystemExit(
            f"trace/log core mismatch: traces={sorted(traces)} logs={sorted(logs)}")

    work = [(str(traces[core]), str(logs[core])) for core in sorted(traces)]
    jobs = max(1, min(args.jobs, len(work), os.cpu_count() or 1))
    if jobs == 1:
        results = [analyze_one(*item) for item in work]
    else:
        # Explicit fork avoids Python 3.14's fork-server listener, which is
        # unavailable in some restricted execution environments.
        context = multiprocessing.get_context("fork")
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=jobs, mp_context=context) as pool:
            futures = [pool.submit(analyze_one, *item) for item in work]
            results = [future.result() for future in futures]
    results.sort(key=lambda result: result["core"])

    output = {
        "version": 1,
        "method": {
            "analysis_only": True,
            "predictor_or_rtl_modified": False,
            "entry_pipeline_offset_cycles": 3,
            "sparse_pipeline_window_cycles": 4,
            "safety_basis": (
                "same frozen certificate sampled only on eligibility, semantic, "
                "count-increase/frontier-increase, and expiry events; monotonic "
                "response/dequeue progress cannot increase the safe bound"),
            "power_claim": (
                "none; clock-cycle reduction is an activity proxy requiring "
                "subsequent synthesized gate-level validation"),
        },
        "cores": results,
        "aggregate": aggregate(results),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
