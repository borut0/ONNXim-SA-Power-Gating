#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path


EXPECTED_COLUMNS = [
    "cycle", "final", "sa_active", "compute_pipeline_empty",
    "compute_front_future_valid", "compute_front_remaining",
    "ex_inst_queue_empty", "ex_front_is_sa",
    "scheduler_core_tile_waiting", "global_issue_drain_bound",
    "core_queue_tickets_exact", "last_dequeued_ticket",
    "tile0_we", "tile0_count_we", "tile0_present", "tile0_candidate_found",
    "tile0_candidate_preload", "tile0_sources_supported",
    "tile0_prefix_count", "tile0_remaining_count", "tile0_operand_used",
    "tile0_operand_allocated", "tile0_operand_valid",
    "tile0_operand_pending_memory", "tile0_operand0_tail_ticket",
    "tile0_operand1_tail_ticket", "tile0_operand0_pending_count",
    "tile0_operand1_pending_count", "tile1_we", "tile1_count_we",
    "tile1_present",
    "tile1_candidate_found", "tile1_candidate_preload",
    "tile1_sources_supported", "tile1_prefix_count",
    "tile1_remaining_count", "tile1_operand_used",
    "tile1_operand_allocated", "tile1_operand_valid",
    "tile1_operand_pending_memory", "tile1_operand0_tail_ticket",
    "tile1_operand1_tail_ticket", "tile1_operand0_pending_count",
    "tile1_operand1_pending_count",
]

SELECTED_COLUMNS = [
    "cycle", "final", "sa_active", "compute_pipeline_empty",
    "ex_inst_queue_empty", "core_queue_tickets_exact",
    "tile0_we", "tile0_count_we", "tile0_present", "tile0_candidate_found",
    "tile0_candidate_preload", "tile0_sources_supported",
    "tile0_operand_used", "tile0_operand_allocated",
    "tile0_operand_valid", "tile0_operand_pending_memory",
    "tile1_we", "tile1_count_we", "tile1_present", "tile1_candidate_found",
    "tile1_candidate_preload", "tile1_sources_supported",
    "tile1_operand_used", "tile1_operand_allocated",
    "tile1_operand_valid", "tile1_operand_pending_memory",
]
SELECTED_INDICES = [EXPECTED_COLUMNS.index(name) for name in SELECTED_COLUMNS]
(
    CYCLE, FINAL, SA_ACTIVE, COMPUTE_PIPELINE_EMPTY,
    EX_INST_QUEUE_EMPTY, CORE_QUEUE_TICKETS_EXACT,
    TILE0_WE, TILE0_COUNT_WE, TILE0_PRESENT, TILE0_CANDIDATE_FOUND,
    TILE0_CANDIDATE_PRELOAD, TILE0_SOURCES_SUPPORTED,
    TILE0_OPERAND_USED, TILE0_OPERAND_ALLOCATED,
    TILE0_OPERAND_VALID, TILE0_OPERAND_PENDING_MEMORY,
    TILE1_WE, TILE1_COUNT_WE, TILE1_PRESENT, TILE1_CANDIDATE_FOUND,
    TILE1_CANDIDATE_PRELOAD, TILE1_SOURCES_SUPPORTED,
    TILE1_OPERAND_USED, TILE1_OPERAND_ALLOCATED,
    TILE1_OPERAND_VALID, TILE1_OPERAND_PENDING_MEMORY,
) = range(len(SELECTED_COLUMNS))


def summarize(path: Path) -> dict:
    weighted = {
        "sa_active_cycles": 0,
        "compute_pipeline_empty_cycles": 0,
        "ticket_exact_cycles": 0,
        "two_resident_preload_candidate_cycles": 0,
        "candidate_with_unsupported_sources_cycles": 0,
        "retention_eligible_input_cycles": 0,
        "response_drain_descriptor_eligible_input_cycles": 0,
        "sa_idle_cycles": 0,
        "idle_compute_drained_cycles": 0,
        "idle_fully_drained_cycles": 0,
        "idle_drained_two_resident_cycles": 0,
        "idle_drained_two_candidate_cycles": 0,
        "idle_drained_two_preload_cycles": 0,
        "idle_drained_two_supported_preload_cycles": 0,
        "unsupported_after_other_preconditions_cycles": 0,
        "unsupported_no_encoded_operands_cycles": 0,
        "unsupported_unallocated_operand_cycles": 0,
        "unsupported_invalid_without_fifo_certificate_cycles": 0,
    }
    idle_first_blocker_cycles = {
        "compute_pipeline_not_empty": 0,
        "execution_queue_not_empty": 0,
        "fewer_than_two_resident_tiles": 0,
        "resident_candidate_missing": 0,
        "resident_candidate_not_preload": 0,
        "resident_sources_unsupported": 0,
        "core_queue_tickets_inexact": 0,
        "eligible": 0,
    }
    tile0_writes = 0
    tile1_writes = 0
    tile0_count_writes = 0
    tile1_count_writes = 0
    previous = None
    row_count = 0

    with path.open() as stream:
        header = stream.readline().rstrip("\r\n").split(",")
        if header != EXPECTED_COLUMNS:
            raise ValueError(f"{path}: unexpected trace columns")
        for line_number, line in enumerate(stream, start=2):
            fields = line.rstrip("\r\n").split(",")
            if len(fields) != len(EXPECTED_COLUMNS):
                raise ValueError(
                    f"{path}:{line_number}: unexpected field count")
            following = tuple(int(fields[index]) for index in SELECTED_INDICES)
            row_count += 1
            if previous is None:
                if following[CYCLE] != 0 or following[FINAL] != 0:
                    raise ValueError(
                        f"{path}: trace must start at cycle 0 and have an end record")
                previous = following
                continue

            if previous[FINAL]:
                raise ValueError(f"{path}: premature final record")
            current = previous
            duration = following[CYCLE] - current[CYCLE]
            if duration < 0:
                raise ValueError(f"{path}: non-monotonic cycle sequence")
            weighted["sa_active_cycles"] += duration * current[SA_ACTIVE]
            weighted["compute_pipeline_empty_cycles"] += (
                duration * current[COMPUTE_PIPELINE_EMPTY]
            )
            weighted["ticket_exact_cycles"] += (
                duration * current[CORE_QUEUE_TICKETS_EXACT]
            )
            idle = not current[SA_ACTIVE]
            compute_drained = current[COMPUTE_PIPELINE_EMPTY]
            execution_drained = current[EX_INST_QUEUE_EMPTY]
            two_present = current[TILE0_PRESENT] and current[TILE1_PRESENT]
            two_candidates = (
                two_present and current[TILE0_CANDIDATE_FOUND] and
                current[TILE1_CANDIDATE_FOUND]
            )
            two_preload = (
                two_candidates and
                current[TILE0_CANDIDATE_PRELOAD] and
                current[TILE1_CANDIDATE_PRELOAD]
            )
            both_supported = (
                current[TILE0_SOURCES_SUPPORTED] and
                current[TILE1_SOURCES_SUPPORTED]
            )
            weighted["two_resident_preload_candidate_cycles"] += (
                duration * two_preload
            )
            unsupported = (
                (current[TILE0_PRESENT] and
                 current[TILE0_CANDIDATE_FOUND] and
                 not current[TILE0_SOURCES_SUPPORTED]) or
                (current[TILE1_PRESENT] and
                 current[TILE1_CANDIDATE_FOUND] and
                 not current[TILE1_SOURCES_SUPPORTED])
            )
            weighted["candidate_with_unsupported_sources_cycles"] += (
                duration * unsupported
            )
            retention_eligible = (
                idle and compute_drained and execution_drained and
                current[CORE_QUEUE_TICKETS_EXACT] and two_preload and
                both_supported
            )
            weighted["retention_eligible_input_cycles"] += (
                duration * retention_eligible
            )
            response_descriptor_eligible = (
                idle and compute_drained and execution_drained and
                current[CORE_QUEUE_TICKETS_EXACT] and two_preload
            )
            weighted["response_drain_descriptor_eligible_input_cycles"] += (
                duration * response_descriptor_eligible
            )
            weighted["sa_idle_cycles"] += duration * idle
            weighted["idle_compute_drained_cycles"] += (
                duration * idle * compute_drained
            )
            fully_drained = idle and compute_drained and execution_drained
            weighted["idle_fully_drained_cycles"] += duration * fully_drained
            weighted["idle_drained_two_resident_cycles"] += (
                duration * fully_drained * two_present
            )
            weighted["idle_drained_two_candidate_cycles"] += (
                duration * fully_drained * two_candidates
            )
            weighted["idle_drained_two_preload_cycles"] += (
                duration * fully_drained * two_preload
            )
            weighted["idle_drained_two_supported_preload_cycles"] += (
                duration * fully_drained * two_preload * both_supported
            )

            unsupported_context = fully_drained and two_preload and not both_supported
            weighted["unsupported_after_other_preconditions_cycles"] += (
                duration * unsupported_context
            )
            if unsupported_context:
                unsupported_tiles = []
                if not current[TILE0_SOURCES_SUPPORTED]:
                    unsupported_tiles.append((
                        current[TILE0_OPERAND_USED],
                        current[TILE0_OPERAND_ALLOCATED],
                        current[TILE0_OPERAND_VALID],
                        current[TILE0_OPERAND_PENDING_MEMORY]))
                if not current[TILE1_SOURCES_SUPPORTED]:
                    unsupported_tiles.append((
                        current[TILE1_OPERAND_USED],
                        current[TILE1_OPERAND_ALLOCATED],
                        current[TILE1_OPERAND_VALID],
                        current[TILE1_OPERAND_PENDING_MEMORY]))
                no_encoded = any(used == 0 for used, _, _, _ in unsupported_tiles)
                unallocated = any(
                    used & ~allocated & 0b11
                    for used, allocated, _, _ in unsupported_tiles)
                invalid_without_certificate = any(
                    used & allocated & ~valid & ~pending & 0b11
                    for used, allocated, valid, pending in unsupported_tiles)
                weighted["unsupported_no_encoded_operands_cycles"] += (
                    duration * no_encoded
                )
                weighted["unsupported_unallocated_operand_cycles"] += (
                    duration * unallocated
                )
                weighted[
                    "unsupported_invalid_without_fifo_certificate_cycles"
                ] += duration * invalid_without_certificate

            if idle:
                if not compute_drained:
                    blocker = "compute_pipeline_not_empty"
                elif not execution_drained:
                    blocker = "execution_queue_not_empty"
                elif not two_present:
                    blocker = "fewer_than_two_resident_tiles"
                elif not two_candidates:
                    blocker = "resident_candidate_missing"
                elif not two_preload:
                    blocker = "resident_candidate_not_preload"
                elif not both_supported:
                    blocker = "resident_sources_unsupported"
                elif not current[CORE_QUEUE_TICKETS_EXACT]:
                    blocker = "core_queue_tickets_inexact"
                else:
                    blocker = "eligible"
                idle_first_blocker_cycles[blocker] += duration
            tile0_writes += current[TILE0_WE]
            tile1_writes += current[TILE1_WE]
            tile0_count_writes += current[TILE0_COUNT_WE]
            tile1_count_writes += current[TILE1_COUNT_WE]
            previous = following

    if row_count < 2:
        raise ValueError(f"{path}: trace must start at cycle 0 and have an end record")
    if not previous[FINAL]:
        raise ValueError(f"{path}: missing final record")

    final_cycle = previous[CYCLE]

    return {
        "path": str(path),
        "cycles": final_cycle,
        "change_records": row_count - 1,
        "compression_cycles_per_record": (
            final_cycle / max(1, row_count - 1)
        ),
        "tile0_write_events": tile0_writes,
        "tile1_write_events": tile1_writes,
        "tile0_count_write_events": tile0_count_writes,
        "tile1_count_write_events": tile1_count_writes,
        "idle_first_blocker_cycles": idle_first_blocker_cycles,
        **weighted,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="number of trace files to scan concurrently")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.jobs == 1:
        summaries = [summarize(path) for path in args.traces]
    else:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=min(args.jobs, len(args.traces)),
                mp_context=multiprocessing.get_context("fork")) as executor:
            summaries = list(executor.map(summarize, args.traces))

    summed_fields = [
        "cycles", "sa_active_cycles", "sa_idle_cycles",
        "idle_compute_drained_cycles", "idle_fully_drained_cycles",
        "idle_drained_two_resident_cycles",
        "idle_drained_two_candidate_cycles",
        "idle_drained_two_preload_cycles",
        "idle_drained_two_supported_preload_cycles",
        "retention_eligible_input_cycles",
        "response_drain_descriptor_eligible_input_cycles",
        "unsupported_after_other_preconditions_cycles",
        "unsupported_no_encoded_operands_cycles",
        "unsupported_unallocated_operand_cycles",
        "unsupported_invalid_without_fifo_certificate_cycles",
    ]
    totals = {
        field: sum(summary[field] for summary in summaries)
        for field in summed_fields
    }
    blocker_totals = {
        blocker: sum(
            summary["idle_first_blocker_cycles"][blocker]
            for summary in summaries)
        for blocker in summaries[0]["idle_first_blocker_cycles"]
    }
    idle_cycles = totals["sa_idle_cycles"]
    unsupported_cycles = totals["unsupported_after_other_preconditions_cycles"]
    aggregate = {
        **totals,
        "idle_first_blocker_cycles": blocker_totals,
        "idle_first_blocker_percent": {
            blocker: 100.0 * cycles / idle_cycles
            for blocker, cycles in blocker_totals.items()
        },
        "retention_funnel_percent_of_idle": {
            field: 100.0 * totals[field] / idle_cycles
            for field in [
                "idle_compute_drained_cycles",
                "idle_fully_drained_cycles",
                "idle_drained_two_resident_cycles",
                "idle_drained_two_candidate_cycles",
                "idle_drained_two_preload_cycles",
                "idle_drained_two_supported_preload_cycles",
                "retention_eligible_input_cycles",
                "response_drain_descriptor_eligible_input_cycles",
            ]
        },
        "unsupported_diagnostic_percent": {
            field: (100.0 * totals[field] / unsupported_cycles
                    if unsupported_cycles else 0.0)
            for field in [
                "unsupported_no_encoded_operands_cycles",
                "unsupported_unallocated_operand_cycles",
                "unsupported_invalid_without_fifo_certificate_cycles",
            ]
        },
    }
    print(json.dumps({"traces": summaries, "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
