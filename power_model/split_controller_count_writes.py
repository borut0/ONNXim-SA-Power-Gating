#!/usr/bin/env python3
"""Split legacy response-count descriptor writes without changing trace time."""

import argparse
from pathlib import Path


OLD_COLUMNS = [
    "cycle", "final", "sa_active", "compute_pipeline_empty",
    "compute_front_future_valid", "compute_front_remaining",
    "ex_inst_queue_empty", "ex_front_is_sa",
    "scheduler_core_tile_waiting", "global_issue_drain_bound",
    "core_queue_tickets_exact", "last_dequeued_ticket",
    "tile0_we", "tile0_present", "tile0_candidate_found",
    "tile0_candidate_preload", "tile0_sources_supported",
    "tile0_prefix_count", "tile0_remaining_count", "tile0_operand_used",
    "tile0_operand_allocated", "tile0_operand_valid",
    "tile0_operand_pending_memory", "tile0_operand0_tail_ticket",
    "tile0_operand1_tail_ticket", "tile0_operand0_pending_count",
    "tile0_operand1_pending_count", "tile1_we", "tile1_present",
    "tile1_candidate_found", "tile1_candidate_preload",
    "tile1_sources_supported", "tile1_prefix_count",
    "tile1_remaining_count", "tile1_operand_used",
    "tile1_operand_allocated", "tile1_operand_valid",
    "tile1_operand_pending_memory", "tile1_operand0_tail_ticket",
    "tile1_operand1_tail_ticket", "tile1_operand0_pending_count",
    "tile1_operand1_pending_count",
]

NEW_COLUMNS = OLD_COLUMNS.copy()
NEW_COLUMNS.insert(13, "tile0_count_we")
NEW_COLUMNS.insert(29, "tile1_count_we")


def convert(source: Path, destination: Path) -> dict:
    previous = [["0"] * 14, ["0"] * 14]
    records = 0
    semantic_writes = [0, 0]
    count_writes = [0, 0]

    with source.open() as input_stream, destination.open("w") as output_stream:
        header = input_stream.readline().rstrip("\r\n").split(",")
        if header != OLD_COLUMNS:
            raise ValueError(f"{source}: expected the 42-column response trace")
        output_stream.write(",".join(NEW_COLUMNS) + "\n")

        for line_number, line in enumerate(input_stream, start=2):
            fields = line.rstrip("\r\n").split(",")
            if len(fields) != len(OLD_COLUMNS):
                raise ValueError(f"{source}:{line_number}: malformed record")

            descriptors = [fields[13:27], fields[28:42]]
            writes = []
            final = fields[1] == "1"
            for tile in range(2):
                semantic = descriptors[tile][:12] != previous[tile][:12]
                count = (not semantic and
                         descriptors[tile][12:] != previous[tile][12:])
                if final:
                    semantic = False
                    count = False
                writes.append(("1" if semantic else "0",
                               "1" if count else "0"))
                semantic_writes[tile] += semantic
                count_writes[tile] += count
                previous[tile] = descriptors[tile]

            converted = (
                fields[:12] + list(writes[0]) + descriptors[0] +
                list(writes[1]) + descriptors[1]
            )
            if len(converted) != len(NEW_COLUMNS):
                raise AssertionError("converted trace field-count mismatch")
            output_stream.write(",".join(converted) + "\n")
            records += 1

    return {
        "records": records,
        "tile0_semantic_writes": semantic_writes[0],
        "tile1_semantic_writes": semantic_writes[1],
        "tile0_count_writes": count_writes[0],
        "tile1_count_writes": count_writes[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for source in args.traces:
        destination = args.output_dir / source.name
        stats = convert(source, destination)
        print(source.name, " ".join(f"{key}={value}"
                                    for key, value in stats.items()))


if __name__ == "__main__":
    main()
