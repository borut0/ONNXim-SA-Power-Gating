#!/usr/bin/env python3
"""Summarize compact ONNXim component-characterization records."""

import argparse
import json
import re
from pathlib import Path


PREFIXES = (
    "COMPONENT_SYSTEM",
    "COMPONENT_CORE",
    "SA_STATE_BREAKDOWN",
    "COMPONENT_NOC",
    "COMPONENT_DRAM",
    "COMPONENT_MEMORY_STAGES",
)


def percentage(value, total):
    return 0.0 if total == 0 else 100.0 * value / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    args = parser.parse_args()

    records = {prefix: [] for prefix in PREFIXES}
    malformed = []
    pattern = re.compile(
        r"(?:^|\] \[info\] )(" + "|".join(PREFIXES) + r") (\{.*\})$"
    )
    with args.log.open("r", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            match = pattern.search(line.rstrip())
            if not match:
                continue
            try:
                records[match.group(1)].append(json.loads(match.group(2)))
            except json.JSONDecodeError:
                malformed.append(line_number)

    cores = records["COMPONENT_CORE"]
    sa_states = records["SA_STATE_BREAKDOWN"]
    total_core_cycles = sum(record["core_cycles"] for record in cores)

    core_fields = (
        "sa_active_cycles",
        "sa_idle_cycles",
        "vector_active_cycles",
        "vector_idle_cycles",
        "sa_or_vector_active_cycles",
        "sa_and_vector_active_cycles",
        "neither_sa_nor_vector_active_cycles",
        "core_request_fifo_nonempty_cycles",
        "core_request_fifo_empty_cycles",
    )
    aggregate_core = {field: sum(r[field] for r in cores) for field in core_fields}
    aggregate_core.update(
        {field.replace("_cycles", "_percent"): percentage(value, total_core_cycles)
         for field, value in list(aggregate_core.items())}
    )

    attribution_names = (
        "compute", "memory_stall", "dependency_stall", "scheduler_stall", "other"
    )
    attribution = {
        name: sum(r["execution_attribution"][name] for r in cores)
        for name in attribution_names
    }
    attribution["percent_of_core_cycles"] = {
        name: percentage(value, total_core_cycles)
        for name, value in attribution.items()
        if name != "percent_of_core_cycles"
    }

    aggregate_sa_states = {}
    for record in sa_states:
        for name, value in record["cycle_counts"].items():
            aggregate_sa_states[name] = aggregate_sa_states.get(name, 0) + value

    systems = records["COMPONENT_SYSTEM"]
    nocs = records["COMPONENT_NOC"]
    drams = records["COMPONENT_DRAM"]
    memory_stages = records["COMPONENT_MEMORY_STAGES"]
    expected_cores = systems[0]["num_cores"] if len(systems) == 1 else None

    checks = {
        "one_system_record": len(systems) == 1,
        "one_noc_record": len(nocs) == 1,
        "one_dram_record": len(drams) == 1,
        "one_memory_stage_record": len(memory_stages) == 1,
        "expected_core_records": expected_cores is not None and len(cores) == expected_cores,
        "expected_sa_state_records": expected_cores is not None and len(sa_states) == expected_cores,
        "all_core_accounting_exact": bool(cores) and all(r["accounting_exact"] for r in cores),
        "all_sa_state_accounting_exact": bool(sa_states) and all(r["accounting_exact"] for r in sa_states),
        "no_malformed_records": not malformed,
    }
    if systems and nocs and drams:
        system = systems[0]
        noc = nocs[0]
        dram = drams[0]
        checks.update({
            "component_core_cycle_total_exact":
                total_core_cycles == system["core_cycles"] * system["num_cores"],
            "request_flow_exact":
                system["core_to_noc_packets"] == system["noc_to_dram_packets"],
            "response_flow_exact":
                system["dram_to_noc_packets"] == system["noc_to_core_packets"],
            "noc_injection_count_exact":
                noc["injected_packets"] ==
                system["core_to_noc_packets"] + system["dram_to_noc_packets"],
            "noc_delivery_count_exact":
                noc["delivered_packets"] ==
                system["noc_to_dram_packets"] + system["noc_to_core_packets"],
            "dram_accept_count_exact":
                dram["accepted_requests"] == system["noc_to_dram_packets"],
            "dram_completion_count_exact":
                dram["completed_requests"] == system["dram_to_noc_packets"],
            "noc_drained": noc["outstanding_packets_at_end"] == 0,
            "dram_drained": dram["outstanding_requests_at_end"] == 0,
            "dram_request_completion_exact":
                dram["accepted_requests"] == dram["completed_requests"],
        })

    result = {
        "source_log": str(args.log),
        "malformed_line_numbers": malformed,
        "record_counts": {name: len(value) for name, value in records.items()},
        "validation": checks,
        "validation_pass": bool(checks) and all(checks.values()),
        "aggregate": {
            "core_cycles": total_core_cycles,
            "core_components": aggregate_core,
            "execution_attribution": attribution,
            "sa_architectural_states": {
                "cycle_counts": aggregate_sa_states,
                "percent_of_core_cycles": {
                    name: percentage(value, total_core_cycles)
                    for name, value in aggregate_sa_states.items()
                },
            },
        },
        "system": systems[0] if len(systems) == 1 else systems,
        "noc": nocs[0] if len(nocs) == 1 else nocs,
        "dram": drams[0] if len(drams) == 1 else drams,
        "memory_request_stages": (
            memory_stages[0] if len(memory_stages) == 1 else memory_stages
        ),
        "interpretation_limits": [
            "SA and vector active percentages may overlap and are reported separately.",
            "NoC demand-active means at least one ONNXim memory packet is in flight; it is not router-level link utilization.",
            "DRAM demand-active means accepted-but-not-completed demand occupancy; transaction bandwidth utilization is reported separately.",
            "Memory-stage averages are request-weighted; component active percentages are cycle-weighted.",
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
