#!/usr/bin/env python3
"""Summarize the opt-in unified PE/weight-bank analytical SA replay."""

import argparse
import json
from pathlib import Path

if __package__:
    from .event_driven_sa import (BANK_DS_RATIO, PE_OFF_RATIO,
                                  replay_event_driven)
else:
    from event_driven_sa import (BANK_DS_RATIO, PE_OFF_RATIO,
                                 replay_event_driven)


PREFIX = "REGATE_EVENT_DRIVEN_SOURCE "


def read_records(path):
    records = []
    for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        position = line.find(PREFIX)
        if position >= 0:
            try:
                records.append(json.loads(line[position + len(PREFIX):]))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: invalid source JSON") from error
    if not records:
        raise ValueError(f"{path}: no {PREFIX.strip()} records")
    return records


def aggregate(records, *, expected_cores=None, beta=None):
    if not records:
        raise ValueError("no SA source records")
    if expected_cores is not None and len(records) != expected_cores:
        raise ValueError("unexpected number of cores")
    core_ids = [item.get("core_id") for item in records]
    if any(type(core_id) is not int or core_id < 0 for core_id in core_ids):
        raise ValueError("invalid core ID")
    if len(set(core_ids)) != len(core_ids):
        raise ValueError("duplicate core ID")
    if any(item.get("model") !=
           "event-driven unified PE and two-bank analytical source" or
           item.get("legacy_whole_sa_predictor_used") is not False
           for item in records):
        raise ValueError("incompatible event-driven source")
    dimensions = {(item.get("array_rows"), item.get("array_columns"),
                   item.get("weight_bytes")) for item in records}
    if len(dimensions) != 1:
        raise ValueError("SA dimensions differ across cores")
    native_horizon = max(item["onnxim_horizon_cycles"] for item in records)
    # Schedule each core first. A stalled core can finish after the native
    # horizon, so all cores must then be accounted on one extended clock.
    preliminary = [replay_event_driven(item, beta=beta) for item in records]
    managed_horizon = max(
        native_horizon,
        *(item["analytical_sa_finish_cycle"] for item in preliminary),
    )
    final = [replay_event_driven(item, horizon=managed_horizon, beta=beta)
             for item in records]
    logic = {mode: sum(item["pe_logic_cycles"][mode] for item in final)
             for mode in ("on", "off", "capacity")}
    banks = {mode: sum(item["weight_bank_pe_cycles"][mode]
                       for item in final)
             for mode in ("on", "ds", "off", "capacity")}
    if logic["on"] + logic["off"] != logic["capacity"]:
        raise ValueError("PE logic partition failed")
    if banks["on"] + banks["ds"] + banks["off"] != banks["capacity"]:
        raise ValueError("bank partition failed")
    logic_equiv = logic["on"] + PE_OFF_RATIO * logic["off"]
    bank_equiv = banks["on"] + BANK_DS_RATIO * banks["ds"]
    logic_managed_ratio = logic_equiv / logic["capacity"]
    bank_managed_ratio = bank_equiv / banks["capacity"]
    combined = ((1 - beta) * logic_managed_ratio +
                beta * bank_managed_ratio) if beta is not None else None
    return {
        "evidence_class": "extrapolated",
        "model": "event-driven unified PE and two-bank analytical replay",
        "state_partition_validation_pass": True,
        "physical_timing_feasibility_status":
            "pending: DS bank wake and any unresolved PE wake require circuit timing and functional rescheduling",
        "unresolved_pe_wake_events":
            sum(item["unresolved_pe_wake_events"] for item in final),
        "same_cycle_state_accounting": True,
        "functional_scheduling_unchanged": True,
        "observed_cores": len(records),
        "onnxim_shared_horizon_cycles": native_horizon,
        "analytical_shared_horizon_cycles": managed_horizon,
        "analytical_sa_tail_extension_cycles": managed_horizon - native_horizon,
        "analytical_sa_timeline_extension_percent":
            100 * (managed_horizon - native_horizon) / native_horizon,
        "analytical_timeline_interpretation":
            "SA-only scenario: each same-core later SA event retains earlier "
            "modeled stalls; native non-SA and memory work are not rescheduled",
        "actual_end_to_end_penalty_percent": None,
        "actual_end_to_end_penalty_status":
            "pending: non-SA instructions, memory and dependent operations were not rescheduled",
        "physical_weight_wake_penalty_status":
            "pending: Slumber DS wake is 13 cycles and ReGate PE wake is 1 cycle; hidden bank wake is an unsupported design assumption",
        "native_onnxim_overlap_intervals":
            (sum(item["native_onnxim_overlap_intervals"] for item in final)
             if all(item["native_onnxim_overlap_intervals"] is not None
                    for item in final) else None),
        "pe_logic_cycles": logic,
        "weight_bank_pe_cycles": banks,
        "structurally_unused_bank_pe_off_cycles":
            sum(item["structurally_unused_bank_pe_off_cycles"]
                for item in final),
        "pe_logic_leakage_saving_percent_on_managed_horizon":
            100 * (1 - logic_managed_ratio),
        "weight_bank_leakage_saving_percent_on_managed_horizon":
            100 * (1 - bank_managed_ratio),
        "beta": beta,
        "beta_evidence_class": "assumed" if beta is not None else "pending",
        "whole_sa_gross_static_opportunity_percent_on_managed_timeline":
            100 * (1 - combined) if combined is not None else None,
        "whole_sa_gross_static_opportunity_percent_on_native_timeline":
            (100 * (1 - combined) if combined is not None and
             managed_horizon == native_horizon else None),
        "whole_sa_static_saving_formula":
            "100*(1-((1-beta)*PE_logic_ratio+beta*two_bank_ratio))",
        "total_energy_saving_percent": None,
        "total_energy_status":
            "pending: SA static fraction and transition/controller/selector/preload dynamic energy",
        "per_core": final,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--expected-cores", type=int)
    parser.add_argument("--beta", type=float,
                        help="explicitly assumed two-bank fraction of baseline SA static power")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = aggregate(read_records(args.log), expected_cores=args.expected_cores,
                       beta=args.beta)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.resolve() == args.log.resolve():
            parser.error("--output must not overwrite the raw simulation log")
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
