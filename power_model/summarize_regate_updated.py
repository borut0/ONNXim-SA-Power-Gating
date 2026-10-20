#!/usr/bin/env python3
"""Aggregate identity-independent updated two-bank ReGate shadow records."""

import argparse
import json
from pathlib import Path

if __package__:
    from .reference_sa_energy import (calculate_reference,
                                      replay_shared_timeline,
                                      calculate_unified_assumed_energy)
else:
    from reference_sa_energy import (calculate_reference,
                                     replay_shared_timeline,
                                     calculate_unified_assumed_energy)


PREFIX = "REGATE_UPDATED_PROFILE "
MODEL = "updated predictor plus two-bank ReGate-like analytical shadow"
STATES = ("capacity", "on", "w_on", "loading", "off", "unclassified")
MULTIPLICITY = (
    "on_zero_banks", "on_one_bank", "on_two_banks",
    "w_on_one_bank", "w_on_two_banks",
)
EVENT_TIMING = (
    "sum_event_start_lateness_cycles", "max_event_start_lateness_cycles",
    "sum_event_finish_lateness_cycles", "max_event_finish_lateness_cycles",
    "simulator_trace_makespan_cycles", "analytical_trace_makespan_cycles",
    "trace_makespan_extension_cycles",
)
VALIDATION_COUNTS = (
    "invalid_intervals", "sequence_errors", "footprint_mismatch_intervals",
    "bank_assignment_conflicts",
)


def nonnegative_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def require_object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def read_records(path):
    records = []
    for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        position = line.find(PREFIX)
        if position < 0:
            continue
        try:
            records.append(json.loads(line[position + len(PREFIX):]))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid updated profile JSON") from error
    if not records:
        raise ValueError(f"{path}: no {PREFIX.strip()} records")
    return records


def aggregate(records, expected_cores=None, reference_energy=False,
              static_fraction=0.5, bank_leakage_share=0.5,
              unified_reference=False,
              transition_on_equivalent_cycles=None,
              overhead_fraction_of_baseline=None,
              assume_zero_overheads=False):
    if unified_reference:
        reference_energy = True
    if not records:
        raise ValueError("updated profile needs at least one core record")
    if expected_cores is not None:
        nonnegative_int(expected_cores, "expected_cores")
    if expected_cores is not None and len(records) != expected_cores:
        raise ValueError(
            f"expected {expected_cores} core records, found {len(records)}"
        )
    core_ids = [nonnegative_int(record.get("core_id"), "core_id")
                for record in records]
    if len(set(core_ids)) != len(core_ids):
        raise ValueError("duplicate core_id in updated profile")
    totals = {field: 0 for field in STATES + MULTIPLICITY}
    preload_write = 0
    preload_while_on = 0
    timing_totals = {field: 0 for field in EVENT_TIMING}
    max_trace_extension = 0
    reference_cores = []
    predictor_whole_sa_cores = []
    simulator_horizons = []
    analytical_horizons = []
    replayed = []
    for record in records:
        core_id = record["core_id"]
        context = f"core {core_id}"
        if type(record.get("version")) is not int or record["version"] != 1:
            raise ValueError("updated profile version must be 1")
        provenance = {
            "model": MODEL,
            "measurement_only": True,
            "control_path_modified": False,
            "functional_scheduling_unchanged": True,
            "timing_evidence_class": "assumed",
            "state_accounting_evidence_class": "derived",
            "energy_saving_status": "pending",
            "energy_saving_percent": None,
        }
        for field, expected in provenance.items():
            if field not in record or record[field] != expected or (
                isinstance(expected, bool) and record[field] is not expected
            ):
                raise ValueError(f"{context} incompatible {field} provenance")
        if record.get("validation_pass") is not True:
            raise ValueError(f"{context} validation failed")
        validation = require_object(record.get("validation"), f"{context} validation")
        for field in ("preload_first_valid", "bank_assignment_valid",
                      "state_partition_exact"):
            if validation.get(field) is not True:
                raise ValueError(f"{context} failed {field}")
        for field in VALIDATION_COUNTS:
            if nonnegative_int(validation.get(field), f"{context} {field}") != 0:
                raise ValueError(f"{context} nonzero {field}")
        rows = nonnegative_int(record.get("array_rows"), f"{context} array_rows")
        columns = nonnegative_int(record.get("array_columns"), f"{context} array_columns")
        if rows == 0 or columns == 0:
            raise ValueError(f"{context} SA dimensions must be positive")
        simulator_horizon = nonnegative_int(
            record.get("onnxim_horizon_cycles"), f"{context} onnxim_horizon_cycles"
        )
        analytical_horizon = nonnegative_int(
            record.get("analytical_horizon_cycles"),
            f"{context} analytical_horizon_cycles",
        )
        if analytical_horizon < simulator_horizon:
            raise ValueError(f"{context} analytical horizon precedes simulator horizon")
        simulator_horizons.append(simulator_horizon)
        analytical_horizons.append(analytical_horizon)
        preload_events = nonnegative_int(
            record.get("preload_events"), f"{context} preload_events"
        )
        gemm_events = nonnegative_int(
            record.get("gemm_events"), f"{context} gemm_events"
        )
        assumed_boundaries = nonnegative_int(
            record.get("assumed_preload_boundary_events"),
            f"{context} assumed_preload_boundary_events",
        )
        if gemm_events and not preload_events:
            raise ValueError(f"{context} GEMM events have no preload boundary")
        if assumed_boundaries > preload_events:
            raise ValueError(f"{context} assumed boundaries exceed preload events")
        event_count = preload_events + gemm_events
        events = record.get("events")
        if not isinstance(events, list) or len(events) != event_count:
            raise ValueError(f"{context} event list does not match event counts")
        counted_preloads = counted_gemms = counted_assumed = 0
        for index, event in enumerate(events):
            event = require_object(event, f"{context} event {index}")
            role = event.get("analytical_role")
            opcode = event.get("opcode")
            assumed = event.get("analytical_preload_boundary_assumed")
            if type(assumed) is not bool:
                raise ValueError(f"{context} event {index} has invalid assumed boundary flag")
            if role == "preload_boundary":
                counted_preloads += 1
                if (assumed and opcode != "GEMM") or (
                    not assumed and opcode != "GEMM_PRELOAD"
                ):
                    raise ValueError(f"{context} event {index} has invalid preload provenance")
                counted_assumed += assumed
            elif role == "gemm_consumer":
                counted_gemms += 1
                if opcode != "GEMM" or assumed:
                    raise ValueError(f"{context} event {index} has invalid GEMM provenance")
            else:
                raise ValueError(f"{context} event {index} has invalid analytical role")
        if (counted_preloads, counted_gemms, counted_assumed) != (
            preload_events, gemm_events, assumed_boundaries
        ):
            raise ValueError(f"{context} event roles do not match event counts")
        states = require_object(record.get("pe_state_cycles"), f"{context} pe_state_cycles")
        for field in STATES + MULTIPLICITY:
            value = nonnegative_int(states.get(field), f"{context} {field}")
            totals[field] += value
        if states["capacity"] != rows * columns * analytical_horizon:
            raise ValueError(f"{context} capacity does not match dimensions and horizon")
        if states["on"] + states["w_on"] + states["loading"] + states["off"] != states["capacity"]:
            raise ValueError(f"{context} state partition mismatch")
        if states["unclassified"] != 0:
            raise ValueError(f"{context} has unclassified state")
        if states["on_zero_banks"] + states["on_one_bank"] + states["on_two_banks"] != states["on"]:
            raise ValueError(f"{context} ON bank split mismatch")
        if states["on_zero_banks"] != 0:
            raise ValueError(f"{context} ON without a resident bank")
        if states["w_on_one_bank"] + states["w_on_two_banks"] != states["w_on"]:
            raise ValueError(f"{context} W_on bank split mismatch")
        exposure = require_object(record.get("preload_write_exposure"),
                                  f"{context} preload_write_exposure")
        write = nonnegative_int(exposure.get("pe_cycles"), f"{context} write exposure")
        while_on = nonnegative_int(exposure.get("while_on_pe_cycles"),
                                   f"{context} while-ON exposure")
        if exposure.get("disjoint_from_state_partition") is not False:
            raise ValueError(f"{context} invalid exposure overlap provenance")
        if write > states["capacity"] or while_on > min(write, states["on"]):
            raise ValueError(f"{context} impossible preload write exposure")
        if write - while_on != states["loading"]:
            raise ValueError(f"{context} preload write exposure disagrees with LOADING state")
        preload_write += write
        preload_while_on += while_on
        timing = require_object(record.get("timing_disagreement"),
                                f"{context} timing_disagreement")
        values = {field: nonnegative_int(timing.get(field), f"{context} {field}")
                  for field in EVENT_TIMING}
        for phase in ("start", "finish"):
            maximum = values[f"max_event_{phase}_lateness_cycles"]
            total = values[f"sum_event_{phase}_lateness_cycles"]
            if maximum > total or total > event_count * maximum:
                raise ValueError(f"{context} impossible event {phase} lateness")
            if maximum > values["analytical_trace_makespan_cycles"]:
                raise ValueError(f"{context} event {phase} lateness exceeds trace bound")
        if (values["simulator_trace_makespan_cycles"] > simulator_horizon or
                values["analytical_trace_makespan_cycles"] > analytical_horizon):
            raise ValueError(f"{context} trace makespan exceeds horizon")
        extension = max(0, values["analytical_trace_makespan_cycles"] -
                        values["simulator_trace_makespan_cycles"])
        if values["trace_makespan_extension_cycles"] != extension:
            raise ValueError(f"{context} trace makespan extension mismatch")
        for field in EVENT_TIMING[:4]:
            if field.startswith("max_"):
                timing_totals[field] = max(timing_totals[field], values[field])
            else:
                timing_totals[field] += values[field]
        timing_totals["sum_core_trace_makespan_extension_cycles"] = (
            timing_totals.get("sum_core_trace_makespan_extension_cycles", 0)
            + extension
        )
        max_trace_extension = max(max_trace_extension, extension)
        if reference_energy:
            reference_cores.append({"core_id": core_id,
                                    **calculate_reference(
                                        record, static_fraction,
                                        bank_leakage_share)})
            predictor = require_object(
                record.get("predictor_reference_whole_sa_logic"),
                f"{context} predictor_reference_whole_sa_logic")
            if (predictor.get("policy_evidence_class") != "extrapolated" or
                    predictor.get("domain") !=
                    "compute logic only; weight banks independent" or
                    predictor.get("break_even_cycles") != 469 or
                    predictor.get("wake_latency_cycles") != 10 or
                    predictor.get("timing_modified") is not False or
                    predictor.get("actual_stall_cycles") is not None):
                raise ValueError(f"{context} incompatible predictor reference")
            episodes = predictor.get("off_episodes")
            if not isinstance(episodes, list):
                raise ValueError(f"{context} missing predictor OFF episodes")
            last_end = 0
            off_cycles = 0
            for episode in episodes:
                episode = require_object(episode, f"{context} OFF episode")
                begin = nonnegative_int(episode.get("start_cycle"),
                                        f"{context} OFF start")
                end = nonnegative_int(episode.get("end_cycle"),
                                      f"{context} OFF end")
                if begin < last_end or end < begin or end > simulator_horizon:
                    raise ValueError(f"{context} invalid predictor OFF episode")
                off_cycles += end - begin
                last_end = end
            if off_cycles != nonnegative_int(predictor.get("off_cycles"),
                                             f"{context} predictor off_cycles"):
                raise ValueError(f"{context} predictor OFF cycle mismatch")
            predictor_whole_sa_cores.append({
                "core_id": core_id,
                "onnxim_horizon_cycles": simulator_horizon,
                "off_cycles": off_cycles,
                "late_wake_events": nonnegative_int(
                    predictor.get("late_wake_events"),
                    f"{context} late_wake_events"),
                "max_local_wake_lateness_cycles": nonnegative_int(
                    predictor.get("max_local_wake_lateness_cycles"),
                    f"{context} max_local_wake_lateness_cycles"),
            })
            if unified_reference:
                managed, replay = replay_shared_timeline(record)
                replayed.append((core_id, managed, replay))
    if totals["on"] + totals["w_on"] + totals["loading"] + totals["off"] != totals["capacity"]:
        raise ValueError("aggregate state partition mismatch")
    output = {
        "version": 1,
        "model": MODEL,
        "measurement_only": True,
        "control_path_modified": False,
        "functional_scheduling_unchanged": True,
        "timing_evidence_class": "assumed",
        "state_accounting_evidence_class": "derived",
        "validation_pass": True,
        "observed_cores": len(records),
        "pe_state_cycles": totals,
        "preload_write_exposure": {
            "pe_cycles": preload_write,
            "while_on_pe_cycles": preload_while_on,
            "disjoint_from_state_partition": False,
        },
        "timing_disagreement": {
            **{field: timing_totals[field] for field in EVENT_TIMING[:4]},
            "sum_core_trace_makespan_extension_cycles":
                timing_totals.get("sum_core_trace_makespan_extension_cycles", 0),
            "max_core_trace_makespan_extension_cycles": max_trace_extension,
            "interpretation": "event lateness sums are diagnostic and may double-count propagated delay; per-core trace extension is non-double-counted, not injected into ONNXim; core extensions do not establish global run penalty",
        },
        "energy_saving_status": "pending",
        "energy_saving_percent": None,
    }
    if reference_energy:
        bank_capacity = sum(item["bank_domain_cycles"]["capacity"]
                            for item in reference_cores)
        logic_capacity = sum(item["pe_logic_cycles"]["capacity"]
                             for item in reference_cores)
        bank_energy = sum(
            item["bank_domain_cycles"]["on"] +
            .94 * item["bank_domain_cycles"]["ss"] +
            .42 * item["bank_domain_cycles"]["ds"]
            for item in reference_cores)
        logic_energy = sum(
            item["pe_logic_cycles"]["on"] +
            .03 * item["pe_logic_cycles"]["off"]
            for item in reference_cores)
        bank_saving = 100 * (1 - bank_energy / bank_capacity)
        one_bank_capacity = sum(
            item["one_bank_always_on_baseline_domain_cycles"]
            for item in reference_cores)
        bank_saving_vs_one_bank = 100 * (1 - bank_energy / one_bank_capacity)
        logic_saving = 100 * (1 - logic_energy / logic_capacity)
        combined_static = (bank_leakage_share * bank_saving +
                           (1 - bank_leakage_share) * logic_saving)
        output["reference_energy"] = {
            "evidence_class": "extrapolated",
            "scope": "offline analytical-shadow gross SA energy opportunity",
            "overall_reference_energy_saving_percent": None,
            "gross_analytical_shadow_energy_opportunity_percent":
                static_fraction * combined_static,
            "one_bank_baseline_shadow_sensitivity_percent":
                static_fraction * (
                    bank_leakage_share * bank_saving_vs_one_bank +
                    (1 - bank_leakage_share) * logic_saving),
            "bank_leakage_saving_percent": bank_saving,
            "bank_leakage_saving_vs_one_bank_percent":
                bank_saving_vs_one_bank,
            "pe_logic_leakage_saving_percent": logic_saving,
            "combined_sa_static_saving_percent": combined_static,
            "static_fraction_assumed": static_fraction,
            "bank_share_of_sa_static_assumed": bank_leakage_share,
            "baselines": {
                "primary": "hypothetical two weight banks always ON and PE logic always ON over the analytical shadow horizon",
                "one_bank_sensitivity": "hypothetical one weight bank always ON and PE logic always ON over the same analytical horizon; bank share is reinterpreted relative to this baseline",
                "onnxim_physical_baseline": "pending",
            },
            "per_core": reference_cores,
            "excluded_energy": reference_cores[0]["excluded_energy"],
            "estimated_performance_penalty_percent": None,
            "performance_penalty_status":
                "pending: runtime gating does not reschedule dependent work",
            "realized_predictor_energy_saving_status": "pending",
            "predictor_whole_sa_logic": {
                "timeline": "ONNXim cycles; separate from analytical shadow",
                "per_core": predictor_whole_sa_cores,
                "overlap_with_analytical_pe_states": "unavailable",
            },
            "sa_tail_schedule_extension_diagnostic": (
                {"status": "derived",
                 "common_onnxim_run_horizon_cycles": simulator_horizons[0],
                 "max_core_analytical_horizon_cycles":
                     max(analytical_horizons),
                 "percent_of_common_run_horizon": 100 *
                     (max(analytical_horizons) - simulator_horizons[0]) /
                     simulator_horizons[0],
                 "interpretation": "analytical SA tail extension only; not gating slowdown"}
                if (simulator_horizons[0] > 0 and
                    len(set(simulator_horizons)) == 1) else
                {"status": "pending",
                 "reason": "cores lack one positive common ONNXim run horizon"}
            ),
        }
    if unified_reference:
        hardware = {(record["array_rows"], record["array_columns"],
                     record.get("weight_bytes")) for record in records}
        if len(hardware) != 1:
            raise ValueError("unified core energy sum requires identical SA dimensions and weight width")
        base_horizon = max(analytical_horizons)
        managed_horizon = max(
            item[1]["analytical_horizon_cycles"] for item in replayed)
        if assume_zero_overheads:
            if transition_on_equivalent_cycles is None:
                transition_on_equivalent_cycles = 0.0
            if overhead_fraction_of_baseline is None:
                overhead_fraction_of_baseline = 0.0
        numeric_energy = (transition_on_equivalent_cycles is not None and
                          overhead_fraction_of_baseline is not None)
        energy_by_core = []
        for core_id, managed, replay in replayed:
            # Every core's banks and logic remain present through the common
            # run horizon, including after its final local SA event.
            managed["analytical_horizon_cycles"] = managed_horizon
            managed["pe_state_cycles"]["capacity"] = (
                managed["array_rows"] * managed["array_columns"] *
                managed_horizon)
            reference = calculate_reference(managed, static_fraction,
                                            bank_leakage_share)
            energy = (calculate_unified_assumed_energy(
                base_horizon, reference, static_fraction,
                bank_leakage_share, transition_on_equivalent_cycles,
                overhead_fraction_of_baseline)
                if numeric_energy else None)
            energy_by_core.append({"core_id": core_id, "replay": replay,
                                   "energy": energy,
                                   "bank_domain_cycles":
                                       reference["bank_domain_cycles"],
                                   "pe_logic_cycles":
                                       reference["pe_logic_cycles"]})
        baseline_one = (sum(core["energy"]["baseline_one_bank_normalized_energy"]
                            for core in energy_by_core) if numeric_energy else None)
        baseline_two = (sum(core["energy"]["baseline_two_bank_normalized_energy"]
                            for core in energy_by_core) if numeric_energy else None)
        managed_total = (sum(core["energy"]["managed_normalized_energy"]
                             for core in energy_by_core) if numeric_energy else None)
        output["unified_reference"] = {
            "evidence_class": "extrapolated",
            "scope": "common-cycle analytical SA trace replay",
            "baseline_ungated_analytical_horizon_cycles": base_horizon,
            "managed_analytical_horizon_cycles": managed_horizon,
            "modeled_sa_schedule_penalty_percent":
                100 * (managed_horizon - base_horizon) / base_horizon,
            "overall_assumed_saving_vs_one_bank_percent":
                100 * (1 - managed_total / baseline_one)
                if numeric_energy else None,
            "overall_assumed_saving_vs_two_bank_percent":
                100 * (1 - managed_total / baseline_two)
                if numeric_energy else None,
            "normalized_baseline_one_bank_energy": baseline_one,
            "normalized_baseline_two_bank_energy": baseline_two,
            "normalized_managed_energy": managed_total,
            "baseline_scope":
                "hypothetical one-bank or two-bank always-ON power on the same ungated two-bank analytical schedule; not an ONNXim one-bank timing result",
            "energy_status": ("extrapolated explicit reference scenario"
                              if numeric_energy else
                              "pending: provide transition and overhead costs, or explicitly choose --assume-zero-overheads"),
            "per_core": energy_by_core,
            "actual_end_to_end_penalty_status":
                "pending: non-SA dependencies and memory timing are not rescheduled",
            "actual_net_circuit_saving_status":
                "pending: transition, controller, selector, routing and extra loading costs are uncharacterized",
            "predictor_replay_limit":
                "ONNXim predictor OFF episodes are accepted only while the ungated analytical SA trace matches ONNXim; later episodes are discarded after timing divergence or managed stall",
            "whole_sa_off_accounting":
                "subset of PE-logic OFF; no second static saving term",
            "local_wake_policy":
                "offline next-access knowledge reserves ON wake guards of 1 PE cycle and 4/13/16 bank cycles; these local wakes cause no modeled stall",
            "transition_cost_units":
                "normalized full-SA static-energy cycles per core, incremental above mode leakage",
            "assumptions": (energy_by_core[0]["energy"]["assumptions"]
                            if numeric_energy else {
                                "static_fraction": static_fraction,
                                "bank_share_of_original_one_bank_sa_static":
                                    bank_leakage_share,
                                "transition_on_equivalent_cycles":
                                    transition_on_equivalent_cycles,
                                "overhead_fraction_of_baseline":
                                    overhead_fraction_of_baseline}),
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--expected-cores", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-energy", action="store_true",
                        help="calculate the explicit 50/50 gross SA reference scenario")
    parser.add_argument("--unified-reference", action="store_true",
                        help="replay matching predictor episodes and compute an assumed common-cycle SA scenario")
    parser.add_argument("--static-fraction", type=float, default=0.5,
                        help="assumed SA static-energy fraction (default: 0.5; not measured)")
    parser.add_argument("--bank-leakage-share", type=float, default=0.5,
                        help="assumed bank share of SA static energy (default: 0.5; not measured)")
    parser.add_argument("--transition-on-equivalent-cycles", type=float,
                        help="incremental transition energy in normalized full-SA static-cycle units per core")
    parser.add_argument("--overhead-fraction-of-baseline", type=float,
                        help="extra controller/selector/loading cost fraction")
    parser.add_argument("--assume-zero-overheads", action="store_true",
                        help="explicit gross scenario with unspecified transition and other overheads set to zero")
    args = parser.parse_args()
    result = aggregate(read_records(args.log), args.expected_cores,
                       args.reference_energy, args.static_fraction,
                       args.bank_leakage_share, args.unified_reference,
                       args.transition_on_equivalent_cycles,
                       args.overhead_fraction_of_baseline,
                       args.assume_zero_overheads)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.resolve() == args.log.resolve():
            parser.error("--output must not overwrite the raw simulation log")
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
