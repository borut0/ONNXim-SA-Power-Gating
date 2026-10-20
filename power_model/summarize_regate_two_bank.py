#!/usr/bin/env python3
"""Aggregate version-2 two-bank ReGate-like ONNXim shadow records.

State exposure is derived from validated ONNXim traces. Optional power factors
are explicit assumptions for a gross static sensitivity only; they are not
ReGate circuit measurements or net energy.
"""

import argparse
import json
import math
import numbers
from pathlib import Path


PREFIX = "REGATE_TWO_BANK_PROFILE "
MODEL = "two-bank ReGate-like analytical shadow"
STATES = (
    "capacity", "on", "on_zero_banks", "on_one_bank", "on_two_banks",
    "w_on_one_bank", "w_on_two_banks", "off", "unclassified",
)
POWER_STATES = (
    "on_one_bank", "on_two_banks", "w_on_one_bank",
    "w_on_two_banks", "off",
)
ROWWISE_SUMS = (
    "mapped_pe_weight_write_events", "row_write_before_issue_events",
    "row_ready_after_first_mac_events", "mapped_pe_over_two_pe_cycles",
    "mapped_pes_ever_over_two", "locally_blocked_pe_weight_write_events",
)
ROWWISE_MAXIMA = (
    "max_row_ready_lateness_cycles", "mapped_pe_peak_live_preloads",
    "max_local_bank_wait_cycles", "max_local_first_mac_delay_cycles",
)
ROWWISE_STATE_FIELDS = (
    "capacity", "on", "on_preload_wave", "on_gemm_wave",
    "on_zero_banks", "on_one_bank", "on_two_banks",
    "w_on_one_bank", "w_on_two_banks", "off", "unclassified",
)
ROWWISE_ASSUMPTIONS = (
    "first_row_lead_cycles_before_start", "row_start_period_cycles",
    "row_write_duration_cycles", "row_order", "row_write_start_equation",
    "row_ready_equation", "bank_release_equation",
    "row_columns_written_simultaneously_assumption",
)
VALIDATION_COUNTERS = (
    "incomplete_intervals", "invalid_intervals", "unknown_weight_intervals",
    "gemm_without_weight_intervals", "footprint_mismatch_intervals",
    "weight_not_ready_intervals", "weight_evicted_before_use_intervals",
    "wave_exceeds_instruction_intervals", "compute_collisions",
    "two_bank_conflicts",
)
ATOMIC_ONLY_FAILURES = (
    "two_bank_conflicts", "weight_evicted_before_use_intervals",
)


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative exact integer")
    return value


def read_records(path):
    records = []
    for line_number, line in enumerate(
            path.read_text(errors="replace").splitlines(), 1):
        position = line.find(PREFIX)
        if position < 0:
            continue
        try:
            records.append(json.loads(line[position + len(PREFIX):]))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{path}:{line_number}: invalid {PREFIX.strip()} JSON"
            ) from error
    if not records:
        raise ValueError(f"{path}: no {PREFIX.strip()} records")
    return records


def _aggregate_rowwise(records, expected_cores):
    """No partial aggregate: every expected core must share one hypothesis."""
    def pending(reason):
        return {
            "available": False,
            "evidence_class": "pending",
            "reason": reason,
            "physical_feasibility_proven": False,
            "actual_stall_cycles": None,
            "net_energy_saving_percent": None,
        }

    if expected_cores is None:
        return pending("--expected-cores is required for a complete rowwise aggregate")
    assumptions = None
    totals = {name: 0 for name in ROWWISE_SUMS}
    totals.update({name: 0 for name in ROWWISE_MAXIMA})
    necessary_pass = True
    for record in records:
        core_id = record["core_id"]
        area = record["array_rows"] * record["array_columns"]
        preload_events = record["preload_events"]
        rowwise = record.get("rowwise_feasibility")
        if not isinstance(rowwise, dict) or rowwise.get("available") is not True:
            detail = (rowwise.get("reason") if isinstance(rowwise, dict)
                      else "rowwise_feasibility missing")
            return pending(f"core {core_id}: rowwise diagnostic pending: {detail}")
        try:
            mapped_writes = _integer(
                rowwise.get("mapped_pe_weight_write_events"),
                "mapped_pe_weight_write_events",
            )
            preissue = _integer(rowwise.get("row_write_before_issue_events"),
                                "row_write_before_issue_events")
        except ValueError as error:
            return pending(f"core {core_id}: {error}")
        instruction_count = record["sa_instruction_intervals"]
        if preload_events == 0:
            if instruction_count != 0 or mapped_writes != 0:
                return pending(f"core {core_id}: zero-preload rowwise state requires an empty trace")
        elif mapped_writes < preload_events:
            return pending(f"core {core_id}: mapped PE writes below preload lower bound")
        if mapped_writes > area * preload_events:
            return pending(f"core {core_id}: mapped PE writes exceed preload footprint bound")
        if preissue > mapped_writes:
            return pending(f"core {core_id}: pre-issue writes exceed mapped writes")
        complete = rowwise.get("secondary_counts_complete")
        if type(complete) is not bool or complete != (preissue == 0):
            return pending(f"core {core_id}: inconsistent secondary_counts_complete")
        if not complete:
            secondary = (ROWWISE_SUMS[2:] + ROWWISE_MAXIMA)
            if any(rowwise.get(name) is not None for name in secondary):
                return pending(f"core {core_id}: incomplete secondary counts must be null")
            return pending(f"core {core_id}: rowwise secondary counts incomplete")
        if (rowwise.get("evidence_class") != "derived" or
                rowwise.get("timing_input_evidence_class") != "assumed"):
            return pending(f"core {core_id}: incompatible rowwise evidence labels")
        if rowwise.get("physical_feasibility_proven") is not False:
            return pending(f"core {core_id}: unsupported physical-feasibility label")
        if rowwise.get("row_columns_written_simultaneously_assumption") is not True:
            return pending(f"core {core_id}: missing simultaneous-column assumption")
        if rowwise.get("row_order") not in ("ascending", "descending"):
            return pending(f"core {core_id}: invalid row order")
        for name in ("first_row_lead_cycles_before_start",
                     "row_start_period_cycles", "row_write_duration_cycles"):
            try:
                value = _integer(rowwise.get(name), name)
            except ValueError as error:
                return pending(f"core {core_id}: {error}")
            if name != "first_row_lead_cycles_before_start" and value == 0:
                return pending(f"core {core_id}: {name} must be positive")
        for name in ("row_write_start_equation", "row_ready_equation",
                     "bank_release_equation"):
            if not isinstance(rowwise.get(name), str) or not rowwise[name].strip():
                return pending(f"core {core_id}: missing {name}")
        current = {name: rowwise[name] for name in ROWWISE_ASSUMPTIONS}
        if assumptions is None:
            assumptions = current
        elif current != assumptions:
            differing = next(name for name in ROWWISE_ASSUMPTIONS
                             if current[name] != assumptions[name])
            return pending(f"core {core_id}: incompatible {differing}")
        if type(rowwise.get("necessary_conditions_pass")) is not bool:
            return pending(f"core {core_id}: missing necessary_conditions_pass")
        necessary_pass &= rowwise["necessary_conditions_pass"]
        for name in ROWWISE_SUMS + ROWWISE_MAXIMA:
            try:
                value = _integer(rowwise.get(name), name)
            except ValueError as error:
                return pending(f"core {core_id}: {error}")
            if name in ROWWISE_SUMS:
                totals[name] += value
            else:
                totals[name] = max(totals[name], value)
        if rowwise["row_ready_after_first_mac_events"] > mapped_writes:
            return pending(f"core {core_id}: late writes exceed mapped writes")
        if rowwise["locally_blocked_pe_weight_write_events"] > mapped_writes:
            return pending(f"core {core_id}: blocked writes exceed mapped writes")
        if rowwise["mapped_pes_ever_over_two"] > area:
            return pending(f"core {core_id}: affected PEs exceed array area")
        if 3 * rowwise["mapped_pes_ever_over_two"] > mapped_writes:
            return pending(f"core {core_id}: affected PEs exceed mapped-write bound")
        if rowwise["mapped_pe_peak_live_preloads"] > preload_events:
            return pending(f"core {core_id}: rowwise peak exceeds preload events")
        if (preload_events > 0 and
                rowwise["mapped_pe_peak_live_preloads"] == 0):
            return pending(f"core {core_id}: complete rowwise peak must be positive")
        has_excess = rowwise["mapped_pe_over_two_pe_cycles"] > 0
        if ((rowwise["mapped_pes_ever_over_two"] > 0) != has_excess or
                (rowwise["mapped_pe_peak_live_preloads"] > 2) != has_excess):
            return pending(f"core {core_id}: rowwise excess/peak/affected-PE counters contradict")
        late = rowwise["row_ready_after_first_mac_events"]
        blocked = rowwise["locally_blocked_pe_weight_write_events"]
        if (late == 0) != (rowwise["max_row_ready_lateness_cycles"] == 0):
            return pending(f"core {core_id}: late-event/max-lateness counters contradict")
        if (blocked == 0) != (rowwise["max_local_bank_wait_cycles"] == 0):
            return pending(f"core {core_id}: blocked-event/max-wait counters contradict")
        if late == 0 and blocked == 0 and rowwise["max_local_first_mac_delay_cycles"]:
            return pending(f"core {core_id}: unexplained local first-MAC delay")
        expected_pass = (rowwise["row_ready_after_first_mac_events"] == 0 and
                         rowwise["mapped_pe_over_two_pe_cycles"] == 0)
        if rowwise["necessary_conditions_pass"] != expected_pass:
            return pending(f"core {core_id}: necessary_conditions_pass contradicts counters")

    return {
        "available": True,
        "evidence_class": "derived",
        "timing_input_evidence_class": "assumed",
        "assumptions": assumptions,
        "observed_cores": len(records),
        "necessary_conditions_pass": necessary_pass,
        "physical_feasibility_proven": False,
        "actual_stall_cycles": None,
        "net_energy_saving_percent": None,
        "stall_status": (
            "pending: local wait fields are conditional diagnostics, "
            "not actual or globally rescheduled stalls"
        ),
        **totals,
    }


def _aggregate_preload_feasibility(records, expected_cores):
    """Validate and aggregate whole-bank fixed-schedule readiness events."""
    def pending(reason):
        return {
            "available": False,
            "evidence_class": "pending",
            "reason": reason,
            "physical_feasibility_proven": False,
            "actual_stall_cycles": None,
        }

    if expected_cores is None:
        return pending("--expected-cores is required for a complete preload feasibility aggregate")
    totals = {
        "audited_preloads": 0,
        "no_added_cycle_preloads": 0,
        "conflicting_preloads": 0,
        "total_required_stall_cycles": 0,
        "max_required_stall_cycles": 0,
    }
    offenders = []
    for record in records:
        core_id = record["core_id"]
        audit = record.get("preload_feasibility_audit")
        if not isinstance(audit, dict) or audit.get("available") is not True:
            return pending(f"core {core_id}: preload_feasibility_audit unavailable")
        if (audit.get("version") != 1 or
                audit.get("evidence_class") != "derived" or
                audit.get("timing_input_evidence_class") != "assumed" or
                audit.get("physical_feasibility_proven") is not False):
            return pending(f"core {core_id}: incompatible preload feasibility evidence labels")
        rowwise = record.get("rowwise_feasibility")
        if not isinstance(rowwise, dict):
            return pending(f"core {core_id}: row timing inputs unavailable")
        try:
            period = _integer(rowwise.get("row_start_period_cycles"),
                              "row_start_period_cycles")
            duration = _integer(rowwise.get("row_write_duration_cycles"),
                                "row_write_duration_cycles")
        except ValueError as error:
            return pending(f"core {core_id}: {error}")
        if period == 0 or duration == 0:
            return pending(f"core {core_id}: row period and duration must be positive")
        events = audit.get("events")
        if not isinstance(events, list) or len(events) != record["preload_events"]:
            return pending(f"core {core_id}: incomplete per-PRELOAD feasibility events")
        core_no_added = 0
        core_conflicts = 0
        core_total = 0
        core_max = 0
        computed_offenders = []
        for expected_index, event in enumerate(events):
            if not isinstance(event, dict):
                return pending(f"core {core_id}: malformed preload event")
            try:
                preload_index = _integer(event.get("preload_index"),
                                         "preload_index")
                issue = _integer(event.get("issue_cycle"), "issue_cycle")
                start = _integer(event.get("start_cycle"), "start_cycle")
                rows = _integer(event.get("rows"), "rows")
                columns = _integer(event.get("columns"), "columns")
                fill = _integer(event.get("fill_span_cycles"),
                                "fill_span_cycles")
                load = _integer(event.get("load_begin"), "load_begin")
                ready = _integer(event.get("bank_ready"), "bank_ready")
                first_use = _integer(event.get("first_use"), "first_use")
                stall = _integer(event.get("required_stall_cycles"),
                                 "required_stall_cycles")
            except ValueError as error:
                return pending(f"core {core_id}: {error}")
            if (preload_index != expected_index or rows == 0 or
                    columns == 0 or rows > record["array_rows"] or
                    columns > record["array_columns"]):
                return pending(f"core {core_id}: invalid preload event ordering or footprint")
            expected_fill = (rows - 1) * period + duration
            if fill != expected_fill or first_use != start:
                return pending(f"core {core_id}: preload fill/first-use equation mismatch")
            planned = event.get("planned_load_begin")
            expected_planned = start - fill if fill <= start else None
            if planned != expected_planned:
                return pending(f"core {core_id}: planned load-begin equation mismatch")
            expected_load = max(issue, expected_planned if expected_planned is not None else issue)
            expected_ready = expected_load + fill
            expected_stall = max(0, expected_ready - first_use)
            if (load != expected_load or ready != expected_ready or
                    stall != expected_stall):
                return pending(f"core {core_id}: causal readiness equation mismatch")
            expected_outcome = "no_added_cycle" if stall == 0 else "conflict"
            if event.get("outcome") != expected_outcome:
                return pending(f"core {core_id}: preload outcome contradicts stall")
            if stall == 0:
                core_no_added += 1
            else:
                core_conflicts += 1
                computed_offenders.append(event)
                offenders.append({"core_id": core_id, **event})
            core_total += stall
            core_max = max(core_max, stall)
        expected_core = {
            "audited_preloads": len(events),
            "no_added_cycle_preloads": core_no_added,
            "conflicting_preloads": core_conflicts,
            "total_required_stall_cycles": core_total,
            "max_required_stall_cycles": core_max,
        }
        for name, value in expected_core.items():
            try:
                actual = _integer(audit.get(name), name)
            except ValueError as error:
                return pending(f"core {core_id}: {error}")
            if actual != value:
                return pending(f"core {core_id}: {name} contradicts events")
        if audit.get("offending_events") != computed_offenders:
            return pending(f"core {core_id}: offending events are not the exact conflict subset")
        for name, value in expected_core.items():
            if name == "max_required_stall_cycles":
                totals[name] = max(totals[name], value)
            else:
                totals[name] += value
    return {
        "available": True,
        "evidence_class": "derived",
        "timing_input_evidence_class": "assumed",
        "physical_feasibility_proven": False,
        "actual_stall_cycles": None,
        "total_interpretation": (
            "sum of independent fixed-schedule per-PRELOAD stalls; not a "
            "globally propagated runtime or an ONNXim cycle change"
        ),
        "offending_events": offenders,
        **totals,
    }


def _aggregate_rowwise_states(records, expected_cores, rowwise_summary):
    """Aggregate only complete version-1 row-schedule state partitions."""
    def pending(reason):
        return {
            "version": 1,
            "available": False,
            "evidence_class": "pending",
            "reason": reason,
            "validation_pass": None,
            "physical_feasibility_proven": False,
            "actual_stall_cycles": None,
            "pe_state_cycles": {name: None for name in ROWWISE_STATE_FIELDS},
            "preload_write_exposure": {
                "evidence_class": "pending",
                "timing_input_evidence_class": "assumed",
                "pe_cycles": None,
                "while_on_pe_cycles": None,
                "disjoint_from_state_partition": False,
            },
            "gross_static_saving_percent": None,
            "net_energy_saving_percent": None,
        }

    if expected_cores is None:
        return pending("--expected-cores is required for a complete rowwise state aggregate")
    if not rowwise_summary.get("available"):
        return pending("rowwise feasibility aggregate is unavailable: " +
                       rowwise_summary.get("reason", "unspecified reason"))
    per_core = []
    aggregate_capacity = 0
    aggregate_states = {name: 0 for name in ROWWISE_STATE_FIELDS}
    aggregate_write = 0
    aggregate_write_on = 0
    all_pass = True
    for record in records:
        core_id = record["core_id"]
        item = record.get("rowwise_state_accounting")
        if not isinstance(item, dict):
            return pending(f"core {core_id}: rowwise_state_accounting missing")
        try:
            if _integer(item.get("version"), "rowwise state version") != 1:
                return pending(f"core {core_id}: unsupported rowwise state version")
        except ValueError as error:
            return pending(f"core {core_id}: {error}")
        if item.get("available") is not True:
            return pending(f"core {core_id}: rowwise state accounting unavailable")
        if (item.get("evidence_class") != "derived" or
                item.get("timing_input_evidence_class") != "assumed"):
            return pending(f"core {core_id}: incompatible rowwise state evidence labels")
        if item.get("physical_feasibility_proven") is not False:
            return pending(f"core {core_id}: unsupported physical-feasibility label")
        validation_pass = item.get("validation_pass")
        if type(validation_pass) is not bool:
            return pending(f"core {core_id}: missing rowwise state validation_pass")
        checks = item.get("required_checks")
        if not isinstance(checks, dict):
            return pending(f"core {core_id}: required_checks missing")
        if checks.get("complete_known_matched_trace") is not True:
            return pending(f"core {core_id}: trace identity checks incomplete")
        timing_pass = checks.get("rowwise_timing_and_two_bank_checks")
        partition_pass = checks.get("state_partition")
        if type(timing_pass) is not bool or type(partition_pass) is not bool:
            return pending(f"core {core_id}: required check result missing")
        if validation_pass != (timing_pass and partition_pass):
            return pending(f"core {core_id}: validation_pass contradicts required checks")
        feasibility = record.get("rowwise_feasibility")
        if (not isinstance(feasibility, dict) or
                type(feasibility.get("necessary_conditions_pass")) is not bool or
                timing_pass != feasibility["necessary_conditions_pass"]):
            return pending(f"core {core_id}: rowwise timing check contradicts feasibility record")
        state = item.get("pe_state_cycles")
        if not isinstance(state, dict):
            return pending(f"core {core_id}: pe_state_cycles missing")
        try:
            values = {name: _integer(state.get(name), name)
                      for name in ROWWISE_STATE_FIELDS}
        except ValueError as error:
            return pending(f"core {core_id}: {error}")
        expected_capacity = (record["array_rows"] * record["array_columns"] *
                             record["core_cycles"])
        if values["capacity"] != expected_capacity:
            return pending(f"core {core_id}: rowwise state capacity mismatch")
        exposure = item.get("preload_write_exposure")
        if not isinstance(exposure, dict):
            return pending(f"core {core_id}: preload_write_exposure missing")
        expected_exposure_class = "derived" if validation_pass else "pending"
        if (exposure.get("evidence_class") != expected_exposure_class or
                exposure.get("timing_input_evidence_class") != "assumed"):
            return pending(f"core {core_id}: incompatible preload-write evidence labels")
        if validation_pass:
            try:
                write = _integer(exposure.get("pe_cycles"), "preload write PE-cycles")
                write_on = _integer(exposure.get("while_on_pe_cycles"),
                                    "preload write while-ON PE-cycles")
            except ValueError as error:
                return pending(f"core {core_id}: {error}")
        else:
            if (exposure.get("pe_cycles") is not None or
                    exposure.get("while_on_pe_cycles") is not None):
                return pending(f"core {core_id}: failed rowwise write exposure must be withheld")
            write = 0
            write_on = 0
        if exposure.get("disjoint_from_state_partition") is not False:
            return pending(f"core {core_id}: write exposure must overlap the state partition")
        if write_on > write or write > expected_capacity:
            return pending(f"core {core_id}: invalid preload-write exposure")
        if validation_pass:
            if values["unclassified"] != 0:
                return pending(f"core {core_id}: validated state has unclassified capacity")
            if values["on"] != values["on_preload_wave"] + values["on_gemm_wave"]:
                return pending(f"core {core_id}: ON opcode split mismatch")
            if values["on"] != (values["on_zero_banks"] +
                                 values["on_one_bank"] +
                                 values["on_two_banks"]):
                return pending(f"core {core_id}: ON bank split mismatch")
            if values["on_zero_banks"] != 0:
                return pending(f"core {core_id}: validated ON state lacks a live bank")
            if values["capacity"] != (values["on"] +
                                       values["w_on_one_bank"] +
                                       values["w_on_two_banks"] +
                                       values["off"]):
                return pending(f"core {core_id}: rowwise PE-state partition mismatch")
        else:
            if values["unclassified"] != values["capacity"] or any(
                    values[name] for name in ROWWISE_STATE_FIELDS
                    if name not in ("capacity", "unclassified")):
                return pending(f"core {core_id}: failed rowwise state is not fail-closed")
        all_pass &= validation_pass
        aggregate_capacity += expected_capacity
        for name in ROWWISE_STATE_FIELDS:
            aggregate_states[name] += values[name]
        aggregate_write += write
        aggregate_write_on += write_on
        per_core.append({"core_id": core_id, "validation_pass": validation_pass,
                         "pe_state_cycles": values,
                         "preload_write_pe_cycles": write,
                         "preload_write_while_on_pe_cycles": write_on})

    # A mixed aggregate fails closed as a whole; no validated subset is
    # presented as complete workload state accounting.
    if all_pass and not rowwise_summary.get("necessary_conditions_pass"):
        return pending("validated states contradict failed rowwise necessary conditions")
    if not all_pass:
        aggregate_states = {name: 0 for name in ROWWISE_STATE_FIELDS}
        aggregate_states["capacity"] = aggregate_capacity
        aggregate_states["unclassified"] = aggregate_capacity
        aggregate_write = None
        aggregate_write_on = None
    return {
        "version": 1,
        "available": True,
        "evidence_class": "derived",
        "timing_input_evidence_class": "assumed",
        "validation_pass": all_pass,
        "physical_feasibility_proven": False,
        "actual_stall_cycles": None,
        "per_core": sorted(per_core, key=lambda item: item["core_id"]),
        "pe_state_cycles": aggregate_states,
        "preload_write_exposure": {
            "evidence_class": "derived" if all_pass else "pending",
            "timing_input_evidence_class": "assumed",
            "pe_cycles": aggregate_write,
            "while_on_pe_cycles": aggregate_write_on,
            "disjoint_from_state_partition": False,
        },
        "gross_static_saving_percent": None,
        "net_energy_saving_percent": None,
        "energy_status": "pending: theoretical PE-state exposure only; no state power, transition, control, or wake evidence",
    }


def _validate_atomic_fail_closed(record, core_id, rows, columns, cycles):
    state = record.get("pe_state_cycles")
    if not isinstance(state, dict):
        raise ValueError(f"core {core_id} lacks fail-closed pe_state_cycles")
    values = {name: _integer(state.get(name), name) for name in STATES}
    capacity = rows * columns * cycles
    if values["capacity"] != capacity or values["unclassified"] != capacity:
        raise ValueError(f"core {core_id} atomic failure is not fail-closed")
    if any(values[name] for name in STATES
           if name not in ("capacity", "unclassified")):
        raise ValueError(f"core {core_id} atomic failure has classified state")
    if _integer(record.get("preload_bank_pe_cycles"),
                "preload_bank_pe_cycles") != 0:
        raise ValueError(f"core {core_id} atomic failure has preload bank cycles")
    validation = record.get("validation")
    if not isinstance(validation, dict):
        raise ValueError(f"core {core_id} lacks validation counters")
    counters = {name: _integer(validation.get(name), name)
                for name in VALIDATION_COUNTERS}
    if any(counters[name] for name in VALIDATION_COUNTERS
           if name not in ATOMIC_ONLY_FAILURES):
        raise ValueError(f"core {core_id} has non-atomic validation failure")
    if not any(counters[name] for name in ATOMIC_ONLY_FAILURES):
        raise ValueError(f"core {core_id} lacks an atomic bank failure counter")
    preload_events = record["preload_events"]
    instruction_count = record["sa_instruction_intervals"]
    if counters["two_bank_conflicts"] > max(0, preload_events - 2):
        raise ValueError(f"core {core_id} two-bank conflicts exceed preload bound")
    if counters["weight_evicted_before_use_intervals"] > instruction_count:
        raise ValueError(f"core {core_id} evicted consumers exceed instruction count")


def aggregate(records, expected_cores=None, power_factors=None,
              factor_source=None, rowwise_diagnostic_only=False,
              rowwise_state_accounting=False):
    if not records:
        raise ValueError("no two-bank profile records")
    if expected_cores is not None:
        if _integer(expected_cores, "expected_cores") == 0:
            raise ValueError("expected_cores must be positive")
        if len(records) != expected_cores:
            raise ValueError(
                f"expected {expected_cores} cores, found {len(records)}"
            )
    if power_factors is None and factor_source is not None:
        raise ValueError("factor_source requires power_factors")
    if rowwise_diagnostic_only and expected_cores is None:
        raise ValueError("rowwise diagnostic-only mode requires expected_cores")
    if rowwise_state_accounting and expected_cores is None:
        raise ValueError("rowwise state-accounting mode requires expected_cores")
    if rowwise_diagnostic_only and rowwise_state_accounting:
        raise ValueError("rowwise modes are mutually exclusive")
    if rowwise_diagnostic_only and power_factors is not None:
        raise ValueError("rowwise diagnostic-only mode rejects power factors")
    if rowwise_state_accounting and power_factors is not None:
        raise ValueError("rowwise state-accounting mode rejects power factors")
    if power_factors is not None:
        if set(power_factors) != set(POWER_STATES):
            raise ValueError("all five two-bank power factors are required")
        if not isinstance(factor_source, str) or not factor_source.strip():
            raise ValueError("an explicit assumed factor source is required")
        for state, factor in power_factors.items():
            if (isinstance(factor, bool) or
                    not isinstance(factor, numbers.Real) or
                    not math.isfinite(factor) or factor < 0):
                raise ValueError(
                    f"{state} power factor must be finite and nonnegative"
                )

    totals = {name: 0 for name in STATES}
    totals["preload_events"] = 0
    totals["gemm_events"] = 0
    totals["preload_bank_pe_cycles"] = 0
    core_ids = set()
    dimensions = set()
    per_core = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("each profile record must be an object")
        if _integer(record.get("version"), "version") != 2:
            raise ValueError("unsupported two-bank profile version")
        if record.get("model") != MODEL:
            raise ValueError("unexpected two-bank profile model")
        if record.get("regate_only") is not True or (
                record.get("combined_with_retention") is not False):
            raise ValueError("mixed or unlabeled power model")
        if type(record.get("validation_pass")) is not bool:
            raise ValueError("two-bank profile lacks boolean validation_pass")
        special_rowwise_mode = (rowwise_diagnostic_only or
                                rowwise_state_accounting)
        if not special_rowwise_mode and record["validation_pass"] is not True:
            raise ValueError("two-bank profile validation failed")
        core_id = _integer(record.get("core_id"), "core_id")
        if core_id in core_ids:
            raise ValueError(f"duplicate two-bank record for core {core_id}")
        core_ids.add(core_id)
        rows = _integer(record.get("array_rows"), "array_rows")
        columns = _integer(record.get("array_columns"), "array_columns")
        cycles = _integer(record.get("core_cycles"), "core_cycles")
        if rows == 0 or columns == 0:
            raise ValueError("array dimensions must be positive")
        dimensions.add((rows, columns))
        instruction_count = _integer(
            record.get("sa_instruction_intervals"),
            "sa_instruction_intervals",
        )
        preload_events = _integer(record.get("preload_events"),
                                  "preload_events")
        gemm_events = _integer(record.get("gemm_events"), "gemm_events")
        if instruction_count != preload_events + gemm_events:
            raise ValueError(f"core {core_id} instruction count mismatch")
        if special_rowwise_mode:
            if record["validation_pass"] is False:
                _validate_atomic_fail_closed(record, core_id,
                                             rows, columns, cycles)
            per_core.append({"core_id": core_id,
                             "legacy_validation_pass":
                                 record["validation_pass"]})
            continue
        state = record.get("pe_state_cycles")
        if not isinstance(state, dict):
            raise ValueError(f"core {core_id} lacks pe_state_cycles")
        values = {name: _integer(state.get(name), name) for name in STATES}
        if values["capacity"] != rows * columns * cycles:
            raise ValueError(f"core {core_id} capacity mismatch")
        if values["on"] != (
                values["on_zero_banks"] + values["on_one_bank"] +
                values["on_two_banks"]):
            raise ValueError(f"core {core_id} ON bank split mismatch")
        if values["on_zero_banks"] or values["unclassified"]:
            raise ValueError(f"core {core_id} has unclassified bank state")
        if values["capacity"] != (
                values["on"] + values["w_on_one_bank"] +
                values["w_on_two_banks"] + values["off"]):
            raise ValueError(f"core {core_id} PE-state partition mismatch")
        for name in ("preload_events", "gemm_events",
                     "preload_bank_pe_cycles"):
            values[name] = _integer(record.get(name), name)
        for name, value in values.items():
            totals[name] += value
        per_core.append({"core_id": core_id, **values})

    if len(dimensions) != 1:
        raise ValueError("inconsistent SA dimensions across cores")
    if expected_cores is not None and core_ids != set(range(expected_cores)):
        raise ValueError("core IDs must be exactly 0..expected_cores-1")
    array_rows, array_columns = next(iter(dimensions))
    rowwise = _aggregate_rowwise(records, expected_cores)
    preload_feasibility = _aggregate_preload_feasibility(
        records, expected_cores
    )
    if rowwise_state_accounting:
        states = _aggregate_rowwise_states(records, expected_cores, rowwise)
        return {
            "version": 3,
            "source_record_version": 2,
            "rowwise_state_accounting_version": 1,
            "model": "two-bank rowwise assumed-schedule logical state aggregate",
            "evidence_class": "derived" if states["available"] else "pending",
            "validation_pass": states["validation_pass"],
            "legacy_atomic_model_preserved": True,
            "regate_only": True,
            "combined_with_retention": False,
            "expected_cores": expected_cores,
            "observed_cores": len(records),
            "array_rows": array_rows,
            "array_columns": array_columns,
            "per_core": sorted(per_core, key=lambda item: item["core_id"]),
            "preload_feasibility_audit": preload_feasibility,
            "rowwise_feasibility": rowwise,
            "rowwise_state_accounting": states,
            "energy_status": "pending: theoretical PE-state exposure only; no power factors, BET, wake latency, or net-saving model",
        }
    if rowwise_diagnostic_only:
        return {
            "version": 2,
            "source_record_version": 2,
            "model": "two-bank rowwise analytical diagnostic aggregate",
            "evidence_class": "derived" if rowwise["available"] else "pending",
            "validation_pass": None,
            "state_accounting_status": (
                "pending: legacy PE-state accounting is not evaluated in "
                "rowwise-diagnostic-only mode"
            ),
            "regate_only": True,
            "combined_with_retention": False,
            "expected_cores": expected_cores,
            "observed_cores": len(records),
            "array_rows": array_rows,
            "array_columns": array_columns,
            "per_core": sorted(per_core, key=lambda item: item["core_id"]),
            "aggregate": {
                **{name: None for name in STATES},
                "preload_events": None,
                "gemm_events": None,
                "preload_bank_pe_cycles": None,
                "ideal_low_power_pe_cycle_exposure_percent": None,
                "gross_static_saving_percent": None,
                "net_energy_saving_percent": None,
            },
            "preload_feasibility_audit": preload_feasibility,
            "rowwise_feasibility": rowwise,
            "energy_status": "pending: diagnostic-only mode has no PE-state or circuit energy result",
        }
    capacity = totals["capacity"]
    low_power = (
        totals["w_on_one_bank"] + totals["w_on_two_banks"] + totals["off"]
    )
    result = {
        "version": 2,
        "source_record_version": 2,
        "model": "two-bank ReGate-like analytical shadow aggregate",
        "evidence_class": "derived",
        "validation_pass": True,
        "regate_only": True,
        "combined_with_retention": False,
        "expected_cores": expected_cores,
        "observed_cores": len(records),
        "array_rows": array_rows,
        "array_columns": array_columns,
        "per_core": sorted(per_core, key=lambda item: item["core_id"]),
        "preload_feasibility_audit": preload_feasibility,
        "rowwise_feasibility": rowwise,
        "aggregate": {
            **totals,
            "ideal_low_power_pe_cycle_exposure_percent": (
                100.0 * low_power / capacity if capacity else None
            ),
            "gross_static_saving_percent": None,
            "net_energy_saving_percent": None,
        },
        "energy_status": (
            "pending: two-bank state power and preload, control, "
            "transition energy need separate circuit evidence"
        ),
    }
    if power_factors is not None:
        modeled = sum(totals[state] * power_factors[state]
                      for state in POWER_STATES)
        result["power_sensitivity"] = {
            "evidence_class": "assumed",
            "result_evidence_class": "extrapolated",
            "source": factor_source,
            "equation": (
                "modeled=sum(state_PE_cycles*explicit_state_factor); "
                "gross_static_saving_percent=100*(1-modeled/capacity)"
            ),
            "factors_relative_to_always_on": dict(power_factors),
            "baseline_normalized_static_energy": capacity,
            "modeled_normalized_static_energy": modeled,
            "limitations": (
                "gross static sensitivity only; excludes preload write, "
                "control, transition energy and timing effects"
            ),
        }
        result["aggregate"]["gross_static_saving_percent"] = (
            100.0 * (1.0 - modeled / capacity) if capacity else None
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--expected-cores", type=int)
    parser.add_argument(
        "--rowwise-diagnostic-only", action="store_true",
        help=("aggregate complete, compatible rowwise diagnostics even when "
              "legacy atomic bank assignment fails; emit no PE-state savings"),
    )
    parser.add_argument(
        "--rowwise-state-accounting", action="store_true",
        help=("aggregate the version-1 fail-closed PE-state partition under "
              "the explicitly supplied assumed row schedule"),
    )
    for state in POWER_STATES:
        parser.add_argument(
            "--" + state.replace("_", "-") + "-power-factor",
            type=float,
        )
    parser.add_argument(
        "--factor-source",
        help="required provenance for explicit assumed power factors",
    )
    args = parser.parse_args()
    supplied = {
        state: getattr(args, state + "_power_factor")
        for state in POWER_STATES
    }
    count = sum(value is not None for value in supplied.values())
    if count not in (0, len(POWER_STATES)):
        parser.error("provide all five power factors or none")
    if count and not args.factor_source:
        parser.error("--factor-source is required with power factors")
    result = aggregate(
        read_records(args.log),
        expected_cores=args.expected_cores,
        power_factors=supplied if count else None,
        factor_source=args.factor_source,
        rowwise_diagnostic_only=args.rowwise_diagnostic_only,
        rowwise_state_accounting=args.rowwise_state_accounting,
    )
    result["source_log"] = str(args.log)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
