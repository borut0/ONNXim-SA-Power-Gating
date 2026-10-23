#!/usr/bin/env python3
"""Compare three *functional* runs using the same two-bank SA timing model.

The simulator emits SHARED_SA_TIMING once per core. This module uses the
recorded row writes and waves directly; it never calls the older analytical
replay, which shifts native ONNXim timestamps without rescheduling the model.
SA energy is normalized to one always-ON two-bank SA core-cycle. The absolute
power cancels.  PE transition energy is represented by ReGate's published
47-cycle PE break-even time; weight-bank transition energy remains an explicit
zero-cost/uncharacterized assumption.  Results are reported for the fixed
two-bank power-share sensitivity sweep beta={0.1,...,0.6}.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

if __package__:
    from .event_driven_sa import BANK_DS_RATIO, PE_OFF_RATIO
else:
    from event_driven_sa import BANK_DS_RATIO, PE_OFF_RATIO


PREFIX = "SHARED_SA_TIMING "
POLICIES = ("always_on", "regate", "ours")
MASK_POLICIES = ("delayed", "optimistic")
BETA_SWEEP = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
PE_BREAK_EVEN_CYCLES = 47
OURS_MINIMUM_OFF_INTERVAL = 48


def _integer(value, label, *, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(f"{label} must be a {'positive' if positive else 'nonnegative'} integer")
    return value


def read_records(path):
    records = []
    for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        position = line.find(PREFIX)
        if position >= 0:
            try:
                records.append(json.loads(line[position + len(PREFIX):]))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: malformed {PREFIX.strip()} JSON") from error
    if not records:
        raise ValueError(f"{path}: no {PREFIX.strip()} records")
    return records


def _account_core(record, horizon, *, mask_policy):
    policy = record.get("policy")
    local_horizon = _integer(record.get("core_cycles"), "core_cycles",
                             positive=True)
    if local_horizon > horizon:
        raise ValueError("core cycle horizon exceeds run horizon")
    rows = _integer(record.get("array_rows"), "array_rows", positive=True)
    columns = _integer(record.get("array_columns"), "array_columns", positive=True)
    area = rows * columns
    events = record.get("events")
    if not isinstance(events, list):
        raise ValueError("SHARED_SA_TIMING lacks events")
    generations = {}
    preload_intervals = []
    all_waves = []
    work = Counter()
    previous_issue = 0
    previous_power_wave = None
    expected_wake_delay_cycles = 0
    expected_lockout_cycles = 0
    expected_lockout_events = 0
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("SA event must be an object")
        tile_id = _integer(event.get("tile_id"), "tile_id")
        sequence = _integer(event.get("preload_sequence"), "preload_sequence",
                            positive=True)
        bank = _integer(event.get("bank"), "bank")
        height = _integer(event.get("mapped_rows"), "mapped_rows", positive=True)
        width = _integer(event.get("mapped_columns"), "mapped_columns",
                         positive=True)
        vectors = _integer(event.get("vectors"), "vectors", positive=True)
        issue = _integer(event.get("issue_cycle"), "issue_cycle")
        start = _integer(event.get("compute_start"), "compute_start")
        start_before = _integer(
            event.get("compute_start_before_power_policy"),
            "compute_start_before_power_policy")
        policy_delay = _integer(event.get("modeled_power_policy_delay"),
                                "modeled_power_policy_delay")
        finish = _integer(event.get("compute_finish"), "compute_finish")
        natural = _integer(event.get("natural_wave_finish"),
                           "natural_wave_finish")
        opcode = event.get("opcode")
        weight_known = event.get("weight_block_known")
        weight_id = _integer(event.get("weight_block_id"), "weight_block_id")
        if (bank > 1 or height > rows or width > columns or
                issue < previous_issue or issue > start_before or
                natural != start + height + width + vectors - 2 or
                finish < natural or finish > local_horizon or start == 0 or
                opcode not in ("GEMM_PRELOAD", "GEMM") or
                type(weight_known) is not bool or
                (weight_known and weight_id == 0)):
            raise ValueError("invalid or unordered functional SA event")
        if start - start_before != policy_delay:
            raise ValueError("power-policy delay does not match compute start")
        prior_live_until = None
        if previous_power_wave is not None:
            prior_start, prior_vectors = previous_power_wave
            prior_live_until = prior_start + prior_vectors
            if start_before < prior_live_until:
                raise ValueError("power-policy input overlaps prior wave")
        if policy == "always_on":
            if policy_delay != 0:
                raise ValueError("always-on policy has a power-policy delay")
        elif policy == "regate":
            eligible_wake = (previous_power_wave is None or
                             start_before > prior_live_until)
            expected = int(eligible_wake)
            if policy_delay != expected:
                raise ValueError("invalid ReGate power-policy delay")
            expected_wake_delay_cycles += policy_delay
        elif policy == "ours":
            if previous_power_wave is None:
                if policy_delay != 1:
                    raise ValueError("invalid first-wave ours wake delay")
                expected_wake_delay_cycles += 1
            elif start_before == prior_live_until:
                if policy_delay != 0:
                    raise ValueError("touching ours wave has a policy delay")
            else:
                unlock = prior_live_until + OURS_MINIMUM_OFF_INTERVAL
                expected = max(1, unlock - start_before)
                if policy_delay != expected:
                    raise ValueError("invalid ours lockout delay")
                expected_wake_delay_cycles += 1
                expected_lockout_cycles += policy_delay - 1
                expected_lockout_events += int(policy_delay > 1)
        previous_power_wave = (start, vectors)
        previous_issue = issue
        key = (tile_id, sequence)
        item = generations.setdefault(key, {
            "bank": bank, "height": height, "width": width,
            "events": [], "write": None})
        if (item["bank"], item["height"], item["width"]) != (
                bank, height, width):
            raise ValueError("preload generation changes bank or footprint")
        item["events"].append(event)
        if opcode == "GEMM_PRELOAD":
            if item["write"] is not None or len(item["events"]) != 1:
                raise ValueError("duplicate or late generation preload")
            write = _integer(event.get("row_write_start"), "row_write_start")
            if (write < issue or write + height > local_horizon or
                    start < write + 1):
                raise ValueError("preload row write is not causal")
            item["write"] = write
            preload_intervals.append((write, write + height))
        elif event.get("row_write_start") is not None:
            raise ValueError("plain GEMM unexpectedly writes weight rows")
        # Policy stalls can change bank choice, core admission, and runtime
        # tile numbering. Compare the semantic SA work as a multiset.
        work[(opcode, weight_id if weight_known else None, weight_known,
              height, width, vectors)] += 1
        all_waves.append((start, vectors, height, width))

    for left, right in zip(sorted(preload_intervals),
                           sorted(preload_intervals)[1:]):
        if left[1] > right[0]:
            raise ValueError("overlapping writes on single preload lane")
    for item in generations.values():
        if item["write"] is None or item["events"][0]["opcode"] != "GEMM_PRELOAD":
            raise ValueError("generation has no first preload")
        starts = [event["compute_start"] for event in item["events"]]
        vectors = [event["vectors"] for event in item["events"]]
        if any(next_start < start + count for start, count, next_start in
               zip(starts, vectors, starts[1:])):
            raise ValueError("overlapping same-generation MAC use")

    # Every mapped wave reaches PE(0,0); checking its MAC interval checks
    # all origin-aligned PE overlaps. Guard intervals can touch by one cycle.
    ordered_waves = sorted(all_waves)
    for prior, later in zip(ordered_waves, ordered_waves[1:]):
        if later[0] < prior[0] + prior[1]:
            raise ValueError("two waves use the same PE MAC simultaneously")

    # The functional scheduler keeps a bank unavailable until the preceding
    # mapped wave has drained, including columns outside the next footprint.
    # Enforce that architectural bank handoff, which also makes generation
    # state intervals disjoint for static accounting.
    by_bank = {bank: [] for bank in (0, 1)}
    for item in generations.values():
        by_bank[item["bank"]].append(item)
    for bank_generations in by_bank.values():
        bank_generations.sort(key=lambda item: item["write"])
        for prior, later in zip(bank_generations, bank_generations[1:]):
            last = max(event["natural_wave_finish"]
                       for event in prior["events"])
            if later["write"] < last:
                raise ValueError("weight bank overwritten before last local use")

    # All mapped rectangles start at PE(0,0).  PE logic is ON for MAC cycles.
    # The wake/guard is part of the transition whose energy is represented by
    # BET below, so counting it again as a full ON leakage cycle would double
    # charge that wake.  Exactly touching MAC intervals coalesce into one ON
    # episode but do not overlap in capacity accounting.
    logic_on = 0
    pe_low_power_episodes = 0
    # Union of all older origin-aligned footprints.  Row r has previously
    # visited columns [0, seen_width_by_row[r]).  This compact skyline lets us
    # count per-PE OFF->ON episodes without iterating a 128x128 rectangle for
    # every wave.
    seen_width_by_row = [0] * rows
    previous_wave = None
    for start, vectors, height, width in ordered_waves:
        seen_overlap = sum(min(width, seen_width_by_row[row])
                           for row in range(height))
        touching_overlap = 0
        logic_on += height * width * vectors
        if previous_wave is not None:
            old_start, old_vectors, old_height, old_width = previous_wave
            if start == old_start + old_vectors:
                touching_overlap = (min(height, old_height) *
                                    min(width, old_width))
            elif (policy == "ours" and
                  start - (old_start + old_vectors) <
                  OURS_MINIMUM_OFF_INTERVAL):
                raise ValueError("ours violates the 48-cycle wave lockout")
        # A PE already seen in any older footprint starts another low-power
        # episode unless it stayed awake across an exactly touching wave.
        pe_low_power_episodes += seen_overlap - touching_overlap
        for row in range(height):
            seen_width_by_row[row] = max(seen_width_by_row[row], width)
        previous_wave = (start, vectors, height, width)

    bank_on = bank_ds = 0
    for bank_generations in by_bank.values():
        for item in bank_generations:
            height, width, write = (item["height"], item["width"],
                                    item["write"])
            consumers = item["events"]
            first_start = consumers[0]["compute_start"]
            last_start = consumers[-1]["compute_start"]
            last_vectors = consumers[-1]["vectors"]
            union = sum(event["vectors"] + 1 for event in consumers)
            union -= sum(
                second["compute_start"] == first["compute_start"] +
                first["vectors"]
                for first, second in zip(consumers, consumers[1:]))
            preload_guard_overlap = height if first_start == write + 1 else 0
            on = height * width * (1 + union) - preload_guard_overlap
            live = height * (width * (last_start + last_vectors - write) +
                             width * (width - 1) // 2)
            ds = live - on
            if ds < 0:
                raise ValueError("negative bank deep-sleep residency")
            bank_on += on
            bank_ds += ds

    logic_capacity = area * horizon
    bank_capacity = 2 * logic_capacity
    if not 0 <= logic_on <= logic_capacity or bank_on + bank_ds > bank_capacity:
        raise ValueError("PE or weight state exceeds functional horizon")
    # ReGate's PE_on wave turns off compute circuitry; a valid loaded weight
    # remains powered after its last known consumer.  Later writes to the
    # same physical position replace its value without an OFF interval.
    # Positions outside a later, smaller footprint remain valid because the
    # functional event stream has no per-position invalidation or zero mask.
    # Origin-aligned rectangles permit a per-row first-seen-width skyline.
    regate_mapped_residency = 0
    delayed_extra = 0
    for bank_generations in by_bank.values():
        seen_width_by_row = [0] * rows
        for item in bank_generations:
            write, height, width = (item["write"], item["height"],
                                    item["width"])
            for row in range(height):
                seen = seen_width_by_row[row]
                if width > seen:
                    regate_mapped_residency += (
                        (width - seen) * (horizon - write - row))
                    seen_width_by_row[row] = width
                if mask_policy == "delayed":
                    # Candidate cells not already holding valid older weights
                    # are ON from this row's write until mask completion.
                    delayed_extra += ((columns - max(width, seen)) *
                                      (height - row))
    if regate_mapped_residency < bank_on:
        raise ValueError("ReGate mapped residency shorter than active use")
    regate_bank_on = bank_on + delayed_extra
    regate_bank_w_on = regate_mapped_residency - bank_on
    if regate_bank_on + regate_bank_w_on > bank_capacity:
        raise ValueError("ReGate candidate power exceeds two-bank capacity")
    wake_counter = _integer(record.get("modeled_pe_wake_delay_cycles"),
                            "modeled_pe_wake_delay_cycles")
    lockout_counter = _integer(record.get("modeled_min_off_lockout_cycles"),
                               "modeled_min_off_lockout_cycles")
    lockout_events_counter = _integer(
        record.get("modeled_min_off_lockout_events"),
        "modeled_min_off_lockout_events")
    if wake_counter != expected_wake_delay_cycles:
        raise ValueError("modeled PE wake-delay counter does not match events")
    if (lockout_counter != expected_lockout_cycles or
            lockout_events_counter != expected_lockout_events):
        raise ValueError("modeled minimum-OFF lockout counters do not match events")
    return {
        "work": work,
        "pe_logic_cycles": {"on": logic_on, "off": logic_capacity - logic_on,
                            "capacity": logic_capacity},
        "pe_low_power_episodes": pe_low_power_episodes,
        "our_weight_bank_cycles": {
            "on": bank_on, "ds": bank_ds,
            "off": bank_capacity - bank_on - bank_ds,
            "capacity": bank_capacity},
        "regate_weight_bank_cycles": {
            "on": regate_bank_on,
            "w_on": regate_bank_w_on,
            "off": bank_capacity - regate_bank_on - regate_bank_w_on,
            "capacity": bank_capacity},
        "regate_delayed_mask_candidate_cycles": delayed_extra,
        "events": len(events),
    }


def aggregate_run(records, *, expected_cores=None, beta, mask_policy="delayed"):
    if not isinstance(records, list) or not records:
        raise ValueError("no functional timing records")
    if expected_cores is not None and len(records) != expected_cores:
        raise ValueError("unexpected number of cores")
    if type(beta) not in (int, float) or not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("beta must be an explicit finite fraction in [0,1]")
    if mask_policy not in MASK_POLICIES:
        raise ValueError("unknown mask policy")
    core_ids = [_integer(r.get("core_id"), "core_id") for r in records]
    if len(set(core_ids)) != len(core_ids):
        raise ValueError("duplicate core ID")
    policies = {r.get("policy") for r in records}
    if len(policies) != 1 or next(iter(policies)) not in POLICIES:
        raise ValueError("mixed or unknown functional policy")
    policy = next(iter(policies))
    dimensions = set()
    for record in records:
        if record.get("schema_version") != 2 or record.get("validation_pass") is not True:
            raise ValueError("invalid or unvalidated functional timing schema")
        if (record.get("two_weight_banks") is not True or
                record.get("bank_ds_wake_hidden_assumption") is not True or
                record.get("pe_break_even_cycles") !=
                PE_BREAK_EVEN_CYCLES or
                record.get("ours_minimum_off_interval_cycles") !=
                OURS_MINIMUM_OFF_INTERVAL):
            raise ValueError("incompatible functional timing assumptions")
        dimensions.add((record.get("array_rows"), record.get("array_columns"),
                        record.get("weight_bytes_per_bank")))
    if len(dimensions) != 1:
        raise ValueError("SA dimensions differ across cores")
    rows, columns, weight_bytes_per_bank = next(iter(dimensions))
    _integer(rows, "array_rows", positive=True)
    _integer(columns, "array_columns", positive=True)
    _integer(weight_bytes_per_bank, "weight_bytes_per_bank", positive=True)
    horizon = max(_integer(record.get("core_cycles"), "core_cycles",
                           positive=True) for record in records)
    accounted = [_account_core(record, horizon, mask_policy=mask_policy)
                 for record in records]
    logic = {key: sum(item["pe_logic_cycles"][key] for item in accounted)
             for key in ("on", "off", "capacity")}
    pe_low_power_episodes = sum(item["pe_low_power_episodes"]
                                for item in accounted)
    our_banks = {key: sum(item["our_weight_bank_cycles"][key]
                          for item in accounted)
                 for key in ("on", "ds", "off", "capacity")}
    regate_banks = {key: sum(item["regate_weight_bank_cycles"][key]
                             for item in accounted)
                    for key in ("on", "w_on", "off", "capacity")}
    work = sum((item["work"] for item in accounted), Counter())
    if logic["on"] + logic["off"] != logic["capacity"]:
        raise ValueError("PE logic state partition failed")
    if sum(our_banks[key] for key in ("on", "ds", "off")) != our_banks["capacity"]:
        raise ValueError("ours bank state partition failed")
    if sum(regate_banks[key] for key in ("on", "w_on", "off")) != \
            regate_banks["capacity"]:
        raise ValueError("ReGate bank state partition failed")
    area = rows * columns
    always_on_cycle_equivalent = len(records) * horizon
    logic_gross_equivalent = (
        logic["on"] + PE_OFF_RATIO * logic["off"]) / area
    # BET converts one complete ON->low-power->ON transition episode into
    # equivalent always-ON leakage cycles.  It is charged for every episode,
    # including short ReGate episodes; consequently net saving may be
    # negative.  Initial/final low-power tails are not complete episodes and
    # are not charged a full round-trip transition.
    transition_equivalent = (
        pe_low_power_episodes * PE_BREAK_EVEN_CYCLES *
        (1 - PE_OFF_RATIO) / area)
    logic_net_equivalent = logic_gross_equivalent + transition_equivalent
    regate_equivalent = ((regate_banks["on"] + regate_banks["w_on"]) /
                         (2 * area))
    our_equivalent = (our_banks["on"] + BANK_DS_RATIO * our_banks["ds"]) / (2 * area)
    gross_policy_equivalent = {
        "always_on": always_on_cycle_equivalent,
        "regate": ((1 - beta) * logic_gross_equivalent +
                   beta * regate_equivalent),
        "ours": ((1 - beta) * logic_gross_equivalent +
                 beta * our_equivalent),
    }[policy]
    net_policy_equivalent = {
        "always_on": always_on_cycle_equivalent,
        "regate": ((1 - beta) * logic_net_equivalent +
                   beta * regate_equivalent),
        "ours": ((1 - beta) * logic_net_equivalent +
                 beta * our_equivalent),
    }[policy]
    return {
        "policy": policy,
        "schema_version": 2,
        "observed_cores": len(records),
        "shared_horizon_cycles": horizon,
        "array_rows": rows,
        "array_columns": columns,
        "weight_bytes_per_bank": weight_bytes_per_bank,
        "beta_two_bank_static_power_share": beta,
        "pe_break_even_cycles": PE_BREAK_EVEN_CYCLES,
        "weight_bank_transition_energy_assumption":
            "zero; uncharacterized",
        "mask_policy": mask_policy,
        "work_signature": work,
        "sa_events": sum(item["events"] for item in accounted),
        "pe_logic_cycles": logic,
        "pe_low_power_episodes": pe_low_power_episodes,
        "pe_transition_energy_equivalent_always_on_core_cycles":
            transition_equivalent if policy != "always_on" else 0,
        "our_weight_bank_cycles": our_banks,
        "regate_weight_bank_cycles": regate_banks,
        "normalized_gross_static_energy_in_always_on_core_cycles":
            gross_policy_equivalent,
        "normalized_net_sa_energy_in_always_on_core_cycles":
            net_policy_equivalent,
        # Backward-compatible alias.  The value is now transition-adjusted.
        "normalized_static_energy_in_always_on_core_cycles":
            net_policy_equivalent,
        "within_run_gross_static_saving_percent":
            100 * (1 - gross_policy_equivalent /
                   always_on_cycle_equivalent),
        "within_run_net_sa_energy_saving_percent":
            100 * (1 - net_policy_equivalent /
                   always_on_cycle_equivalent),
        "simulator_counters": {
            key: sum(_integer(record.get(key, 0), key) for record in records)
            for key in ("sa_instructions", "preloads", "modeled_preload_cycles",
                        "modeled_pe_wake_delay_cycles", "modeled_bank_wait_cycles",
                        "modeled_min_off_lockout_cycles",
                        "modeled_min_off_lockout_events",
                        "modeled_sa_completion_cycles")},
    }


def _compare_one_beta(reference, regate, ours, *, expected_cores,
                      beta, mask_policy):
    runs = {name: aggregate_run(records, expected_cores=expected_cores,
                                beta=beta, mask_policy=mask_policy)
            for name, records in (("always_on", reference),
                                  ("regate", regate), ("ours", ours))}
    if any(runs[name]["policy"] != name for name in POLICIES):
        raise ValueError("wrong policy in one or more functional runs")
    reference_run = runs["always_on"]
    for name in ("regate", "ours"):
        candidate = runs[name]
        if (candidate["work_signature"] != reference_run["work_signature"] or
                any(candidate[key] != reference_run[key]
                    for key in ("observed_cores", "array_rows", "array_columns",
                                "weight_bytes_per_bank"))):
            raise ValueError(f"{name} run does not match reference SA work")
    reference_energy = reference_run[
        "normalized_net_sa_energy_in_always_on_core_cycles"]
    regate_energy = runs["regate"][
        "normalized_net_sa_energy_in_always_on_core_cycles"]
    ours_energy = runs["ours"][
        "normalized_net_sa_energy_in_always_on_core_cycles"]
    if reference_energy <= 0 or regate_energy <= 0:
        raise ValueError("reference or ReGate static energy is nonpositive")
    horizon = reference_run["shared_horizon_cycles"]
    regate_saving = 100 * (1 - regate_energy / reference_energy)
    ours_saving = 100 * (1 - ours_energy / reference_energy)
    return {
        "beta": beta,
        "regate_net_sa_energy_saving_percent_vs_always_on":
            regate_saving,
        "ours_net_sa_energy_saving_percent_vs_always_on":
            ours_saving,
        "ours_net_sa_energy_saving_percentage_point_gain_vs_regate":
            ours_saving - regate_saving,
        "ours_additional_net_sa_energy_saving_percent_vs_regate":
            100 * (1 - ours_energy / regate_energy),
        "normalized_net_sa_energy_in_reference_core_cycles": {
            "always_on": reference_energy, "regate": regate_energy,
            "ours": ours_energy},
        "runs": runs,
    }


def _fixed_sa_order_serial_scenario(reference):
    """Replay PE(0,0) delays on the always-on SA order.

    This deliberately assumes that each added delay shifts every later event
    and the core endpoint by the same amount.  The event log does not contain
    the non-SA dependency or memory graph needed for an exact workload replay.
    """
    per_core = []
    for record in reference:
        prior_live_until = None
        regate_delay = ours_delay = 0
        regate_wakes = ours_wakes = ours_lockouts = 0
        for event in record["events"]:
            start = event["compute_start"]
            vectors = event["vectors"]
            if prior_live_until is None:
                regate_delay += 1
                ours_delay += 1
                regate_wakes += 1
                ours_wakes += 1
            else:
                gap = start - prior_live_until
                if gap < 0:
                    raise ValueError("reference SA waves overlap in fixed-order replay")
                if gap > 0:
                    regate_delay += 1
                    regate_wakes += 1
                    ours_wakes += 1
                    delay = max(1, OURS_MINIMUM_OFF_INTERVAL - gap)
                    ours_delay += delay
                    ours_lockouts += int(delay > 1)
            prior_live_until = start + vectors
        endpoint = record["core_cycles"]
        per_core.append({
            "core_id": record["core_id"],
            "reference_core_cycles": endpoint,
            "regate_serial_policy_delay_cycles": regate_delay,
            "ours_serial_policy_delay_cycles": ours_delay,
            "regate_explicit_wake_events": regate_wakes,
            "ours_explicit_wake_events": ours_wakes,
            "ours_lockout_events": ours_lockouts,
            "regate_serial_endpoint_cycles": endpoint + regate_delay,
            "ours_serial_endpoint_cycles": endpoint + ours_delay,
        })
    reference_horizon = max(item["reference_core_cycles"] for item in per_core)
    regate_horizon = max(item["regate_serial_endpoint_cycles"]
                         for item in per_core)
    ours_horizon = max(item["ours_serial_endpoint_cycles"]
                       for item in per_core)
    return {
        "kind": "fixed SA order, serial no-overlap analytical scenario",
        "exact_end_to_end_causal_replay": False,
        "assumption": "every inserted PE(0,0) delay shifts all later SA work and the same core's endpoint; memory and non-SA scheduling remain unmodeled",
        "reference_horizon_cycles": reference_horizon,
        "regate_horizon_cycles": regate_horizon,
        "ours_horizon_cycles": ours_horizon,
        "regate_scenario_performance_penalty_percent":
            100 * (regate_horizon / reference_horizon - 1),
        "ours_scenario_performance_penalty_percent":
            100 * (ours_horizon / reference_horizon - 1),
        "per_core": per_core,
    }


def compare_runs(reference, regate, ours, *, expected_cores=None,
                 mask_policy="delayed"):
    sensitivity = [_compare_one_beta(
        reference, regate, ours, expected_cores=expected_cores,
        beta=beta, mask_policy=mask_policy) for beta in BETA_SWEEP]
    representative = sensitivity[0]
    runs = representative["runs"]
    reference_run = runs["always_on"]
    horizon = reference_run["shared_horizon_cycles"]
    return {
        "model": "matched two-bank functional SA timing comparison",
        "evidence_class":
            "functional timing with PE-BET transition energy and assumed static power ratios",
        "state_partition_validation_pass": True,
        "matched_sa_work_validation_pass": True,
        "whole_workload_identity_status":
            "not independently certified: run the same model/config/trace in all three modes; emitted records verify only an SA instruction multiset",
        "observed_cores": reference_run["observed_cores"],
        "beta_two_bank_static_power_share_sweep": list(BETA_SWEEP),
        "pe_break_even_cycles": PE_BREAK_EVEN_CYCLES,
        "ours_minimum_off_interval_cycles": OURS_MINIMUM_OFF_INTERVAL,
        "regate_mask_policy": mask_policy,
        "reference_horizon_cycles": horizon,
        "regate_horizon_cycles": runs["regate"]["shared_horizon_cycles"],
        "ours_horizon_cycles": runs["ours"]["shared_horizon_cycles"],
        "observed_regate_horizon_delta_percent":
            100 * (runs["regate"]["shared_horizon_cycles"] / horizon - 1),
        "observed_ours_horizon_delta_percent":
            100 * (runs["ours"]["shared_horizon_cycles"] / horizon - 1),
        "observed_horizon_interpretation":
            "independent full-simulator schedules; a negative difference may reflect changed scheduling or contention and is not an isolated power-policy speedup",
        "fixed_sa_order_serial_scenario":
            _fixed_sa_order_serial_scenario(reference),
        "beta_sensitivity": [
            {key: value for key, value in item.items()
             if key != "runs"} for item in sensitivity],
        "physical_bank_wake_feasibility":
            "assumed: DS weight-bank wake is fully hidden by PE wake; circuit timing is not established",
        "energy_scope":
            "SA static leakage plus PE transition energy represented by the published 47-cycle BET; weight-bank transition, dynamic, controller, selector, and preload energy excluded",
        "weight_bank_transition_energy_assumption":
            "zero for both matched two-bank policies; uncharacterized and must be sensitivity-tested later",
        "runs_at_representative_beta_0_1": {
            name: {key: value for key, value in run.items()
                   if key != "work_signature"}
            for name, run in runs.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--always-on-log", type=Path, required=True)
    parser.add_argument("--regate-log", type=Path, required=True)
    parser.add_argument("--ours-log", type=Path, required=True)
    parser.add_argument("--expected-cores", type=int)
    parser.add_argument("--mask-policy", choices=MASK_POLICIES,
                        default="delayed")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    inputs = (args.always_on_log, args.regate_log, args.ours_log)
    if len({path.resolve() for path in inputs}) != 3:
        parser.error("three separate functional run logs are required")
    if args.output and args.output.resolve() in {path.resolve() for path in inputs}:
        parser.error("--output must not overwrite a raw simulation log")
    result = compare_runs(*(read_records(path) for path in inputs),
                          expected_cores=args.expected_cores,
                          mask_policy=args.mask_policy)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
