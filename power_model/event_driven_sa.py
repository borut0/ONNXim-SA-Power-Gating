"""Event-driven PE and two-bank state replay with analytical SA stalls.

This is an assumed architectural replay of ONNXim's emitted SA events. It
does not alter ONNXim's functional scheduler or establish an end-to-end
performance result. A bank is ON while written or consumed, DS while its
contents have a later consumer, and OFF otherwise. The PE logic follows the
diagonal activation wave, with one ON wake cycle before each local use.
"""

from collections import defaultdict
import math


PE_OFF_RATIO = 0.03  # ReGate reference; not a measurement of this design.
BANK_DS_RATIO = 0.42  # Slumber 128-B reference extrapolated to each PE bank.
PE_WAKE_CYCLES = 1  # ReGate reference; DS wake is assumed hidden by it.


def _integer(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _union(spans, horizon):
    merged = []
    for begin, end in sorted(spans):
        if begin < 0 or end < begin or end > horizon:
            raise ValueError("SA state interval exceeds shared horizon")
        if begin == end:
            continue
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((begin, end))
    return merged


def _length(spans):
    return sum(end - begin for begin, end in spans)


def _intersection_length(left, right):
    i = j = total = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) -
                     max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def replay_event_driven(record, *, horizon=None, beta=None):
    """Return a unified analytical PE/bank trace with a pending beta by default.

    Each emitted preload instance denotes one bank generation. Its mapped
    rectangle is origin aligned. The proposed one-row-per-cycle write schedule
    begins no earlier than the shifted issue cycle. When the native start is
    too early, later SA issue/start bounds on this core inherit the delay.
    Non-SA instructions and memory are not rescheduled.
    """
    rows = _integer(record.get("array_rows"), "array_rows")
    columns = _integer(record.get("array_columns"), "array_columns")
    weight_bytes = _integer(record.get("weight_bytes"), "weight_bytes")
    baseline_horizon = _integer(record.get("onnxim_horizon_cycles"),
                                "onnxim_horizon_cycles")
    if not rows or not columns or not weight_bytes or not baseline_horizon:
        raise ValueError("invalid array, horizon, or weight width")
    if beta is not None and (type(beta) not in (int, float) or
                             not math.isfinite(beta) or not 0 <= beta <= 1):
        raise ValueError("explicit beta must be in [0,1]")
    events = record.get("events")
    if not isinstance(events, list):
        raise ValueError("missing analytical SA events")
    if record.get("validation_pass") is not True:
        raise ValueError("source SA event validation did not pass")

    copies = {}
    previous_issue = 0
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("SA event must be an object")
        instance = _integer(event.get("preload_instance"), "preload_instance")
        bank = _integer(event.get("assumed_bank"), "assumed_bank")
        mapped_rows = _integer(event.get("tile_m"), "tile_m")
        mapped_cols = _integer(event.get("tile_k"), "tile_k")
        vectors = _integer(event.get("tile_n"), "tile_n")
        issue = _integer(event.get("issue_cycle"), "issue_cycle")
        if (not instance or bank > 1 or not mapped_rows or not mapped_cols or
                not vectors or mapped_rows > rows or mapped_cols > columns or
                issue < previous_issue):
            raise ValueError("invalid or unordered SA event")
        previous_issue = issue
        item = copies.setdefault(instance, {"bank": bank, "rows": mapped_rows,
                                           "columns": mapped_cols,
                                           "events": [], "preload": None})
        if (item["bank"], item["rows"], item["columns"]) != (
                bank, mapped_rows, mapped_cols):
            raise ValueError("bank generation changes footprint")
        role = event.get("analytical_role")
        if role == "preload_boundary":
            if item["preload"] is not None:
                raise ValueError("duplicate preload for bank generation")
            item["preload"] = event
        elif role != "gemm_consumer":
            raise ValueError("unknown SA event role")
        item["events"].append(event)
    if any(item["preload"] is None or
           item["events"][0] is not item["preload"]
           for item in copies.values()):
        raise ValueError("bank generation lacks a preceding preload")

    native_windows = []
    for event in events:
        begin = _integer(event["onnxim_compute_start"], "ONNXim start")
        end = _integer(event["onnxim_compute_finish"], "ONNXim finish")
        expected = rows + columns - 2 + max(event["tile_n"], 4)
        if end - begin != expected or end > baseline_horizon:
            raise ValueError("invalid ONNXim compute interval")
        native_windows.append((begin, end))
    native_overlap_pairs = 0
    latest_end = 0
    for begin, end in sorted(native_windows):
        if begin < latest_end:
            native_overlap_pairs += 1
        latest_end = max(latest_end, end)

    # The native windows may overlap because waves can be pipelined. A new
    # wave need only wait for earlier MAC use at the same PE, not for the
    # entire earlier array window to drain. All mapped rectangles start at
    # PE(0,0), so the per-PE MAC bound reduces to start >= old_start +
    # old_vectors. Preload writes use one row lane and two assigned banks.
    stream_owner = {}
    scheduled = defaultdict(list)
    managed_events = []
    last_consumer_index = {instance: max(index for index, event in
                                         enumerate(events)
                                         if event["preload_instance"] == instance)
                           for instance in copies}
    bank_owner = {}
    lane_free = 0
    mac_frontier = 0
    pe00_on_until = 0
    accumulated_shift = 0
    preload_rows = {}
    native_sa_finish = 0
    for index, event in enumerate(events):
        instance = event["preload_instance"]
        bank = event["assumed_bank"]
        stream = _integer(event.get("tile_stream_id"), "tile_stream_id")
        native_issue = event["issue_cycle"]
        issue = native_issue + accumulated_shift
        mapped_rows = event["tile_m"]
        mapped_cols = event["tile_k"]
        vectors = event["tile_n"]
        native_start = _integer(event.get("onnxim_compute_start"),
                                "onnxim_compute_start")
        native_finish = _integer(event.get("onnxim_compute_finish"),
                                 "onnxim_compute_finish")
        native_sa_finish = max(native_sa_finish, native_finish)
        start = max(native_start + accumulated_shift, issue, mac_frontier)
        wave_finish = start + mapped_rows + mapped_cols + vectors - 2
        if native_issue > native_start or (native_start + mapped_rows +
                                          mapped_cols + vectors - 2 >
                                          native_finish):
            raise ValueError("native SA interval cannot contain mapped wave")
        role = event["analytical_role"]
        if role == "preload_boundary":
            prior = bank_owner.get(bank)
            if prior is not None and last_consumer_index[prior] >= index:
                raise ValueError("bank is overwritten before its last consumer")
            if prior is not None:
                prior_cols = copies[prior]["columns"]
                # A row write replaces only the columns mapped by this
                # generation; old columns outside that overlap are irrelevant.
                release = max(old_start + min(prior_cols, mapped_cols) - 1 +
                              old_vectors for old_start, _, old_vectors, _
                              in scheduled[prior])
            else:
                release = 0
            # Row 0 writes in [write,write+1). PE(0,0) first uses it
            # at start, so start >= write+1. Rows r then write at write+r.
            start = max(start, issue + PE_WAKE_CYCLES,
                        lane_free + PE_WAKE_CYCLES,
                        release + PE_WAKE_CYCLES)
            write = start - PE_WAKE_CYCLES
            preload_rows[instance] = write
            lane_free = write + mapped_rows
            bank_owner[bank] = instance
            stream_owner[stream] = instance
        else:
            if stream_owner.get(stream) != instance:
                raise ValueError("GEMM consumes an invalid bank generation")
            if bank_owner.get(bank) != instance:
                raise ValueError("GEMM consumes an overwritten bank generation")
            # PE(0,0) needs one guard cycle if it was previously OFF.
            if pe00_on_until < start:
                start = max(start, issue + PE_WAKE_CYCLES)
        extra = start - (native_start + accumulated_shift)
        accumulated_shift += extra
        finish = start + native_finish - native_start
        wave_finish = start + mapped_rows + mapped_cols + vectors - 2
        mac_frontier = start + vectors
        pe00_on_until = start + vectors
        scheduled[instance].append((start, wave_finish, vectors, issue))
        managed_events.append({"preload_instance": instance,
                               "bank": bank, "role": role,
                               "mapped_rows": mapped_rows,
                               "mapped_columns": mapped_cols,
                               "vectors": vectors,
                               "compute_start": start,
                               "compute_finish": finish,
                               "mapped_wave_finish": wave_finish,
                               "issue_cycle": issue,
                               "native_issue_cycle": native_issue,
                               "native_compute_start": native_start,
                               "native_compute_finish": native_finish,
                               "added_delay_cycles": extra})

    analytical_sa_finish = max((event["compute_finish"] for event in
                                managed_events), default=0)
    managed_horizon = max(baseline_horizon, analytical_sa_finish)
    if horizon is not None:
        horizon = _integer(horizon, "shared horizon")
        if horizon < managed_horizon:
            raise ValueError("shared horizon ends before managed SA work")
        managed_horizon = horizon
    area = rows * columns
    domains_per_bank = area  # Each PE has two independently gated banks.
    useful_mac_events = 0
    assumed_boundaries = 0
    preload_write_cycles = 0
    unresolved_pe_wake = 0

    # Count PE-logic interval unions without materializing one Python interval
    # object per PE and event.  Every mapped rectangle is origin aligned.  For
    # a PE at (r,c), an event's ON span is [start+r+c-1,
    # start+r+c+vectors).  The r+c term cancels when two consecutive spans
    # are compared, so only an exact MAC-boundary touch overlaps one guard
    # cycle.  The grid stores last_MAC_end-(r+c).
    last_logic_end_offset = [[-1] * columns for _ in range(rows)]
    logic_on = 0
    for event in managed_events:
        mapped_rows = event["mapped_rows"]
        mapped_cols = event["mapped_columns"]
        vectors = event["vectors"]
        start = event["compute_start"]
        overlap = 0
        new_end_offset = start + vectors
        replacement = [new_end_offset] * mapped_cols
        for row in range(mapped_rows):
            state_row = last_logic_end_offset[row]
            overlap += state_row[:mapped_cols].count(start)
            state_row[:mapped_cols] = replacement
        logic_on += mapped_rows * mapped_cols * (vectors + 1) - overlap
        useful_mac_events += mapped_rows * mapped_cols * vectors

    # Each bank's union of origin-aligned footprints can be represented by
    # the widest mapped prefix ever seen in each row.
    mapped_width_by_bank_row = [[0] * rows for _ in range(2)]
    bank_on_cycles = 0
    bank_ds_cycles = 0
    for instance, item in copies.items():
        bank, mapped_rows, mapped_cols = (item["bank"], item["rows"],
                                           item["columns"])
        preload = next(e for e in managed_events
                       if e["preload_instance"] == instance and
                       e["role"] == "preload_boundary")
        if item["preload"].get("analytical_preload_boundary_assumed") is True:
            assumed_boundaries += 1
        base_write = preload_rows[instance]
        issue = preload["issue_cycle"]
        if base_write < issue or base_write < 0:
            raise ValueError("row write would begin before its instruction issue")
        first_start = scheduled[instance][0][0]
        if base_write + 1 > first_start:
            raise ValueError("row write finishes after first local use")

        for row in range(mapped_rows):
            mapped_width_by_bank_row[bank][row] = max(
                mapped_width_by_bank_row[bank][row], mapped_cols)
        preload_write_cycles += mapped_rows * mapped_cols

        # Consumer ON-union length is identical at every mapped PE because
        # diagonal r+c shifts all of a generation's spans equally.  Exactly
        # contiguous MAC spans overlap by the one-cycle wake guard.
        consumers = scheduled[instance]
        consumer_union = sum(vectors + 1 for _, _, vectors, _ in consumers)
        for previous, current in zip(consumers, consumers[1:]):
            if current[0] == previous[0] + previous[2]:
                consumer_union -= 1

        # The preload write overlaps the first consumer guard in column 0.
        bank_on_cycles += (mapped_rows * mapped_cols *
                           (1 + consumer_union) - mapped_rows)

        last_start, _, last_vectors, _ = consumers[-1]
        # For column c, live length is
        # last_start+c+last_vectors-base_write-1; row offsets cancel.
        live_cycles = mapped_rows * (
            mapped_cols * (last_start + last_vectors - base_write - 1) +
            mapped_cols * (mapped_cols - 1) // 2)
        on_inside_live = (mapped_rows * mapped_cols * consumer_union -
                          mapped_rows)
        generation_ds = live_cycles - on_inside_live
        if generation_ds < 0:
            raise ValueError("negative bank deep-sleep residency")
        bank_ds_cycles += generation_ds

    logic_capacity = area * managed_horizon
    mapped_bank_pe_count = sum(sum(widths)
                               for widths in mapped_width_by_bank_row)
    bank_capacity = 2 * domains_per_bank * managed_horizon
    logic = {"on": logic_on, "off": logic_capacity - logic_on,
             "capacity": logic_capacity}
    banks = {"on": bank_on_cycles, "ds": bank_ds_cycles,
             "off": bank_capacity - bank_on_cycles - bank_ds_cycles,
             "capacity": bank_capacity}
    if min(logic.values()) < 0 or min(banks.values()) < 0:
        raise ValueError("negative PE or bank state residency")
    logic_equiv = logic["on"] + PE_OFF_RATIO * logic["off"]
    bank_equiv = banks["on"] + BANK_DS_RATIO * banks["ds"]
    logic_saving = 100 * (1 - logic_equiv / logic_capacity)
    bank_saving = 100 * (1 - bank_equiv / bank_capacity)
    # A numeric whole-SA result requires the measured or explicitly assumed
    # baseline static-power share of the two banks. No default is used.
    combined = ((1 - beta) * logic_equiv / logic_capacity +
                beta * bank_equiv / bank_capacity
                if beta is not None else None)
    return {
        "core_id": record.get("core_id"),
        "evidence_class": "extrapolated",
        "model": "event-driven unified PE and two-bank analytical replay",
        "same_cycle_state_accounting": True,
        "state_partition_validation_pass": True,
        "physical_timing_feasibility_status":
            "pending: unresolved PE wake timing and 13-cycle Slumber DS wake are not rescheduled"
            if unresolved_pe_wake else
            "pending: Slumber DS wake is 13 cycles versus assumed 1-cycle PE advance; circuit and stall feasibility unproven",
        "functional_scheduling_unchanged": True,
        "array_rows": rows, "array_columns": columns,
        "onnxim_horizon_cycles": baseline_horizon,
        "native_sa_finish_cycle": native_sa_finish,
        "analytical_sa_finish_cycle": analytical_sa_finish,
        "analytical_added_sa_delay_cycles": accumulated_shift,
        "analytical_sa_tail_extension_cycles":
            max(0, analytical_sa_finish - baseline_horizon),
        "managed_native_horizon_cycles": managed_horizon,
        "managed_horizon_cycles": managed_horizon,
        "wake_policy": "assumed one-cycle advance signal before local wave use; functional issue timing and DS wake feasibility pending",
        "unresolved_pe_wake_events": unresolved_pe_wake,
        "preload_row_write_pe_cycles": preload_write_cycles,
        "native_onnxim_overlap_intervals": native_overlap_pairs,
        "native_overlap_interpretation":
            "pipelined waves retain overlap; same-PE MAC and bank-write conflicts delay later SA waves",
        "actual_end_to_end_penalty_percent": None,
        "actual_end_to_end_penalty_status":
            "pending: ONNXim functional scheduling is unchanged and circuit wake stalls were not injected",
        "physical_weight_wake_penalty_status":
            "pending: assumed hidden DS wake conflicts with Slumber Table 1's 13-cycle DS wake versus ReGate Table 3's 1-cycle PE wake; requires circuit characterization or a different bank circuit",
        "pe_logic_cycles": logic,
        "weight_bank_pe_cycles": banks,
        "structurally_unused_bank_pe_count":
            2 * area - mapped_bank_pe_count,
        "structurally_unused_bank_pe_off_cycles":
            (2 * area - mapped_bank_pe_count) * managed_horizon,
        "useful_mac_events": useful_mac_events,
        "analytical_preload_boundaries_assumed": assumed_boundaries,
        "pe_logic_leakage_saving_percent_on_managed_horizon": logic_saving,
        "weight_bank_leakage_saving_percent_on_managed_horizon": bank_saving,
        "beta": beta,
        "whole_sa_gross_static_opportunity_percent_on_managed_timeline":
            100 * (1 - combined) if combined is not None else None,
        "whole_sa_gross_static_opportunity_percent_on_native_timeline":
            (100 * (1 - combined) if combined is not None and
             managed_horizon == baseline_horizon else None),
        "whole_sa_static_saving_formula":
            "100*(1-((1-beta)*(PE_ON+0.03*PE_OFF)/"
            "(PE_COUNT*T_managed)+beta*(BANK_ON+0.42*BANK_DS)/"
            "(2*PE_COUNT*T_managed)))",
        "assumptions": [
            "two independently gated weight banks per PE",
            "one origin-aligned mapped weight rectangle per preload generation",
            "one-row-per-cycle writes are aligned immediately ahead of each adjusted diagonal wave; instruction issue is a causality bound",
            "one shared assumed row-write lane across both weight banks",
            "native and adjusted compute windows may overlap; same-PE MAC use is exclusive",
            "added SA delay shifts subsequent same-core SA issue and compute bounds; non-SA work is not rescheduled",
            "one PE wake cycle before local wave use where issue timing permits",
            "DS weight wake is hypothetically hidden by PE advance for gross static accounting only",
            "the hidden-DS-wake hypothesis is user-selected and unsupported by Slumber/ReGate timing; physical penalty remains pending",
            "no SS bank state, idle-gap threshold, or global whole-SA power state",
            "PE OFF leakage 0.03 from ReGate; Slumber's 128-B DS ratio 0.42 extrapolated to each PE bank",
            "transition, controller, selector, and preload dynamic energy excluded",
        ],
        "reference_parameters": {
            "pe_off_to_on_static_ratio": {
                "value": PE_OFF_RATIO, "evidence_class": "reference",
                "publication": "ReGate: Enabling Power Gating in Neural Processing Units",
                "locator": "Figure 21 caption, logic OFF leakage example",
                "url": "https://arxiv.org/pdf/2508.02536"},
            "pe_wake_cycles": {
                "value": PE_WAKE_CYCLES, "evidence_class": "reference",
                "publication": "ReGate: Enabling Power Gating in Neural Processing Units",
                "locator": "Table 3, SA (PE); Section 4.4 synthesis discussion",
                "url": "https://arxiv.org/pdf/2508.02536"},
            "bank_ds_to_on_static_ratio": {
                "value": BANK_DS_RATIO, "evidence_class": "extrapolated",
                "publication": "Slumber: Static-Power Management for GPGPU Register Files",
                "locator": "Table 1, Deep Sleep row (128-B register-file block)",
                "url": "https://www.cs.ucr.edu/~hzama001/publications/Slumber.pdf",
                "extrapolation": "applied to each independently gated PE weight bank"}},
        "limitation": "analytical SA clock with assumed row-wise writes and per-PE wave; non-SA and memory work are not rescheduled; physical DS wake remains unproven",
        "managed_events": managed_events,
    }
