"""ReGate PE-state abstractions over an existing analytical SA event timeline.

This module never changes the proposed event-driven two-bank replay.  The
``matched_two_bank`` result applies ReGate-style ON/W_ON/OFF control to that
replay's *same two-bank hardware and clock*; it is an ablation, not a literal
ReGate implementation.  ``literal_one_bank`` accounts one stationary weight
register per PE, and is unavailable if two generations need that register at
the same time.  The caller must not turn an unavailable result into a saving.

The delayed mask scenario powers all candidate columns of each *loaded* row
until the mapped row stream ends.  It does not claim that ReGate physically
writes every zero in the complete array: that preload detail is not specified
by the paper.  The optimistic scenario powers only mapping-required columns.
Numerical zero-weight gating is absent because ONNXim supplies dimensions,
not stationary-weight values.
"""

import math

if __package__:
    from .event_driven_sa import PE_OFF_RATIO
else:
    from event_driven_sa import PE_OFF_RATIO


MASK_POLICIES = ("optimistic", "delayed")


def _positive_int(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _post_mask_cycles(generation, mask_ready):
    """Sum max(0, last_local_use-mask_ready) over its mapped rectangle."""
    width = generation["columns"]
    height = generation["rows"]
    last = max(event["compute_start"] + event["vectors"]
               for event in generation["consumers"])
    total = 0
    for row in range(height):
        offset = last + row - mask_ready
        first_column = max(0, 1 - offset)
        if first_column < width:
            count = width - first_column
            total += count * (offset + first_column) + count * (count - 1) // 2
    return total


def _generations(replay):
    rows = _positive_int(replay.get("array_rows"), "array_rows")
    columns = _positive_int(replay.get("array_columns"), "array_columns")
    horizon = _positive_int(replay.get("managed_horizon_cycles"),
                            "managed_horizon_cycles")
    if replay.get("state_partition_validation_pass") is not True:
        raise ValueError("source replay state partition did not pass")
    events = replay.get("managed_events")
    if not isinstance(events, list):
        raise ValueError("source replay has no managed events")
    generations = {}
    for event in events:
        instance = _positive_int(event.get("preload_instance"),
                                 "preload_instance")
        mapped_rows = _positive_int(event.get("mapped_rows"), "mapped_rows")
        mapped_cols = _positive_int(event.get("mapped_columns"),
                                   "mapped_columns")
        vectors = _positive_int(event.get("vectors"), "vectors")
        start = _nonnegative_int(event.get("compute_start"), "compute_start")
        _nonnegative_int(event.get("issue_cycle"), "issue_cycle")
        if mapped_rows > rows or mapped_cols > columns:
            raise ValueError("mapped footprint exceeds physical SA")
        if start + mapped_rows + mapped_cols + vectors - 2 > horizon:
            raise ValueError("mapped wave exceeds managed horizon")
        item = generations.setdefault(instance, {
            "instance": instance, "rows": mapped_rows,
            "columns": mapped_cols, "bank": event.get("bank"),
            "preload": None, "consumers": []})
        if (item["rows"], item["columns"], item["bank"]) != (
                mapped_rows, mapped_cols, event.get("bank")):
            raise ValueError("preload generation changes footprint or bank")
        item["consumers"].append(event)
        if event.get("role") == "preload_boundary":
            if item["preload"] is not None:
                raise ValueError("duplicate preload boundary")
            item["preload"] = event
        elif event.get("role") != "gemm_consumer":
            raise ValueError("unknown SA event role")
    for item in generations.values():
        preload = item["preload"]
        if preload is None or item["consumers"][0] is not preload:
            raise ValueError("generation lacks preceding preload boundary")
        write = preload["compute_start"] - 1
        if write < preload["issue_cycle"] or write < 0:
            raise ValueError("preload write would precede issue")
        if write + item["rows"] > horizon:
            raise ValueError("preload row stream exceeds horizon")
        item["write_start"] = write
        item["mask_ready"] = write + item["rows"]
    return rows, columns, horizon, sorted(generations.values(),
                                          key=lambda item: item["write_start"])


def replay_regate_baseline(replay, *, mask_policy="delayed", beta=None):
    """Return matched two-bank ablation and a fail-closed literal one-bank case.

    ``beta`` is the assumed fraction of an *always-on two-bank* SA's static
    power that belongs to both weight banks.  For feasible one-bank ReGate,
    its one-bank weight share is derived as beta/(2-beta), which preserves the
    same assumed logic power and per-bank power in both designs.
    """
    if mask_policy not in MASK_POLICIES:
        raise ValueError("unknown preload-mask policy")
    if beta is not None and (type(beta) not in (int, float) or
                             not math.isfinite(beta) or not 0 <= beta <= 1):
        raise ValueError("beta must be a finite fraction in [0,1]")
    rows, columns, horizon, generations = _generations(replay)
    area = rows * columns
    logic = replay.get("pe_logic_cycles")
    banks = replay.get("weight_bank_pe_cycles")
    if not isinstance(logic, dict) or not isinstance(banks, dict):
        raise ValueError("source replay lacks PE or bank state cycles")
    if (logic.get("capacity") != area * horizon or
            logic.get("on", -1) + logic.get("off", -1) != area * horizon or
            banks.get("capacity") != 2 * area * horizon or
            sum(banks.get(mode, -1) for mode in ("on", "ds", "off")) !=
            2 * area * horizon):
        raise ValueError("source replay state partition is invalid")

    extra_candidate_cycles = 0
    one_bank_powered = 0
    one_bank_conflicts = []
    release_offset_by_column = [0] * columns
    release_owner_by_column = [None] * columns
    same_bank_release_by_column = [[0] * columns for _ in range(2)]
    delayed_same_bank_overlap = []
    for generation in generations:
        height = generation["rows"]
        width = generation["columns"]
        base = generation["write_start"]
        mask_ready = generation["mask_ready"]
        if mask_policy == "delayed":
            prior_release = same_bank_release_by_column[generation["bank"]]
            if any(prior_release[column] > base
                   for column in range(width, columns)):
                delayed_same_bank_overlap.append(generation["instance"])
            extra_candidate_cycles += ((columns - width) * height *
                                       (height + 1) // 2)
            candidate_width = columns
        else:
            candidate_width = width
        candidate_cycles = candidate_width * height * (height + 1) // 2
        one_bank_powered += candidate_cycles + _post_mask_cycles(
            generation, mask_ready)

        # At PE(r,c), last use is last_start+r+c+vectors. New generation
        # writes that row at next_base+r. The r terms cancel. Keep a frontier
        # per candidate column; using a single widest-column bound would
        # falsely flag an optimistic narrow overwrite as a conflict.
        conflicting = [(column, release_offset_by_column[column])
                       for column in range(candidate_width)
                       if base < release_offset_by_column[column]]
        if conflicting:
            conflict_column, needed_start = max(conflicting,
                                                key=lambda pair: pair[1])
            one_bank_conflicts.append({
                "preload_instance": generation["instance"],
                "prior_preload_instance":
                    release_owner_by_column[conflict_column],
                "conflicting_column": conflict_column,
                "write_start": base,
                "prior_latest_required_write_start": needed_start,
            })
        for column in range(candidate_width):
            if mask_ready > release_offset_by_column[column]:
                release_offset_by_column[column] = mask_ready
                release_owner_by_column[column] = generation["instance"]
        last_use_offset = max(event["compute_start"] + event["vectors"]
                              for event in generation["consumers"])
        same_bank_release = same_bank_release_by_column[generation["bank"]]
        for column in range(width):
            release_offset = last_use_offset + column
            if release_offset > release_offset_by_column[column]:
                release_offset_by_column[column] = release_offset
                release_owner_by_column[column] = generation["instance"]
            same_bank_release[column] = max(same_bank_release[column],
                                            release_offset)

    two_bank_on = banks["on"]
    two_bank_w_on = banks["ds"] + extra_candidate_cycles
    two_bank_off = banks["off"] - extra_candidate_cycles
    if two_bank_off < 0 and not delayed_same_bank_overlap:
        raise ValueError("delayed preload exceeds two-bank OFF capacity")
    two_bank_capacity = banks["capacity"]
    if (not delayed_same_bank_overlap and
            two_bank_on + two_bank_w_on + two_bank_off != two_bank_capacity):
        raise ValueError("matched two-bank state partition failed")
    logic_equivalent = logic["on"] + PE_OFF_RATIO * logic["off"]
    two_bank_equivalent = two_bank_on + two_bank_w_on
    two_bank_ratio = ((1 - beta) * logic_equivalent / logic["capacity"] +
                      beta * two_bank_equivalent / two_bank_capacity
                      if beta is not None and not delayed_same_bank_overlap
                      else None)

    literal = {
        "status": "one_bank_conflict" if one_bank_conflicts else "feasible",
        "conflict_count": len(one_bank_conflicts),
        "conflicts": one_bank_conflicts,
        "pe_cycles": None,
        "energy": None,
        "energy_status": "one-bank conflict; rescheduling or reloads required"
                         if one_bank_conflicts else
                         "beta not supplied; one-bank weight share unknown"
                         if beta is None else
                         "gross static leakage only; transitions excluded",
    }
    if not one_bank_conflicts:
        one_bank_on = logic["on"]
        one_bank_w_on = one_bank_powered - one_bank_on
        one_bank_off = area * horizon - one_bank_powered
        if min(one_bank_w_on, one_bank_off) < 0:
            raise ValueError("literal one-bank state partition failed")
        literal["pe_cycles"] = {
            "on": one_bank_on, "w_on": one_bank_w_on,
            "off": one_bank_off, "capacity": area * horizon}
        if beta is not None:
            one_bank_weight_share = beta / (2 - beta)
            one_bank_ratio = (
                one_bank_on + one_bank_weight_share * one_bank_w_on +
                PE_OFF_RATIO * one_bank_off) / (area * horizon)
            literal["energy"] = {
                "one_bank_weight_share": one_bank_weight_share,
                "normalized_static_energy_ratio_vs_one_bank_always_on":
                    one_bank_ratio,
                "gross_static_saving_percent_vs_one_bank_always_on":
                    100 * (1 - one_bank_ratio),
                "baseline_power_conversion_two_bank_to_one_bank":
                    2 / (2 - beta),
            }

    return {
        "model": "ReGate SA power-policy abstraction",
        "evidence_class": "analytical policy sensitivity",
        "source_timeline": "existing proposed managed SA events",
        "mask_policy": mask_policy,
        "mask_policy_meaning": (
            "candidate columns in each loaded row stay powered until that "
            "mapped row stream ends; physical full-array padding writes "
            "are not asserted" if mask_policy == "delayed" else
            "mapping-required columns only; zero detection before a "
            "destination register is powered is assumed"),
        "numeric_zero_weights_modeled": False,
        "spatial_mask_provenance":
            "mapping-derived tile_m-by-tile_k structural footprint; ReGate's actual-value row/column masks are unavailable",
        "preload_candidate_extra_weight_cycles": extra_candidate_cycles,
        "pe_logic_cycles": dict(logic),
        "matched_two_bank": {
            "interpretation": "ReGate-style ON/W_ON/OFF policy on proposed two-bank hardware, not literal ReGate; weights optimistically retire OFF immediately after their last known consumer",
            "status": ("unresolved_padded_column_overlap"
                       if delayed_same_bank_overlap else "feasible"),
            "overlapping_preload_instances": delayed_same_bank_overlap,
            "weight_bank_cycles": {
                "on": two_bank_on, "w_on": two_bank_w_on,
                "off": two_bank_off, "capacity": two_bank_capacity}
                if not delayed_same_bank_overlap else None,
            "beta": beta,
            "normalized_static_energy_ratio_vs_two_bank_always_on":
                two_bank_ratio,
            "gross_static_saving_percent_vs_two_bank_always_on":
                100 * (1 - two_bank_ratio) if two_bank_ratio is not None
                else None,
        },
        "literal_one_bank": literal,
        "limitations": [
            "ReGate's exact destination-register power timing during preload is unspecified",
            "no numerical zero-weight detection because ONNXim lacks weight values",
            "one-bank result is unavailable if simultaneous generations need one PE register",
            "native non-SA and memory execution is not rescheduled",
        ],
    }
