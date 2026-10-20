"""Explicit, cross-paper SA leakage reference scenario from an updated shadow.

This is an offline opportunity calculation.  It does not replay the runtime
predictor or change ONNXim scheduling.  In particular, Slumber's 128-byte GPU
register block is an assumed storage *domain* grouping PE weights according
to the simulation's reported weight byte width, not a 128-byte register in
each PE.
"""

from collections import defaultdict
import copy
import math


BLOCK_BYTES = 128
SLUMBER_POWER = {"on": 1.0, "ss": 0.94, "ds": 0.42, "off": 0.0}
SLUMBER_WAKE = {"ss": 4, "ds": 13, "off": 16}
REGATE_PE_OFF_POWER = 0.03
REGATE_PE_WAKE = 1
REGATE_PE_BET = 47


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _merge(spans, horizon):
    result = []
    for begin, end in sorted(spans):
        if begin > end or end > horizon:
            raise ValueError("reference interval exceeds analytical horizon")
        if begin == end:
            continue
        if result and begin <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((begin, end))
    return result


def _bank_gap(counts, begin, end, retained, next_access):
    """Choose a gross-leakage mode with a complete ON-power wake guard."""
    if begin >= end:
        return
    length = end - begin
    if not retained:
        wake = min(SLUMBER_WAKE["off"], length) if next_access else 0
        counts["off"] += length - wake
        counts["on"] += wake
        counts["wake_on"] += wake
        return
    choices = []
    for mode in ("ss", "ds"):
        wake = min(SLUMBER_WAKE[mode], length) if next_access else 0
        dwell = length - wake
        saving = dwell * (1 - SLUMBER_POWER[mode])
        choices.append((saving, -SLUMBER_WAKE[mode], mode, dwell, wake))
    saving, _, mode, dwell, wake = max(choices)
    if saving <= 0:
        counts["on"] += length
    else:
        counts[mode] += dwell
        counts["on"] += wake
        counts["wake_on"] += wake


def _bank_domain_states(access, resident, horizon):
    access = _merge(access, horizon)
    resident = _merge(resident, horizon)
    counts = {"on": 0, "ss": 0, "ds": 0, "off": 0, "wake_on": 0}
    gap_start = 0
    for access_start, access_end in access + [(horizon, horizon)]:
        # An access interrupts the gap; split it where live data ceases to
        # require retention. A new retention interval begins only after a
        # write access, so an OFF -> retained transition cannot occur here.
        point = gap_start
        for retain_start, retain_end in resident:
            if retain_end <= point or retain_start >= access_start:
                continue
            if retain_start > point:
                _bank_gap(counts, point, min(retain_start, access_start),
                          False, False)
                point = retain_start
            stop = min(retain_end, access_start)
            _bank_gap(counts, point, stop, True, stop == access_start and
                      access_start < horizon)
            point = stop
        if point < access_start:
            _bank_gap(counts, point, access_start, False,
                      access_start < horizon)
        counts["on"] += access_end - access_start
        gap_start = access_end
    if sum(counts[key] for key in ("on", "ss", "ds", "off")) != horizon:
        raise ValueError("128-B bank-domain state partition mismatch")
    return counts


def _logic_pe_states(access, horizon, wake_spans=(), forced_off_spans=()):
    # A whole-array wake powers every PE logic domain, even before its first
    # useful MAC.  It is charged here, rather than as a second OFF saving.
    access = _merge(list(access) + list(wake_spans), horizon)
    forced_off = _merge(forced_off_spans, horizon)
    if any(left < right_end and right < left_end
           for left, left_end in access
           for right, right_end in forced_off):
        raise ValueError("whole-SA OFF overlaps compute or wake")
    on = off = off_episodes = 0
    base_on_spans = []
    point = 0
    for begin, end in access + [(horizon, horizon)]:
        gap = begin - point
        if begin < horizon and gap >= REGATE_PE_BET + REGATE_PE_WAKE:
            off += gap - REGATE_PE_WAKE
            on += REGATE_PE_WAKE
            base_on_spans.append((begin - REGATE_PE_WAKE, begin))
            off_episodes += 1
        elif begin == horizon and gap >= REGATE_PE_BET:
            off += gap
            off_episodes += 1
        else:
            on += gap
            base_on_spans.append((point, begin))
        on += end - begin
        base_on_spans.append((begin, end))
        point = end
    # The full-array control supersedes a local PE's one-cycle wake guard.
    # The subsequent full-array wake is present in `access` and stays ON.
    override = sum(min(a_end, b_end) - max(a_begin, b_begin)
                   for a_begin, a_end in _merge(base_on_spans, horizon)
                   for b_begin, b_end in forced_off
                   if a_begin < b_end and b_begin < a_end)
    on -= override
    off += override
    if on + off != horizon:
        raise ValueError("PE compute-logic state partition mismatch")
    return on, off, off_episodes


def calculate_reference(record, static_fraction=0.5, bank_leakage_share=0.5):
    """Return a gross same-shadow-horizon scenario and constituent accounting.

    The fractions are explicit scenario assumptions. Bank transition energy,
    controller and selector energy, and preload dynamic energy are excluded.
    A separate actual performance penalty is intentionally not inferred.
    """
    for name, value in (("static_fraction", static_fraction),
                        ("bank_leakage_share", bank_leakage_share)):
        if type(value) not in (float, int) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0,1]")
    rows = _integer(record.get("array_rows"), "array_rows")
    columns = _integer(record.get("array_columns"), "array_columns")
    weight_bytes = _integer(record.get("weight_bytes"), "weight_bytes")
    horizon = _integer(record.get("analytical_horizon_cycles"),
                       "analytical_horizon_cycles")
    if (rows == 0 or columns == 0 or horizon == 0 or weight_bytes == 0 or
            BLOCK_BYTES % weight_bytes):
        raise ValueError("reference model needs compatible nonzero SA dimensions, horizon, and weight bytes")
    area = rows * columns
    weights_per_domain = BLOCK_BYTES // weight_bytes
    domains_per_bank = (area + weights_per_domain - 1) // weights_per_domain
    events = record.get("events")
    if not isinstance(events, list):
        raise ValueError("reference model needs complete analytical events")

    copies = {}
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"event {index} must be an object")
        instance = _integer(event.get("preload_instance"), "preload_instance")
        bank = _integer(event.get("assumed_bank"), "assumed_bank")
        m = _integer(event.get("tile_m"), "tile_m")
        k = _integer(event.get("tile_k"), "tile_k")
        n = _integer(event.get("tile_n"), "tile_n")
        start = _integer(event.get("analytical_compute_start"),
                         "analytical_compute_start")
        finish = _integer(event.get("analytical_compute_finish"),
                          "analytical_compute_finish")
        if (instance == 0 or bank not in (0, 1) or m == 0 or k == 0 or
                n == 0 or m > rows or k > columns or
                finish != start + m + k + n - 2 or finish > horizon):
            raise ValueError(f"event {index} has invalid analytical wave")
        copy = copies.setdefault(instance, {"bank": bank, "m": m, "k": k,
                                            "events": [], "preload": None})
        if (copy["bank"], copy["m"], copy["k"]) != (bank, m, k):
            raise ValueError("copy changes bank or footprint")
        copy["events"].append(event)
        if event.get("analytical_role") == "preload_boundary":
            if copy["preload"] is not None:
                raise ValueError("copy has duplicate preload")
            load = _integer(event.get("loading_start_cycle"),
                            "loading_start_cycle")
            load_end = _integer(event.get("loading_finish_cycle"),
                                "loading_finish_cycle")
            if load_end != load + m or load_end > horizon or load_end > start:
                raise ValueError("invalid row-wise preload interval")
            copy["preload"] = event
        elif event.get("analytical_role") != "gemm_consumer":
            raise ValueError("invalid analytical event role")
    if any(copy["preload"] is None for copy in copies.values()):
        raise ValueError("consumer without a preload copy")

    bank_access = defaultdict(list)
    bank_resident = defaultdict(list)
    pe_access = defaultdict(list)
    raw_write_pe_cycles = 0
    for copy in copies.values():
        bank, m, k = copy["bank"], copy["m"], copy["k"]
        raw_write_pe_cycles += m * k
        load_start = copy["preload"]["loading_start_cycle"]
        for row in range(m):
            column = 0
            while column < k:
                group = (row * columns + column) // weights_per_domain
                last_column = min(k, (group + 1) * weights_per_domain -
                                  row * columns)
                key = (bank, group)
                write = load_start + row
                bank_access[key].append((write, write + 1))
                last_use = write + 1
                for event in copy["events"]:
                    begin = event["analytical_compute_start"] + row + column
                    end = event["analytical_compute_start"] + row + \
                        (last_column - 1) + event["tile_n"]
                    bank_access[key].append((begin, end))
                    last_use = max(last_use, end)
                    for c in range(column, last_column):
                        pe_begin = event["analytical_compute_start"] + row + c
                        pe_access[row * columns + c].append(
                            (pe_begin, pe_begin + event["tile_n"]))
                bank_resident[key].append((write + 1, last_use))
                column = last_column

    bank_counts = {key: 0 for key in ("on", "ss", "ds", "off", "wake_on")}
    for bank in (0, 1):
        for group in range(domains_per_bank):
            counts = _bank_domain_states(bank_access[(bank, group)],
                                         bank_resident[(bank, group)], horizon)
            for key in bank_counts:
                bank_counts[key] += counts[key]
    bank_capacity = 2 * domains_per_bank * horizon
    if sum(bank_counts[key] for key in SLUMBER_POWER) != bank_capacity:
        raise ValueError("aggregate bank-domain partition mismatch")
    bank_normalized_energy = sum(bank_counts[key] * SLUMBER_POWER[key]
                                 for key in SLUMBER_POWER)
    bank_saving = 100 * (1 - bank_normalized_energy / bank_capacity)
    one_bank_baseline = domains_per_bank * horizon
    bank_saving_vs_one_bank = 100 * (1 -
                                     bank_normalized_energy / one_bank_baseline)

    logic_on = logic_off = logic_off_episodes = 0
    wake_spans = record.get("analytical_whole_sa_wake_spans", [])
    if not isinstance(wake_spans, list):
        raise ValueError("analytical whole-SA wake spans must be a list")
    wake_spans = [tuple(span) for span in wake_spans]
    _merge(wake_spans, horizon)
    off_spans = record.get("analytical_whole_sa_off_spans", [])
    if not isinstance(off_spans, list):
        raise ValueError("analytical whole-SA OFF spans must be a list")
    off_spans = [tuple(span) for span in off_spans]
    _merge(off_spans, horizon)
    active_compute_pe_cycles = 0
    for pe in range(area):
        active_compute_pe_cycles += sum(end - begin for begin, end in
                                        _merge(pe_access[pe], horizon))
        on, off, episodes = _logic_pe_states(pe_access[pe], horizon,
                                             wake_spans, off_spans)
        logic_on += on
        logic_off += off
        logic_off_episodes += episodes
    logic_capacity = area * horizon
    if logic_on + logic_off != logic_capacity:
        raise ValueError("aggregate PE-logic partition mismatch")
    whole_sa_off_pe_cycles = area * sum(end - begin for begin, end in off_spans)
    if whole_sa_off_pe_cycles > logic_off:
        raise ValueError("whole-SA OFF is not a subset of PE-logic OFF")
    states = record.get("pe_state_cycles")
    exposure = record.get("preload_write_exposure")
    if (not isinstance(states, dict) or
            states.get("on") != active_compute_pe_cycles or
            not isinstance(exposure, dict) or
            exposure.get("pe_cycles") != raw_write_pe_cycles):
        raise ValueError("analytical events disagree with PE state or write exposure")
    logic_energy = logic_on + REGATE_PE_OFF_POWER * logic_off
    logic_saving = 100 * (1 - logic_energy / logic_capacity)
    combined_static_saving = (bank_leakage_share * bank_saving +
                              (1 - bank_leakage_share) * logic_saving)
    overall_gross_saving = static_fraction * combined_static_saving
    one_bank_shadow_sensitivity = static_fraction * (
        bank_leakage_share * bank_saving_vs_one_bank +
        (1 - bank_leakage_share) * logic_saving)
    return {
        "evidence_class": "extrapolated",
        "scope": "SA gross reference energy opportunity, analytical shadow horizon",
        "baseline": "hypothetical two weight banks continuously ON plus PE logic continuously ON over the analytical horizon",
        "overall_reference_energy_saving_percent": None,
        "gross_analytical_shadow_energy_opportunity_percent":
            overall_gross_saving,
        "one_bank_baseline_shadow_sensitivity_percent":
            one_bank_shadow_sensitivity,
        "bank_leakage_saving_percent": bank_saving,
        "bank_leakage_saving_vs_one_bank_percent": bank_saving_vs_one_bank,
        "pe_logic_leakage_saving_percent": logic_saving,
        "combined_sa_static_saving_percent": combined_static_saving,
        "reconciliation_equation": "static_fraction * (bank_share * bank_saving + (1-bank_share) * pe_logic_saving)",
        "assumptions": {
            "static_fraction": static_fraction,
            "bank_share_of_sa_static": bank_leakage_share,
            "weight_bytes": weight_bytes,
            "weight_power_domain_bytes": BLOCK_BYTES,
            "weights_per_power_domain": weights_per_domain,
            "power_domains_per_bank": domains_per_bank,
            "bank_on_ss_ds_off_power_ratios": SLUMBER_POWER,
            "bank_ss_ds_off_wake_cycles": SLUMBER_WAKE,
            "bank_policy": "offline gross-leakage-maximizing SS/DS selection; OFF only without live weight; wake cycles charged ON",
            "pe_logic_off_power_ratio": REGATE_PE_OFF_POWER,
            "pe_logic_wake_cycles": REGATE_PE_WAKE,
            "pe_logic_break_even_cycles": REGATE_PE_BET,
            "dynamic_energy_fraction_unchanged": 1 - static_fraction,
        },
        "bank_domain_cycles": {"capacity": bank_capacity, **bank_counts},
        "one_bank_always_on_baseline_domain_cycles": one_bank_baseline,
        "pe_logic_cycles": {"capacity": logic_capacity, "on": logic_on,
                             "off": logic_off,
                             "off_episodes": logic_off_episodes},
        "analytical_whole_sa_wake_spans": wake_spans,
        "analytical_whole_sa_off_spans": off_spans,
        "whole_sa_off_pe_cycles_subset": whole_sa_off_pe_cycles,
        "excluded_energy": ["bank transition", "PE transition",
                            "controller", "selector", "isolation/routing",
                            "additional preload dynamic"],
        "estimated_performance_penalty_percent": None,
        "performance_penalty_status": "pending: no managed schedule or dependency-rescheduled makespan",
        "actual_end_to_end_saving_status": "pending",
        "one_bank_baseline_scope": "hypothetical one-bank always-ON normalization on the same analytical horizon; static shares remain assumed",
    }


def replay_shared_timeline(record):
    """Replay certified ONNXim OFF episodes against one analytical SA clock.

    This is a trace-replay extrapolation.  An episode is accepted only while
    the ungated analytical SA event trace agrees with ONNXim's SA event trace.
    Following a schedule divergence, original predictor decisions are no
    longer projected.  Loading, bank replacement, and compute retain their
    analytical resource and same-stream dependencies.
    """
    events = record.get("events")
    policy = record.get("predictor_reference_whole_sa_logic")
    base_horizon = _integer(record.get("analytical_horizon_cycles"),
                            "analytical_horizon_cycles")
    if not isinstance(events, list) or not isinstance(policy, dict):
        raise ValueError("unified replay needs events and predictor episodes")
    original_episodes = policy.get("off_episodes")
    if not isinstance(original_episodes, list):
        raise ValueError("unified replay needs predictor OFF episodes")
    if any(not isinstance(event, dict) for event in events):
        raise ValueError("unified replay requires complete event objects")
    # Reconstruct the ungated two-bank schedule before using its timestamps.
    # The emitted schedule must follow the declared issue, load, compute and
    # bank-release dependencies exactly; otherwise an apparent wake penalty
    # could really be an invalid source trace.
    final_index = {}
    for index, event in enumerate(events):
        final_index[_integer(event.get("preload_instance"),
                             "preload_instance")] = index
    baseline_load_ready = baseline_compute_ready = 0
    baseline_bank_ready = [0, 0]
    baseline_bank_owner = [None, None]
    baseline_weight_ready = {}
    baseline_stream_owner = {}
    previous_issue = 0
    for index, event in enumerate(events):
        issue = _integer(event.get("issue_cycle"), "issue_cycle")
        if issue < previous_issue:
            raise ValueError("analytical events are not in issue order")
        previous_issue = issue
        stream = _integer(event.get("tile_stream_id"), "tile_stream_id")
        instance = _integer(event.get("preload_instance"),
                            "preload_instance")
        bank = _integer(event.get("assumed_bank"), "assumed_bank")
        rows = _integer(event.get("tile_m"), "tile_m")
        columns = _integer(event.get("tile_k"), "tile_k")
        vectors = _integer(event.get("tile_n"), "tile_n")
        if bank not in (0, 1) or min(instance, rows, columns, vectors) == 0:
            raise ValueError("invalid analytical bank or event dimensions")
        if event.get("analytical_role") == "preload_boundary":
            if instance in baseline_weight_ready:
                raise ValueError("duplicate preload instance")
            previous = baseline_bank_owner[bank]
            if previous is not None and final_index[previous] >= index:
                raise ValueError("preload overwrites a bank with a live consumer")
            load = max(issue, baseline_load_ready, baseline_bank_ready[bank])
            load_end = load + rows
            if (event.get("loading_start_cycle") != load or
                    event.get("loading_finish_cycle") != load_end):
                raise ValueError("analytical preload does not match lane/bank schedule")
            baseline_load_ready = load_end
            baseline_weight_ready[instance] = load_end
            baseline_stream_owner[stream] = instance
            baseline_bank_owner[bank] = instance
        elif event.get("analytical_role") == "gemm_consumer":
            if baseline_stream_owner.get(stream) != instance:
                raise ValueError("GEMM does not consume its stream's preload")
            if baseline_bank_owner[bank] != instance:
                raise ValueError("GEMM reads a different weight bank")
            if (event.get("loading_start_cycle") != 0 or
                    event.get("loading_finish_cycle") != 0):
                raise ValueError("GEMM unexpectedly writes a bank")
        else:
            raise ValueError("invalid analytical event role")
        start = max(issue, baseline_compute_ready,
                    baseline_weight_ready[instance])
        finish = start + rows + columns + vectors - 2
        if (event.get("analytical_compute_start") != start or
                event.get("analytical_compute_finish") != finish):
            raise ValueError("analytical compute does not match lane/weight schedule")
        baseline_compute_ready = finish
        if index == final_index[instance]:
            baseline_bank_ready[bank] = finish
    raw_episodes = []
    previous_end = 0
    for index, episode in enumerate(original_episodes):
        if not isinstance(episode, dict):
            raise ValueError(f"predictor episode {index} must be an object")
        begin = _integer(episode.get("start_cycle"), "episode start")
        end = _integer(episode.get("end_cycle"), "episode end")
        if (begin >= end or begin < previous_end or
                end > record.get("onnxim_horizon_cycles", -1)):
            raise ValueError("predictor episodes are unordered or out of bounds")
        raw_episodes.append((begin, end))
        previous_end = end
    if sum(end - begin for begin, end in raw_episodes) != policy.get("off_cycles"):
        raise ValueError("predictor episode count disagrees with OFF cycles")

    baseline_waves = []
    first_timing_mismatch = math.inf
    for index, event in enumerate(events):
        start = _integer(event.get("analytical_compute_start"),
                         f"event {index} analytical start")
        finish = _integer(event.get("analytical_compute_finish"),
                          f"event {index} analytical finish")
        onnx_start = _integer(event.get("onnxim_compute_start"),
                              f"event {index} ONNXim start")
        onnx_finish = _integer(event.get("onnxim_compute_finish"),
                               f"event {index} ONNXim finish")
        if finish <= start or finish > base_horizon:
            raise ValueError("invalid analytical compute wave")
        baseline_waves.append((start, finish))
        if (start, finish) != (onnx_start, onnx_finish):
            first_timing_mismatch = min(first_timing_mismatch,
                                        start, onnx_start)

    accepted = []
    discarded = []
    for begin, end in raw_episodes:
        reason = None
        if end + 10 > first_timing_mismatch:
            reason = "analytical and ONNXim event timing diverged"
        elif end > base_horizon:
            reason = "episode exceeds analytical baseline horizon"
        elif end - begin < REGATE_PE_BET:
            reason = "short OFF episode lacks a compatible PE OFF interval"
        elif any(begin < finish and start < end
                 for start, finish in baseline_waves):
            reason = "episode overlaps an analytical compute wave"
        if reason:
            discarded.append({"start_cycle": begin, "end_cycle": end,
                              "reason": reason})
        else:
            accepted.append((begin, end))

    managed = copy.deepcopy(record)
    managed_events = managed["events"]
    load_lane_ready = compute_lane_ready = 0
    bank_replace_ready = [0, 0]
    bank_owner = [None, None]
    weight_ready = {}
    stream_delay = defaultdict(int)
    used_episodes = []
    wake_spans = []
    projection_disabled = False
    for index, (original, event) in enumerate(zip(events, managed_events)):
        bank = _integer(original.get("assumed_bank"), "assumed_bank")
        if bank not in (0, 1):
            raise ValueError("assumed bank must be 0 or 1")
        instance = _integer(original.get("preload_instance"),
                            "preload_instance")
        stream = _integer(original.get("tile_stream_id"), "tile_stream_id")
        rows = _integer(original.get("tile_m"), "tile_m")
        columns = _integer(original.get("tile_k"), "tile_k")
        vectors = _integer(original.get("tile_n"), "tile_n")
        if not (instance and rows and columns and vectors):
            raise ValueError("unified event lacks a positive copy or dimension")
        issue = _integer(original.get("issue_cycle"), "issue_cycle")
        baseline_start = _integer(original.get("analytical_compute_start"),
                                  "analytical_compute_start")
        baseline_finish = _integer(original.get("analytical_compute_finish"),
                                   "analytical_compute_finish")
        effective_issue = issue + stream_delay[stream]
        if original.get("analytical_role") == "preload_boundary":
            previous_copy = bank_owner[bank]
            if (previous_copy is not None and
                    final_index[previous_copy] >= index):
                raise ValueError("preload overwrites a bank with a live consumer")
            original_load = _integer(original.get("loading_start_cycle"),
                                     "loading_start_cycle")
            load_start = max(original_load + stream_delay[stream],
                             effective_issue, load_lane_ready,
                             bank_replace_ready[bank])
            load_finish = load_start + rows
            event["loading_start_cycle"] = load_start
            event["loading_finish_cycle"] = load_finish
            load_lane_ready = load_finish
            weight_ready[instance] = load_finish
            bank_owner[bank] = instance
        elif original.get("analytical_role") != "gemm_consumer":
            raise ValueError("invalid analytical event role")
        if instance not in weight_ready:
            raise ValueError("consumer precedes its preload")
        start = max(baseline_start + stream_delay[stream], effective_issue,
                    compute_lane_ready, weight_ready[instance])
        if not projection_disabled:
            for episode in accepted:
                begin, end = episode
                if episode in used_episodes or begin > baseline_start:
                    continue
                if end > baseline_start:
                    continue  # baseline overlap was already rejected
                used_episodes.append(episode)
                wake_spans.append((end, end + 10))
                if start < end + 10:
                    start = end + 10
        finish = start + rows + columns + vectors - 2
        event["analytical_compute_start"] = start
        event["analytical_compute_finish"] = finish
        compute_lane_ready = finish
        if index == final_index[instance]:
            bank_replace_ready[bank] = finish
        delay = max(0, finish - baseline_finish)
        stream_delay[stream] = max(stream_delay[stream], delay)
        if start > baseline_start:
            projection_disabled = True
    if not projection_disabled:
        for episode in accepted:
            if episode not in used_episodes:
                used_episodes.append(episode)
                # With no subsequent SA use, no wake is needed in this run.
    for episode in accepted:
        if episode not in used_episodes:
            discarded.append({"start_cycle": episode[0],
                              "end_cycle": episode[1],
                              "reason": "predictor projection disabled after managed stall"})
    managed_horizon = max(base_horizon, compute_lane_ready, load_lane_ready,
                          *(end for _, end in wake_spans))
    managed["analytical_horizon_cycles"] = managed_horizon
    managed["analytical_whole_sa_wake_spans"] = wake_spans
    managed["pe_state_cycles"] = {"capacity": (
        _integer(record.get("array_rows"), "array_rows") *
        _integer(record.get("array_columns"), "array_columns") *
        managed_horizon), "on": sum(
        _integer(event["tile_m"], "tile_m") *
        _integer(event["tile_k"], "tile_k") *
        _integer(event["tile_n"], "tile_n") for event in managed_events)}
    managed["preload_write_exposure"] = {
        "pe_cycles": sum(_integer(event["tile_m"], "tile_m") *
                         _integer(event["tile_k"], "tile_k")
                         for event in managed_events
                         if event["analytical_role"] == "preload_boundary")}
    managed["analytical_whole_sa_off_spans"] = used_episodes
    return managed, {
        "evidence_class": "extrapolated",
        "scope": "SA analytical trace replay using matching-prefix predictor episodes",
        "ungated_analytical_horizon_cycles": base_horizon,
        "managed_analytical_horizon_cycles": managed_horizon,
        "accepted_predictor_off_episodes": [
            {"start_cycle": begin, "end_cycle": end}
            for begin, end in used_episodes],
        "accepted_predictor_off_cycles": sum(end - begin
                                              for begin, end in used_episodes),
        "discarded_predictor_off_episodes": discarded,
        "predictor_projection_disabled_after_stall": projection_disabled,
        "whole_sa_wake_spans_charged_on": wake_spans,
        "modeled_sa_schedule_penalty_percent":
            100 * (managed_horizon - base_horizon) / base_horizon
            if base_horizon else None,
        "end_to_end_performance_penalty_status": "pending: dependent non-SA work is not replayed",
    }


def calculate_unified_assumed_energy(baseline_horizon, managed_reference,
                                     static_fraction, bank_leakage_share,
                                     transition_on_equivalent_cycles,
                                     overhead_fraction_of_baseline):
    """One-core SA scenario against explicit one- and two-bank baselines.

    Transition and other overhead inputs are incremental normalized costs.
    All parameters must be supplied. Zero means an explicitly excluded cost,
    not a measurement of zero.
    """
    baseline_horizon = _integer(baseline_horizon, "baseline_horizon")
    if baseline_horizon == 0:
        raise ValueError("baseline horizon must be positive")
    for name, value in (("static_fraction", static_fraction),
                        ("bank_leakage_share", bank_leakage_share),
                        ("transition_on_equivalent_cycles",
                         transition_on_equivalent_cycles),
                        ("overhead_fraction_of_baseline",
                         overhead_fraction_of_baseline)):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and explicit")
    if (not 0 < static_fraction <= 1 or
            not 0 <= bank_leakage_share <= 1 or
            transition_on_equivalent_cycles < 0 or
            overhead_fraction_of_baseline < 0):
        raise ValueError("invalid assumed energy parameter")
    bank = managed_reference["bank_domain_cycles"]
    logic = managed_reference["pe_logic_cycles"]
    domains_per_bank = managed_reference["assumptions"][
        "power_domains_per_bank"]
    area = logic["capacity"] // (bank["capacity"] //
                                 (2 * domains_per_bank))
    bank_energy = sum(bank[mode] * SLUMBER_POWER[mode]
                      for mode in SLUMBER_POWER)
    logic_energy = logic["on"] + REGATE_PE_OFF_POWER * logic["off"]
    bank_energy_per_baseline_bank = bank_energy / domains_per_bank
    logic_energy_per_pe = logic_energy / area
    baseline_one_bank = baseline_horizon / static_fraction
    baseline_two_bank = (baseline_horizon / static_fraction +
                         bank_leakage_share * baseline_horizon)
    unchanged_dynamic = baseline_one_bank - baseline_horizon
    managed = (unchanged_dynamic +
               bank_leakage_share * bank_energy_per_baseline_bank +
               (1 - bank_leakage_share) * logic_energy_per_pe +
               transition_on_equivalent_cycles +
               overhead_fraction_of_baseline * baseline_one_bank)
    return {
        "evidence_class": "extrapolated",
        "scope": "SA analytical shadow energy with explicit reference assumptions",
        "baseline_one_bank_normalized_energy": baseline_one_bank,
        "baseline_two_bank_normalized_energy": baseline_two_bank,
        "managed_normalized_energy": managed,
        "overall_saving_vs_one_bank_percent":
            100 * (1 - managed / baseline_one_bank),
        "overall_saving_vs_two_bank_percent":
            100 * (1 - managed / baseline_two_bank),
        "normalized_unchanged_dynamic_energy": unchanged_dynamic,
        "normalized_bank_energy":
            bank_leakage_share * bank_energy_per_baseline_bank,
        "normalized_logic_energy":
            (1 - bank_leakage_share) * logic_energy_per_pe,
        "normalized_transition_energy": transition_on_equivalent_cycles,
        "normalized_other_overhead_energy":
            overhead_fraction_of_baseline * baseline_one_bank,
        "assumptions": {
            "static_fraction": static_fraction,
            "bank_share_of_original_one_bank_sa_static": bank_leakage_share,
            "workload_dynamic_energy_unchanged": True,
            "transition_on_equivalent_cycles": transition_on_equivalent_cycles,
            "overhead_fraction_of_baseline": overhead_fraction_of_baseline,
            "zero_transition_or_overhead_inputs_mean_excluded": True,
            "second_bank_included_in_managed_energy": True,
        },
        "net_circuit_energy_saving_status": "pending: circuit transition, controller, selector, routing and extra loading costs",
    }


def calculate_aligned_overall(*, baseline_horizon, managed_horizon,
                              pe_count, bank_domain_count,
                              bank_mode_cycles, logic_mode_cycles,
                              static_fraction, bank_leakage_share,
                              bank_transition_on_equivalent_cycles,
                              logic_transition_on_equivalent_cycles,
                              overhead_fraction_of_baseline=0.0,
                              whole_sa_off_pe_cycles=0):
    """Evaluate a supplied common-timeline scenario, with no hidden charges.

    The caller must establish that both horizons and every mode counter refer
    to the same workload and architecture, including the same number of bank
    domains in baseline and managed designs. Dynamic energy is assumed
    unchanged. Transition inputs are *incremental* ON-equivalent domain/PE
    cycles above the leakage already charged by the mode counters; wake-guard
    cycles charged as ON must not be entered again. The whole-SA count is a
    *subset* of logic OFF cycles, never an extra energy term.
    """
    baseline_horizon = _integer(baseline_horizon, "baseline_horizon")
    managed_horizon = _integer(managed_horizon, "managed_horizon")
    pe_count = _integer(pe_count, "pe_count")
    bank_domain_count = _integer(bank_domain_count, "bank_domain_count")
    whole_sa_off_pe_cycles = _integer(whole_sa_off_pe_cycles,
                                      "whole_sa_off_pe_cycles")
    if not all((baseline_horizon, managed_horizon, pe_count,
                bank_domain_count)):
        raise ValueError("aligned scenario requires positive horizons and domains")
    for name, value in (("static_fraction", static_fraction),
                        ("bank_leakage_share", bank_leakage_share),
                        ("overhead_fraction_of_baseline",
                         overhead_fraction_of_baseline),
                        ("bank_transition_on_equivalent_cycles",
                         bank_transition_on_equivalent_cycles),
                        ("logic_transition_on_equivalent_cycles",
                         logic_transition_on_equivalent_cycles)):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be nonnegative and explicit")
    if not 0 < static_fraction <= 1 or bank_leakage_share > 1:
        raise ValueError("fractions must lie in their physical ranges")
    if not isinstance(bank_mode_cycles, dict) or not isinstance(logic_mode_cycles, dict):
        raise ValueError("aligned scenario requires complete mode counters")
    bank = {mode: _integer(bank_mode_cycles.get(mode), f"bank {mode}")
            for mode in SLUMBER_POWER}
    logic = {mode: _integer(logic_mode_cycles.get(mode), f"logic {mode}")
             for mode in ("on", "off")}
    if sum(bank.values()) != bank_domain_count * managed_horizon:
        raise ValueError("bank modes do not partition managed horizon")
    if sum(logic.values()) != pe_count * managed_horizon:
        raise ValueError("logic modes do not partition managed horizon")
    if (whole_sa_off_pe_cycles > logic["off"] or
            whole_sa_off_pe_cycles % pe_count):
        raise ValueError("whole-SA OFF must be a subset of PE-logic OFF")
    bank_on_equivalent = (
        sum(bank[mode] * SLUMBER_POWER[mode] for mode in SLUMBER_POWER) +
        bank_transition_on_equivalent_cycles) / bank_domain_count
    logic_on_equivalent = (
        logic["on"] + REGATE_PE_OFF_POWER * logic["off"] +
        logic_transition_on_equivalent_cycles) / pe_count
    baseline_total = baseline_horizon / static_fraction
    unchanged_dynamic = baseline_total - baseline_horizon
    managed_total = (unchanged_dynamic +
                     bank_leakage_share * bank_on_equivalent +
                     (1 - bank_leakage_share) * logic_on_equivalent +
                     overhead_fraction_of_baseline * baseline_total)
    return {
        "evidence_class": "assumed",
        "scope": "supplied aligned common-timeline SA scenario",
        "assumptions": {
            "same_bank_domain_count_in_baseline_and_managed": True,
            "workload_dynamic_energy_unchanged": True,
            "transition_inputs_incremental_above_mode_cycle_leakage": True,
        },
        "overall_energy_saving_percent": 100 * (1 - managed_total / baseline_total),
        "modeled_performance_penalty_percent":
            100 * (managed_horizon - baseline_horizon) / baseline_horizon,
        "normalized_baseline_energy": baseline_total,
        "normalized_managed_energy": managed_total,
        "normalized_unchanged_dynamic_energy": unchanged_dynamic,
        "normalized_bank_static_plus_transition_energy":
            bank_leakage_share * bank_on_equivalent,
        "normalized_logic_static_plus_transition_energy":
            (1 - bank_leakage_share) * logic_on_equivalent,
        "normalized_other_overhead_energy":
            overhead_fraction_of_baseline * baseline_total,
        "whole_sa_off_pe_cycles_subset": whole_sa_off_pe_cycles,
    }
