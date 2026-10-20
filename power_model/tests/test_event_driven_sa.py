import unittest

from power_model.event_driven_sa import replay_event_driven
from power_model.summarize_regate_event_driven import aggregate


def union_length(spans):
    total = 0
    end = -1
    for begin, finish in sorted(spans):
        if begin >= end:
            total += finish - begin
        elif finish > end:
            total += finish - end
        end = max(end, finish)
    return total


def brute_state_cycles(result):
    rows = result["array_rows"]
    columns = result["array_columns"]
    horizon = result["managed_horizon_cycles"]
    logic = {}
    bank_on = {}
    bank_live = {}
    generations = {}
    for item in result["managed_events"]:
        generations.setdefault(item["preload_instance"], []).append(item)
        for row in range(item["mapped_rows"]):
            for col in range(item["mapped_columns"]):
                pe = row * columns + col
                begin = item["compute_start"] + row + col - 1
                finish = begin + item["vectors"] + 1
                logic.setdefault(pe, []).append((begin, finish))
                bank_on.setdefault((item["bank"], pe), []).append(
                    (begin, finish))
    for items in generations.values():
        preload = items[0]
        base_write = preload["compute_start"] - 1
        for row in range(preload["mapped_rows"]):
            for col in range(preload["mapped_columns"]):
                pe = row * columns + col
                write = base_write + row
                key = (preload["bank"], pe)
                bank_on.setdefault(key, []).append((write, write + 1))
                last = max(item["compute_start"] + row + col +
                           item["vectors"] for item in items)
                bank_live.setdefault(key, []).append((write + 1, last))
    logic_on = sum(union_length(logic.get(pe, []))
                   for pe in range(rows * columns))
    bank_on_cycles = 0
    bank_ds_cycles = 0
    for bank in (0, 1):
        for pe in range(rows * columns):
            key = (bank, pe)
            on = []
            for begin, finish in sorted(bank_on.get(key, [])):
                if not on or begin > on[-1][1]:
                    on.append([begin, finish])
                else:
                    on[-1][1] = max(on[-1][1], finish)
            live = bank_live.get(key, [])
            bank_on_cycles += sum(finish - begin for begin, finish in on)
            for begin, finish in live:
                covered = sum(max(0, min(finish, on_finish) -
                                  max(begin, on_begin))
                              for on_begin, on_finish in on)
                bank_ds_cycles += finish - begin - covered
    return {
        "logic": {"on": logic_on,
                  "off": rows * columns * horizon - logic_on,
                  "capacity": rows * columns * horizon},
        "banks": {"on": bank_on_cycles, "ds": bank_ds_cycles,
                  "off": 2 * rows * columns * horizon - bank_on_cycles -
                  bank_ds_cycles,
                  "capacity": 2 * rows * columns * horizon},
    }


def event(instance, stream, bank, role, issue, start, *, vectors=4,
          rows=2, columns=2):
    return {
        "preload_instance": instance, "tile_stream_id": stream,
        "assumed_bank": bank, "analytical_role": role,
        "analytical_preload_boundary_assumed": False,
        "tile_m": rows, "tile_k": columns, "tile_n": vectors,
        "issue_cycle": issue,
        "onnxim_compute_start": start,
        "onnxim_compute_finish": start + 2 + max(vectors, 4),
    }


def source():
    return {
        "core_id": 0,
        "model": "event-driven unified PE and two-bank analytical source",
        "legacy_whole_sa_predictor_used": False,
        "validation_pass": True,
        "array_rows": 2, "array_columns": 2, "weight_bytes": 2,
        "onnxim_horizon_cycles": 18,
        "events": [
            event(1, 0, 0, "preload_boundary", 0, 1),
            event(2, 1, 1, "preload_boundary", 4, 5),
            event(1, 0, 0, "gemm_consumer", 11, 12),
        ],
    }


class EventDrivenSaTest(unittest.TestCase):
    def test_overlapping_native_windows_are_preserved(self):
        result = replay_event_driven(source())
        self.assertEqual(result["native_onnxim_overlap_intervals"], 1)
        self.assertEqual(result["managed_events"][0]["compute_start"], 1)
        self.assertEqual(result["managed_events"][1]["compute_start"], 5)
        self.assertEqual(result["managed_native_horizon_cycles"], 18)
        self.assertEqual(result["useful_mac_events"], 48)
        self.assertEqual(result["unresolved_pe_wake_events"], 0)
        self.assertIsNone(result["actual_end_to_end_penalty_percent"])
        logic = result["pe_logic_cycles"]
        banks = result["weight_bank_pe_cycles"]
        self.assertEqual(logic["on"] + logic["off"], logic["capacity"])
        self.assertEqual(banks["on"] + banks["ds"] + banks["off"],
                         banks["capacity"])
        self.assertGreater(banks["ds"], 0)

    def test_explicit_beta_only(self):
        self.assertIsNone(replay_event_driven(source())["beta"])
        result = aggregate([source()], beta=0.2)
        self.assertEqual(result["beta_evidence_class"], "assumed")
        self.assertIsNotNone(result[
            "whole_sa_gross_static_opportunity_percent_on_native_timeline"])
        self.assertIsNone(result["total_energy_saving_percent"])

    def test_no_sa_events_leave_logic_and_both_banks_off(self):
        item = source()
        item["events"] = []
        item["onnxim_horizon_cycles"] = 5
        result = replay_event_driven(item)
        self.assertEqual(result["pe_logic_cycles"]["off"], 20)
        self.assertEqual(result["weight_bank_pe_cycles"]["off"], 40)

    def test_same_pe_mac_conflict_delays_later_wave(self):
        item = source()
        item["events"][1]["issue_cycle"] = 3
        item["events"][1]["onnxim_compute_start"] = 4
        item["events"][1]["onnxim_compute_finish"] = 10
        result = replay_event_driven(item)
        self.assertEqual(result["managed_events"][1]["compute_start"], 5)
        self.assertEqual(result["managed_events"][1]["added_delay_cycles"], 1)
        self.assertEqual(result["managed_events"][2]["issue_cycle"], 12)

    def test_same_bank_generation_overwrite_fails_closed(self):
        item = source()
        item["events"][1]["assumed_bank"] = 0
        with self.assertRaisesRegex(ValueError, "before its last consumer"):
            replay_event_driven(item)

    def test_fused_preload_inserts_causal_stall_and_propagates_it(self):
        item = source()
        item["events"][0]["issue_cycle"] = 1
        result = replay_event_driven(item)
        self.assertEqual(result["managed_events"][0]["compute_start"], 2)
        self.assertEqual(result["managed_events"][0]["added_delay_cycles"], 1)
        self.assertEqual(result["managed_events"][1]["issue_cycle"], 5)
        self.assertEqual(result["analytical_added_sa_delay_cycles"], 1)

    def test_early_instruction_issue_does_not_start_physical_row_write(self):
        item = source()
        item["array_rows"] = item["array_columns"] = 4
        item["onnxim_horizon_cycles"] = 40
        item["events"] = [
            event(1, 0, 0, "preload_boundary", 0, 10,
                  rows=4, columns=1),
            event(2, 1, 1, "preload_boundary", 1, 20,
                  rows=4, columns=1),
            event(3, 2, 0, "preload_boundary", 2, 30,
                  rows=4, columns=1),
        ]
        for entry in item["events"]:
            entry["onnxim_compute_finish"] = (
                entry["onnxim_compute_start"] + 10)
        result = replay_event_driven(item)
        self.assertEqual(result["preload_row_write_pe_cycles"], 12)
        self.assertEqual(result["native_onnxim_overlap_intervals"], 0)

    def test_each_consumer_wake_uses_its_own_issue_cycle(self):
        item = source()
        item["events"][2]["issue_cycle"] = 12
        result = replay_event_driven(item)
        self.assertEqual(result["unresolved_pe_wake_events"], 0)
        self.assertEqual(result["managed_events"][2]["compute_start"], 13)

    def test_single_preload_lane_delays_next_row_stream(self):
        item = source()
        item["array_rows"] = item["array_columns"] = 4
        item["onnxim_horizon_cycles"] = 20
        item["events"] = [
            event(1, 0, 0, "preload_boundary", 0, 1,
                  rows=4, columns=1, vectors=1),
            event(2, 1, 1, "preload_boundary", 1, 2,
                  rows=4, columns=1, vectors=1),
        ]
        for entry in item["events"]:
            entry["onnxim_compute_finish"] = (
                entry["onnxim_compute_start"] + 10)
        result = replay_event_driven(item)
        self.assertEqual(result["managed_events"][1]["compute_start"], 5)
        self.assertEqual(result["managed_events"][1]["added_delay_cycles"], 3)

    def test_bank_overwrite_after_last_consumer_waits_for_local_use(self):
        item = source()
        item["events"] = [
            event(1, 0, 0, "preload_boundary", 0, 1),
            event(2, 1, 0, "preload_boundary", 2, 3),
        ]
        result = replay_event_driven(item)
        self.assertEqual(result["managed_events"][1]["compute_start"], 7)
        self.assertEqual(result["analytical_added_sa_delay_cycles"], 4)

    def test_structural_edge_weight_banks_remain_off(self):
        item = source()
        item["array_rows"] = item["array_columns"] = 4
        for entry in item["events"]:
            entry["onnxim_compute_finish"] = entry["onnxim_compute_start"] + 10
        item["onnxim_horizon_cycles"] = 22
        result = replay_event_driven(item)
        self.assertEqual(result["structurally_unused_bank_pe_count"], 24)
        self.assertEqual(result["structurally_unused_bank_pe_off_cycles"],
                         24 * 22)

    def test_early_pe_bank_turns_off_before_diagonal_tail(self):
        item = source()
        item["events"] = item["events"][:1]
        item["onnxim_horizon_cycles"] = 7
        banks = replay_event_driven(item)["weight_bank_pe_cycles"]
        # Bank 0 at PE(0,0) is OFF on [5,7) while PE(1,1)
        # still consumes the same generation on [3,7).
        self.assertEqual(banks["off"], 34)

    def test_contiguous_bank_switch_keeps_pe_on(self):
        item = source()
        item["array_rows"] = item["array_columns"] = 1
        item["onnxim_horizon_cycles"] = 6
        item["events"] = [
            event(1, 0, 0, "preload_boundary", 0, 1, vectors=1,
                  rows=1, columns=1),
            event(2, 1, 1, "preload_boundary", 1, 2, vectors=1,
                  rows=1, columns=1),
        ]
        for entry in item["events"]:
            entry["onnxim_compute_finish"] = entry["onnxim_compute_start"] + 4
        result = replay_event_driven(item)
        self.assertEqual(result["pe_logic_cycles"]["on"], 3)

    def test_contiguous_plain_gemm_needs_no_second_pe_wake(self):
        item = source()
        item["array_rows"] = item["array_columns"] = 1
        item["onnxim_horizon_cycles"] = 6
        item["events"] = [
            event(1, 0, 0, "preload_boundary", 0, 1, vectors=1,
                  rows=1, columns=1),
            event(1, 0, 0, "gemm_consumer", 2, 2, vectors=1,
                  rows=1, columns=1),
        ]
        for entry in item["events"]:
            entry["onnxim_compute_finish"] = entry["onnxim_compute_start"] + 4
        result = replay_event_driven(item)
        self.assertEqual(result["unresolved_pe_wake_events"], 0)

    def test_aggregate_formulas_match_brute_per_pe_intervals(self):
        item = source()
        item["array_rows"] = item["array_columns"] = 4
        item["onnxim_horizon_cycles"] = 30
        item["events"] = [
            event(1, 0, 0, "preload_boundary", 0, 1, vectors=2,
                  rows=4, columns=4),
            event(1, 0, 0, "gemm_consumer", 3, 3, vectors=2,
                  rows=4, columns=4),
            event(2, 1, 1, "preload_boundary", 4, 5, vectors=1,
                  rows=2, columns=3),
            event(3, 2, 0, "preload_boundary", 6, 7, vectors=3,
                  rows=3, columns=2),
        ]
        for entry in item["events"]:
            entry["onnxim_compute_finish"] = (
                entry["onnxim_compute_start"] + 10)
        result = replay_event_driven(item)
        brute = brute_state_cycles(result)
        self.assertEqual(result["pe_logic_cycles"], brute["logic"])
        self.assertEqual(result["weight_bank_pe_cycles"], brute["banks"])


if __name__ == "__main__":
    unittest.main()
