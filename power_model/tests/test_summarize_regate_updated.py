import json
import tempfile
import unittest
from pathlib import Path

from power_model.summarize_regate_updated import aggregate, read_records
from power_model.reference_sa_energy import (
    calculate_aligned_overall, calculate_reference,
    replay_shared_timeline)


def unified_record():
    item = record()
    item.update({
        "array_rows": 1, "array_columns": 1, "weight_bytes": 2,
        "onnxim_horizon_cycles": 502,
        "analytical_horizon_cycles": 502,
        "preload_events": 2, "gemm_events": 0,
        "events": [
            {"input_index": i, "tile_stream_id": i,
             "preload_instance": i + 1, "assumed_bank": 0,
             "opcode": "GEMM_PRELOAD", "analytical_role": "preload_boundary",
             "analytical_preload_boundary_assumed": False,
             "tile_m": 1, "tile_k": 1, "tile_n": 1,
             "issue_cycle": load, "loading_start_cycle": load,
             "loading_finish_cycle": load + 1,
             "analytical_compute_start": load + 1,
             "analytical_compute_finish": load + 2,
             "onnxim_compute_start": load + 1,
             "onnxim_compute_finish": load + 2}
            for i, load in enumerate((0, 500))],
        "pe_state_cycles": {
            "capacity": 502, "on": 2, "on_zero_banks": 0,
            "on_one_bank": 2, "on_two_banks": 0,
            "w_on": 0, "w_on_one_bank": 0, "w_on_two_banks": 0,
            "loading": 2, "off": 498, "unclassified": 0},
        "preload_write_exposure": {
            "pe_cycles": 2, "while_on_pe_cycles": 0,
            "disjoint_from_state_partition": False},
        "timing_disagreement": {
            "sum_event_start_lateness_cycles": 0,
            "max_event_start_lateness_cycles": 0,
            "sum_event_finish_lateness_cycles": 0,
            "max_event_finish_lateness_cycles": 0,
            "simulator_trace_makespan_cycles": 502,
            "analytical_trace_makespan_cycles": 502,
            "trace_makespan_extension_cycles": 0},
        "predictor_reference_whole_sa_logic": {
            "policy_evidence_class": "extrapolated",
            "domain": "compute logic only; weight banks independent",
            "break_even_cycles": 469, "wake_latency_cycles": 10,
            "off_cycles": 499,
            "off_episodes": [{"start_cycle": 2, "end_cycle": 501}],
            "late_wake_events": 1,
            "max_local_wake_lateness_cycles": 10,
            "actual_stall_cycles": None, "timing_modified": False},
    })
    return item


def record(core_id=0):
    return {
        "version": 1,
        "core_id": core_id,
        "model": "updated predictor plus two-bank ReGate-like analytical shadow",
        "measurement_only": True,
        "control_path_modified": False,
        "functional_scheduling_unchanged": True,
        "timing_evidence_class": "assumed",
        "state_accounting_evidence_class": "derived",
        "energy_saving_status": "pending",
        "energy_saving_percent": None,
        "array_rows": 2,
        "array_columns": 2,
        "onnxim_horizon_cycles": 4,
        "analytical_horizon_cycles": 5,
        "preload_events": 1,
        "assumed_preload_boundary_events": 0,
        "gemm_events": 1,
        "events": [
            {
                "opcode": "GEMM_PRELOAD",
                "analytical_role": "preload_boundary",
                "analytical_preload_boundary_assumed": False,
            },
            {
                "opcode": "GEMM",
                "analytical_role": "gemm_consumer",
                "analytical_preload_boundary_assumed": False,
            },
        ],
        "validation_pass": True,
        "validation": {
            "preload_first_valid": True,
            "bank_assignment_valid": True,
            "state_partition_exact": True,
            "invalid_intervals": 0,
            "sequence_errors": 0,
            "footprint_mismatch_intervals": 0,
            "bank_assignment_conflicts": 0,
        },
        "pe_state_cycles": {
            "capacity": 20,
            "on": 5,
            "on_zero_banks": 0,
            "on_one_bank": 3,
            "on_two_banks": 2,
            "w_on": 4,
            "w_on_one_bank": 3,
            "w_on_two_banks": 1,
            "loading": 2,
            "off": 9,
            "unclassified": 0,
        },
        "preload_write_exposure": {
            "pe_cycles": 3,
            "while_on_pe_cycles": 1,
            "disjoint_from_state_partition": False,
        },
        "timing_disagreement": {
            "sum_event_start_lateness_cycles": 2,
            "max_event_start_lateness_cycles": 2,
            "sum_event_finish_lateness_cycles": 3,
            "max_event_finish_lateness_cycles": 3,
            "simulator_trace_makespan_cycles": 3,
            "analytical_trace_makespan_cycles": 4,
            "trace_makespan_extension_cycles": 1,
        },
    }


class UpdatedSummaryTest(unittest.TestCase):
    def test_unified_replay_wake_stall_and_energy_scope(self):
        item = unified_record()
        managed, replay = replay_shared_timeline(item)
        self.assertEqual(managed["events"][1]["analytical_compute_start"], 511)
        self.assertEqual(managed["analytical_horizon_cycles"], 512)
        self.assertEqual(replay["accepted_predictor_off_cycles"], 499)
        self.assertEqual(managed["analytical_whole_sa_off_spans"], [(2, 501)])
        self.assertEqual(managed["analytical_whole_sa_wake_spans"], [(501, 511)])
        self.assertAlmostEqual(
            replay["modeled_sa_schedule_penalty_percent"], 1000 / 502)
        reference = calculate_reference(managed)
        self.assertEqual(reference["pe_logic_cycles"]["off"], 499)
        self.assertEqual(reference["whole_sa_off_pe_cycles_subset"], 499)
        self.assertEqual(reference["pe_logic_cycles"]["on"], 13)
        pending = aggregate([item], expected_cores=1,
                            unified_reference=True)["unified_reference"]
        self.assertIsNone(pending["overall_assumed_saving_vs_one_bank_percent"])
        self.assertIn("pending", pending["energy_status"])
        assumed = aggregate([item], expected_cores=1,
                            unified_reference=True,
                            assume_zero_overheads=True)["unified_reference"]
        self.assertAlmostEqual(assumed["modeled_sa_schedule_penalty_percent"],
                               1000 / 502)
        self.assertIsInstance(assumed["overall_assumed_saving_vs_one_bank_percent"],
                              float)
        self.assertGreater(
            assumed["overall_assumed_saving_vs_two_bank_percent"],
            assumed["overall_assumed_saving_vs_one_bank_percent"])

    def test_unified_replay_discards_mismatched_predictor_episode(self):
        item = unified_record()
        item["events"][1]["onnxim_compute_start"] = 500
        managed, replay = replay_shared_timeline(item)
        self.assertEqual(managed["analytical_horizon_cycles"], 502)
        self.assertEqual(replay["accepted_predictor_off_cycles"], 0)
        self.assertEqual(len(replay["discarded_predictor_off_episodes"]), 1)

    def test_unified_replay_rejects_unreconstructable_analytical_schedule(self):
        item = unified_record()
        item["events"][1]["loading_start_cycle"] = 499
        with self.assertRaisesRegex(ValueError, "analytical preload"):
            replay_shared_timeline(item)
        item = unified_record()
        item["events"][1]["analytical_compute_start"] = 500
        with self.assertRaisesRegex(ValueError, "analytical compute"):
            replay_shared_timeline(item)

    def test_aligned_overall_charges_extended_horizon_and_avoids_double_count(self):
        inputs = {
            "baseline_horizon": 10, "managed_horizon": 12,
            "pe_count": 4, "bank_domain_count": 2,
            "bank_mode_cycles": {"on": 24, "ss": 0, "ds": 0, "off": 0},
            "logic_mode_cycles": {"on": 48, "off": 0},
            "static_fraction": .5, "bank_leakage_share": .5,
            "bank_transition_on_equivalent_cycles": 0,
            "logic_transition_on_equivalent_cycles": 0,
        }
        scenario = calculate_aligned_overall(**inputs)
        self.assertAlmostEqual(scenario["overall_energy_saving_percent"], -10)
        self.assertAlmostEqual(scenario["modeled_performance_penalty_percent"], 20)
        self.assertAlmostEqual(scenario["normalized_managed_energy"], 22)
        charged = calculate_aligned_overall(
            **{**inputs, "bank_transition_on_equivalent_cycles": 2})
        self.assertAlmostEqual(charged["normalized_managed_energy"], 22.5)
        inputs["logic_mode_cycles"] = {"on": 28, "off": 20}
        without_whole_sa = calculate_aligned_overall(**inputs)
        with_whole_sa = calculate_aligned_overall(
            **inputs, whole_sa_off_pe_cycles=20)
        self.assertEqual(without_whole_sa["overall_energy_saving_percent"],
                         with_whole_sa["overall_energy_saving_percent"])
        with self.assertRaisesRegex(ValueError, "subset"):
            calculate_aligned_overall(**inputs, whole_sa_off_pe_cycles=21)

    def test_reference_energy_uses_128_byte_domains_and_keeps_penalty_pending(self):
        item = record()
        item.update({
            "array_rows": 1, "array_columns": 1,
            "weight_bytes": 2,
            "onnxim_horizon_cycles": 100,
            "analytical_horizon_cycles": 100,
            "preload_events": 1, "gemm_events": 0,
            "events": [{
                "input_index": 0, "preload_instance": 1,
                "assumed_bank": 0, "tile_m": 1, "tile_k": 1, "tile_n": 1,
                "opcode": "GEMM_PRELOAD", "analytical_role": "preload_boundary",
                "analytical_preload_boundary_assumed": False,
                "loading_start_cycle": 0, "loading_finish_cycle": 1,
                "analytical_compute_start": 1,
                "analytical_compute_finish": 2,
            }],
            "pe_state_cycles": {
                "capacity": 100, "on": 1, "on_zero_banks": 0,
                "on_one_bank": 1, "on_two_banks": 0,
                "w_on": 0, "w_on_one_bank": 0, "w_on_two_banks": 0,
                "loading": 1, "off": 98, "unclassified": 0,
            },
            "preload_write_exposure": {
                "pe_cycles": 1, "while_on_pe_cycles": 0,
                "disjoint_from_state_partition": False,
            },
            "timing_disagreement": {
                "sum_event_start_lateness_cycles": 0,
                "max_event_start_lateness_cycles": 0,
                "sum_event_finish_lateness_cycles": 0,
                "max_event_finish_lateness_cycles": 0,
                "simulator_trace_makespan_cycles": 2,
                "analytical_trace_makespan_cycles": 2,
                "trace_makespan_extension_cycles": 0,
            },
            "predictor_reference_whole_sa_logic": {
                "policy_evidence_class": "extrapolated",
                "domain": "compute logic only; weight banks independent",
                "break_even_cycles": 469, "wake_latency_cycles": 10,
                "off_cycles": 78,
                "off_episodes": [{"start_cycle": 2, "end_cycle": 80}],
                "late_wake_events": 0,
                "max_local_wake_lateness_cycles": 0,
                "actual_stall_cycles": None, "timing_modified": False,
            },
        })
        result = aggregate([item], expected_cores=1, reference_energy=True)
        scenario = result["reference_energy"]
        self.assertEqual(scenario["per_core"][0]["bank_domain_cycles"]["capacity"], 200)
        self.assertAlmostEqual(scenario["bank_leakage_saving_percent"], 99)
        self.assertAlmostEqual(scenario["pe_logic_leakage_saving_percent"], 95.06)
        self.assertAlmostEqual(
            scenario["gross_analytical_shadow_energy_opportunity_percent"],
            48.515)
        self.assertIsNone(scenario["overall_reference_energy_saving_percent"])
        self.assertIsNone(scenario["estimated_performance_penalty_percent"])
        self.assertEqual(
            scenario["sa_tail_schedule_extension_diagnostic"]
                    ["percent_of_common_run_horizon"], 0)
        self.assertEqual(scenario["predictor_whole_sa_logic"]["per_core"][0]["off_cycles"], 78)

        # A 128-B domain covers all 64 two-byte weights in this row. A
        # one-column tile still has one shared bank domain per bank.
        wide = json.loads(json.dumps(item))
        wide["array_columns"] = 64
        wide["pe_state_cycles"]["capacity"] = 6400
        wide["pe_state_cycles"]["off"] = 6398
        self.assertEqual(calculate_reference(wide)["bank_domain_cycles"]["capacity"], 200)

        # The second bank being OFF is a 50% saving only against a
        # hypothetical *two*-bank-always-ON baseline. It saves nothing
        # against one original bank always ON.
        one_bank_case = json.loads(json.dumps(item))
        one_bank_case["analytical_horizon_cycles"] = 2
        one_bank_case["pe_state_cycles"]["capacity"] = 2
        one_bank_case["pe_state_cycles"]["off"] = 0
        bank_comparison = calculate_reference(one_bank_case)
        self.assertAlmostEqual(bank_comparison["bank_leakage_saving_percent"], 50)
        self.assertAlmostEqual(
            bank_comparison["bank_leakage_saving_vs_one_bank_percent"], 0)

        bad = json.loads(json.dumps(item))
        bad["events"][0]["tile_k"] = 2
        with self.assertRaises(ValueError):
            calculate_reference(bad)
        bad = json.loads(json.dumps(item))
        bad.pop("weight_bytes")
        with self.assertRaisesRegex(ValueError, "weight_bytes"):
            calculate_reference(bad)

    def test_aggregate_preserves_partition_and_pending_energy(self):
        result = aggregate([record(0), record(1)], expected_cores=2)
        self.assertTrue(result["validation_pass"])
        self.assertEqual(result["pe_state_cycles"]["capacity"], 40)
        self.assertEqual(result["pe_state_cycles"]["off"], 18)
        self.assertEqual(result["preload_write_exposure"]["pe_cycles"], 6)
        self.assertEqual(
            result["timing_disagreement"]["sum_core_trace_makespan_extension_cycles"],
            2,
        )
        self.assertEqual(
            result["timing_disagreement"]["max_core_trace_makespan_extension_cycles"],
            1,
        )
        self.assertIsNone(result["energy_saving_percent"])

    def test_rejects_nonzero_unclassified(self):
        item = record()
        item["pe_state_cycles"]["unclassified"] = 1
        with self.assertRaisesRegex(ValueError, "unclassified"):
            aggregate([item])

    def test_reads_prefixed_log_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.log"
            path.write_text("[info] REGATE_UPDATED_PROFILE " + json.dumps(record()) + "\n")
            self.assertEqual(len(read_records(path)), 1)

    def test_rejects_invalid_exposure_values_and_overlap(self):
        for field, value in (("pe_cycles", -1), ("pe_cycles", True),
                             ("pe_cycles", 21), ("while_on_pe_cycles", 4)):
            item = record()
            item["preload_write_exposure"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                aggregate([item])
        item = record()
        item["preload_write_exposure"]["disjoint_from_state_partition"] = True
        with self.assertRaisesRegex(ValueError, "provenance"):
            aggregate([item])
        item = record()
        item["preload_write_exposure"]["pe_cycles"] = 4
        with self.assertRaisesRegex(ValueError, "LOADING state"):
            aggregate([item])

    def test_rejects_impossible_timing_values(self):
        cases = (
            ("sum_event_start_lateness_cycles", -1),
            ("max_event_start_lateness_cycles", 3),
            ("max_event_finish_lateness_cycles", 0),
            ("simulator_trace_makespan_cycles", 5),
            ("analytical_trace_makespan_cycles", 6),
            ("trace_makespan_extension_cycles", 2),
        )
        for field, value in cases:
            item = record()
            item["timing_disagreement"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                aggregate([item])

    def test_rejects_false_provenance_and_detailed_validation(self):
        for field, value in (("measurement_only", False),
                             ("control_path_modified", True),
                             ("timing_evidence_class", "measured"),
                             ("model", "unrelated")):
            item = record()
            item[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "provenance"):
                aggregate([item])
        item = record()
        item["validation"]["state_partition_exact"] = False
        with self.assertRaisesRegex(ValueError, "state_partition_exact"):
            aggregate([item])
        item = record()
        item["validation"]["invalid_intervals"] = 1
        with self.assertRaisesRegex(ValueError, "invalid_intervals"):
            aggregate([item])

    def test_rejects_capacity_and_bank_partition_inconsistency(self):
        item = record()
        item["analytical_horizon_cycles"] = 4
        with self.assertRaisesRegex(ValueError, "capacity"):
            aggregate([item])
        item = record()
        item["pe_state_cycles"]["on_zero_banks"] = 1
        item["pe_state_cycles"]["on_one_bank"] = 2
        with self.assertRaisesRegex(ValueError, "resident bank"):
            aggregate([item])

    def test_rejects_inconsistent_preload_event_counts(self):
        item = record()
        item["preload_events"] = 0
        with self.assertRaisesRegex(ValueError, "no preload boundary"):
            aggregate([item])
        for assumed in (-1, 2):
            item = record()
            item["assumed_preload_boundary_events"] = assumed
            with self.subTest(assumed=assumed), self.assertRaises(ValueError):
                aggregate([item])
        item = record()
        item["events"][0]["analytical_role"] = "gemm_consumer"
        with self.assertRaisesRegex(ValueError, "event .* provenance"):
            aggregate([item])
        item = record()
        item["events"] = item["events"][:1]
        with self.assertRaisesRegex(ValueError, "event list"):
            aggregate([item])


if __name__ == "__main__":
    unittest.main()
