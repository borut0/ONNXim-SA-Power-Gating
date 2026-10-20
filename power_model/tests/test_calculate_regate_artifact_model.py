import unittest

from power_model.calculate_regate_artifact_model import calculate
from power_model.summarize_sa_region_profile import aggregate


def make_summary(*cores, rows=4, columns=4):
    return {
        "version": 4,
        "validation_pass": True,
        "array_rows": rows,
        "array_columns": columns,
        "core_cycles": sum(item["core_cycles"] for item in cores),
        "whole_sa_active_cycles": sum(
            item["whole_sa_active_cycles"] for item in cores
        ),
        "sa_instruction_intervals": sum(
            item["sa_instruction_intervals"] for item in cores
        ),
        "modeled_useful_mac_count": sum(
            item["modeled_useful_mac_count"] for item in cores
        ),
        "per_core_regate_inputs": list(cores),
    }


def core(core_id=0, execution=100, active=10, instructions=2, macs=80):
    return {
        "core_id": core_id,
        "core_cycles": execution,
        "whole_sa_active_cycles": active,
        "sa_instruction_intervals": instructions,
        "modeled_useful_mac_count": macs,
    }


def region_record(core_id, macs):
    histogram = [{"length_cycles": 6, "episode_count": 1}]
    return {
        "version": 4,
        "core_id": core_id,
        "array_rows": 4,
        "array_columns": 4,
        "core_cycles": 10,
        "sa_instruction_intervals": 1,
        "modeled_useful_mac_count": macs,
        "invalid_dimension_intervals": 0,
        "clipped_dimension_intervals": 0,
        "whole_sa_active_cycles": 4,
        "whole_sa_idle_cycles": 6,
        "whole_sa_committed_cycles": 4,
        "whole_sa_uncommitted_cycles": 6,
        "whole_sa_idle_episode_length_histogram": histogram,
        "validation_pass": True,
        "granularities": [{
            "region_size_rows": 4,
            "region_size_columns": 4,
            "region_rows": 1,
            "region_columns": 1,
            "region_count": 1,
            "capacity_region_cycles": 10,
            "modeled_active_region_cycles": 4,
            "modeled_idle_region_cycles": 6,
            "temporally_idle_region_cycles": 6,
            "spatially_idle_region_cycles_during_sa_active": 0,
            "conservative_reserved_region_cycles": 4,
            "conservative_available_region_cycles": 6,
            "conservative_temporally_uncommitted_region_cycles": 6,
            "conservative_spatially_available_region_cycles_during_commitment": 0,
            "idle_episode_length_histogram": histogram,
            "accounting_exact": True,
        }],
    }


def region_record_v5(core_id, macs):
    record = region_record(core_id, macs)
    record["version"] = 5
    # 4x4 array, 4 active cycles, 6 idle cycles.
    record["pe_state_opportunity"] = {
        "completed_intervals": 1,
        "incomplete_intervals": 0,
        "invalid_dimension_intervals": 0,
        "insufficient_duration_intervals": 0,
        "union_active_cycles": 4,
        "active_capacity_pe_cycles": 64,
        "on_compute_pe_cycles": macs,
        "resident_weight_pe_cycles": 48,
        "w_on_candidate_pe_cycles": 48 - macs,
        "off_unmapped_candidate_pe_cycles": 16,
        "temporal_idle_unclassified_pe_cycles": 96,
        "total_capacity_pe_cycles": 160,
        "accounting_exact": True,
    }
    return record


def region_record_v6(core_id, macs):
    record = region_record_v5(core_id, macs)
    record["version"] = 6
    # This synthetic fixture has no issue-to-start gap. The analytical state
    # partition is therefore exact under the documented wave/lifetime model.
    record["regate_analytical_wave_lifetime"] = {
        "analytical_on_pe_cycles": macs,
        "simulator_active_capacity_pe_cycles": 64,
        "simulator_active_non_wave_pe_cycles": 64 - macs,
        "resident_weight_pe_cycles": 48,
        "pre_start_committed_pe_cycles": 0,
        "analytical_w_on_pe_cycles": 48 - macs,
        "structural_off_candidate_pe_cycles": 160 - 48,
        "unclassified_pe_cycles": 0,
        "total_capacity_pe_cycles": 160,
        "completed_intervals": 1,
        "incomplete_intervals": 0,
        "invalid_dimension_intervals": 0,
        "wave_exceeds_instruction_intervals": 0,
        "overlapping_wave_intervals": 0,
        "unknown_weight_block_intervals": 0,
        "gemm_without_preload_intervals": 0,
        "weight_block_mismatch_intervals": 0,
        "footprint_mismatch_intervals": 0,
        "wave_overlap_free": True,
        "weight_lifetime_complete": True,
        "partition_exact_under_model": True,
    }
    record["granularities"][0]["regate_analytical_wave"] = {
        "analytical_on_region_cycles": 4,
        "analytical_w_on_region_cycles": 0,
        "structural_off_candidate_region_cycles": 6,
        "preload_committed_unclassified_region_cycles": 0,
        "capacity_region_cycles": 10,
        "partition_exact_under_model": True,
    }
    return record


class ReGateArtifactModelTest(unittest.TestCase):
    def test_pinned_artifact_golden_vectors_at_128_by_128(self):
        # Independently evaluated from the pinned ReGate artifact equations:
        # utilization is computed from the original SA time, delay is then
        # added, execution time is max(original execution, adjusted SA time),
        # and HW/Full uses a 0.03 low-power factor with one-cycle PE delay.
        vectors = [
            # execution, active, FLOPs, ops, util, delay, adjusted SA,
            # adjusted execution, normalized ReGate energy
            (1000, 256, 8_388_608, 4, 1.0, 2, 258, 1000, 280.26),
            (1000, 256, 4_194_304, 4, 0.5, 2, 258, 1000, 155.13),
            (256, 256, 4_194_304, 4, 0.5, 2, 258, 258, 132.87),
            (1000, 1000, 16_384_000, 2, 0.5, 2, 1002, 1002, 516.03),
            (1000, 128, 8_388_608, 8, 1.0, 1, 129, 1000, 155.13),
        ]
        for execution, active, flops, instructions, expected_u, expected_o, \
                expected_sa, expected_execution, expected_energy in vectors:
            result = calculate(
                make_summary(
                    core(
                        execution=execution,
                        active=active,
                        instructions=instructions,
                        macs=flops // 2,
                    ),
                    rows=128,
                    columns=128,
                )
            )["per_core"][0]
            self.assertAlmostEqual(result["sa_flops_utilization"], expected_u)
            self.assertEqual(
                result["estimated_pe_delay_overhead_cycles"], expected_o
            )
            self.assertEqual(
                result["estimated_adjusted_sa_cycles"], expected_sa
            )
            self.assertEqual(
                result["estimated_adjusted_execution_cycles"],
                expected_execution,
            )
            self.assertAlmostEqual(
                result["normalized_regate_static_energy"], expected_energy
            )

    def test_non_mxu_equivalent_zero_sa_case(self):
        result = calculate(
            make_summary(
                core(execution=1000, active=0, instructions=0, macs=0),
                rows=128,
                columns=128,
            )
        )["per_core"][0]
        self.assertEqual(result["estimated_pe_delay_overhead_cycles"], 0)
        self.assertAlmostEqual(result["sa_flops_utilization"], 0.0)
        self.assertAlmostEqual(
            result["normalized_regate_static_energy"], 30.0
        )

    def test_energy_equation_without_delay(self):
        result = calculate(
            make_summary(core()), low_power_factor=0.03, pe_delay_cycles=0
        )
        self.assertEqual(
            result["reference"]["artifact_source_sha256"],
            "0174f85e4354f8ccb0f10c34b5c05ed0912179bda29f4087a240809c0baab800",
        )
        measured = result["per_core"][0]
        self.assertAlmostEqual(measured["sa_flops_utilization"], 0.5)
        self.assertAlmostEqual(
            measured["normalized_regate_static_energy"], 7.85
        )
        self.assertAlmostEqual(
            result["aggregate"]["normalized_static_energy_saving_percent"],
            92.15,
        )

    def test_artifact_delay_ordering_and_exposed_overhead(self):
        result = calculate(
            make_summary(
                core(execution=10, active=10, instructions=10, macs=160)
            ),
            low_power_factor=0.03,
            pe_delay_cycles=1,
        )
        measured = result["per_core"][0]
        self.assertEqual(measured["estimated_pe_delay_overhead_cycles"], 3)
        self.assertEqual(measured["estimated_adjusted_sa_cycles"], 13)
        self.assertEqual(measured["estimated_adjusted_execution_cycles"], 13)
        self.assertAlmostEqual(
            result["aggregate"]["estimated_performance_overhead_percent"],
            30.0,
        )

        partial = calculate(
            make_summary(core()), low_power_factor=0.03, pe_delay_cycles=1
        )["per_core"][0]
        self.assertAlmostEqual(partial["sa_flops_utilization"], 0.5)
        self.assertEqual(partial["estimated_adjusted_sa_cycles"], 12)
        self.assertAlmostEqual(
            partial["normalized_regate_static_energy"], 8.82
        )

    def test_all_on_factor_produces_no_saving_without_delay(self):
        result = calculate(
            make_summary(core()), low_power_factor=1.0, pe_delay_cycles=0
        )
        self.assertAlmostEqual(
            result["aggregate"]["normalized_static_energy_saving_percent"],
            0.0,
        )

    def test_utilization_cap_is_visible(self):
        result = calculate(
            make_summary(core(active=1, instructions=1, macs=17)),
            pe_delay_cycles=0,
        )
        measured = result["per_core"][0]
        self.assertGreater(measured["raw_sa_flops_utilization"], 1.0)
        self.assertEqual(measured["sa_flops_utilization"], 1.0)
        self.assertTrue(measured["utilization_clamped_to_one"])
        self.assertTrue(result["warnings"])

    def test_rejects_summary_without_work_counts(self):
        with self.assertRaisesRegex(ValueError, "work-count inputs"):
            calculate({"version": 3, "validation_pass": True})

    def test_rejects_aggregate_mismatch(self):
        summary = make_summary(core())
        summary["modeled_useful_mac_count"] += 1
        with self.assertRaisesRegex(ValueError, "does not match"):
            calculate(summary)

    def test_region_summary_preserves_per_core_regate_inputs(self):
        summary = aggregate([region_record(1, 32), region_record(0, 16)])
        self.assertTrue(summary["validation_pass"])
        self.assertEqual(summary["version"], 4)
        self.assertEqual(summary["modeled_useful_mac_count"], 48)
        self.assertEqual(
            [item["core_id"] for item in summary["per_core_regate_inputs"]],
            [0, 1],
        )

    def test_region_summary_aggregates_v5_pe_state_candidates(self):
        summary = aggregate([
            region_record_v5(1, 32),
            region_record_v5(0, 16),
        ])
        self.assertTrue(summary["validation_pass"])
        self.assertEqual(summary["version"], 5)
        opportunity = summary["pe_state_opportunity"]
        self.assertEqual(opportunity["on_compute_pe_cycles"], 48)
        self.assertEqual(opportunity["resident_weight_pe_cycles"], 96)
        self.assertEqual(
            opportunity["temporal_idle_unclassified_pe_cycles"], 192
        )
        self.assertEqual(
            opportunity["temporal_idle_classification"], "unclassified"
        )
        self.assertTrue(opportunity["accounting_exact"])
        self.assertTrue(opportunity["not_energy_result"])

    def test_region_summary_rejects_v5_pe_state_mismatch(self):
        record = region_record_v5(0, 16)
        record["pe_state_opportunity"][
            "temporal_idle_unclassified_pe_cycles"
        ] += 1
        summary = aggregate([record])
        self.assertFalse(summary["validation_pass"])
        self.assertIn(
            "aggregate PE-state opportunity accounting mismatch",
            summary["validation_errors"],
        )

    def test_region_summary_preserves_v6_wave_lifetime_partition(self):
        summary = aggregate([
            region_record_v6(1, 32),
            region_record_v6(0, 16),
        ])
        self.assertTrue(summary["validation_pass"])
        self.assertEqual(summary["version"], 6)
        wave = summary["regate_analytical_wave_lifetime"]
        self.assertEqual(wave["analytical_on_pe_cycles"], 48)
        self.assertEqual(wave["analytical_w_on_pe_cycles"], 48)
        self.assertEqual(wave["structural_off_candidate_pe_cycles"], 224)
        self.assertTrue(wave["partition_exact_under_model"])
        region = summary["granularities"][0]["regate_analytical_wave"]
        self.assertEqual(region["analytical_on_region_cycles"], 8)
        self.assertEqual(region["structural_off_candidate_region_cycles"], 12)
        self.assertTrue(region["partition_exact_under_model"])

    def test_region_summary_rejects_false_exact_v6_region_partition(self):
        record = region_record_v6(0, 16)
        record["granularities"][0]["regate_analytical_wave"][
            "structural_off_candidate_region_cycles"
        ] += 1
        summary = aggregate([record])
        self.assertFalse(summary["validation_pass"])
        self.assertTrue(any(
            "analytical state partition failed" in item
            for item in summary["validation_errors"]
        ))


if __name__ == "__main__":
    unittest.main()
