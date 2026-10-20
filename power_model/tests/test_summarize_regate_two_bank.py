import unittest

from power_model.summarize_regate_two_bank import aggregate


def rowwise(**overrides):
    item = {
        "available": True,
        "evidence_class": "derived",
        "timing_input_evidence_class": "assumed",
        "secondary_counts_complete": True,
        "physical_feasibility_proven": False,
        "first_row_lead_cycles_before_start": 1,
        "row_start_period_cycles": 1,
        "row_write_duration_cycles": 1,
        "row_order": "ascending",
        "row_columns_written_simultaneously_assumption": True,
        "row_write_start_equation": (
            "start_cycle - first_row_lead_cycles_before_start + "
            "ordered_row_index * row_start_period_cycles"
        ),
        "row_ready_equation": (
            "row_write_start + row_write_duration_cycles"
        ),
        "bank_release_equation": (
            "max(row_ready, last local modeled MAC end across the "
            "preload and its exactly matched later GEMMs)"
        ),
        "necessary_conditions_pass": True,
        "mapped_pe_weight_write_events": 2,
        "row_write_before_issue_events": 0,
        "row_ready_after_first_mac_events": 0,
        "max_row_ready_lateness_cycles": 0,
        "mapped_pe_peak_live_preloads": 2,
        "mapped_pe_over_two_pe_cycles": 0,
        "mapped_pes_ever_over_two": 0,
        "locally_blocked_pe_weight_write_events": 0,
        "max_local_bank_wait_cycles": 0,
        "max_local_first_mac_delay_cycles": 0,
    }
    item.update(overrides)
    return item


def rowwise_state(valid=True, **state_overrides):
    state = {
        "capacity": 6,
        "on": 4 if valid else 0,
        "on_preload_wave": 4 if valid else 0,
        "on_gemm_wave": 0,
        "on_zero_banks": 0,
        "on_one_bank": 1 if valid else 0,
        "on_two_banks": 3 if valid else 0,
        "w_on_one_bank": 0,
        "w_on_two_banks": 2 if valid else 0,
        "off": 0,
        "unclassified": 0 if valid else 6,
    }
    state.update(state_overrides)
    return {
        "version": 1,
        "available": True,
        "evidence_class": "derived",
        "timing_input_evidence_class": "assumed",
        "validation_pass": valid,
        "required_checks": {
            "complete_known_matched_trace": True,
            "rowwise_timing_and_two_bank_checks": valid,
            "state_partition": valid,
        },
        "pe_state_cycles": state,
        "preload_write_exposure": {
            "evidence_class": "derived" if valid else "pending",
            "timing_input_evidence_class": "assumed",
            "pe_cycles": 2 if valid else None,
            "while_on_pe_cycles": 1 if valid else None,
            "disjoint_from_state_partition": False,
        },
        "physical_feasibility_proven": False,
    }


def preload_feasibility(events):
    offenders = [event for event in events
                 if event["required_stall_cycles"] != 0]
    stalls = [event["required_stall_cycles"] for event in events]
    return {
        "version": 1,
        "available": True,
        "evidence_class": "derived",
        "timing_input_evidence_class": "assumed",
        "physical_feasibility_proven": False,
        "audited_preloads": len(events),
        "no_added_cycle_preloads": len(events) - len(offenders),
        "conflicting_preloads": len(offenders),
        "total_required_stall_cycles": sum(stalls),
        "max_required_stall_cycles": max(stalls, default=0),
        "events": events,
        "offending_events": offenders,
    }


def atomic_failed_record(core_id=0):
    item = record(core_id)
    item["preload_events"] = 3
    item["sa_instruction_intervals"] = 3
    item["validation_pass"] = False
    item["pe_state_cycles"] = {
        name: (6 if name in ("capacity", "unclassified") else 0)
        for name in item["pe_state_cycles"]
    }
    item["preload_bank_pe_cycles"] = 0
    item["validation"] = {
        "incomplete_intervals": 0,
        "invalid_intervals": 0,
        "unknown_weight_intervals": 0,
        "gemm_without_weight_intervals": 0,
        "footprint_mismatch_intervals": 0,
        "weight_not_ready_intervals": 0,
        "weight_evicted_before_use_intervals": 0,
        "wave_exceeds_instruction_intervals": 0,
        "compute_collisions": 0,
        "two_bank_conflicts": 1,
    }
    item["rowwise_feasibility"] = rowwise(
        mapped_pe_weight_write_events=3,
    )
    return item


def record(core_id=0):
    return {
        "version": 2,
        "core_id": core_id,
        "model": "two-bank ReGate-like analytical shadow",
        "regate_only": True,
        "combined_with_retention": False,
        "validation_pass": True,
        "array_rows": 1,
        "array_columns": 1,
        "core_cycles": 6,
        "sa_instruction_intervals": 2,
        "preload_events": 2,
        "gemm_events": 0,
        "preload_bank_pe_cycles": 1,
        "pe_state_cycles": {
            "capacity": 6,
            "on": 4,
            "on_zero_banks": 0,
            "on_one_bank": 1,
            "on_two_banks": 3,
            "w_on_one_bank": 0,
            "w_on_two_banks": 2,
            "off": 0,
            "unclassified": 0,
        },
    }


class TwoBankSummaryTest(unittest.TestCase):
    def test_validated_state_exposure_with_energy_pending(self):
        result = aggregate([record()], expected_cores=1)
        self.assertEqual(result["version"], 2)
        self.assertEqual(result["source_record_version"], 2)
        self.assertFalse(result["rowwise_feasibility"]["available"])
        self.assertEqual(
            result["aggregate"]["ideal_low_power_pe_cycle_exposure_percent"],
            100.0 * 2 / 6,
        )
        self.assertIsNone(result["aggregate"]["gross_static_saving_percent"])
        self.assertIsNone(result["aggregate"]["net_energy_saving_percent"])

    def test_explicit_factors_are_assumed_static_sensitivity(self):
        factors = {
            "on_one_bank": 1.0,
            "on_two_banks": 1.1,
            "w_on_one_bank": 0.2,
            "w_on_two_banks": 0.3,
            "off": 0.05,
        }
        result = aggregate(
            [record()], power_factors=factors,
            factor_source="hypothetical test values",
        )
        self.assertEqual(
            result["power_sensitivity"]["evidence_class"], "assumed"
        )
        self.assertAlmostEqual(
            result["aggregate"]["gross_static_saving_percent"],
            100.0 * (1.0 - (1.0 + 3.3 + 0.6) / 6.0),
        )
        self.assertIsNone(result["aggregate"]["net_energy_saving_percent"])

    def test_invalid_trace_cannot_produce_saving(self):
        invalid = record()
        invalid["validation_pass"] = False
        with self.assertRaisesRegex(ValueError, "validation failed"):
            aggregate([invalid])

    def test_partition_and_bank_split_are_checked(self):
        invalid = record()
        invalid["pe_state_cycles"]["w_on_two_banks"] = 1
        with self.assertRaisesRegex(ValueError, "partition mismatch"):
            aggregate([invalid])
        invalid = record()
        invalid["pe_state_cycles"]["on_two_banks"] = 2
        with self.assertRaisesRegex(ValueError, "ON bank split mismatch"):
            aggregate([invalid])

    def test_rejects_mixed_dimensions_and_instruction_mismatch(self):
        other = record(core_id=1)
        other["array_rows"] = 2
        other["pe_state_cycles"]["capacity"] = 12
        other["pe_state_cycles"]["off"] = 6
        with self.assertRaisesRegex(ValueError, "inconsistent SA dimensions"):
            aggregate([record(), other])
        invalid = record()
        invalid["sa_instruction_intervals"] = 3
        with self.assertRaisesRegex(ValueError, "instruction count mismatch"):
            aggregate([invalid])
        missing = record()
        del missing["sa_instruction_intervals"]
        with self.assertRaisesRegex(ValueError, "sa_instruction_intervals"):
            aggregate([missing])

    def test_rejects_missing_power_factor_or_provenance(self):
        factors = {
            "on_one_bank": 1.0,
            "on_two_banks": 1.0,
            "w_on_one_bank": 0.2,
            "w_on_two_banks": 0.3,
        }
        with self.assertRaisesRegex(ValueError, "all five"):
            aggregate([record()], power_factors=factors,
                      factor_source="test")
        factors["off"] = 0.0
        with self.assertRaisesRegex(ValueError, "source"):
            aggregate([record()], power_factors=factors)

    def test_rowwise_aggregates_separately_from_validated_state(self):
        first = record(0)
        second = record(1)
        first["rowwise_feasibility"] = rowwise()
        second["rowwise_feasibility"] = rowwise(
            necessary_conditions_pass=False,
            row_ready_after_first_mac_events=1,
            max_row_ready_lateness_cycles=2,
            max_local_first_mac_delay_cycles=2,
        )
        result = aggregate([first, second], expected_cores=2)
        diagnostic = result["rowwise_feasibility"]
        self.assertTrue(result["validation_pass"])
        self.assertTrue(diagnostic["available"])
        self.assertFalse(diagnostic["necessary_conditions_pass"])
        self.assertEqual(diagnostic["mapped_pe_weight_write_events"], 4)
        self.assertEqual(diagnostic["row_ready_after_first_mac_events"], 1)
        self.assertEqual(diagnostic["mapped_pe_peak_live_preloads"], 2)
        self.assertEqual(diagnostic["max_row_ready_lateness_cycles"], 2)
        self.assertFalse(diagnostic["physical_feasibility_proven"])
        self.assertIsNone(diagnostic["actual_stall_cycles"])

    def test_whole_bank_preload_feasibility_is_recomputed_and_aggregated(self):
        item = record()
        item["rowwise_feasibility"] = rowwise()
        events = [
            {
                "preload_index": 0,
                "instruction_index": 0,
                "weight_block_id": 1,
                "issue_cycle": 0,
                "start_cycle": 1,
                "rows": 1,
                "columns": 1,
                "fill_span_cycles": 1,
                "planned_load_begin": 0,
                "load_begin": 0,
                "bank_ready": 1,
                "first_use": 1,
                "required_stall_cycles": 0,
                "outcome": "no_added_cycle",
            },
            {
                "preload_index": 1,
                "instruction_index": 1,
                "weight_block_id": 2,
                "issue_cycle": 3,
                "start_cycle": 3,
                "rows": 1,
                "columns": 1,
                "fill_span_cycles": 1,
                "planned_load_begin": 2,
                "load_begin": 3,
                "bank_ready": 4,
                "first_use": 3,
                "required_stall_cycles": 1,
                "outcome": "conflict",
            },
        ]
        item["preload_feasibility_audit"] = preload_feasibility(events)
        result = aggregate([item], expected_cores=1)
        audit = result["preload_feasibility_audit"]
        self.assertTrue(audit["available"])
        self.assertEqual(audit["audited_preloads"], 2)
        self.assertEqual(audit["no_added_cycle_preloads"], 1)
        self.assertEqual(audit["conflicting_preloads"], 1)
        self.assertEqual(audit["total_required_stall_cycles"], 1)
        self.assertEqual(audit["max_required_stall_cycles"], 1)
        self.assertEqual(audit["offending_events"][0]["core_id"], 0)
        self.assertEqual(audit["offending_events"][0]["preload_index"], 1)

        broken = record()
        broken["rowwise_feasibility"] = rowwise()
        broken["preload_feasibility_audit"] = preload_feasibility(events)
        broken["preload_feasibility_audit"]["events"][1][
            "required_stall_cycles"] = 0
        pending = aggregate([broken], expected_cores=1)[
            "preload_feasibility_audit"]
        self.assertFalse(pending["available"])
        self.assertIn("causal readiness equation", pending["reason"])

    def test_diagnostic_only_allows_atomic_failure_without_state_saving(self):
        first = atomic_failed_record(0)
        second = atomic_failed_record(1)
        second["rowwise_feasibility"] = rowwise(
            mapped_pe_weight_write_events=3,
            mapped_pe_peak_live_preloads=1,
        )
        result = aggregate([first, second], expected_cores=2,
                           rowwise_diagnostic_only=True)
        self.assertEqual(result["version"], 2)
        self.assertIsNone(result["validation_pass"])
        self.assertTrue(result["rowwise_feasibility"]["available"])
        self.assertEqual(
            result["rowwise_feasibility"]["mapped_pe_weight_write_events"], 6
        )
        self.assertIsNone(result["aggregate"]["capacity"])
        self.assertIsNone(
            result["aggregate"]["ideal_low_power_pe_cycle_exposure_percent"]
        )
        self.assertIsNone(result["aggregate"]["gross_static_saving_percent"])
        self.assertIsNone(result["aggregate"]["net_energy_saving_percent"])
        self.assertNotIn("power_sensitivity", result)

    def test_rowwise_pending_on_missing_partial_or_incompatible_core(self):
        first = atomic_failed_record(0)
        second = atomic_failed_record(1)
        del second["rowwise_feasibility"]
        result = aggregate([first, second], expected_cores=2,
                           rowwise_diagnostic_only=True)
        self.assertFalse(result["rowwise_feasibility"]["available"])
        self.assertIn("core 1", result["rowwise_feasibility"]["reason"])
        second["rowwise_feasibility"] = rowwise(
            mapped_pe_weight_write_events=3,
            secondary_counts_complete=False,
            row_write_before_issue_events=1,
            **{name: None for name in (
                "row_ready_after_first_mac_events",
                "mapped_pe_over_two_pe_cycles",
                "mapped_pes_ever_over_two",
                "locally_blocked_pe_weight_write_events",
                "max_row_ready_lateness_cycles",
                "mapped_pe_peak_live_preloads",
                "max_local_bank_wait_cycles",
                "max_local_first_mac_delay_cycles",
            )},
        )
        result = aggregate([first, second], expected_cores=2,
                           rowwise_diagnostic_only=True)
        self.assertIn("incomplete", result["rowwise_feasibility"]["reason"])
        second["rowwise_feasibility"] = rowwise(
            mapped_pe_weight_write_events=3, row_order="descending")
        result = aggregate([first, second], expected_cores=2,
                           rowwise_diagnostic_only=True)
        self.assertIn("row_order", result["rowwise_feasibility"]["reason"])
        second["rowwise_feasibility"] = rowwise(
            mapped_pe_weight_write_events=3,
            bank_release_equation="different equation"
        )
        result = aggregate([first, second], expected_cores=2,
                           rowwise_diagnostic_only=True)
        self.assertIn("bank_release_equation",
                      result["rowwise_feasibility"]["reason"])
        second["rowwise_feasibility"] = rowwise(
            mapped_pe_weight_write_events=3,
            timing_input_evidence_class="reference"
        )
        result = aggregate([first, second], expected_cores=2,
                           rowwise_diagnostic_only=True)
        self.assertIn("evidence labels",
                      result["rowwise_feasibility"]["reason"])

    def test_diagnostic_only_rejects_non_atomic_or_non_fail_closed_trace(self):
        invalid = atomic_failed_record()
        invalid["validation"]["unknown_weight_intervals"] = 1
        with self.assertRaisesRegex(ValueError, "non-atomic"):
            aggregate([invalid], expected_cores=1,
                      rowwise_diagnostic_only=True)
        invalid = atomic_failed_record()
        invalid["pe_state_cycles"]["on"] = 1
        with self.assertRaisesRegex(ValueError, "classified state"):
            aggregate([invalid], expected_cores=1,
                      rowwise_diagnostic_only=True)
        invalid = atomic_failed_record()
        invalid["validation"]["two_bank_conflicts"] = 0
        with self.assertRaisesRegex(ValueError, "lacks an atomic"):
            aggregate([invalid], expected_cores=1,
                      rowwise_diagnostic_only=True)

    def test_diagnostic_only_requires_core_count_and_rejects_power_factors(self):
        with self.assertRaisesRegex(ValueError, "requires expected_cores"):
            aggregate([atomic_failed_record()], rowwise_diagnostic_only=True)
        with self.assertRaisesRegex(ValueError, "rejects power factors"):
            aggregate([atomic_failed_record()], expected_cores=1,
                      power_factors={"off": 0.1},
                      factor_source="test",
                      rowwise_diagnostic_only=True)

    def test_rowwise_rejects_impossible_per_core_counts(self):
        cases = (
            ("mapped_pe_weight_write_events", 0, "lower bound"),
            ("mapped_pe_weight_write_events", 4, "footprint bound"),
            ("row_ready_after_first_mac_events", 4, "late writes"),
            ("locally_blocked_pe_weight_write_events", 4,
             "blocked writes"),
            ("mapped_pes_ever_over_two", 2, "array area"),
            ("mapped_pe_peak_live_preloads", 4, "preload events"),
            ("mapped_pe_peak_live_preloads", 0, "must be positive"),
        )
        for field, value, message in cases:
            with self.subTest(field=field):
                item = atomic_failed_record()
                item["rowwise_feasibility"][field] = value
                diagnostic = aggregate(
                    [item], expected_cores=1,
                    rowwise_diagnostic_only=True,
                )["rowwise_feasibility"]
                self.assertFalse(diagnostic["available"])
                self.assertIn(message, diagnostic["reason"])

    def test_rowwise_rejects_inconsistent_completeness_and_pass_flag(self):
        item = atomic_failed_record()
        item["rowwise_feasibility"]["necessary_conditions_pass"] = False
        diagnostic = aggregate([item], expected_cores=1,
                               rowwise_diagnostic_only=True)["rowwise_feasibility"]
        self.assertIn("contradicts counters", diagnostic["reason"])
        item = atomic_failed_record()
        item["rowwise_feasibility"]["row_write_before_issue_events"] = 1
        diagnostic = aggregate([item], expected_cores=1,
                               rowwise_diagnostic_only=True)["rowwise_feasibility"]
        self.assertIn("secondary_counts_complete", diagnostic["reason"])
        item["rowwise_feasibility"]["secondary_counts_complete"] = False
        diagnostic = aggregate([item], expected_cores=1,
                               rowwise_diagnostic_only=True)["rowwise_feasibility"]
        self.assertIn("must be null", diagnostic["reason"])
        for field in (
            "row_ready_after_first_mac_events",
            "mapped_pe_over_two_pe_cycles",
            "mapped_pes_ever_over_two",
            "locally_blocked_pe_weight_write_events",
            "max_row_ready_lateness_cycles",
            "mapped_pe_peak_live_preloads",
            "max_local_bank_wait_cycles",
            "max_local_first_mac_delay_cycles",
        ):
            item["rowwise_feasibility"][field] = None
        diagnostic = aggregate([item], expected_cores=1,
                               rowwise_diagnostic_only=True)["rowwise_feasibility"]
        self.assertIn("incomplete", diagnostic["reason"])

    def test_atomic_failure_bounds_and_contiguous_core_ids(self):
        item = atomic_failed_record()
        item["validation"]["two_bank_conflicts"] = 2
        with self.assertRaisesRegex(ValueError, "preload bound"):
            aggregate([item], expected_cores=1,
                      rowwise_diagnostic_only=True)
        item = atomic_failed_record()
        item["validation"]["weight_evicted_before_use_intervals"] = 4
        with self.assertRaisesRegex(ValueError, "instruction count"):
            aggregate([item], expected_cores=1,
                      rowwise_diagnostic_only=True)
        with self.assertRaisesRegex(ValueError, "core IDs"):
            aggregate([atomic_failed_record(1)], expected_cores=1,
                      rowwise_diagnostic_only=True)

    def test_rowwise_overlap_and_event_maxima_must_agree(self):
        cases = (
            ({"mapped_pes_ever_over_two": 1}, "excess/peak"),
            ({"mapped_pe_peak_live_preloads": 3}, "excess/peak"),
            ({"mapped_pe_over_two_pe_cycles": 1,
              "necessary_conditions_pass": False}, "excess/peak"),
            ({"max_row_ready_lateness_cycles": 1}, "late-event"),
            ({"row_ready_after_first_mac_events": 1,
              "necessary_conditions_pass": False}, "late-event"),
            ({"max_local_bank_wait_cycles": 1}, "blocked-event"),
            ({"locally_blocked_pe_weight_write_events": 1},
             "blocked-event"),
            ({"max_local_first_mac_delay_cycles": 1},
             "unexplained local first-MAC"),
        )
        for changes, message in cases:
            with self.subTest(changes=changes):
                item = atomic_failed_record()
                item["rowwise_feasibility"].update(changes)
                diagnostic = aggregate(
                    [item], expected_cores=1,
                    rowwise_diagnostic_only=True,
                )["rowwise_feasibility"]
                self.assertFalse(diagnostic["available"])
                self.assertIn(message, diagnostic["reason"])

    def test_affected_pes_need_three_writes_each(self):
        item = atomic_failed_record()
        item["array_rows"] = 2
        item["pe_state_cycles"]["capacity"] = 12
        item["pe_state_cycles"]["unclassified"] = 12
        item["rowwise_feasibility"].update({
            "mapped_pe_over_two_pe_cycles": 2,
            "mapped_pes_ever_over_two": 2,
            "mapped_pe_peak_live_preloads": 3,
            "necessary_conditions_pass": False,
        })
        diagnostic = aggregate([item], expected_cores=1,
                               rowwise_diagnostic_only=True)["rowwise_feasibility"]
        self.assertIn("mapped-write bound", diagnostic["reason"])

    def test_rowwise_state_mode_aggregates_validated_theoretical_states(self):
        first = atomic_failed_record(0)
        second = atomic_failed_record(1)
        first["rowwise_state_accounting"] = rowwise_state()
        second["rowwise_state_accounting"] = rowwise_state()
        result = aggregate([first, second], expected_cores=2,
                           rowwise_state_accounting=True)
        self.assertEqual(result["version"], 3)
        self.assertTrue(result["validation_pass"])
        states = result["rowwise_state_accounting"]
        self.assertEqual(states["pe_state_cycles"]["capacity"], 12)
        self.assertEqual(states["pe_state_cycles"]["on"], 8)
        self.assertEqual(states["pe_state_cycles"]["w_on_two_banks"], 4)
        self.assertEqual(states["pe_state_cycles"]["unclassified"], 0)
        self.assertEqual(states["preload_write_exposure"]["pe_cycles"], 4)
        self.assertIsNone(states["gross_static_saving_percent"])
        self.assertIsNone(states["net_energy_saving_percent"])

    def test_rowwise_state_mode_fails_closed_across_cores(self):
        first = atomic_failed_record(0)
        second = atomic_failed_record(1)
        first["rowwise_state_accounting"] = rowwise_state()
        second["rowwise_feasibility"] = rowwise(
            mapped_pe_weight_write_events=3,
            necessary_conditions_pass=False,
            row_ready_after_first_mac_events=1,
            max_row_ready_lateness_cycles=1,
            max_local_first_mac_delay_cycles=1,
        )
        second["rowwise_state_accounting"] = rowwise_state(False)
        result = aggregate([first, second], expected_cores=2,
                           rowwise_state_accounting=True)
        states = result["rowwise_state_accounting"]
        self.assertFalse(states["validation_pass"])
        self.assertEqual(states["pe_state_cycles"]["capacity"], 12)
        self.assertEqual(states["pe_state_cycles"]["unclassified"], 12)
        self.assertEqual(states["pe_state_cycles"]["on"], 0)
        self.assertIsNone(states["preload_write_exposure"]["pe_cycles"])

    def test_rowwise_state_mode_rejects_incompatible_or_contradictory_inputs(self):
        first = atomic_failed_record(0)
        second = atomic_failed_record(1)
        first["rowwise_state_accounting"] = rowwise_state()
        second["rowwise_state_accounting"] = rowwise_state()
        second["rowwise_feasibility"] = rowwise(
            mapped_pe_weight_write_events=3, row_order="descending")
        result = aggregate([first, second], expected_cores=2,
                           rowwise_state_accounting=True)
        self.assertFalse(result["rowwise_state_accounting"]["available"])
        self.assertIn("rowwise feasibility aggregate is unavailable",
                      result["rowwise_state_accounting"]["reason"])

        item = atomic_failed_record()
        item["rowwise_state_accounting"] = rowwise_state()
        item["rowwise_state_accounting"]["required_checks"][
            "rowwise_timing_and_two_bank_checks"] = False
        item["rowwise_state_accounting"]["validation_pass"] = False
        item["rowwise_state_accounting"]["required_checks"][
            "state_partition"] = True
        item["rowwise_state_accounting"]["pe_state_cycles"] = {
            name: (6 if name in ("capacity", "unclassified") else 0)
            for name in item["rowwise_state_accounting"]["pe_state_cycles"]
        }
        item["rowwise_state_accounting"]["preload_write_exposure"]["pe_cycles"] = 0
        item["rowwise_state_accounting"]["preload_write_exposure"]["while_on_pe_cycles"] = 0
        result = aggregate([item], expected_cores=1,
                           rowwise_state_accounting=True)
        self.assertFalse(result["rowwise_state_accounting"]["available"])
        self.assertIn("contradicts feasibility record",
                      result["rowwise_state_accounting"]["reason"])

        item = atomic_failed_record()
        item["rowwise_state_accounting"] = rowwise_state()
        item["rowwise_state_accounting"]["preload_write_exposure"][
            "evidence_class"] = "measured"
        result = aggregate([item], expected_cores=1,
                           rowwise_state_accounting=True)
        self.assertFalse(result["rowwise_state_accounting"]["available"])
        self.assertIn("preload-write evidence labels",
                      result["rowwise_state_accounting"]["reason"])

    def test_rowwise_state_empty_trace_is_all_off(self):
        item = record()
        item.update({
            "sa_instruction_intervals": 0,
            "preload_events": 0,
            "gemm_events": 0,
            "preload_bank_pe_cycles": 0,
            "rowwise_feasibility": rowwise(
                mapped_pe_weight_write_events=0,
                mapped_pe_peak_live_preloads=0,
            ),
            "rowwise_state_accounting": rowwise_state(
                True, on=0, on_preload_wave=0, on_one_bank=0,
                on_two_banks=0, w_on_two_banks=0, off=6,
            ),
        })
        item["rowwise_state_accounting"]["preload_write_exposure"]["pe_cycles"] = 0
        item["rowwise_state_accounting"]["preload_write_exposure"]["while_on_pe_cycles"] = 0
        result = aggregate([item], expected_cores=1,
                           rowwise_state_accounting=True)
        states = result["rowwise_state_accounting"]["pe_state_cycles"]
        self.assertEqual(states["off"], 6)
        self.assertEqual(states["unclassified"], 0)


if __name__ == "__main__":
    unittest.main()
