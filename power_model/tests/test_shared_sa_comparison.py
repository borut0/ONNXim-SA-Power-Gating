import copy
import unittest

from power_model.shared_sa_comparison import (
    _fixed_sa_order_serial_scenario, aggregate_run, compare_runs)


def event(opcode="GEMM_PRELOAD", *, start=1, write=0, vectors=2,
          rows=2, columns=2, sequence=1, tile=1, bank=0,
          start_before=None, policy_delay=None, issue=0):
    start_before = start if start_before is None else start_before
    policy_delay = (start - start_before if policy_delay is None else
                    policy_delay)
    return {
        "tile_id": tile, "preload_sequence": sequence,
        "opcode": opcode, "bank": bank, "mapped_rows": rows,
        "mapped_columns": columns, "vectors": vectors,
        "weight_block_id": 7, "weight_block_known": True,
        "issue_cycle": issue,
        "row_write_start": write if opcode == "GEMM_PRELOAD" else None,
        "compute_start": start,
        "compute_start_before_power_policy": start_before,
        "modeled_power_policy_delay": policy_delay,
        "natural_wave_finish": start + rows + columns + vectors - 2,
        "compute_finish": start + rows + columns + vectors - 2,
    }


def first_gated_wave(**kwargs):
    return event(start=2, start_before=1, **kwargs)


def record(policy="always_on", *, horizon=10, events=None, wake_count=None,
           lockout_cycles=0, lockout_events=0):
    events = ([event()] if policy == "always_on" else [first_gated_wave()]) \
        if events is None else events
    if wake_count is None:
        wake_count = int(policy != "always_on" and bool(events))
    return {
        "schema_version": 2, "validation_pass": True,
        "core_id": 0, "policy": policy, "core_cycles": horizon,
        "array_rows": 2, "array_columns": 2,
        "weight_bytes_per_bank": 8,
        "two_weight_banks": True,
        "bank_ds_wake_hidden_assumption": True,
        "pe_break_even_cycles": 47,
        "ours_minimum_off_interval_cycles": 48,
        "sa_instructions": len(events),
        "preloads": sum(e["opcode"] == "GEMM_PRELOAD" for e in events),
        "modeled_preload_cycles": 2,
        "modeled_pe_wake_delay_cycles": wake_count,
        "modeled_bank_wait_cycles": 0,
        "modeled_min_off_lockout_cycles": lockout_cycles,
        "modeled_min_off_lockout_events": lockout_events,
        "modeled_sa_completion_cycles": 5,
        "events": copy.deepcopy(events),
    }


class SharedSaComparisonTest(unittest.TestCase):
    def test_matched_three_run_static_and_performance(self):
        reference = record()
        regate = record("regate", horizon=11)
        ours = record("ours", horizon=12)
        result = compare_runs([reference], [regate], [ours])
        self.assertAlmostEqual(
            result["observed_regate_horizon_delta_percent"], 10)
        self.assertAlmostEqual(
            result["observed_ours_horizon_delta_percent"], 20)
        self.assertEqual(result["reference_horizon_cycles"], 10)
        self.assertEqual(result["ours_horizon_cycles"], 12)
        self.assertTrue(result["matched_sa_work_validation_pass"])
        self.assertEqual(result["beta_two_bank_static_power_share_sweep"],
                         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        self.assertEqual(len(result["beta_sensitivity"]), 6)
        beta_half = result["beta_sensitivity"][4]
        self.assertGreater(beta_half[
            "ours_net_sa_energy_saving_percent_vs_always_on"], 0)
        self.assertGreater(beta_half[
            "regate_net_sa_energy_saving_percent_vs_always_on"], 0)

    def test_no_sa_work_accounts_whole_array_off(self):
        ours = aggregate_run([record("ours", events=[])], beta=0.5)
        self.assertEqual(ours["pe_logic_cycles"]["off"], 40)
        self.assertEqual(ours["our_weight_bank_cycles"]["off"], 80)
        self.assertEqual(ours["sa_events"], 0)

    def test_banks_retained_between_same_generation_consumers(self):
        events = [first_gated_wave(vectors=2),
                  event("GEMM", start=52, start_before=51, vectors=2)]
        ours = aggregate_run([record("ours", horizon=60, events=events,
                                     wake_count=2)],
                             beta=0.5)
        self.assertGreater(ours["our_weight_bank_cycles"]["ds"], 0)
        self.assertEqual(
            sum(ours["our_weight_bank_cycles"][mode]
                for mode in ("on", "ds", "off")),
            ours["our_weight_bank_cycles"]["capacity"])

    def test_delayed_mask_has_additional_candidate_power(self):
        events = [first_gated_wave(rows=1, columns=1, vectors=1)]
        delayed = aggregate_run(
            [record("regate", events=events)], beta=0.5)
        optimistic = aggregate_run(
            [record("regate", events=events)], beta=0.5,
            mask_policy="optimistic")
        self.assertEqual(delayed["regate_weight_bank_cycles"]["on"],
                         optimistic["regate_weight_bank_cycles"]["on"] + 1)
        self.assertEqual(delayed["regate_weight_bank_cycles"]["w_on"],
                         optimistic["regate_weight_bank_cycles"]["w_on"])

    def test_regate_reports_explicit_on_w_on_off_partition(self):
        events = [first_gated_wave(vectors=2),
                  event("GEMM", start=11, start_before=10, vectors=2)]
        regate = aggregate_run(
            [record("regate", horizon=20, events=events, wake_count=2)],
            beta=0.5)
        states = regate["regate_weight_bank_cycles"]
        self.assertGreater(states["on"], 0)
        self.assertGreater(states["w_on"], 0)
        self.assertGreater(states["off"], 0)
        self.assertEqual(states["on"] + states["w_on"] + states["off"],
                         states["capacity"])

    def test_regate_keeps_valid_mapped_weights_after_last_consumer(self):
        states = aggregate_run([record("regate", horizon=10)],
                               beta=0.2)["regate_weight_bank_cycles"]
        # 2 columns in row 0 are loaded at cycle 0; row 1 at cycle 1.
        # Both rows remain resident through cycle 9, despite no later GEMM.
        self.assertEqual(states, {"on": 16, "w_on": 22,
                                  "off": 42, "capacity": 80})

    def test_regate_same_cell_overwrite_keeps_continuous_residency(self):
        events = [first_gated_wave(rows=1, columns=1, vectors=1),
                  event(rows=1, columns=1, vectors=1, write=5, start=7,
                        start_before=6, issue=5, tile=2, sequence=2)]
        states = aggregate_run(
            [record("regate", horizon=10, events=events, wake_count=2)],
            beta=0.2,
            mask_policy="optimistic")["regate_weight_bank_cycles"]
        self.assertEqual(states, {"on": 6, "w_on": 4,
                                  "off": 70, "capacity": 80})

    def test_regate_unwritten_cells_survive_smaller_preload(self):
        events = [first_gated_wave(rows=2, columns=2, vectors=1),
                  event(rows=1, columns=1, vectors=1, write=5, start=7,
                        start_before=6, issue=5, tile=2, sequence=2)]
        states = aggregate_run(
            [record("regate", horizon=10, events=events, wake_count=2)],
            beta=0.2,
            mask_policy="optimistic")["regate_weight_bank_cycles"]
        self.assertEqual(states["on"] + states["w_on"], 38)
        self.assertEqual(states["off"], 42)

    def test_fixed_sa_order_serial_scenario_uses_reference_wave_gaps(self):
        events = [event(vectors=2),
                  event("GEMM", start=10, issue=10, vectors=2)]
        scenario = _fixed_sa_order_serial_scenario(
            [record("always_on", horizon=20, events=events)])
        self.assertFalse(scenario["exact_end_to_end_causal_replay"])
        self.assertEqual(scenario["regate_horizon_cycles"], 22)
        self.assertEqual(scenario["ours_horizon_cycles"], 62)
        self.assertEqual(scenario["per_core"][0][
            "ours_serial_policy_delay_cycles"], 42)
        self.assertEqual(scenario["per_core"][0][
            "regate_explicit_wake_events"], 2)
        self.assertEqual(scenario["per_core"][0][
            "ours_explicit_wake_events"], 2)

    def test_shorter_independent_run_is_diagnostic_not_policy_speedup(self):
        result = compare_runs([record(horizon=10)],
                              [record("regate", horizon=9)],
                              [record("ours", horizon=9)])
        self.assertAlmostEqual(result["observed_ours_horizon_delta_percent"],
                               -10)
        self.assertAlmostEqual(result["fixed_sa_order_serial_scenario"][
            "ours_scenario_performance_penalty_percent"], 10)

    def test_invalid_preload_causality_fails_closed(self):
        broken = first_gated_wave(write=2)
        with self.assertRaisesRegex(ValueError, "preload row write is not causal"):
            aggregate_run([record("ours", events=[broken])], beta=0.5)

    def test_missing_event_work_fails_comparison(self):
        with self.assertRaisesRegex(ValueError, "does not match reference"):
            compare_runs([record()], [record("regate")],
                         [record("ours", events=[])])

    def test_physical_bank_choice_may_differ_by_policy(self):
        second_bank = first_gated_wave(bank=1, tile=900, sequence=3)
        changed_core = record("ours", events=[second_bank])
        changed_core["core_id"] = 5
        result = compare_runs([record()], [record("regate")],
                              [changed_core])
        self.assertTrue(result["matched_sa_work_validation_pass"])

    def test_duplicate_core_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "duplicate core ID"):
            aggregate_run([record(), record()], beta=0.5)

    def test_assumption_mismatch_fails_closed(self):
        bad = record("ours")
        bad["bank_ds_wake_hidden_assumption"] = False
        with self.assertRaisesRegex(ValueError, "incompatible functional"):
            aggregate_run([bad], beta=0.5)

    def test_regate_charges_bet_for_short_episode_without_clamping(self):
        events = [first_gated_wave(vectors=2),
                  event("GEMM", start=11, start_before=10, vectors=2)]
        regate = aggregate_run(
            [record("regate", horizon=20, events=events, wake_count=2)],
            beta=0.5)
        self.assertEqual(regate["pe_low_power_episodes"], 4)
        self.assertGreater(
            regate["normalized_net_sa_energy_in_always_on_core_cycles"],
            regate["normalized_gross_static_energy_in_always_on_core_cycles"])

    def test_touching_waves_are_coalesced_without_transition_charge(self):
        events = [first_gated_wave(vectors=2),
                  event("GEMM", start=4, vectors=2)]
        regate = aggregate_run(
            [record("regate", horizon=10, events=events)], beta=0.5)
        self.assertEqual(regate["pe_low_power_episodes"], 0)
        self.assertEqual(
            regate["pe_transition_energy_equivalent_always_on_core_cycles"],
            0)

    def test_ours_requires_48_cycle_noncontiguous_wave_gap(self):
        short = [first_gated_wave(vectors=2),
                 event("GEMM", start=51, start_before=50, vectors=2)]
        with self.assertRaisesRegex(ValueError, "ours lockout"):
            aggregate_run([record("ours", horizon=60, events=short)],
                          beta=0.5)
        boundary = [first_gated_wave(vectors=2),
                    event("GEMM", start=52, start_before=51, vectors=2)]
        ours = aggregate_run(
            [record("ours", horizon=60, events=boundary, wake_count=2)],
            beta=0.5)
        self.assertEqual(ours["pe_low_power_episodes"], 4)

    def test_event_cannot_finish_past_its_own_core_horizon(self):
        long_core = record("regate", horizon=20)
        short_core = record("regate", horizon=4)
        short_core["core_id"] = 1
        with self.assertRaisesRegex(ValueError, "invalid or unordered"):
            aggregate_run([long_core, short_core], beta=0.5)

    def test_event_power_delay_must_match_start_difference(self):
        malformed = event(start=2, start_before=1, policy_delay=0)
        with self.assertRaisesRegex(ValueError, "power-policy delay"):
            aggregate_run([record("regate", events=[malformed])], beta=0.5)

    def test_regate_rejects_non_unit_power_delay(self):
        malformed = event(start=3, start_before=1, policy_delay=2)
        with self.assertRaisesRegex(ValueError, "ReGate power-policy delay"):
            aggregate_run([record("regate", events=[malformed])], beta=0.5)

    def test_preload_delayed_first_wave_still_pays_explicit_wake(self):
        first = event(start=11, start_before=10, issue=0)
        for policy in ("regate", "ours"):
            with self.subTest(policy=policy):
                result = aggregate_run(
                    [record(policy, horizon=20, events=[first])], beta=0.5)
                self.assertEqual(result["simulator_counters"][
                    "modeled_pe_wake_delay_cycles"], 1)
                missing_wake = event(start=10, start_before=10, issue=0)
                with self.assertRaisesRegex(ValueError, "delay"):
                    aggregate_run([record(policy, horizon=20,
                                          events=[missing_wake])], beta=0.5)

    def test_ours_wake_and_lockout_boundary_gaps(self):
        for gap in (1, 47, 48, 60):
            with self.subTest(gap=gap):
                before = 4 + gap  # first wave starts at 2 and has 2 vectors
                delay = max(1, 48 - gap)
                lockout = delay - 1
                events = [first_gated_wave(),
                          event("GEMM", start=before + delay,
                                start_before=before, issue=before)]
                result = aggregate_run(
                    [record("ours", horizon=80, events=events,
                            wake_count=2, lockout_cycles=lockout,
                            lockout_events=int(lockout > 0))], beta=0.5)
                counters = result["simulator_counters"]
                self.assertEqual(counters["modeled_pe_wake_delay_cycles"], 2)
                self.assertEqual(counters["modeled_min_off_lockout_cycles"],
                                 lockout)

    def test_policy_delay_counters_must_reconcile(self):
        delayed = first_gated_wave()
        malformed = record("regate", events=[delayed])
        malformed["modeled_pe_wake_delay_cycles"] = 0
        with self.assertRaisesRegex(ValueError, "wake-delay counter"):
            aggregate_run([malformed], beta=0.5)

    def test_ours_lockout_delay_and_counters_reconcile(self):
        events = [
            first_gated_wave(vectors=2),
            event("GEMM", start=52, start_before=10, policy_delay=42,
                  vectors=2),
        ]
        valid = record("ours", horizon=60, events=events, wake_count=2)
        valid["modeled_min_off_lockout_cycles"] = 41
        valid["modeled_min_off_lockout_events"] = 1
        ours = aggregate_run([valid], beta=0.5)
        self.assertEqual(
            ours["simulator_counters"]["modeled_min_off_lockout_cycles"],
            41)
        malformed = copy.deepcopy(valid)
        malformed["modeled_min_off_lockout_events"] = 0
        with self.assertRaisesRegex(ValueError, "lockout counters"):
            aggregate_run([malformed], beta=0.5)


if __name__ == "__main__":
    unittest.main()
