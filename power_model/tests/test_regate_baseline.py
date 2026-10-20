import copy
import unittest

from power_model.event_driven_sa import replay_event_driven
from power_model.regate_baseline import replay_regate_baseline
from power_model.summarize_regate_baseline import compare
from power_model.tests.test_event_driven_sa import event, source


class RegateBaselineTest(unittest.TestCase):
    def test_single_generation_state_partition_and_delayed_mask(self):
        item = source()
        item["array_rows"] = item["array_columns"] = 2
        item["onnxim_horizon_cycles"] = 8
        item["events"] = [event(1, 0, 0, "preload_boundary", 0, 2,
                                vectors=1, rows=1, columns=1)]
        item["events"][0]["onnxim_compute_finish"] = 8
        proposed = replay_event_driven(item, beta=0.2)
        optimistic = replay_regate_baseline(proposed, mask_policy="optimistic",
                                            beta=0.2)
        delayed = replay_regate_baseline(proposed, mask_policy="delayed",
                                         beta=0.2)
        self.assertEqual(optimistic["literal_one_bank"]["status"], "feasible")
        self.assertEqual(optimistic["literal_one_bank"]["pe_cycles"],
                         {"on": 2, "w_on": 0, "off": 30, "capacity": 32})
        self.assertEqual(delayed["literal_one_bank"]["pe_cycles"],
                         {"on": 2, "w_on": 1, "off": 29, "capacity": 32})
        self.assertEqual(delayed["preload_candidate_extra_weight_cycles"], 1)
        for result in (optimistic, delayed):
            states = result["matched_two_bank"]["weight_bank_cycles"]
            self.assertEqual(states["on"] + states["w_on"] + states["off"],
                             states["capacity"])

    def test_two_simultaneous_generations_fail_literal_case_only(self):
        proposed = replay_event_driven(source(), beta=0.2)
        result = replay_regate_baseline(proposed, beta=0.2)
        self.assertGreater(result["literal_one_bank"]["conflict_count"], 0)
        self.assertIsNone(result["literal_one_bank"]["energy"])
        self.assertIsNotNone(result["matched_two_bank"][
            "normalized_static_energy_ratio_vs_two_bank_always_on"])

    def test_comparison_does_not_mutate_proposed_replay(self):
        item = source()
        original = copy.deepcopy(item)
        result = compare([item], beta=0.2)
        self.assertEqual(item, original)
        self.assertTrue(result["proposed_policy_unchanged"])
        self.assertIsNotNone(result[
            "additional_static_saving_percent_vs_matched_two_bank_regate_style"])
        self.assertIsNone(result[
            "proposed_additional_static_saving_percent_vs_literal_one_bank_regate"])

    def test_no_sa_work_is_off_in_both_policies(self):
        item = source()
        item["events"] = []
        result = replay_regate_baseline(replay_event_driven(item), beta=0.2)
        self.assertEqual(result["literal_one_bank"]["pe_cycles"]["off"],
                         2 * 2 * 18)
        self.assertEqual(result["matched_two_bank"]["weight_bank_cycles"][
            "off"], 2 * 2 * 2 * 18)

    def test_zero_matched_baseline_energy_has_undefined_incremental_ratio(self):
        item = source()
        item["events"] = []
        result = compare([item], beta=1)
        self.assertEqual(result["matched_two_bank_regate_style_energy_ratio"],
                         0)
        self.assertIsNone(result[
            "additional_static_saving_percent_vs_matched_two_bank_regate_style"])

    def test_one_bank_narrow_overwrite_does_not_conflict_with_high_column(self):
        item = source()
        item["events"] = [
            event(1, 0, 0, "preload_boundary", 0, 1, vectors=1,
                  rows=1, columns=2),
            event(2, 1, 1, "preload_boundary", 1, 3, vectors=1,
                  rows=1, columns=1),
        ]
        item["onnxim_horizon_cycles"] = 9
        for entry in item["events"]:
            entry["onnxim_compute_finish"] = entry["onnxim_compute_start"] + 6
        proposed = replay_event_driven(item)
        baseline = replay_regate_baseline(proposed,
                                          mask_policy="optimistic")
        self.assertEqual(baseline["literal_one_bank"]["conflict_count"], 0)

    def test_delayed_mask_fails_closed_when_prior_high_column_still_live(self):
        item = source()
        item["events"] = [
            event(1, 0, 0, "preload_boundary", 0, 1, vectors=1,
                  rows=1, columns=2),
            event(2, 1, 0, "preload_boundary", 1, 3, vectors=1,
                  rows=1, columns=1),
        ]
        item["onnxim_horizon_cycles"] = 12
        for entry in item["events"]:
            entry["onnxim_compute_finish"] = entry["onnxim_compute_start"] + 6
        replay = replay_event_driven(item)
        delayed = replay_regate_baseline(replay, mask_policy="delayed",
                                         beta=0.2)
        self.assertEqual(delayed["matched_two_bank"]["status"],
                         "unresolved_padded_column_overlap")
        self.assertIsNone(delayed["matched_two_bank"]["weight_bank_cycles"])
        self.assertIsNone(delayed["matched_two_bank"][
            "normalized_static_energy_ratio_vs_two_bank_always_on"])
        comparison = compare([item], beta=0.2)
        self.assertIsNone(comparison[
            "additional_static_saving_percent_vs_matched_two_bank_regate_style"])


if __name__ == "__main__":
    unittest.main()
