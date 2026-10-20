import unittest

from power_model.summarize_regate_event_driven import aggregate


def record(core_id, events):
    return {
        "core_id": core_id,
        "model": "event-driven unified PE and two-bank analytical source",
        "legacy_whole_sa_predictor_used": False,
        "validation_pass": True,
        "array_rows": 2,
        "array_columns": 2,
        "weight_bytes": 2,
        "onnxim_horizon_cycles": 6,
        "events": events,
    }


class SummarizeRegateEventDrivenTest(unittest.TestCase):
    def test_stalled_core_extends_both_cores_accounting_horizon(self):
        event = {
            "preload_instance": 1,
            "tile_stream_id": 0,
            "assumed_bank": 0,
            "analytical_role": "preload_boundary",
            "analytical_preload_boundary_assumed": False,
            "tile_m": 1,
            "tile_k": 1,
            "tile_n": 1,
            "issue_cycle": 0,
            "onnxim_compute_start": 0,
            "onnxim_compute_finish": 6,
        }
        result = aggregate([record(0, [event]), record(1, [])],
                           expected_cores=2, beta=0.1)
        self.assertEqual(result["onnxim_shared_horizon_cycles"], 6)
        self.assertEqual(result["analytical_shared_horizon_cycles"], 7)
        self.assertEqual(result["analytical_sa_tail_extension_cycles"], 1)
        self.assertEqual(result["pe_logic_cycles"]["capacity"], 2 * 4 * 7)
        self.assertEqual(result["weight_bank_pe_cycles"]["capacity"],
                         2 * 2 * 4 * 7)
        self.assertTrue(all(item["managed_horizon_cycles"] == 7
                            for item in result["per_core"]))
        self.assertIsNone(result[
            "whole_sa_gross_static_opportunity_percent_on_native_timeline"])
        self.assertIsNotNone(result[
            "whole_sa_gross_static_opportunity_percent_on_managed_timeline"])
        self.assertIsNone(result["actual_end_to_end_penalty_percent"])


if __name__ == "__main__":
    unittest.main()
