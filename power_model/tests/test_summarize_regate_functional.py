import unittest

from power_model.summarize_regate_functional import aggregate


def record(core_id=0, validation_pass=True):
    return {
        "version": 1,
        "model": "ReGate logical-control shadow approximation",
        "core_id": core_id,
        "array_rows": 4,
        "array_columns": 4,
        "regate_only": True,
        "combined_with_retention": False,
        "validation_pass": validation_pass,
        "pe_state_cycles": {
            "capacity": 100,
            "on": 20,
            "w_on": 30,
            "off": 40,
            "unclassified": 10,
            "partition_exact_under_model": True,
        },
    }


class ReGateFunctionalSummaryTest(unittest.TestCase):
    provenance = {
        "evidence_class": "reference",
        "locator": "test fixture",
    }

    def test_reference_factor_gives_gross_static_saving(self):
        result = aggregate(
            [record()],
            0.03,
            0.03,
            self.provenance,
            expected_cores=1,
        )
        self.assertAlmostEqual(
            result["aggregate"]["normalized_regate_gross_static_energy"],
            32.1,
        )
        self.assertAlmostEqual(
            result["aggregate"]["gross_static_saving_percent"],
            67.9,
        )
        self.assertEqual(result["net_energy_saving_status"], "pending")

    def test_failed_shadow_validation_is_rejected(self):
        with self.assertRaises(ValueError):
            aggregate(
                [record(validation_pass=False)],
                0.03,
                0.03,
                self.provenance,
            )

    def test_partition_mismatch_is_rejected(self):
        item = record()
        item["pe_state_cycles"]["off"] = 39
        with self.assertRaises(ValueError):
            aggregate([item], 0.03, 0.03, self.provenance)

    def test_retention_combination_is_rejected(self):
        item = record()
        item["combined_with_retention"] = True
        with self.assertRaises(ValueError):
            aggregate([item], 0.03, 0.03, self.provenance)

    def test_fractional_cycle_count_is_rejected(self):
        item = record()
        item["pe_state_cycles"]["on"] = 20.5
        with self.assertRaises(ValueError):
            aggregate([item], 0.03, 0.03, self.provenance)

    def test_boolean_cycle_count_is_rejected(self):
        item = record()
        item["pe_state_cycles"]["on"] = True
        with self.assertRaises(ValueError):
            aggregate([item], 0.03, 0.03, self.provenance)

    def test_wrong_schema_version_is_rejected(self):
        item = record()
        item["version"] = 2
        with self.assertRaises(ValueError):
            aggregate([item], 0.03, 0.03, self.provenance)

    def test_missing_parameter_provenance_is_rejected(self):
        with self.assertRaises(ValueError):
            aggregate([record()], 0.03, 0.03, {})


if __name__ == "__main__":
    unittest.main()
