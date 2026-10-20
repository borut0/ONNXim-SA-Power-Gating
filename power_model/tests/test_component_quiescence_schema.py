import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUMMARIZER = ROOT / "power_model" / "summarize_component_characterization.py"


class ComponentQuiescenceSchemaTest(unittest.TestCase):
    def test_v1_demand_idle_log_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "old.log"
            log.write_text(
                'COMPONENT_NOC {"version":1,"demand_idle_cycles":10}\n'
            )
            result = subprocess.run(
                ["python3", str(SUMMARIZER), str(log)],
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("v1 demand-idle is rejected", result.stderr)

    def test_current_builder_has_no_demand_idle_fallback(self):
        source = (
            ROOT / "power_model" / "workload_suite" /
            "build_component_selection_evidence.py"
        ).read_text()
        self.assertNotIn("demand_idle", source)
        self.assertIn("modeled_quiescent_percent", source)
        self.assertIn("schema-v2 modeled quiescence required", source)


if __name__ == "__main__":
    unittest.main()
