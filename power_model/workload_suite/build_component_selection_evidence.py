#!/usr/bin/env python3
"""Build compact, reproducible evidence for selecting the systolic array."""

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_DIR = ROOT / "power_model" / "workload_suite" / "summaries"
OUT_DIR = ROOT / "power_model" / "workload_suite" / "evidence"

WORKLOADS = [
    "alexnet", "resnet18", "resnet34", "resnet50", "resnet101",
    "vgg11", "vgg16", "vgg19", "wide_resnet50_2", "gpt2",
    "opt125m_12l_full", "opt_1_3b_final", "opt_2_7b_final",
    "opt_6_7b_final", "opt_66b_final", "llama3_8b_32_8_final",
]


def summary_path(name):
    candidates = [SUMMARY_DIR / f"{name}.final.json", SUMMARY_DIR / f"{name}.json"]
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit(f"missing canonical summary for {name}")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    sources = []
    totals = {
        "core_cycles": 0, "sa_idle_cycles": 0,
        "fifo_empty_cycles": 0, "icnt_cycles": 0, "noc_quiescent_cycles": 0,
        "dram_cycles": 0, "dram_quiescent_cycles": 0,
        "spad_idle_cycles": 0,
    }

    for name in WORKLOADS:
        path = summary_path(name)
        data = json.loads(path.read_text())
        if data.get("validation_pass") is not True:
            raise SystemExit(f"component validation did not pass: {path}")
        if data["noc"].get("version") != 2 or data["dram"].get("version") != 2:
            raise SystemExit(
                f"schema-v2 modeled quiescence required; rerun workload: {path}")
        if not data["noc"].get("modeled_quiescence_available") or not data["dram"].get("modeled_quiescence_available"):
            raise SystemExit(f"modeled quiescence unavailable: {path}")
        core = data["aggregate"]["core_components"]
        dram = data["dram"]
        row = {
            "schema_version": 2,
            "workload": name,
            "summary": str(path.relative_to(ROOT)),
            "validation_pass": True,
            "core_cycles": data["aggregate"]["core_cycles"],
            "sa_idle_cycles": core["sa_idle_cycles"],
            "sa_idle_percent": core["sa_idle_percent"],
            "core_request_fifo_empty_percent": core["core_request_fifo_empty_percent"],
            "noc_modeled_quiescent_percent": data["noc"]["modeled_quiescent_percent"],
            "dram_modeled_quiescent_percent": dram["modeled_quiescent_percent"],
            "all_sram_devices_modeled_idle_percent":
                core["all_sram_devices_modeled_idle_percent"],
            "dram_transaction_bandwidth_utilization_percent": dram["transaction_bandwidth_utilization_percent"],
        }
        rows.append(row)
        totals["core_cycles"] += row["core_cycles"]
        totals["sa_idle_cycles"] += row["sa_idle_cycles"]
        totals["fifo_empty_cycles"] += core["core_request_fifo_empty_cycles"]
        totals["icnt_cycles"] += data["system"]["icnt_cycles"]
        totals["noc_quiescent_cycles"] += data["noc"]["modeled_quiescent_cycles"]
        totals["dram_cycles"] += dram["dram_cycles"]
        totals["dram_quiescent_cycles"] += dram["modeled_quiescent_cycles"]
        totals["spad_idle_cycles"] += core["all_sram_devices_modeled_idle_cycles"]
        sources.append({
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
        })

    csv_path = OUT_DIR / "component_selection_v2.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    weighted = {
        "workload_count": len(rows),
        **totals,
        "sa_idle_percent": 100 * totals["sa_idle_cycles"] / totals["core_cycles"],
        "core_request_fifo_empty_percent": 100 * totals["fifo_empty_cycles"] / totals["core_cycles"],
        "noc_modeled_quiescent_percent": 100 * totals["noc_quiescent_cycles"] / totals["icnt_cycles"],
        "dram_modeled_quiescent_percent": 100 * totals["dram_quiescent_cycles"] / totals["dram_cycles"],
        "all_sram_devices_modeled_idle_percent": 100 * totals["spad_idle_cycles"] / totals["core_cycles"],
    }
    evidence = {
        "definition": "Cycle-weighted sums over independently validated compact component summaries.",
        "excluded_metric": {
            "metric": "vector-unit activity/idleness",
            "reason": "Excluded from component-selection claims because the current simulator vector counter is suspected to be incorrect; no quantitative conclusion is drawn from it.",
        },
        "weighted_aggregate": weighted,
        "sources": sources,
        "component_selection_csv": {
            "path": str(csv_path.relative_to(ROOT)),
            "sha256": sha256(csv_path),
        },
    }
    (OUT_DIR / "component_selection_v2_manifest.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
