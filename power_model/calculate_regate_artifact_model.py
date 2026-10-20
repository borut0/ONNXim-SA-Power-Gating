#!/usr/bin/env python3
"""Apply ReGate's released PE-level SA equations to an ONNXim profile.

This calculator preserves the equation ordering in ReGate's public artifact
rather than simulating its PE control network.  ONNXim supplies per-core
workload aggregates, whereas the artifact applies the equations per operator;
the result is consequently an artifact-equation aggregate approximation, not
artifact parity.  Results are derived, normalized estimates, not measured
Joules or a circuit-level reproduction.
"""

import argparse
import json
import math
from pathlib import Path


REGATE_ARTIFACT_COMMIT = "004dbfccb90cfc0a61bb79bbee3a5770b4b3897b"
REGATE_POWER_MODEL_SHA256 = (
    "0174f85e4354f8ccb0f10c34b5c05ed0912179bda29f4087a240809c0baab800"
)
REGATE_POWER_MODEL = (
    "trace_util/llm_ops_generator/power_analysis_lib.py:328-356,432-540"
)


def percentage(numerator, denominator):
    return 0.0 if denominator == 0 else 100.0 * numerator / denominator


def analyze_core(item, array_rows, array_columns, low_power_factor,
                 pe_delay_cycles):
    core_id = int(item["core_id"])
    execution_cycles = int(item["core_cycles"])
    sa_active_cycles = int(item["whole_sa_active_cycles"])
    sa_instruction_count = int(item["sa_instruction_intervals"])
    useful_mac_count = int(item["modeled_useful_mac_count"])

    values = {
        "core_cycles": execution_cycles,
        "whole_sa_active_cycles": sa_active_cycles,
        "sa_instruction_intervals": sa_instruction_count,
        "modeled_useful_mac_count": useful_mac_count,
    }
    if any(value < 0 for value in values.values()):
        raise ValueError(f"core {core_id} contains a negative ReGate input")
    if sa_active_cycles > execution_cycles:
        raise ValueError(
            f"core {core_id} SA-active cycles exceed execution cycles"
        )
    if sa_active_cycles == 0 and useful_mac_count != 0:
        raise ValueError(
            f"core {core_id} has useful MACs but no SA-active cycles"
        )
    if sa_active_cycles != 0 and sa_instruction_count == 0:
        raise ValueError(
            f"core {core_id} has SA-active cycles but no SA instructions"
        )

    active_mac_capacity = array_rows * array_columns * sa_active_cycles
    useful_flop_count = 2 * useful_mac_count
    active_flop_capacity = 2 * active_mac_capacity
    raw_utilization = (
        0.0
        if active_mac_capacity == 0
        else useful_mac_count / active_mac_capacity
    )
    # ReGate's released model explicitly caps FLOP utilization at one.  Keep
    # the uncapped value and flag visible so this never hides an inconsistency.
    utilization = min(raw_utilization, 1.0)
    utilization_clamped = raw_utilization > 1.0

    if sa_active_cycles == 0:
        delay_overhead_cycles = 0
    else:
        delay_from_active_time = math.ceil(
            sa_active_cycles * pe_delay_cycles / array_rows
        )
        delay_from_instruction_count = math.ceil(
            sa_instruction_count * pe_delay_cycles
        )
        delay_overhead_cycles = min(
            delay_from_active_time, delay_from_instruction_count
        )

    adjusted_sa_cycles = sa_active_cycles + delay_overhead_cycles
    adjusted_execution_cycles = max(execution_cycles, adjusted_sa_cycles)

    # Normalize full-array ON static power to one energy unit per cycle.
    # ReGate computes utilization before adding delay, then uses the adjusted
    # SA/execution times in its static-energy expression; mirror that ordering.
    baseline_energy = float(execution_cycles)
    regate_energy = (
        utilization * adjusted_sa_cycles
        + low_power_factor * (1.0 - utilization) * adjusted_sa_cycles
        + low_power_factor
        * (adjusted_execution_cycles - adjusted_sa_cycles)
    )

    return {
        "core_id": core_id,
        **values,
        "active_mac_capacity": active_mac_capacity,
        "modeled_useful_flop_count": useful_flop_count,
        "active_flop_capacity": active_flop_capacity,
        "raw_sa_flops_utilization": raw_utilization,
        "sa_flops_utilization": utilization,
        "utilization_clamped_to_one": utilization_clamped,
        "estimated_pe_delay_overhead_cycles": delay_overhead_cycles,
        "estimated_adjusted_sa_cycles": adjusted_sa_cycles,
        "estimated_adjusted_execution_cycles": adjusted_execution_cycles,
        "normalized_always_on_static_energy": baseline_energy,
        "normalized_regate_static_energy": regate_energy,
        "normalized_static_energy_saving_percent": percentage(
            baseline_energy - regate_energy, baseline_energy
        ),
    }


def calculate(summary, low_power_factor=0.03, pe_delay_cycles=1):
    if not 0.0 <= low_power_factor <= 1.0:
        raise ValueError("low-power factor must be between zero and one")
    if pe_delay_cycles < 0:
        raise ValueError("PE delay must be nonnegative")
    if not summary.get("validation_pass", False):
        raise ValueError("source SA-region summary did not pass validation")
    if int(summary.get("version", 0)) < 4:
        raise ValueError(
            "source summary lacks ReGate work-count inputs; rerun with SA "
            "region profiler version 4"
        )

    array_rows = int(summary["array_rows"])
    array_columns = int(summary["array_columns"])
    if array_rows <= 0 or array_columns <= 0:
        raise ValueError("array dimensions must be positive")
    if array_rows != array_columns:
        raise ValueError(
            "ReGate artifact delay equation requires a square SA dimension"
        )

    inputs = summary.get("per_core_regate_inputs", [])
    if not inputs:
        raise ValueError("source summary contains no per-core ReGate inputs")
    cores = [
        analyze_core(
            item,
            array_rows,
            array_columns,
            low_power_factor,
            pe_delay_cycles,
        )
        for item in inputs
    ]

    aggregate_checks = {
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
    }
    for name, calculated in aggregate_checks.items():
        if name in summary and int(summary[name]) != calculated:
            raise ValueError(
                f"aggregate {name} does not match per-core inputs: "
                f"{summary[name]} != {calculated}"
            )

    baseline_energy = sum(
        item["normalized_always_on_static_energy"] for item in cores
    )
    regate_energy = sum(
        item["normalized_regate_static_energy"] for item in cores
    )
    original_global_cycles = max(item["core_cycles"] for item in cores)
    adjusted_global_cycles = max(
        item["estimated_adjusted_execution_cycles"] for item in cores
    )

    warnings = []
    clamped_cores = [
        item["core_id"] for item in cores if item["utilization_clamped_to_one"]
    ]
    if clamped_cores:
        warnings.append(
            "Raw utilization exceeded one and was visibly capped exactly as "
            f"in the ReGate artifact for cores {clamped_cores}."
        )

    return {
        "version": 1,
        "model": (
            "ReGate PE-level artifact-equation workload-aggregate approximation"
        ),
        "evidence_class": "derived_from_measured_onnxim_and_reference_regate_inputs",
        "source_validation_pass": True,
        "array_rows": array_rows,
        "array_columns": array_columns,
        "reference": {
            "paper": "ReGate: Enabling Power Gating in Neural Processing Units",
            "venue": "MICRO 2025",
            "artifact_commit": REGATE_ARTIFACT_COMMIT,
            "artifact_equations": REGATE_POWER_MODEL,
            "artifact_source_sha256": REGATE_POWER_MODEL_SHA256,
            "low_power_factor": low_power_factor,
            "pe_power_on_off_delay_cycles": pe_delay_cycles,
            "configuration_scope": "ReGate HW/Full PE-level SA path",
        },
        "input_evidence": {
            "core_and_sa_cycles": "measured by cycle-level ONNXim",
            "sa_instruction_count": "measured by cycle-level ONNXim",
            "useful_mac_count": (
                "derived from issued instruction metadata as "
                "sum(tile_m*tile_k*tile_n)"
            ),
            "low_power_factor_and_pe_delay": "reference from ReGate",
            "energy_and_performance_results": "derived",
        },
        "scope": {
            "aggregation": "per-core workload aggregate",
            "timing": (
                "reference artifact delay equation; no ONNXim scheduling or "
                "execution timing is modified"
            ),
            "energy_unit": (
                "full-array ON static-power-cycle equivalent; not Joules"
            ),
            "bet_filter_applied": False,
            "utilization_equation": (
                "2*useful_MACs/(2*array_rows*array_columns*SA_active_cycles); "
                "the factor of two converts one MAC to two FLOPs and cancels"
            ),
        },
        "per_core": cores,
        "aggregate": {
            **aggregate_checks,
            "normalized_always_on_static_energy": baseline_energy,
            "normalized_regate_static_energy": regate_energy,
            "normalized_static_energy_saving_percent": percentage(
                baseline_energy - regate_energy, baseline_energy
            ),
            "estimated_original_global_cycles": original_global_cycles,
            "estimated_adjusted_global_cycles": adjusted_global_cycles,
            "estimated_performance_overhead_percent": percentage(
                adjusted_global_cycles - original_global_cycles,
                original_global_cycles,
            ),
        },
        "warnings": warnings,
        "limitations": [
            "ONNXim does not expose cycle-accurate PE_on wavefront activity.",
            "The public ReGate artifact operates per operator; this calculator "
            "applies the same equations to per-core workload aggregates.",
            "The model does not distinguish W_on leakage from complete OFF "
            "leakage and does not inspect numerical zero weights.",
            "Controller, switch, isolation, routing, and transition energies "
            "are not independently measured by this calculation.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--low-power-factor", type=float, default=0.03)
    parser.add_argument("--pe-delay-cycles", type=int, default=1)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    result = calculate(
        summary,
        low_power_factor=args.low_power_factor,
        pe_delay_cycles=args.pe_delay_cycles,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
