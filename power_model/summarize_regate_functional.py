#!/usr/bin/env python3
"""Aggregate ReGate logical-control shadow records.

The result is an extrapolated gross-static sensitivity estimate. It is not
measured energy and does not include transition, controller, routing,
isolation, or wake-stall costs. Power factors must be selected explicitly.
"""

import argparse
import json
import numbers
from pathlib import Path


PREFIX = "REGATE_FUNCTIONAL_PROFILE "
REGATE_ARTIFACT_COMMIT = "004dbfccb90cfc0a61bb79bbee3a5770b4b3897b"
REGATE_POWER_MODEL = (
    "trace_util/llm_ops_generator/power_analysis_lib.py:328-356,432-540"
)
REGATE_HW_FULL_LOW_POWER_FACTOR = 0.03


def percentage(numerator, denominator):
    return 0.0 if denominator == 0 else 100.0 * numerator / denominator


def exact_integer(value, description):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{description} must be an exact integer")
    return value


def read_records(path):
    records = []
    for line_number, line in enumerate(
            path.read_text(errors="replace").splitlines(), 1):
        marker = line.find(PREFIX)
        if marker < 0:
            continue
        payload = line[marker + len(PREFIX):]
        try:
            records.append(json.loads(payload))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{path}:{line_number}: invalid {PREFIX.strip()} JSON"
            ) from error
    if not records:
        raise ValueError(f"{path}: no {PREFIX.strip()} records")
    return records


def aggregate(records, w_on_power_factor, off_power_factor,
              parameter_source, expected_cores=None):
    for name, value in (
            ("W_on power factor", w_on_power_factor),
            ("OFF power factor", off_power_factor)):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f"{name} must be a real number")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between zero and one")
    if not isinstance(parameter_source, dict) or not parameter_source:
        raise ValueError("power-factor provenance must be a nonempty object")
    if parameter_source.get("evidence_class") not in {
            "reference", "assumed"}:
        raise ValueError(
            "power-factor provenance must be reference or assumed"
        )
    if not isinstance(parameter_source.get("locator"), str) or not (
            parameter_source["locator"].strip()):
        raise ValueError("power-factor provenance requires a locator")
    if not records:
        raise ValueError("no ReGate functional records")

    for item in records:
        if not isinstance(item, dict):
            raise ValueError("each ReGate functional record must be an object")
        if exact_integer(item.get("version"), "record version") != 1:
            raise ValueError("unsupported ReGate functional record version")
        if item.get("model") != (
                "ReGate logical-control shadow approximation"):
            raise ValueError("unexpected ReGate functional record model")
    core_ids = [
        exact_integer(item.get("core_id"), "core_id")
        for item in records
    ]
    if len(set(core_ids)) != len(core_ids):
        raise ValueError("duplicate ReGate record for a core")
    if expected_cores is not None:
        exact_integer(expected_cores, "expected core count")
        if expected_cores <= 0:
            raise ValueError("expected core count must be positive")
    if expected_cores is not None and len(records) != expected_cores:
        raise ValueError(
            f"expected {expected_cores} cores, found {len(records)}"
        )

    dimensions = {
        (
            exact_integer(item.get("array_rows"), "array_rows"),
            exact_integer(item.get("array_columns"), "array_columns"),
        )
        for item in records
    }
    if len(dimensions) != 1:
        raise ValueError("inconsistent SA dimensions across cores")
    array_rows, array_columns = next(iter(dimensions))
    if array_rows <= 0 or array_columns <= 0:
        raise ValueError("SA dimensions must be positive")

    totals = {
        "capacity": 0,
        "on": 0,
        "w_on": 0,
        "off": 0,
        "unclassified": 0,
    }
    per_core = []
    for item in sorted(records, key=lambda value: value["core_id"]):
        core_id = exact_integer(item["core_id"], "core_id")
        if item.get("regate_only") is not True:
            raise ValueError(f"core {core_id} is not marked ReGate-only")
        if item.get("combined_with_retention") is not False:
            raise ValueError(
                f"core {core_id} is combined with retention"
            )
        if item.get("validation_pass") is not True:
            raise ValueError(
                f"core {core_id} ReGate shadow validation failed"
            )
        states = item.get("pe_state_cycles")
        if not isinstance(states, dict):
            raise ValueError(
                f"core {core_id} pe_state_cycles must be an object"
            )
        values = {
            name: exact_integer(
                states.get(name), f"core {core_id} {name} PE cycles"
            )
            for name in totals
        }
        if any(value < 0 for value in values.values()):
            raise ValueError(f"core {core_id} has negative PE-state cycles")
        classified = (
            values["on"] + values["w_on"] + values["off"] +
            values["unclassified"]
        )
        if classified != values["capacity"]:
            raise ValueError(
                f"core {core_id} PE-state partition does not equal capacity"
            )
        if states.get("partition_exact_under_model") is not True:
            raise ValueError(
                f"core {core_id} partition is not exact under the model"
            )

        baseline = float(values["capacity"])
        # Fail closed: unresolved PE-cycles remain at ON power.
        modeled = (
            values["on"] + values["unclassified"] +
            w_on_power_factor * values["w_on"] +
            off_power_factor * values["off"]
        )
        per_core.append({
            "core_id": core_id,
            "pe_state_cycles": values,
            "normalized_always_on_static_energy": baseline,
            "normalized_regate_gross_static_energy": modeled,
            "gross_static_saving_percent": percentage(
                baseline - modeled, baseline
            ),
        })
        for name in totals:
            totals[name] += values[name]

    baseline = float(totals["capacity"])
    modeled = (
        totals["on"] + totals["unclassified"] +
        w_on_power_factor * totals["w_on"] +
        off_power_factor * totals["off"]
    )
    return {
        "version": 1,
        "model": "ReGate logical-control shadow gross-static estimate",
        "evidence_class": "extrapolated",
        "source_validation_pass": True,
        "regate_only": True,
        "combined_with_retention": False,
        "array_rows": array_rows,
        "array_columns": array_columns,
        "power_parameters": {
            "w_on_over_on": w_on_power_factor,
            "off_over_on": off_power_factor,
            "provenance": parameter_source,
            "state_mapping_evidence_class": "assumed",
            "state_mapping": (
                "the selected scalar factors are applied independently to "
                "shadow W_on and OFF PE-cycle counts"
            ),
        },
        "per_core": per_core,
        "aggregate": {
            "pe_state_cycles": totals,
            "normalized_always_on_static_energy": baseline,
            "normalized_regate_gross_static_energy": modeled,
            "gross_static_saving_percent": percentage(
                baseline - modeled, baseline
            ),
        },
        "reference": {
            "paper": (
                "ReGate: Enabling Power Gating in Neural Processing Units"
            ),
            "paper_locator": "Section 4.1 and Table 3",
            "artifact_commit": REGATE_ARTIFACT_COMMIT,
            "artifact_locator": REGATE_POWER_MODEL,
        },
        "excluded_costs": [
            "power-state transition energy",
            "controller and PE_on distribution energy",
            "power-switch, isolation, and routing overhead",
            "physical wake delay and structural-hazard stalls",
        ],
        "net_energy_saving_status": "pending",
        "interpretation": (
            "Gross static saving versus an always-ON PE baseline. "
            "Unclassified PE-cycles are conservatively charged at ON power. "
            "Transferring scalar power factors to shadow PE states is a "
            "cross-model sensitivity extrapolation, not the ReGate artifact "
            "equation. This is not measured Joules or net ReGate energy."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--expected-cores", type=int)
    parser.add_argument(
        "--regate-hw-full-reference",
        action="store_true",
        help=(
            "explicitly use the pinned ReGate HW/Full low-power factor 0.03 "
            "as a sensitivity factor for both W_on and OFF"
        ),
    )
    parser.add_argument("--w-on-power-factor", type=float)
    parser.add_argument("--off-power-factor", type=float)
    args = parser.parse_args()

    explicit = (
        args.w_on_power_factor is not None or
        args.off_power_factor is not None
    )
    if args.regate_hw_full_reference and explicit:
        parser.error(
            "select the ReGate reference or explicit factors, not both"
        )
    if args.regate_hw_full_reference:
        w_on_factor = REGATE_HW_FULL_LOW_POWER_FACTOR
        off_factor = REGATE_HW_FULL_LOW_POWER_FACTOR
        source = {
            "evidence_class": "reference",
            "locator": (
                f"ReGate artifact {REGATE_ARTIFACT_COMMIT}, "
                f"{REGATE_POWER_MODEL}"
            ),
            "reference_value": REGATE_HW_FULL_LOW_POWER_FACTOR,
            "limitation": (
                "the artifact uses one aggregate low-power factor; applying "
                "it separately to W_on and OFF is not an artifact equation"
            ),
        }
    elif (args.w_on_power_factor is not None and
          args.off_power_factor is not None):
        w_on_factor = args.w_on_power_factor
        off_factor = args.off_power_factor
        source = {
            "evidence_class": "assumed",
            "locator": "explicit command-line factors",
            "limitation": (
                "not a circuit measurement or ReGate artifact state ratio"
            ),
        }
    else:
        parser.error(
            "use --regate-hw-full-reference or provide both "
            "--w-on-power-factor and --off-power-factor"
        )

    result = aggregate(
        read_records(args.log),
        w_on_factor,
        off_factor,
        source,
        expected_cores=args.expected_cores,
    )
    result["source_log"] = str(args.log)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
