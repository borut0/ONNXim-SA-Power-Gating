#!/usr/bin/env python3
"""Generate parameterized PMOS-header transient decks from a validated load.

The input is an aggregate leakage deck produced by
generate_hspice_aggregate_load.py.  This script changes only the array supply:
the ideal virtual-VDD source is replaced by a real ASAP7 PMOS header bank and
a two-state sleep command.  It preserves all measured cell states, PEX cell
models, load capacitances, and sequential initialization from the input deck.

Candidate header multiplicities are derived from measured ACTIVE power and
the directly measured unit-header I-V curve.  They are screening points, not
a final physical size or a claim of worst-case active-current adequacy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def spice_number(value: float) -> str:
    return f"{value:.12g}"


def parse_int_list(text: str) -> list[int]:
    values = sorted({int(value) for value in text.split(",")})
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def parse_float_list(text: str) -> list[float]:
    values = sorted({float(value) for value in text.split(",")})
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated numbers")
    return values


def derive_candidates(unit_data: dict, active_power_w: float) -> list[dict]:
    vdd = float(unit_data["vdd_v"])
    active_current_a = active_power_w / vdd
    candidates = []
    for raw_droop, raw_current in unit_data[
        "on_current_a_by_droop_percent"
    ].items():
        droop_percent = float(raw_droop)
        unit_current_a = float(raw_current)
        candidates.append(
            {
                "droop_percent": droop_percent,
                "unit_current_a": unit_current_a,
                "multiplicity": math.ceil(active_current_a / unit_current_a),
            }
        )
    candidates.sort(key=lambda item: item["multiplicity"])
    return candidates


def transform_aggregate_deck(
    aggregate_text: str,
    model_name: str,
    multiplicity: int,
    vdd: float,
    sleep_slew_ps: float,
) -> str:
    lines = aggregate_text.splitlines()
    supply_indices = [
        index for index, line in enumerate(lines)
        if line.strip().upper().startswith("VDD_SOURCE VVIRTUAL 0 ")
    ]
    if len(supply_indices) != 1:
        raise RuntimeError(
            "validated aggregate deck must contain exactly one "
            "'VDD_SOURCE vvirtual 0 ...' line"
        )

    transient_indices = [
        index for index, line in enumerate(lines)
        if line.strip().lower().startswith(".tran ")
    ]
    if len(transient_indices) != 1:
        raise RuntimeError("aggregate deck must contain exactly one .tran")
    transient_index = transient_indices[0]
    if not any(line.strip().lower() == ".end" for line in lines[transient_index:]):
        raise RuntimeError("aggregate deck has no .end after .tran")

    # Keep the validated load and initialization, but discard its old analysis.
    body = lines[:transient_index]
    supply_index = supply_indices[0]
    if supply_index >= len(body):
        raise RuntimeError("VDD source unexpectedly appears after .tran")

    slew_ns = sleep_slew_ps / 1000.0
    entry_start_ns = 5.0
    entry_end_ns = entry_start_ns + slew_ns
    wake_start_ns = 15.0
    wake_end_ns = wake_start_ns + slew_ns
    simulation_end_ns = 25.0

    replacement = [
        f"VDD_SOURCE vdd 0 {spice_number(vdd)}",
        (
            "VSLEEP sleep 0 PWL("
            f"0 0 {spice_number(entry_start_ns)}n 0 "
            f"{spice_number(entry_end_ns)}n {spice_number(vdd)} "
            f"{spice_number(wake_start_ns)}n {spice_number(vdd)} "
            f"{spice_number(wake_end_ns)}n 0 "
            f"{spice_number(simulation_end_ns)}n 0)"
        ),
        (
            "MHEADER vvirtual sleep vdd vdd "
            f"{model_name} w=81n l=20n nfin=3 m={multiplicity}"
        ),
    ]
    body[supply_index:supply_index + 1] = replacement

    entry_trigger_v = 0.5 * vdd
    entry_target_v = 0.1 * vdd
    wake_target_v = 0.9 * vdd
    analysis = [
        "",
        f".tran 1p {spice_number(simulation_end_ns)}n",
        "* Direct static and rail-state measurements",
        ".measure tran p_on_vdd AVG par('-v(vdd)*i(VDD_SOURCE)') "
        "FROM=4n TO=4.8n",
        ".measure tran vvirtual_on_avg AVG v(vvirtual) FROM=4n TO=4.8n",
        ".measure tran p_off_vdd AVG par('-v(vdd)*i(VDD_SOURCE)') "
        "FROM=13n TO=14.5n",
        ".measure tran p_off_logic1 AVG par('-v(logic1)*i(VLOGIC1)') "
        "FROM=13n TO=14.5n",
        ".measure tran vvirtual_off_avg AVG v(vvirtual) FROM=13n TO=14.5n",
        "* Electrical entry/wake definitions: sleep 50% to rail 10%/90%",
        (
            ".measure tran entry_latency "
            f"TRIG v(sleep) VAL={spice_number(entry_trigger_v)} RISE=1 "
            f"TARG v(vvirtual) VAL={spice_number(entry_target_v)} FALL=1"
        ),
        (
            ".measure tran wake_latency "
            f"TRIG v(sleep) VAL={spice_number(entry_trigger_v)} FALL=1 "
            f"TARG v(vvirtual) VAL={spice_number(wake_target_v)} RISE=1"
        ),
        "* Raw fixed-window energies; offline analysis applies baseline subtraction",
        ".measure tran entry_vdd_energy_raw INTEG "
        "par('-v(vdd)*i(VDD_SOURCE)') FROM=5n TO=10n",
        ".measure tran entry_logic1_energy_raw INTEG "
        "par('-v(logic1)*i(VLOGIC1)') FROM=5n TO=10n",
        ".measure tran entry_control_energy_net INTEG "
        "par('-v(sleep)*i(VSLEEP)') FROM=5n TO=10n",
        ".measure tran entry_control_energy_abs INTEG "
        "par('abs(v(sleep)*i(VSLEEP))') FROM=5n TO=10n",
        ".measure tran wake_vdd_energy_raw INTEG "
        "par('-v(vdd)*i(VDD_SOURCE)') FROM=15n TO=20n",
        ".measure tran wake_logic1_energy_raw INTEG "
        "par('-v(logic1)*i(VLOGIC1)') FROM=15n TO=20n",
        ".measure tran wake_control_energy_net INTEG "
        "par('-v(sleep)*i(VSLEEP)') FROM=15n TO=20n",
        ".measure tran wake_control_energy_abs INTEG "
        "par('abs(v(sleep)*i(VSLEEP))') FROM=15n TO=20n",
        ".measure tran peak_wake_current MAX par('-i(VDD_SOURCE)') "
        "FROM=15n TO=20n",
        ".end",
        "",
    ]
    return "\n".join(body + analysis)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-deck", required=True, type=Path)
    parser.add_argument("--unit-measurements", required=True, type=Path)
    parser.add_argument("--active-power-w", required=True, type=float)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--header-multiplicities", type=parse_int_list)
    parser.add_argument(
        "--sleep-slew-ps", type=parse_float_list, default=[50.0, 250.0, 1000.0]
    )
    args = parser.parse_args()

    if args.active_power_w <= 0:
        raise ValueError("--active-power-w must be positive")
    aggregate_text = args.aggregate_deck.read_text()
    unit_data = json.loads(args.unit_measurements.read_text())
    if unit_data.get("status") != "direct_hspice_measurement":
        raise RuntimeError("unit measurement file is not marked as direct HSPICE data")

    candidates = derive_candidates(unit_data, args.active_power_w)
    multiplicities = (
        args.header_multiplicities
        if args.header_multiplicities is not None
        else sorted({item["multiplicity"] for item in candidates})
    )
    vdd = float(unit_data["vdd_v"])
    model_name = str(unit_data["model_name"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for multiplicity in multiplicities:
        for sleep_slew_ps in args.sleep_slew_ps:
            slew_tag = f"{sleep_slew_ps:g}".replace(".", "p")
            output = args.output_dir / (
                f"{args.label}_m{multiplicity}_slew{slew_tag}ps.sp"
            )
            output.write_text(
                transform_aggregate_deck(
                    aggregate_text, model_name, multiplicity, vdd,
                    sleep_slew_ps
                )
            )
            outputs.append(
                {
                    "path": str(output),
                    "multiplicity": multiplicity,
                    "sleep_slew_ps": sleep_slew_ps,
                }
            )

    summary = {
        "label": args.label,
        "aggregate_deck": str(args.aggregate_deck),
        "unit_measurements": str(args.unit_measurements),
        "active_power_w": args.active_power_w,
        "active_average_current_a": args.active_power_w / vdd,
        "derived_candidates": candidates,
        "generated_multiplicities": multiplicities,
        "sleep_slew_ps": args.sleep_slew_ps,
        "outputs": outputs,
        "direct_measurements": [
            "aggregate cell states and loads",
            "unit PMOS ON/OFF I-V",
            "ACTIVE average power",
        ],
        "assumptions_or_sensitivity_parameters": [
            "RVT PMOS header topology",
            "parallel-device multiplicity",
            "sleep-command slew",
            "candidate sizing uses average ACTIVE current and is not final timing signoff",
            "output isolation and sleep-driver implementation are not included",
        ],
    }
    summary_path = args.output_dir / f"{args.label}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
