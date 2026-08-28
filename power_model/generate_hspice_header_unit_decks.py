#!/usr/bin/env python3
"""Generate ASAP7 PMOS-header unit-characterization HSPICE decks.

This is the first, load-independent checkpoint in the SA power-gating flow.
It measures a single 3-fin RVT PMOS header's ON I-V curve, OFF leakage curve,
and verifies that HSPICE device multiplicity scales current linearly.  These
measurements bracket, but do not select, the array-level header size.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def spice_number(value: float) -> str:
    return f"{value:.12g}"


def common_lines(model: Path, temperature_c: float) -> list[str]:
    return [
        f".include '{model}'",
        ".option post=2 runlvl=6 accurate",
        f".temp {spice_number(temperature_c)}",
        "",
    ]


def header_device(instance: str, drain: str, gate: str, source: str,
                  model_name: str, multiplicity: int = 1) -> str:
    # Geometry matches the PMOS in ASAP7 INVx1_ASAP7_75t_R CDL.
    return (
        f"M{instance} {drain} {gate} {source} {source} {model_name} "
        f"w=81n l=20n nfin=3 m={multiplicity}"
    )


def make_on_iv(model: Path, model_name: str, vdd: float,
               temperature_c: float, label: str) -> str:
    points = (
        ("2p5pct", 0.975 * vdd),
        ("5pct", 0.950 * vdd),
        ("7p5pct", 0.925 * vdd),
        ("10pct", 0.900 * vdd),
    )
    start = 0.80 * vdd
    step = vdd / 1400.0  # 0.5 mV at VDD=0.7 V; includes every point above.
    lines = [
        f"* {label}: one-unit ASAP7 RVT PMOS header ON I-V",
        *common_lines(model, temperature_c),
        f"VDD_SOURCE vdd 0 {spice_number(vdd)}",
        "VSLEEP sleep 0 0",
        "VCLAMP vvirtual 0 0",
        header_device("HEADER", "vvirtual", "sleep", "vdd", model_name),
        "",
        f".dc VCLAMP {spice_number(start)} {spice_number(vdd)} "
        f"{spice_number(step)}",
    ]
    for name, virtual_voltage in points:
        lines.append(
            f".measure dc ion_drop_{name} FIND par('-i(VDD_SOURCE)') "
            f"AT={spice_number(virtual_voltage)}"
        )
    lines.extend([".end", ""])
    return "\n".join(lines)


def make_off_iv(model: Path, model_name: str, vdd: float,
                temperature_c: float, label: str) -> str:
    points = (
        ("0", 0.0),
        ("0p25vdd", 0.25 * vdd),
        ("0p50vdd", 0.50 * vdd),
        ("0p90vdd", 0.90 * vdd),
    )
    step = vdd / 140.0  # 5 mV at VDD=0.7 V.
    lines = [
        f"* {label}: one-unit ASAP7 RVT PMOS header OFF leakage I-V",
        *common_lines(model, temperature_c),
        f"VDD_SOURCE vdd 0 {spice_number(vdd)}",
        f"VSLEEP sleep 0 {spice_number(vdd)}",
        "VCLAMP vvirtual 0 0",
        header_device("HEADER", "vvirtual", "sleep", "vdd", model_name),
        "",
        f".dc VCLAMP 0 {spice_number(vdd)} {spice_number(step)}",
    ]
    for name, virtual_voltage in points:
        lines.append(
            f".measure dc ioff_vvirtual_{name} FIND par('-i(VDD_SOURCE)') "
            f"AT={spice_number(virtual_voltage)}"
        )
    lines.extend([".end", ""])
    return "\n".join(lines)


def make_multiplicity(model: Path, model_name: str, vdd: float,
                      temperature_c: float, label: str) -> str:
    vvirtual = 0.95 * vdd
    lines = [
        f"* {label}: ASAP7 PMOS-header device multiplicity validation",
        *common_lines(model, temperature_c),
        "VSLEEP sleep 0 0",
    ]
    for multiplicity in (1, 10, 100):
        lines.extend(
            [
                f"VDD_M{multiplicity} vdd_m{multiplicity} 0 "
                f"{spice_number(vdd)}",
                f"VCLAMP_M{multiplicity} vv_m{multiplicity} 0 "
                f"{spice_number(vvirtual)}",
                header_device(
                    f"M{multiplicity}", f"vv_m{multiplicity}", "sleep",
                    f"vdd_m{multiplicity}", model_name, multiplicity
                ),
            ]
        )
    lines.extend(
        [
            "",
            ".tran 0.1p 10p",
            ".measure tran ion_m1 AVG par('-i(VDD_M1)') FROM=5p TO=10p",
            ".measure tran ion_m10 AVG par('-i(VDD_M10)') FROM=5p TO=10p",
            ".measure tran ion_m100 AVG par('-i(VDD_M100)') FROM=5p TO=10p",
            ".measure tran ratio_m10_m1 PARAM='ion_m10/ion_m1'",
            ".measure tran ratio_m100_m1 PARAM='ion_m100/ion_m1'",
            ".end",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--label", default="asap7_tt_25c")
    parser.add_argument("--model-name", default="pmos_rvt")
    parser.add_argument("--vdd", default=0.7, type=float)
    parser.add_argument("--temperature-c", default=25.0, type=float)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "multiplicity": args.output_dir / f"header_unit_multiplicity_{args.label}.sp",
        "on_iv": args.output_dir / f"header_unit_on_iv_{args.label}.sp",
        "off_iv": args.output_dir / f"header_unit_off_iv_{args.label}.sp",
    }
    outputs["multiplicity"].write_text(
        make_multiplicity(
            args.model, args.model_name, args.vdd, args.temperature_c,
            args.label
        )
    )
    outputs["on_iv"].write_text(
        make_on_iv(
            args.model, args.model_name, args.vdd, args.temperature_c,
            args.label
        )
    )
    outputs["off_iv"].write_text(
        make_off_iv(
            args.model, args.model_name, args.vdd, args.temperature_c,
            args.label
        )
    )

    summary = {
        "label": args.label,
        "model": str(args.model),
        "model_name": args.model_name,
        "vdd_v": args.vdd,
        "temperature_c": args.temperature_c,
        "unit_geometry": {"width_nm": 81, "length_nm": 20, "nfin": 3},
        "outputs": {name: str(path) for name, path in outputs.items()},
        "scope": (
            "unit-device characterization only; no array header size is "
            "selected by these decks"
        ),
    }
    summary_path = args.output_dir / f"header_unit_{args.label}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
