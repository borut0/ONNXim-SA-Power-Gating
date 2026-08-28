#!/usr/bin/env python3
"""Generate a multiplicity-compressed ASAP7 aggregate-load HSPICE deck.

The deck preserves measured cell pin states and exact synthesis-stage fanout
pin capacitance groups.  Sequential cells are asynchronously initialized to
their measured QN state before leakage measurement.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_subckts(path: Path):
    lines = path.read_text(errors="replace").splitlines()
    subckts = {}
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped.lower().startswith(".subckt "):
            index += 1
            continue
        tokens = stripped.split()
        index += 1
        while index < len(lines) and lines[index].lstrip().startswith("+"):
            tokens.extend(lines[index].lstrip()[1:].split())
            index += 1
        name = tokens[1]
        pins = tuple(
            token for token in tokens[2:]
            if token.upper() != "PARAMS:" and "=" not in token
        )
        subckts[name] = pins
    return subckts


def parse_assignments(text: str):
    result = {}
    if not text:
        return result
    for assignment in text.split(";"):
        name, value = assignment.split("=", 1)
        result[name.upper()] = value
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-loads", required=True, type=Path)
    parser.add_argument("--pex", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    subckts = parse_subckts(args.pex)
    rows = []
    with args.state_loads.open(newline="") as stream:
        for row in csv.DictReader(stream):
            row["count"] = int(row["count"])
            row["state"] = parse_assignments(row["pin_state"])
            row["loads"] = parse_assignments(row["output_loads_ff"])
            rows.append(row)

    missing_subckts = sorted({row["cell"] for row in rows} - set(subckts))
    if missing_subckts:
        raise RuntimeError(f"missing PEX subcircuits: {missing_subckts}")

    deck = [
        f"* {args.label}: multiplicity-compressed ASAP7 aggregate load",
        f".include '{args.model}'",
        f".include '{args.pex}'",
        ".option post=2 runlvl=6",
        ".temp 25",
        "",
        "VDD_SOURCE vvirtual 0 0.7",
        "VLOGIC1 logic1 0 0.7",
        "",
    ]

    total_cells = 0
    total_external_load_ff = 0.0
    sequential_groups = 0

    for group, row in enumerate(rows):
        cell = row["cell"]
        count = row["count"]
        state = row["state"]
        loads = {pin: float(value) for pin, value in row["loads"].items()}
        total_cells += count
        total_external_load_ff += sum(loads.values()) * count

        special_nodes = {}
        if cell.upper().startswith("DFFASRHQN"):
            sequential_groups += 1
            qn = state.get("QN")
            if qn not in ("0", "1"):
                raise RuntimeError(f"unsupported sequential state: {cell} {state}")
            if state.get("RESETN") != "1" or state.get("SETN") != "1":
                raise RuntimeError(
                    f"expected inactive async controls at idle: {cell} {state}"
                )
            control = "SETN" if qn == "0" else "RESETN"
            node = f"init_{group}"
            special_nodes[control] = node
            deck.append(
                f"VINIT{group} {node} 0 PWL(0 0 1n 0 1.01n 0.7 4n 0.7)"
            )

        nodes = []
        output_nodes = []
        for raw_pin in subckts[cell]:
            pin = raw_pin.upper()
            if pin == "VSS":
                nodes.append("0")
            elif pin == "VDD":
                nodes.append("vvirtual")
            elif pin in loads:
                node = f"g{group}_{pin.lower()}"
                nodes.append(node)
                output_nodes.append((pin, node, loads[pin]))
            elif pin in special_nodes:
                nodes.append(special_nodes[pin])
            elif pin in state:
                if state[pin] == "0":
                    nodes.append("0")
                elif state[pin] == "1":
                    nodes.append("logic1")
                else:
                    raise RuntimeError(f"unknown pin state: {cell}/{pin}={state[pin]}")
            else:
                raise RuntimeError(f"unmapped PEX pin: {cell}/{raw_pin}")

        deck.append(f"XG{group} {' '.join(nodes)} {cell} M={count}")
        for output_number, (pin, node, load_ff) in enumerate(output_nodes):
            aggregate_load_ff = load_ff * count
            if aggregate_load_ff > 0:
                deck.append(
                    f"CG{group}_{output_number} {node} 0 {aggregate_load_ff:.9f}f"
                )

    deck.extend(
        [
            "",
            ".tran 1p 4n",
            ".measure tran ivdd_avg AVG i(VDD_SOURCE) FROM=3n TO=4n",
            ".measure tran pleak_avg AVG "
            "par('-v(vvirtual)*i(VDD_SOURCE)') FROM=3n TO=4n",
            ".end",
            "",
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(deck))
    summary = {
        "label": args.label,
        "aggregate_groups": len(rows),
        "sequential_groups": sequential_groups,
        "represented_cells": total_cells,
        "external_output_load_ff": total_external_load_ff,
        "output": str(args.output),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
