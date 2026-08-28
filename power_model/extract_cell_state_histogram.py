#!/usr/bin/env python3
"""Extract final gate-pin state populations from a mapped Verilog/VCD pair.

The script is intentionally streaming.  It scans the VCD event section once
to retain only each scalar identifier's final value, then scans only the VCD
header to associate those values with standard-cell instance ports.  It does
not emit per-instance or per-cycle data.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


CELL_INSTANCE_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_$]*_ASAP7_75t_R)\s+(\S+)\s*\(\s*$"
)
PORT_RE = re.compile(r"^\s*\.([A-Za-z_][A-Za-z0-9_$]*)\s*\(")


def parse_netlist(path: Path):
    instance_to_cell: dict[str, str] = {}
    cell_ports: dict[str, set[str]] = defaultdict(set)
    current_cell: str | None = None
    current_instance: str | None = None

    with path.open(errors="replace") as stream:
        for line in stream:
            if current_instance is None:
                match = CELL_INSTANCE_RE.match(line)
                if match:
                    current_cell, current_instance = match.groups()
                    # Verilog escaped identifiers include a leading '\\',
                    # while Icarus emits the corresponding VCD scope without
                    # that escape marker.
                    current_instance = current_instance.lstrip("\\")
                    instance_to_cell[current_instance] = current_cell
                continue

            port_match = PORT_RE.match(line)
            if port_match:
                cell_ports[current_cell].add(port_match.group(1))
            if line.lstrip().startswith(");"):
                current_cell = None
                current_instance = None

    return instance_to_cell, {
        cell: tuple(sorted(ports)) for cell, ports in cell_ports.items()
    }


def final_scalar_values(path: Path):
    values: dict[str, str] = {}
    in_events = False
    dump_directive: str | None = None

    with path.open(errors="replace") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not in_events:
                if line == "$enddefinitions $end":
                    in_events = True
                continue

            # A VCD $dumpoff block writes X for every signal.  Those X values
            # mean "recording disabled", not the circuit's electrical state,
            # so retain the last state from the preceding active dump window.
            if line == "$dumpoff":
                dump_directive = "off"
                continue
            if line in ("$dumpon", "$dumpvars"):
                dump_directive = "on"
                continue
            if line == "$end" and dump_directive is not None:
                dump_directive = None
                continue
            if dump_directive == "off":
                continue

            if len(line) >= 2 and line[0] in "01xXzZ":
                values[line[1:]] = line[0].lower()

    return values


def accumulate_states(path: Path, instance_to_cell, cell_ports, final_values):
    states: Counter[tuple[str, str]] = Counter()
    scopes: list[str] = []
    active_instance: str | None = None
    active_cell: str | None = None
    active_pins: dict[str, str] = {}
    observed_instances = 0
    missing_pin_instances = 0
    unknown_instances = 0

    def finish_instance():
        nonlocal observed_instances, missing_pin_instances, unknown_instances
        if active_instance is None or active_cell is None:
            return

        expected = cell_ports[active_cell]
        missing = any(pin not in active_pins for pin in expected)
        unknown = missing or any(
            active_pins.get(pin, "?") not in ("0", "1") for pin in expected
        )
        signature = ";".join(
            f"{pin}={active_pins.get(pin, '?')}" for pin in expected
        )
        states[(active_cell, signature)] += 1
        observed_instances += 1
        missing_pin_instances += int(missing)
        unknown_instances += int(unknown)

    with path.open(errors="replace") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if line == "$enddefinitions $end":
                break

            if line.startswith("$scope "):
                fields = line.split()
                scopes.append(fields[2])
                candidate = scopes[-1]
                if (
                    len(scopes) >= 2
                    and scopes[-2] == "dut"
                    and candidate in instance_to_cell
                ):
                    active_instance = candidate
                    active_cell = instance_to_cell[candidate]
                    active_pins = {}
                continue

            if line == "$upscope $end":
                if active_instance is not None and scopes[-1] == active_instance:
                    finish_instance()
                    active_instance = None
                    active_cell = None
                    active_pins = {}
                if scopes:
                    scopes.pop()
                continue

            if active_instance is not None and line.startswith("$var "):
                fields = line.split()
                if len(fields) >= 6 and fields[2] == "1":
                    identifier = fields[3]
                    reference = fields[4]
                    if reference in cell_ports[active_cell]:
                        active_pins[reference] = final_values.get(identifier, "?")

    return {
        "states": states,
        "observed_instances": observed_instances,
        "missing_pin_instances": missing_pin_instances,
        "unknown_instances": unknown_instances,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--netlist", required=True, type=Path)
    parser.add_argument("--vcd", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    instance_to_cell, cell_ports = parse_netlist(args.netlist)
    final_values = final_scalar_values(args.vcd)
    result = accumulate_states(
        args.vcd, instance_to_cell, cell_ports, final_values
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("cell", "pin_state", "count"))
        for (cell, signature), count in sorted(result["states"].items()):
            writer.writerow((cell, signature, count))

    summary = {
        "label": args.label,
        "mapped_instances": len(instance_to_cell),
        "observed_instances": result["observed_instances"],
        "unique_cell_types": len(cell_ports),
        "unique_cell_states": len(result["states"]),
        "missing_pin_instances": result["missing_pin_instances"],
        "unknown_instances": result["unknown_instances"],
        "final_scalar_identifiers": len(final_values),
        "output": str(args.output),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
