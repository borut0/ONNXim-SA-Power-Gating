#!/usr/bin/env python3
"""Create a compact state-and-fanout-load population for ASAP7 cells.

Fanout load is the sum of nominal TT Liberty input-pin capacitances connected
to each mapped output net.  It intentionally excludes post-route wire/PDN
capacitance, which is unavailable in the synthesis-stage design.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


CELL_INSTANCE_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_$]*_ASAP7_75t_R)\s+(\S+)\s*\(\s*$"
)
CONNECTION_RE = re.compile(
    r"^\s*\.([A-Za-z_][A-Za-z0-9_$]*)\s*\((.*)\)\s*,?\s*$"
)
CELL_RE = re.compile(r"^\s*cell\s*\(([^)]+)\)\s*\{")
PIN_RE = re.compile(r"^\s*pin\s*\(([^)]+)\)\s*\{")
DIRECTION_RE = re.compile(r"^\s*direction\s*:\s*([A-Za-z_]+)\s*;")
CAP_RE = re.compile(
    r"^\s*capacitance\s*:\s*([-+0-9.eE]+)\s*;"
)


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", errors="replace")
    return path.open(errors="replace")


def parse_liberties(paths):
    pins: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for path in paths:
        depth = 0
        cell = None
        cell_parent_depth = None
        pin = None
        pin_parent_depth = None
        direction = None
        capacitance = None

        with open_text(path) as stream:
            for line in stream:
                cell_match = CELL_RE.match(line) if cell is None else None
                if cell_match:
                    cell = cell_match.group(1).strip()
                    cell_parent_depth = depth

                pin_match = PIN_RE.match(line) if cell and pin is None else None
                if pin_match:
                    pin = pin_match.group(1).strip()
                    pin_parent_depth = depth
                    direction = None
                    capacitance = None

                if pin is not None:
                    match = DIRECTION_RE.match(line)
                    if match:
                        direction = match.group(1).lower()
                    match = CAP_RE.match(line)
                    if match:
                        capacitance = float(match.group(1))

                depth += line.count("{") - line.count("}")

                if pin is not None and depth == pin_parent_depth:
                    entry = {
                        "direction": direction,
                        "capacitance_ff": capacitance,
                    }
                    previous = pins[cell].get(pin)
                    if previous is not None and previous != entry:
                        raise RuntimeError(f"conflicting Liberty pin {cell}/{pin}")
                    pins[cell][pin] = entry
                    pin = None
                    pin_parent_depth = None

                if cell is not None and depth == cell_parent_depth:
                    cell = None
                    cell_parent_depth = None

    return dict(pins)


def iter_instances(path: Path):
    cell = None
    instance = None
    connections = None
    with path.open(errors="replace") as stream:
        for line in stream:
            if instance is None:
                match = CELL_INSTANCE_RE.match(line)
                if match:
                    cell, instance = match.groups()
                    instance = instance.lstrip("\\")
                    connections = {}
                continue

            match = CONNECTION_RE.match(line)
            if match:
                pin, net = match.groups()
                connections[pin] = net.strip()
            if line.lstrip().startswith(");"):
                yield cell, instance, connections
                cell = None
                instance = None
                connections = None


def build_load_map(netlist: Path, liberty_pins):
    instance_to_cell = {}
    net_load_ff = defaultdict(float)
    net_driver_count = Counter()
    missing_cells = Counter()
    missing_input_caps = Counter()
    total_sink_cap_ff = 0.0

    for cell, instance, connections in iter_instances(netlist):
        instance_to_cell[instance] = cell
        metadata = liberty_pins.get(cell)
        if metadata is None:
            missing_cells[cell] += 1
            continue
        for pin, net in connections.items():
            info = metadata.get(pin)
            if info is None:
                continue
            if info["direction"] in ("input", "inout"):
                capacitance = info["capacitance_ff"]
                if capacitance is None:
                    missing_input_caps[(cell, pin)] += 1
                else:
                    net_load_ff[net] += capacitance
                    total_sink_cap_ff += capacitance
            if info["direction"] in ("output", "inout"):
                net_driver_count[net] += 1

    output_loads = {}
    total_attached_output_load_ff = 0.0
    for cell, instance, connections in iter_instances(netlist):
        metadata = liberty_pins.get(cell, {})
        loads = []
        for pin, net in connections.items():
            info = metadata.get(pin)
            if info and info["direction"] in ("output", "inout"):
                load = net_load_ff.get(net, 0.0)
                loads.append((pin, load))
                if net_driver_count[net] == 1:
                    total_attached_output_load_ff += load
        output_loads[instance] = tuple(sorted(loads))

    return {
        "instance_to_cell": instance_to_cell,
        "output_loads": output_loads,
        "missing_cells": missing_cells,
        "missing_input_caps": missing_input_caps,
        "total_sink_cap_ff": total_sink_cap_ff,
        "total_attached_output_load_ff": total_attached_output_load_ff,
        "multi_driver_nets": sum(1 for count in net_driver_count.values() if count > 1),
    }


def final_scalar_values(path: Path):
    values = {}
    in_events = False
    dump_directive = None
    with path.open(errors="replace") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not in_events:
                if line == "$enddefinitions $end":
                    in_events = True
                continue
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


def accumulate(path: Path, load_map, liberty_pins, final_values):
    states = Counter()
    scopes = []
    active_instance = None
    active_cell = None
    active_pins = {}
    observed = missing = unknown = 0

    def finish():
        nonlocal observed, missing, unknown
        if active_instance is None:
            return
        expected = tuple(sorted(liberty_pins[active_cell]))
        has_missing = any(pin not in active_pins for pin in expected)
        has_unknown = has_missing or any(
            active_pins.get(pin, "?") not in ("0", "1") for pin in expected
        )
        state = ";".join(f"{pin}={active_pins.get(pin, '?')}" for pin in expected)
        loads = ";".join(
            f"{pin}={load:.6f}"
            for pin, load in load_map["output_loads"].get(active_instance, ())
        )
        states[(active_cell, state, loads)] += 1
        observed += 1
        missing += int(has_missing)
        unknown += int(has_unknown)

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
                    and candidate in load_map["instance_to_cell"]
                ):
                    active_instance = candidate
                    active_cell = load_map["instance_to_cell"][candidate]
                    active_pins = {}
                continue
            if line == "$upscope $end":
                if active_instance is not None and scopes[-1] == active_instance:
                    finish()
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
                    if reference in liberty_pins[active_cell]:
                        active_pins[reference] = final_values.get(identifier, "?")

    return states, observed, missing, unknown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--netlist", required=True, type=Path)
    parser.add_argument("--vcd", required=True, type=Path)
    parser.add_argument("--liberty-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    liberty_paths = sorted(args.liberty_dir.glob("*.lib")) + sorted(
        args.liberty_dir.glob("*.lib.gz")
    )
    liberty_pins = parse_liberties(liberty_paths)
    load_map = build_load_map(args.netlist, liberty_pins)
    final_values = final_scalar_values(args.vcd)
    states, observed, missing, unknown = accumulate(
        args.vcd, load_map, liberty_pins, final_values
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("cell", "pin_state", "output_loads_ff", "count"))
        for (cell, state, loads), count in sorted(states.items()):
            writer.writerow((cell, state, loads, count))

    summary = {
        "label": args.label,
        "liberty_files": len(liberty_paths),
        "liberty_cell_types": len(liberty_pins),
        "mapped_instances": len(load_map["instance_to_cell"]),
        "observed_instances": observed,
        "aggregate_state_load_groups": len(states),
        "missing_pin_instances": missing,
        "unknown_instances": unknown,
        "missing_liberty_cell_types": dict(load_map["missing_cells"]),
        "missing_input_capacitances": {
            f"{cell}/{pin}": count
            for (cell, pin), count in load_map["missing_input_caps"].items()
        },
        "multi_driver_nets": load_map["multi_driver_nets"],
        "total_sink_pin_capacitance_ff": load_map["total_sink_cap_ff"],
        "total_attached_output_load_ff": load_map[
            "total_attached_output_load_ff"
        ],
        "primary_input_or_undriven_load_ff": (
            load_map["total_sink_cap_ff"]
            - load_map["total_attached_output_load_ff"]
        ),
        "output": str(args.output),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
