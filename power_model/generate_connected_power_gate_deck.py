#!/usr/bin/env python3
"""Generate a connected ASAP7 SA power-gating HSPICE deck.

The mapped standard-cell graph is preserved exactly.  Each mapped cell is
instantiated from the extracted ASAP7 cell library with its VDD connected to
one switched virtual rail.  Only top-level design inputs are driven.  This is
deliberately different from the multiplicity-compressed leakage model, whose
fixed internal-pin sources make it unsuitable for rail-collapse transitions.

The extracted cell models include intra-cell parasitics.  No extra Liberty
fanout capacitance is added because doing so would double-count cell input-pin
capacitance.  Synthesis-stage interconnect/PDN and external isolation loads are
not represented and must be reported as limitations.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


CELL_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_$]*_ASAP7_75t_R)\s+(\S+)\s*\(\s*$"
)
CONNECTION_RE = re.compile(
    r"^\s*\.([A-Za-z_][A-Za-z0-9_$]*)\s*\((.*)\)\s*,?\s*$"
)
DECL_RE = re.compile(
    r"^\s*(input|output)\s*(?:\[\s*(\d+)\s*:\s*(\d+)\s*\])?\s*"
    r"([A-Za-z_][A-Za-z0-9_$]*)\s*;"
)
SUBCKT_RE = re.compile(r"^\s*\.subckt\s+(\S+)\s+(.*?)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class Instance:
    cell: str
    name: str
    connections: dict[str, str]


def parse_instances(path: Path) -> list[Instance]:
    instances = []
    cell = name = None
    connections = None
    with path.open(errors="replace") as stream:
        for line in stream:
            if name is None:
                match = CELL_RE.match(line)
                if match:
                    cell, name = match.groups()
                    connections = {}
                continue
            match = CONNECTION_RE.match(line)
            if match:
                pin, net = match.groups()
                connections[pin.upper()] = net.strip()
            if line.lstrip().startswith(");"):
                instances.append(Instance(cell, name.lstrip("\\"), connections))
                cell = name = connections = None
    if name is not None:
        raise RuntimeError(f"unterminated instance {name}")
    return instances


def parse_ports(path: Path):
    inputs = []
    outputs = []
    with path.open(errors="replace") as stream:
        for line in stream:
            match = DECL_RE.match(line)
            if not match:
                continue
            direction, msb, lsb, name = match.groups()
            if msb is None:
                bits = [name]
            else:
                hi, lo = int(msb), int(lsb)
                step = -1 if hi >= lo else 1
                bits = [f"{name}[{bit}]" for bit in range(hi, lo + step, step)]
            (inputs if direction == "input" else outputs).extend(bits)
    return inputs, outputs


def parse_subckts(path: Path):
    subckts = {}
    with path.open(errors="replace") as stream:
        for line in stream:
            match = SUBCKT_RE.match(line)
            if match:
                name, pins = match.groups()
                subckts[name.lower()] = (name, pins.split())
    return subckts


def spice_name(raw: str, mapping: dict[str, str]) -> str:
    raw = raw.strip()
    if raw in ("1'b0", "1'h0"):
        return "0"
    if raw in ("1'b1", "1'h1"):
        return "vvdd"
    if raw not in mapping:
        mapping[raw] = f"n{len(mapping):08d}"
    return mapping[raw]


def piecewise_source(name: str, node: str, points):
    pairs = [f"{time:.3f}n {voltage:.6f}" for time, voltage in points]
    lines = [f"V{name} {node} 0 PWL("]
    for index in range(0, len(pairs), 4):
        lines.append("+ " + " ".join(pairs[index:index + 4]))
    lines.append("+ )")
    return "\n".join(lines)


def clock_points(first_rise: float, last_fall: float, slew_ns: float, period=1.0):
    points = [(0.0, 0.0), (first_rise - slew_ns, 0.0)]
    rise = first_rise
    while rise + 0.5 <= last_fall + 1e-12:
        points.extend(
            [
                (rise, 0.7),
                (rise + 0.5 - slew_ns, 0.7),
                (rise + 0.5, 0.0),
            ]
        )
        rise += period
    points.append((last_fall + period, 0.0))
    return points


def static_bit_value(port: str, pattern: str) -> float:
    if pattern == "zero":
        return 0.0
    match = re.match(r"(activation_in|weight_in|psum_in)\[(\d+)\]", port)
    if not match:
        return 0.0
    bus, bit_text = match.groups()
    bit = int(bit_text)
    # Deterministic, nontrivial operands; values are stimuli, not model inputs.
    if bus == "activation_in":
        return 0.7 if bit % 5 in (0, 2) else 0.0
    if bus == "weight_in":
        return 0.7 if bit % 7 in (1, 3, 6) else 0.0
    return 0.7 if bit % 11 in (0, 4) else 0.0


def stress_bit_points(port: str, first_compute_rise: float,
                      compute_edges: int, stop: float):
    """Alternate activation/psum operands before every compute edge."""
    match = re.match(r"(activation_in|psum_in)\[(\d+)\]", port)
    if not match:
        return None
    _, bit_text = match.groups()
    current_high = int(bit_text) % 2 == 0
    points = [(0.0, 0.7 if current_high else 0.0)]
    for edge in range(compute_edges):
        transition_start = first_compute_rise - 0.25 + edge
        next_high = not current_high
        points.extend(
            [
                (transition_start, 0.7 if current_high else 0.0),
                (transition_start + 0.05, 0.7 if next_high else 0.0),
            ]
        )
        current_high = next_high
    points.append((stop, 0.7 if current_high else 0.0))
    return points


def generate(args):
    netlist = Path(args.netlist)
    pex = Path(args.pex)
    instances = parse_instances(netlist)
    inputs, outputs = parse_ports(netlist)
    subckts = parse_subckts(pex)

    if not instances:
        raise RuntimeError("no mapped ASAP7 instances found")
    required_cells = sorted({instance.cell for instance in instances})
    missing_cells = [cell for cell in required_cells if cell.lower() not in subckts]
    if missing_cells:
        raise RuntimeError(f"cells absent from extracted library: {missing_cells}")

    mapping = {}
    for port in inputs + outputs:
        base = port.replace("[", "_").replace("]", "")
        mapping[port] = f"p_{base}"

    signal_lines = []
    pin_errors = []
    for index, instance in enumerate(instances):
        subckt_name, pin_order = subckts[instance.cell.lower()]
        nodes = []
        for pin in pin_order:
            normalized = pin.upper()
            if normalized in ("VSS", "VGND"):
                nodes.append("0")
            elif normalized in ("VDD", "VPWR"):
                nodes.append("vvdd")
            elif normalized in instance.connections:
                nodes.append(spice_name(instance.connections[normalized], mapping))
            else:
                pin_errors.append(f"{instance.cell}/{instance.name}/{pin}")
        signal_lines.append(
            f"XG{index:06d} {' '.join(nodes)} {subckt_name}"
            f" $ original={instance.name}"
        )
    if pin_errors:
        raise RuntimeError(f"missing mapped pin connections: {pin_errors[:20]}")

    clock_slew_ns = 0.05
    if args.pattern in ("nonzero", "stress"):
        preload_edges = args.rows
        compute_edges = args.rows + args.cols + 4
    else:
        preload_edges = 0
        compute_edges = 0
    last_rise = (
        2.0 if args.pattern == "zero"
        else preload_edges + compute_edges + 1.0
    )
    last_fall = last_rise + 0.5
    quiet_start = last_fall + 1.5
    entry_start = quiet_start + 2.0
    entry_end = entry_start + args.sleep_slew_ps / 1000.0
    wake_start = entry_end + args.off_hold_ns
    wake_end = wake_start + args.sleep_slew_ps / 1000.0
    stop = (
        quiet_start + 1.0
        if args.mode == "active-stress"
        else wake_end + args.wake_hold_ns
    )

    stimulus = []
    stimulus.append(piecewise_source(
        "CLK", mapping["clk"], clock_points(1.0, last_fall, clock_slew_ns)
    ))
    if args.pattern == "zero":
        # A drained, non-retentive domain may be reset and have every boundary
        # input clamped low before its header opens.  Do this outside the entry
        # measurement window so the measured transition is only power gating.
        reset_low = quiet_start - 1.0
        reset_points = [
            (0.0, 0.0), (0.700, 0.0), (0.750, 0.7),
            (reset_low, 0.7), (reset_low + 0.05, 0.0), (stop, 0.0),
        ]
    else:
        reset_points = [(0.0, 0.0), (0.700, 0.0), (0.750, 0.7), (stop, 0.7)]
    stimulus.append(piecewise_source("RESETN", mapping["rst_n"], reset_points))
    if args.pattern in ("nonzero", "stress"):
        preload_off = preload_edges + 1.25
        compute_on = preload_off + 0.50
        compute_off = preload_edges + compute_edges + 1.25
        first_compute_rise = preload_edges + 2.0
        stimulus.append(piecewise_source(
            "WSHIFT", mapping["weight_shift_en"],
            [(0.0, 0.0), (1.70, 0.0), (1.75, 0.7),
             (preload_off, 0.7), (preload_off + 0.05, 0.0), (stop, 0.0)]
        ))
        stimulus.append(piecewise_source(
            "COMPUTE", mapping["compute_en"],
            [(0.0, 0.0), (compute_on, 0.0), (compute_on + 0.05, 0.7),
             (compute_off, 0.7), (compute_off + 0.05, 0.0), (stop, 0.0)]
        ))
    else:
        stimulus.append(f"VWSHIFT {mapping['weight_shift_en']} 0 0")
        stimulus.append(f"VCOMPUTE {mapping['compute_en']} 0 0")

    driven = {"clk", "rst_n", "weight_shift_en", "compute_en"}
    for index, port in enumerate(inputs):
        if port in driven:
            continue
        if args.pattern == "stress":
            points = stress_bit_points(
                port, first_compute_rise, compute_edges, stop
            )
            if points is not None:
                stimulus.append(piecewise_source(
                    f"IN{index:04d}", mapping[port], points
                ))
                continue
        value = static_bit_value(port, args.pattern)
        stimulus.append(f"VIN{index:04d} {mapping[port]} 0 {value:.1f}")

    if args.mode == "active-stress":
        stimulus.append("VSLEEP sleep 0 0")
    else:
        sleep_points = [
            (0.0, 0.0),
            (entry_start, 0.0),
            (entry_end, 0.7),
            (wake_start, 0.7),
            (wake_end, 0.0),
            (stop, 0.0),
        ]
        stimulus.append(piecewise_source("SLEEP", "sleep", sleep_points))

    on_from = entry_start - 1.0
    on_to = entry_start - 0.2
    off_from = wake_start - 1.0
    off_to = wake_start - 0.2
    active_from = active_to = None
    if args.mode == "active-stress":
        active_from = first_compute_rise - 0.10
        active_to = last_fall
        measures = [
            f".measure tran p_active_vdd AVG par('-v(vdd)*i(VDD_SRC)') FROM={active_from:.3f}n TO={active_to:.3f}n",
            f".measure tran i_active_peak MAX par('-i(VDD_SRC)') FROM={active_from:.3f}n TO={active_to:.3f}n",
            f".measure tran vvirtual_active_avg AVG v(vvdd) FROM={active_from:.3f}n TO={active_to:.3f}n",
            f".measure tran vvirtual_active_min MIN v(vvdd) FROM={active_from:.3f}n TO={active_to:.3f}n",
        ]
        energy_status = "not_applicable_active_stress"
    elif args.mode == "duration-sweep":
        event_end = stop
        measures = [
            f".measure tran p_on_vdd AVG par('-v(vdd)*i(VDD_SRC)') FROM={on_from:.3f}n TO={on_to:.3f}n",
            f".measure tran vvirtual_at_wake FIND v(vvdd) AT={wake_start:.6f}n",
            f".measure tran vvirtual_sleep_min MIN v(vvdd) FROM={entry_start:.6f}n TO={wake_start:.6f}n",
            f".measure tran vvirtual_recovered FIND v(vvdd) AT={event_end:.6f}n",
            f".measure tran wake_peak_current MAX par('-i(VDD_SRC)') FROM={wake_start:.6f}n TO={event_end:.6f}n",
            f".measure tran sleep_vdd_energy INTEG par('-v(vdd)*i(VDD_SRC)') FROM={entry_start:.6f}n TO={wake_start:.6f}n",
            f".measure tran sleep_control_energy INTEG par('abs(v(sleep)*i(VSLEEP))') FROM={entry_start:.6f}n TO={wake_start:.6f}n",
            f".measure tran recovery_vdd_energy INTEG par('-v(vdd)*i(VDD_SRC)') FROM={wake_start:.6f}n TO={event_end:.6f}n",
            f".measure tran recovery_control_energy INTEG par('abs(v(sleep)*i(VSLEEP))') FROM={wake_start:.6f}n TO={event_end:.6f}n",
            f".measure tran event_vdd_energy INTEG par('-v(vdd)*i(VDD_SRC)') FROM={entry_start:.6f}n TO={event_end:.6f}n",
            f".measure tran event_control_energy INTEG par('abs(v(sleep)*i(VSLEEP))') FROM={entry_start:.6f}n TO={event_end:.6f}n",
            ".measure tran wake_90_latency TRIG v(sleep) VAL=0.35 FALL=1 TARG v(vvdd) VAL=0.63 RISE=1",
            ".measure tran wake_95_latency TRIG v(sleep) VAL=0.35 FALL=1 TARG v(vvdd) VAL=0.665 RISE=1",
        ]
        energy_status = "fixed_duration_whole_event"
    else:
        measures = [
            f".measure tran p_on_vdd AVG par('-v(vdd)*i(VDD_SRC)') FROM={on_from:.3f}n TO={on_to:.3f}n",
            f".measure tran p_off_vdd AVG par('-v(vdd)*i(VDD_SRC)') FROM={off_from:.3f}n TO={off_to:.3f}n",
            f".measure tran vvirtual_off AVG v(vvdd) FROM={off_from:.3f}n TO={off_to:.3f}n",
            f".measure tran vvirtual_min MIN v(vvdd) FROM={entry_start:.3f}n TO={wake_start:.3f}n",
            ".measure tran entry_latency TRIG v(sleep) VAL=0.35 RISE=1 TARG v(vvdd) VAL=0.07 FALL=1",
            ".measure tran wake_latency TRIG v(sleep) VAL=0.35 FALL=1 TARG v(vvdd) VAL=0.63 RISE=1",
            f".measure tran wake_peak_current MAX par('-i(VDD_SRC)') FROM={wake_start:.3f}n TO={stop:.3f}n",
        ]
        energy_status = "latency_discovery_only"
    if (
        args.mode == "power-gate"
        and (args.entry_latency_ns is not None or args.wake_latency_ns is not None)
    ):
        if args.entry_latency_ns is None or args.wake_latency_ns is None:
            raise RuntimeError("both measured entry and wake latencies are required")
        # Latencies are measured from the 50% sleep-control crossing.  Include
        # the first half of the specified control slew in each raw-energy
        # window so integration begins at the physical control transition.
        half_slew_ns = args.sleep_slew_ps / 2000.0
        entry_energy_end = entry_start + half_slew_ns + args.entry_latency_ns
        wake_energy_end = wake_start + half_slew_ns + args.wake_latency_ns
        if entry_energy_end >= wake_start or wake_energy_end > stop:
            raise RuntimeError("energy windows exceed the simulated transition intervals")
        measures.extend(
            [
                f".measure tran entry_vdd_energy INTEG par('-v(vdd)*i(VDD_SRC)') FROM={entry_start:.6f}n TO={entry_energy_end:.6f}n",
                f".measure tran entry_control_energy INTEG par('abs(v(sleep)*i(VSLEEP))') FROM={entry_start:.6f}n TO={entry_energy_end:.6f}n",
                f".measure tran wake_vdd_energy INTEG par('-v(vdd)*i(VDD_SRC)') FROM={wake_start:.6f}n TO={wake_energy_end:.6f}n",
                f".measure tran wake_control_energy INTEG par('abs(v(sleep)*i(VSLEEP))') FROM={wake_start:.6f}n TO={wake_energy_end:.6f}n",
            ]
        )
        energy_status = "threshold_bounded_using_measured_latencies"

    deck = [
        f"* Connected {args.rows}x{args.cols} SA power-gating characterization",
        f"* pattern={args.pattern} header_m={args.header_m}",
        f".include '{args.model}'",
        f".include '{args.pex}'",
        f".option post={args.post} runlvl=5 accurate",
        ".temp 25",
        "VDD_SRC vdd 0 0.7",
        f"MSLEEP vvdd sleep vdd vdd pmos_rvt L=20n W=81n nfin=3 m={args.header_m}",
        *stimulus,
        "",
        "* Mapped connected standard-cell graph.",
        *signal_lines,
        "",
        f".tran {args.tran_step_ps:g}p {stop:.3f}n",
        *measures,
        ".end",
        "",
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(deck))
    summary = {
        "rows": args.rows,
        "cols": args.cols,
        "pattern": args.pattern,
        "mode": args.mode,
        "mapped_instances": len(instances),
        "mapped_cell_types": len(required_cells),
        "header_m": args.header_m,
        "sleep_slew_ps": args.sleep_slew_ps,
        "tran_step_ps": args.tran_step_ps,
        "post": args.post,
        "entry_latency_ns_for_energy": args.entry_latency_ns,
        "wake_latency_ns_for_energy": args.wake_latency_ns,
        "transition_energy_status": energy_status,
        "active_measurement_window_ns": (
            [active_from, active_to] if active_from is not None else None
        ),
        "sleep_plateau_ns": (
            args.off_hold_ns if args.mode == "duration-sweep" else None
        ),
        "sleep_command_50pct_duration_ns": (
            args.off_hold_ns + args.sleep_slew_ps / 1000.0
            if args.mode == "duration-sweep" else None
        ),
        "whole_event_measurement_window_ns": (
            [entry_start, stop] if args.mode == "duration-sweep" else None
        ),
        "entry_start_ns": entry_start,
        "wake_start_ns": wake_start,
        "stop_ns": stop,
        "top_inputs": len(inputs),
        "top_outputs": len(outputs),
        "missing_cells": missing_cells,
        "missing_pin_connections": len(pin_errors),
        "model_limitations": [
            "synthesis-stage interconnect and PDN parasitics excluded",
            "external output isolation and its load excluded",
            "lumped RVT PMOS header represents an equivalent switch network",
            "rail-only wake excludes architectural reset and weight reload",
        ],
        "output": str(output),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--netlist", required=True)
    parser.add_argument("--pex", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--header-m", type=int, required=True)
    parser.add_argument("--sleep-slew-ps", type=float, default=50.0)
    parser.add_argument("--off-hold-ns", type=float, default=10.0)
    parser.add_argument("--wake-hold-ns", type=float, default=5.0)
    parser.add_argument("--tran-step-ps", type=float, default=1.0)
    parser.add_argument("--post", type=int, choices=(0, 2), default=2)
    parser.add_argument("--entry-latency-ns", type=float)
    parser.add_argument("--wake-latency-ns", type=float)
    parser.add_argument("--mode", choices=("power-gate", "active-stress", "duration-sweep"),
                        default="power-gate")
    parser.add_argument("--pattern", choices=("zero", "nonzero", "stress"),
                        required=True)
    args = parser.parse_args()
    if (
        args.rows <= 0 or args.cols <= 0 or args.header_m <= 0
        or args.tran_step_ps <= 0
    ):
        parser.error("rows, cols, header-m, and tran-step-ps must be positive")
    if args.mode == "active-stress" and args.pattern != "stress":
        parser.error("active-stress mode requires --pattern stress")
    if args.mode != "active-stress" and args.pattern == "stress":
        parser.error("stress pattern requires --mode active-stress")
    if args.mode == "duration-sweep" and args.pattern != "zero":
        parser.error("duration-sweep mode requires --pattern zero")
    generate(args)


if __name__ == "__main__":
    main()
