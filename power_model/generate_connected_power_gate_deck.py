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


def spice_name(raw: str, mapping: dict[str, str], high_node: str = "vvdd") -> str:
    raw = raw.strip()
    if raw in ("1'b0", "1'h0"):
        return "0"
    if raw in ("1'b1", "1'h1"):
        return high_node
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


def virtual_supply_lines(supply_mode: str, header_m: int,
                         sleep_gate_node: str = "sleep") -> list[str]:
    if supply_mode == "direct":
        return ["VDD_SRC vvdd 0 0.7"]
    if supply_mode == "header":
        return [
            "VDD_SRC vdd 0 0.7",
            f"MSLEEP vvdd {sleep_gate_node} vdd vdd pmos_rvt "
            f"L=20n W=81n nfin=3 m={header_m}",
        ]
    raise ValueError(f"unsupported supply mode: {supply_mode}")


def sleep_driver_lines(driver_cells: list[str], subckts: dict) -> tuple[list[str], str]:
    """Instantiate a real VDD-powered non-inverting sleep-control chain.

    The PWL source drives only the first cell input.  Every driver's internal
    and output energy is consequently drawn through VDD_SRC.  No default chain
    is selected: callers must name every extracted cell explicitly.
    """
    if not driver_cells:
        return [], "sleep"
    lines = []
    previous = "sleep_cmd"
    for index, requested_name in enumerate(driver_cells):
        record = subckts.get(requested_name.lower())
        if record is None:
            raise RuntimeError(
                f"sleep-driver cell absent from extracted library: {requested_name}"
            )
        subckt_name, pin_order = record
        output = "sleep" if index == len(driver_cells) - 1 else f"sleep_drv_{index}"
        nodes = []
        for pin in pin_order:
            normalized = pin.upper()
            if normalized in ("VSS", "VGND"):
                nodes.append("0")
            elif normalized in ("VDD", "VPWR"):
                nodes.append("vdd")
            elif normalized == "A":
                nodes.append(previous)
            elif normalized == "Y":
                nodes.append(output)
            else:
                raise RuntimeError(
                    f"unsupported sleep-driver pin {pin} on {subckt_name}"
                )
        lines.append(f"XSLPDRV{index:02d} {' '.join(nodes)} {subckt_name}")
        previous = output
    return lines, "sleep_cmd"


def instance_supply_node(instance_name: str,
                         always_on_instances: set[str]) -> str:
    return "vdd" if instance_name in always_on_instances else "vvdd"


def clock_points(first_rise: float, last_fall: float, slew_ns: float, period=1.0):
    points = [(0.0, 0.0), (first_rise - slew_ns, 0.0)]
    rise = first_rise
    while rise + 0.5 <= last_fall + 1e-12:
        # Preserve the LOW plateau until the requested slew immediately before
        # every rising edge.  Without this point, SPICE linearly interpolates
        # from the preceding falling edge to the next rise across the entire
        # half-cycle, creating an unintended slow clock ramp.
        if rise > first_rise:
            points.append((rise - slew_ns, 0.0))
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


def signed_bits(value: int, width: int) -> list[int]:
    encoded = value & ((1 << width) - 1)
    return [(encoded >> bit) & 1 for bit in range(width)]


def functional_bit_points(port: str, stop: float, change_start: float,
                          change_end: float):
    """Reproduce the signed 1x1 sequence in rtl/tb_ws_pe.sv."""
    match = re.match(r"(activation_in|weight_in|psum_in)\[(\d+)\]", port)
    if not match:
        return None
    bus, bit_text = match.groups()
    bit = int(bit_text)
    if bus == "weight_in":
        initial = signed_bits(-3, 16)[bit]
        return [(0.0, 0.7 * initial), (stop, 0.7 * initial)]
    if bus == "activation_in":
        first, second = signed_bits(7, 16)[bit], signed_bits(-4, 16)[bit]
    else:
        first, second = signed_bits(100, 32)[bit], signed_bits(-20, 32)[bit]
    return [
        (0.0, 0.7 * first),
        (change_start, 0.7 * first),
        (change_end, 0.7 * second),
        (stop, 0.7 * second),
    ]


def functional_sequence_bit_points(port: str, stop: float,
                                   first_compute_rise: float,
                                   period_ns: float, weight: int,
                                   vectors: list[tuple[int, int]]):
    """Drive a fixed weight and an arbitrary activation/partial-sum sequence."""
    match = re.match(r"(activation_in|weight_in|psum_in)\[(\d+)\]", port)
    if not match:
        return None
    bus, bit_text = match.groups()
    bit = int(bit_text)
    width = 32 if bus == "psum_in" else 16
    if bus == "weight_in":
        level = 0.7 * signed_bits(weight, width)[bit]
        return [(0.0, level), (stop, level)]
    operand_index = 0 if bus == "activation_in" else 1
    bits = [signed_bits(vector[operand_index], width)[bit] for vector in vectors]
    points = [(0.0, 0.7 * bits[0])]
    for index, value in enumerate(bits[1:], start=1):
        transition_start = first_compute_rise + (index - 1) * period_ns + 0.05
        points.extend([
            (transition_start, 0.7 * bits[index - 1]),
            (transition_start + 0.05, 0.7 * value),
        ])
    points.append((stop, 0.7 * bits[-1]))
    return points


def signed_wrap(value: int, width: int) -> int:
    encoded = value & ((1 << width) - 1)
    sign = 1 << (width - 1)
    return encoded - (1 << width) if encoded & sign else encoded


def functional_sequence_expectations(weight: int,
                                     vectors: list[tuple[int, int]]) -> list[dict]:
    return [
        {
            "label": index,
            "activation": signed_wrap(activation, 16),
            "psum": signed_wrap(activation * weight + psum, 32),
        }
        for index, (activation, psum) in enumerate(vectors, start=1)
    ]


def functional_measures(mapping: dict[str, str], latency_cycles: int,
                        first_compute_rise: float,
                        period_ns: float) -> list[str]:
    if latency_cycles not in (1, 2):
        raise ValueError("functional latency must be one or two cycles")
    first_output_sample = first_compute_rise + period_ns - 0.10
    second_output_sample = first_compute_rise + 2 * period_ns - 0.10
    first_psum_sample = (
        first_output_sample if latency_cycles == 1 else second_output_sample
    )
    second_psum_sample = (
        second_output_sample if latency_cycles == 1
        else first_compute_rise + 3 * period_ns - 0.10
    )
    measures = []
    samples = (
        # Sample immediately before the following rising edge.  Unlike
        # the zero-delay RTL testbench, the extracted gate-level circuit needs
        # most of the cycle for clock-to-Q and combinational settling.
        ("a1", "activation_out", 16, first_output_sample),
        ("p1", "psum_out", 32, first_psum_sample),
        ("a2", "activation_out", 16, second_output_sample),
        ("p2", "psum_out", 32, second_psum_sample),
    )
    for label, bus, width, sample_ns in samples:
        for bit in range(width):
            measures.append(
                f".measure tran {label}_b{bit:02d} FIND "
                f"v({mapping[f'{bus}[{bit}]']}) AT={sample_ns:.3f}n"
            )
    return measures


def functional_sequence_measures(mapping: dict[str, str], latency_cycles: int,
                                 first_compute_rise: float, period_ns: float,
                                 vector_count: int) -> list[str]:
    if latency_cycles not in (1, 2):
        raise ValueError("functional latency must be one or two cycles")
    measures = []
    for index in range(1, vector_count + 1):
        activation_sample = first_compute_rise + index * period_ns - 0.10
        psum_sample = (
            first_compute_rise + (index - 1 + latency_cycles) * period_ns - 0.10
        )
        for label, bus, width, sample_ns in (
            (f"a{index}", "activation_out", 16, activation_sample),
            (f"p{index}", "psum_out", 32, psum_sample),
        ):
            for bit in range(width):
                measures.append(
                    f".measure tran {label}_b{bit:02d} FIND "
                    f"v({mapping[f'{bus}[{bit}]']}) AT={sample_ns:.3f}n"
                )
    return measures


def diagnostic_measures(nodes: list[str]) -> list[str]:
    """Sample explicitly requested SPICE nodes around the second input change."""
    times = (3.20, 3.40, 3.49, 3.51, 3.60, 3.80, 3.90, 3.99, 4.01, 4.20)
    measures = []
    for node_index, node in enumerate(nodes):
        for time_index, sample_ns in enumerate(times):
            measures.append(
                f".measure tran diag_n{node_index:02d}_t{time_index:02d} "
                f"FIND v({node}) AT={sample_ns:.3f}n"
            )
    return measures


def retained_state_measures(mapping: dict[str, str], nets: list[str],
                            before_ns: float, during_ns: float,
                            after_ns: float) -> list[str]:
    measures = []
    for index, net in enumerate(nets):
        node = mapping[net]
        for phase, sample_ns in (
            ("before", before_ns),
            ("during", during_ns),
            ("after", after_ns),
        ):
            measures.append(
                f".measure tran retain_{index:02d}_{phase} "
                f"FIND v({node}) AT={sample_ns:.6f}n"
            )
    return measures


def generate(args):
    netlist = Path(args.netlist)
    pex = Path(args.pex)
    instances = parse_instances(netlist)
    inputs, outputs = parse_ports(netlist)
    subckts = parse_subckts(pex)

    always_on_instances: set[str] = set()
    if args.always_on_instance_file:
        instance_file = Path(args.always_on_instance_file)
        always_on_instances = {
            line.strip()
            for line in instance_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }

    if not instances:
        raise RuntimeError("no mapped ASAP7 instances found")
    required_cells = sorted({instance.cell for instance in instances})
    instance_names = {instance.name for instance in instances}
    unknown_always_on = sorted(always_on_instances - instance_names)
    if unknown_always_on:
        raise RuntimeError(
            f"always-on instance names absent from netlist: {unknown_always_on}"
        )
    if always_on_instances and args.supply_mode != "header":
        raise RuntimeError("always-on instances require header supply mode")
    non_sequential_always_on = sorted(
        instance.name for instance in instances
        if instance.name in always_on_instances
        and not instance.cell.upper().startswith("DFF")
    )
    if non_sequential_always_on:
        raise RuntimeError(
            "always-on partition contains non-DFF cells: "
            f"{non_sequential_always_on}"
        )
    retained_nets = sorted(
        instance.connections["QN"]
        for instance in instances
        if instance.name in always_on_instances and "QN" in instance.connections
    )
    missing_cells = [cell for cell in required_cells if cell.lower() not in subckts]
    if missing_cells:
        raise RuntimeError(f"cells absent from extracted library: {missing_cells}")
    if args.sleep_driver_cell and args.supply_mode != "header":
        raise RuntimeError("sleep-driver cells require header supply mode")
    sleep_drivers, sleep_source_node = sleep_driver_lines(
        args.sleep_driver_cell, subckts
    )

    mapping = {}
    for port in inputs + outputs:
        base = port.replace("[", "_").replace("]", "")
        mapping[port] = f"p_{base}"

    signal_lines = []
    pin_errors = []
    for index, instance in enumerate(instances):
        subckt_name, pin_order = subckts[instance.cell.lower()]
        instance_supply = instance_supply_node(
            instance.name, always_on_instances
        )
        nodes = []
        for pin in pin_order:
            normalized = pin.upper()
            if normalized in ("VSS", "VGND"):
                nodes.append("0")
            elif normalized in ("VDD", "VPWR"):
                nodes.append(instance_supply)
            elif normalized in instance.connections:
                nodes.append(spice_name(
                    instance.connections[normalized], mapping, instance_supply
                ))
            else:
                pin_errors.append(f"{instance.cell}/{instance.name}/{pin}")
        signal_lines.append(
            f"XG{index:06d} {' '.join(nodes)} {subckt_name}"
            f" $ original={instance.name}"
        )
    if pin_errors:
        raise RuntimeError(f"missing mapped pin connections: {pin_errors[:20]}")

    functional_vectors = (
        args.functional_vectors
        if args.functional_vectors
        else [(7, 100), (-4, -20)]
    )
    functional_weight = args.functional_weight
    functional_expected = functional_sequence_expectations(
        functional_weight, functional_vectors
    )
    clock_slew_ns = 0.05
    if args.pattern in ("nonzero", "stress"):
        preload_edges = args.rows
        compute_edges = args.rows + args.cols + 4
    else:
        preload_edges = 0
        compute_edges = 0
    if args.pattern == "functional":
        preload_edges = 1
        compute_edges = len(functional_vectors)
        functional_period = args.functional_period_ns
        preload_rise = 1.0 + functional_period
        first_compute_rise = 1.0 + 2 * functional_period
        last_compute_rise = (
            first_compute_rise + (compute_edges - 1) * functional_period
        )
        last_rise = first_compute_rise + compute_edges * functional_period
    else:
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
        if args.mode in ("active-stress", "functional-check")
        else wake_end + args.wake_hold_ns
    )

    stimulus = []
    stimulus.append(piecewise_source(
        "CLK", mapping["clk"], clock_points(
            1.0, last_fall, clock_slew_ns,
            args.functional_period_ns if args.pattern == "functional" else 1.0,
        )
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
    if args.pattern in ("nonzero", "stress", "functional"):
        if args.pattern == "functional":
            preload_on = preload_rise - 0.30
            preload_off = preload_rise + 0.25
            compute_on = first_compute_rise - 0.25
            compute_off = last_compute_rise + 0.50
        else:
            preload_on = 1.70
            preload_off = preload_edges + 1.25
            compute_on = preload_off + 0.50
            compute_off = preload_edges + compute_edges + 1.25
            first_compute_rise = preload_edges + 2.0
        stimulus.append(piecewise_source(
            "WSHIFT", mapping["weight_shift_en"],
            [(0.0, 0.0), (preload_on, 0.0), (preload_on + 0.05, 0.7),
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
        if args.pattern == "functional":
            points = functional_sequence_bit_points(
                port, stop, first_compute_rise, args.functional_period_ns,
                functional_weight, functional_vectors,
            )
            if points is not None:
                stimulus.append(piecewise_source(
                    f"IN{index:04d}", mapping[port], points
                ))
                continue
        value = static_bit_value(port, args.pattern)
        stimulus.append(f"VIN{index:04d} {mapping[port]} 0 {value:.1f}")

    if args.mode in ("active-stress", "functional-check"):
        if args.supply_mode == "header":
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
        stimulus.append(piecewise_source("SLEEP", sleep_source_node, sleep_points))

    on_from = entry_start - 1.0
    on_to = entry_start - 0.2
    off_from = wake_start - 1.0
    off_to = wake_start - 0.2
    source_voltage_node = "vvdd" if args.supply_mode == "direct" else "vdd"
    active_from = active_to = None
    if args.mode == "active-stress":
        active_from = first_compute_rise - 0.10
        active_to = last_fall
        measures = [
            f".measure tran p_active_vdd AVG par('-v({source_voltage_node})*i(VDD_SRC)') FROM={active_from:.3f}n TO={active_to:.3f}n",
            f".measure tran i_active_peak MAX par('-i(VDD_SRC)') FROM={active_from:.3f}n TO={active_to:.3f}n",
            f".measure tran vvirtual_active_avg AVG v(vvdd) FROM={active_from:.3f}n TO={active_to:.3f}n",
            f".measure tran vvirtual_active_min MIN v(vvdd) FROM={active_from:.3f}n TO={active_to:.3f}n",
        ]
        energy_status = "not_applicable_active_stress"
    elif args.mode == "functional-check":
        active_from = preload_rise - 0.10
        active_to = (
            first_compute_rise
            + compute_edges * args.functional_period_ns + 0.20
        )
        measures = [
            f".measure tran vvirtual_functional_min MIN v(vvdd) FROM={active_from:.3f}n TO={active_to:.3f}n",
            *functional_sequence_measures(
                mapping, args.functional_latency_cycles,
                first_compute_rise, args.functional_period_ns,
                len(functional_vectors),
            ),
            *diagnostic_measures(args.diagnostic_node),
        ]
        energy_status = "not_applicable_functional_check"
    elif args.mode == "duration-sweep":
        event_end = stop
        measures = [
            f".measure tran p_on_vdd AVG par('-v(vdd)*i(VDD_SRC)') FROM={on_from:.3f}n TO={on_to:.3f}n",
            f".measure tran p_off_vdd AVG par('-v(vdd)*i(VDD_SRC)') FROM={off_from:.3f}n TO={off_to:.3f}n",
            f".measure tran vvirtual_off AVG v(vvdd) FROM={off_from:.3f}n TO={off_to:.3f}n",
            f".measure tran vvirtual_at_wake FIND v(vvdd) AT={wake_start:.6f}n",
            f".measure tran vvirtual_sleep_min MIN v(vvdd) FROM={entry_start:.6f}n TO={wake_start:.6f}n",
            f".measure tran vvirtual_recovered FIND v(vvdd) AT={event_end:.6f}n",
            f".measure tran wake_peak_current MAX par('-i(VDD_SRC)') FROM={wake_start:.6f}n TO={event_end:.6f}n",
            f".measure tran sleep_vdd_energy INTEG par('-v(vdd)*i(VDD_SRC)') FROM={entry_start:.6f}n TO={wake_start:.6f}n",
            f".measure tran sleep_control_energy INTEG par('abs(v({sleep_source_node})*i(VSLEEP))') FROM={entry_start:.6f}n TO={wake_start:.6f}n",
            f".measure tran recovery_vdd_energy INTEG par('-v(vdd)*i(VDD_SRC)') FROM={wake_start:.6f}n TO={event_end:.6f}n",
            f".measure tran recovery_control_energy INTEG par('abs(v({sleep_source_node})*i(VSLEEP))') FROM={wake_start:.6f}n TO={event_end:.6f}n",
            f".measure tran event_vdd_energy INTEG par('-v(vdd)*i(VDD_SRC)') FROM={entry_start:.6f}n TO={event_end:.6f}n",
            f".measure tran event_control_energy INTEG par('abs(v({sleep_source_node})*i(VSLEEP))') FROM={entry_start:.6f}n TO={event_end:.6f}n",
            ".measure tran entry_10_latency TRIG v(sleep) VAL=0.35 RISE=1 TARG v(vvdd) VAL=0.07 FALL=1",
            ".measure tran wake_90_latency TRIG v(sleep) VAL=0.35 FALL=1 TARG v(vvdd) VAL=0.63 RISE=1",
            ".measure tran wake_95_latency TRIG v(sleep) VAL=0.35 FALL=1 TARG v(vvdd) VAL=0.665 RISE=1",
            *retained_state_measures(
                mapping, retained_nets,
                entry_start - 0.10, wake_start - 0.10, stop - 0.10,
            ),
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
            *retained_state_measures(
                mapping, retained_nets,
                entry_start - 0.10, wake_start - 0.10, stop - 0.10,
            ),
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
                f".measure tran entry_control_energy INTEG par('abs(v({sleep_source_node})*i(VSLEEP))') FROM={entry_start:.6f}n TO={entry_energy_end:.6f}n",
                f".measure tran wake_vdd_energy INTEG par('-v(vdd)*i(VDD_SRC)') FROM={wake_start:.6f}n TO={wake_energy_end:.6f}n",
                f".measure tran wake_control_energy INTEG par('abs(v({sleep_source_node})*i(VSLEEP))') FROM={wake_start:.6f}n TO={wake_energy_end:.6f}n",
            ]
        )
        energy_status = "threshold_bounded_using_measured_latencies"

    deck = [
        f"* Connected {args.rows}x{args.cols} SA power-gating characterization",
        (
            f"* pattern={args.pattern} supply_mode={args.supply_mode} "
            f"header_m={args.header_m if args.supply_mode == 'header' else 'not_applicable'}"
        ),
        f".include '{args.model}'",
        f".include '{args.pex}'",
        f".option post={args.post} runlvl=5 accurate",
        ".temp 25",
        *virtual_supply_lines(args.supply_mode, args.header_m),
        *sleep_drivers,
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
        "supply_mode": args.supply_mode,
        "mapped_instances": len(instances),
        "mapped_cell_types": len(required_cells),
        "header_m": args.header_m if args.supply_mode == "header" else None,
        "always_on_instance_file": args.always_on_instance_file,
        "always_on_instances": sorted(always_on_instances),
        "always_on_instance_count": len(always_on_instances),
        "retained_state_nets": retained_nets,
        "retained_state_sample_ns": (
            {
                "before": entry_start - 0.10,
                "during": wake_start - 0.10,
                "after": stop - 0.10,
            }
            if args.mode in ("power-gate", "duration-sweep") and retained_nets
            else None
        ),
        "sleep_slew_ps": args.sleep_slew_ps,
        "sleep_driver_cells": args.sleep_driver_cell,
        "sleep_driver_energy_accounting": (
            "driver internal/output energy included through VDD_SRC; ideal PWL source drives only the first driver input"
            if args.sleep_driver_cell else
            "ideal PWL source directly drives the header gate"
        ),
        "tran_step_ps": args.tran_step_ps,
        "post": args.post,
        "entry_latency_ns_for_energy": args.entry_latency_ns,
        "wake_latency_ns_for_energy": args.wake_latency_ns,
        "functional_latency_cycles": (
            args.functional_latency_cycles
            if args.mode == "functional-check" else None
        ),
        "functional_period_ns": (
            args.functional_period_ns
            if args.mode == "functional-check" else None
        ),
        "functional_second_operand_transition_ns": (
            [first_compute_rise + 0.05, first_compute_rise + 0.10]
            if args.mode == "functional-check" and len(functional_vectors) > 1
            else None
        ),
        "functional_second_operand_settle_ns": (
            args.functional_period_ns - 0.10
            if args.mode == "functional-check" else None
        ),
        "transition_energy_status": energy_status,
        "active_measurement_window_ns": (
            [active_from, active_to] if active_from is not None else None
        ),
        "functional_expectations": (
            {
                "weight_signed": functional_weight,
                "vectors": functional_expected,
            }
            if args.mode == "functional-check" else None
        ),
        "diagnostic_nodes": args.diagnostic_node,
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
        "entry_start_ns": (
            entry_start
            if args.mode in ("power-gate", "duration-sweep") else None
        ),
        "wake_start_ns": (
            wake_start
            if args.mode in ("power-gate", "duration-sweep") else None
        ),
        "stop_ns": stop,
        "top_inputs": len(inputs),
        "top_outputs": len(outputs),
        "missing_cells": missing_cells,
        "missing_pin_connections": len(pin_errors),
        "model_limitations": [
            "synthesis-stage interconnect and PDN parasitics excluded",
            "external output isolation and its load excluded",
            "rail-only wake excludes architectural reset and weight reload",
        ] + (
            ["lumped RVT PMOS header represents an equivalent switch network"]
            if args.supply_mode == "header" else
            ["direct-supply diagnostic excludes the PMOS header"]
        ) + (
            ["listed weight-register cells use the unswitched VDD rail; all other mapped cells use the switched virtual rail"]
            if always_on_instances else []
        ),
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
    parser.add_argument(
        "--always-on-instance-file",
        help="text file containing exact mapped instance names tied to unswitched VDD",
    )
    parser.add_argument(
        "--supply-mode", choices=("header", "direct"), default="header",
        help="header uses the PMOS bank; direct is a functional diagnostic only",
    )
    parser.add_argument("--sleep-slew-ps", type=float, default=50.0)
    parser.add_argument(
        "--sleep-driver-cell", action="append", default=[],
        help=(
            "extracted non-inverting buffer cell in source-to-header order; "
            "repeat to build an explicit VDD-powered sleep-driver chain"
        ),
    )
    parser.add_argument("--off-hold-ns", type=float, default=10.0)
    parser.add_argument("--wake-hold-ns", type=float, default=5.0)
    parser.add_argument("--tran-step-ps", type=float, default=1.0)
    parser.add_argument("--post", type=int, choices=(0, 2), default=2)
    parser.add_argument("--entry-latency-ns", type=float)
    parser.add_argument("--wake-latency-ns", type=float)
    parser.add_argument(
        "--functional-latency-cycles", type=int, choices=(1, 2), default=2,
        help="registered MAC latency used only by functional-check mode",
    )
    parser.add_argument(
        "--functional-period-ns", type=float, default=1.0,
        help="clock period used only by functional-check mode",
    )
    parser.add_argument(
        "--functional-weight", type=int, default=-3,
        help="fixed signed 16-bit weight used by functional-check mode",
    )
    parser.add_argument(
        "--functional-vector", action="append", default=[],
        metavar="ACTIVATION,PSUM",
        help=(
            "signed functional operand pair; repeat to create a sequence "
            "(default: 7,100 then -4,-20)"
        ),
    )
    parser.add_argument(
        "--diagnostic-node", action="append", default=[],
        help="SPICE node to sample around 3.5-4.0 ns; repeat as needed",
    )
    parser.add_argument("--mode", choices=("power-gate", "active-stress", "duration-sweep", "functional-check"),
                        default="power-gate")
    parser.add_argument("--pattern", choices=("zero", "nonzero", "stress", "functional"),
                        required=True)
    args = parser.parse_args()
    args.functional_vectors = []
    for item in args.functional_vector:
        try:
            activation_text, psum_text = item.split(",", 1)
            activation, psum = int(activation_text), int(psum_text)
        except ValueError:
            parser.error(
                f"invalid --functional-vector {item!r}; expected ACTIVATION,PSUM"
            )
        if not -(1 << 15) <= activation < (1 << 15):
            parser.error(f"functional activation {activation} is outside signed 16-bit range")
        if not -(1 << 31) <= psum < (1 << 31):
            parser.error(f"functional psum {psum} is outside signed 32-bit range")
        args.functional_vectors.append((activation, psum))
    if not -(1 << 15) <= args.functional_weight < (1 << 15):
        parser.error("functional weight is outside signed 16-bit range")
    if (
        args.rows <= 0 or args.cols <= 0 or args.header_m <= 0
        or args.tran_step_ps <= 0 or args.functional_period_ns <= 0
    ):
        parser.error("rows, cols, header-m, and tran-step-ps must be positive")
    if args.mode == "active-stress" and args.pattern != "stress":
        parser.error("active-stress mode requires --pattern stress")
    if args.mode == "functional-check" and args.pattern != "functional":
        parser.error("functional-check mode requires --pattern functional")
    if args.mode == "functional-check" and args.functional_period_ns <= 0.10:
        parser.error("functional-check period must exceed the 0.10 ns input transition")
    if args.mode != "active-stress" and args.pattern == "stress":
        parser.error("stress pattern requires --mode active-stress")
    if args.mode != "functional-check" and args.pattern == "functional":
        parser.error("functional pattern requires --mode functional-check")
    if args.mode == "duration-sweep" and args.pattern not in ("zero", "nonzero"):
        parser.error("duration-sweep mode requires --pattern zero or nonzero")
    if args.supply_mode == "direct" and args.mode not in (
        "functional-check", "active-stress"
    ):
        parser.error(
            "direct supply is restricted to functional-check and active-stress"
        )
    generate(args)


if __name__ == "__main__":
    main()
