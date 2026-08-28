#!/usr/bin/env python3
"""Normalize measured SA power-gating savings against static-energy baselines.

The numerator comes from summarize_controller_net_energy.py: exact RTL sleep
episodes evaluated with the directly measured HSPICE duration-energy curve,
minus 100%-annotated controller energy for the net result.  Denominators use
the same HSPICE ON-state reference power, avoiding a mixed-model comparison
with OpenROAD ACTIVE/PRELOAD dynamic power.
"""

import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path

from summarize_controller_trace import summarize


def percent(numerator, denominator):
    return 100.0 * numerator / denominator if denominator else None


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("net_energy", type=Path)
    parser.add_argument("traces", nargs="*", type=Path)
    parser.add_argument(
        "--trace-summary", type=Path,
        help="existing summarize_controller_trace.py JSON output")
    parser.add_argument(
        "--curve", type=Path,
        default=base / "hspice_inputs/m300_duration_energy_tt25c.json")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--controller-area-um2", type=float)
    parser.add_argument("--sa-area-um2", type=float)
    args = parser.parse_args()

    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if bool(args.traces) == bool(args.trace_summary):
        parser.error("provide traces or --trace-summary, but not both")
    if (args.controller_area_um2 is None) != (args.sa_area_um2 is None):
        parser.error("provide both area arguments or neither")

    if args.trace_summary:
        trace_aggregate = json.loads(
            args.trace_summary.read_text())["aggregate"]
        traces = None
    elif args.jobs == 1:
        traces = [summarize(path) for path in args.traces]
    else:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=min(args.jobs, len(args.traces)),
                mp_context=multiprocessing.get_context("fork")) as executor:
            traces = list(executor.map(summarize, args.traces))

    energy = json.loads(args.net_energy.read_text())
    curve = json.loads(args.curve.read_text())
    aggregate = energy["aggregate"]

    if traces is None:
        total_cycles = int(trace_aggregate["cycles"])
        active_cycles = int(trace_aggregate["sa_active_cycles"])
        idle_cycles = int(trace_aggregate["sa_idle_cycles"])
        core_count = len(energy["cores"])
    else:
        total_cycles = sum(trace["cycles"] for trace in traces)
        active_cycles = sum(trace["sa_active_cycles"] for trace in traces)
        idle_cycles = sum(trace["sa_idle_cycles"] for trace in traces)
        core_count = len(traces)
    if active_cycles + idle_cycles != total_cycles:
        raise ValueError("SA active/idle cycles do not sum to total cycles")
    if aggregate["workload_cycles"] != total_cycles:
        raise ValueError("net-energy and controller-trace cycle totals differ")

    sleep_cycles = int(aggregate["sleep_cycles"])
    if sleep_cycles > idle_cycles:
        raise ValueError("sleep cycles exceed SA-idle cycles")

    frequency_hz = float(curve["frequency_ghz_for_cycle_mapping"]) * 1e9
    pe_count = int(curve["extrapolation"]["target_pe_count"])
    on_power_w = float(curve["on_reference_power_w_per_pe"]) * pe_count
    off_power_w = float(curve["steady_off_power_w_per_pe"]) * pe_count

    total_static_reference_j = on_power_w * total_cycles / frequency_hz
    idle_static_reference_j = on_power_w * idle_cycles / frequency_hz
    sleep_window_on_reference_j = on_power_w * sleep_cycles / frequency_hz
    ideal_sleep_window_delta_j = (
        (on_power_w - off_power_w) * sleep_cycles / frequency_hz)

    gross_j = float(aggregate["gross_sa_power_gating_savings_j"])
    controller_j = float(aggregate["controller_energy_j"])
    net_j = float(aggregate["net_savings_j"])
    if abs((gross_j - controller_j) - net_j) > 1e-15:
        raise ValueError("gross - controller does not equal net savings")

    output = {
        "scope": {
            "cores": core_count,
            "target_array_per_core": "128x128",
            "target_pe_count_per_core": pe_count,
            "frequency_hz": frequency_hz,
            "total_core_cycles": total_cycles,
            "sa_active_cycles": active_cycles,
            "sa_idle_cycles": idle_cycles,
            "power_gated_sleep_cycles": sleep_cycles,
            "sleep_episodes": aggregate["sleep_episodes"],
        },
        "direct_hspice_static_power": {
            "on_reference_w_per_pe":
                float(curve["on_reference_power_w_per_pe"]),
            "steady_off_w_per_pe":
                float(curve["steady_off_power_w_per_pe"]),
            "steady_off_reduction_percent": percent(
                on_power_w - off_power_w, on_power_w),
        },
        "extrapolated_128x128_static_power": {
            "on_reference_w_per_array": on_power_w,
            "steady_off_w_per_array": off_power_w,
            "scaling_basis": curve["extrapolation"]["basis"],
        },
        "opportunity": {
            "power_gated_percent_of_sa_idle_cycles":
                percent(sleep_cycles, idle_cycles),
            "power_gated_percent_of_total_core_cycles":
                percent(sleep_cycles, total_cycles),
        },
        "energy_j": {
            "total_execution_on_static_reference": total_static_reference_j,
            "sa_idle_on_static_reference": idle_static_reference_j,
            "sleep_window_on_static_reference": sleep_window_on_reference_j,
            "ideal_sleep_window_delta_without_transition_cost":
                ideal_sleep_window_delta_j,
            "measured_gross_sa_savings": gross_j,
            "measured_controller_cost": controller_j,
            "measured_net_savings": net_j,
        },
        "normalized_reduction_percent": {
            "gross_of_total_execution_static":
                percent(gross_j, total_static_reference_j),
            "net_of_total_execution_static":
                percent(net_j, total_static_reference_j),
            "gross_of_sa_idle_static":
                percent(gross_j, idle_static_reference_j),
            "net_of_sa_idle_static":
                percent(net_j, idle_static_reference_j),
            "gross_of_sleep_window_on_static":
                percent(gross_j, sleep_window_on_reference_j),
            "net_of_sleep_window_on_static":
                percent(net_j, sleep_window_on_reference_j),
            "realized_fraction_of_ideal_sleep_window_delta":
                percent(gross_j, ideal_sleep_window_delta_j),
            "controller_cost_as_percent_of_gross":
                percent(controller_j, gross_j),
        },
        "area": None,
        "claims": {
            "supported": [
                "steady OFF versus ON static-power reduction",
                "gross and controller-inclusive net reduction relative to "
                "the HSPICE static-energy reference",
                "power-gated cycle coverage of ONNXim SA-idle cycles",
            ],
            "not_supported_by_these_inputs": [
                "exact reduction of total SA energy including utilization-"
                "dependent ACTIVE and PRELOAD dynamic energy",
                "accelerator- or system-level energy reduction",
            ],
        },
    }

    if args.controller_area_um2 is not None:
        output["area"] = {
            "controller_area_um2": args.controller_area_um2,
            "extrapolated_128x128_sa_area_um2": args.sa_area_um2,
            "controller_percent_of_sa_area": percent(
                args.controller_area_um2, args.sa_area_um2),
        }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
