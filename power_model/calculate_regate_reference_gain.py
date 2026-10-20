#!/usr/bin/env python3
"""Apply ReGate-style delay/BET references to SA idle-episode histograms.

This is a sensitivity calculation, not a claim that ReGate's 7 nm hardware
parameters apply to the local design.  The energy-equivalent result is an
upper bound because an idle histogram plus BET does not reveal residual OFF
leakage or the local circuit's transition-energy curve.
"""

import argparse
import json
from pathlib import Path


def percentage(numerator, denominator):
    return 0.0 if denominator == 0 else 100.0 * numerator / denominator


def analyze_histogram(histogram, capacity_cycles, idle_cycles, delay, bet):
    # ReGate gates only when the interval is longer than BET and longer than
    # twice the power on/off delay (MICRO 2025, Section 4.3).
    minimum_exclusive = max(bet, 2 * delay)
    total_episodes = 0
    qualifying_episodes = 0
    qualifying_idle_cycles = 0
    break_even_compensated_cycles = 0
    reconstructed_idle_cycles = 0

    for item in histogram:
        length = int(item["length_cycles"])
        count = int(item["episode_count"])
        total_episodes += count
        reconstructed_idle_cycles += length * count
        if length > minimum_exclusive:
            qualifying_episodes += count
            qualifying_idle_cycles += length * count
            # BET is the duration at which transition cost and leakage saving
            # balance.  Subtracting BET gives an idealized energy-equivalent
            # duration beyond break-even, not measured Joules.
            break_even_compensated_cycles += (length - bet) * count

    if reconstructed_idle_cycles != idle_cycles:
        raise ValueError(
            "idle histogram does not reconstruct the reported idle cycles: "
            f"{reconstructed_idle_cycles} != {idle_cycles}"
        )

    return {
        "power_on_off_delay_cycles": delay,
        "break_even_cycles": bet,
        "qualification_rule": f"episode_length > max({bet}, 2*{delay})",
        "idle_episode_count": total_episodes,
        "qualifying_episode_count": qualifying_episodes,
        "qualifying_episode_percent": percentage(
            qualifying_episodes, total_episodes
        ),
        "qualifying_idle_cycles": qualifying_idle_cycles,
        "qualifying_idle_cycle_coverage_percent": percentage(
            qualifying_idle_cycles, idle_cycles
        ),
        "qualifying_idle_cycles_percent_of_capacity": percentage(
            qualifying_idle_cycles, capacity_cycles
        ),
        "break_even_compensated_cycles": break_even_compensated_cycles,
        "upper_bound_net_static_gain_percent": percentage(
            break_even_compensated_cycles, capacity_cycles
        ),
    }


def find_granularity(summary, size):
    for item in summary["granularities"]:
        if (
            int(item["region_size_rows"]) == size
            and int(item["region_size_columns"]) == size
        ):
            return item
    raise ValueError(f"summary does not contain {size}x{size} granularity")


def calculate(summary, pe_delay, pe_bet, full_delay, full_bet):
    if not summary.get("validation_pass", False):
        raise ValueError("source SA-region summary did not pass validation")
    if int(summary.get("version", 0)) < 3:
        raise ValueError(
            "source summary has no continuous idle-episode histograms; "
            "rerun with SA region profiler version 3"
        )

    array_rows = int(summary["array_rows"])
    array_columns = int(summary["array_columns"])
    if array_rows != array_columns:
        raise ValueError("current reference calculator requires a square SA")

    pe = find_granularity(summary, 1)
    full = find_granularity(summary, array_rows)
    pe_result = analyze_histogram(
        pe["idle_episode_length_histogram"],
        int(pe["capacity_region_cycles"]),
        int(pe["modeled_idle_region_cycles"]),
        pe_delay,
        pe_bet,
    )
    full_result = analyze_histogram(
        full["idle_episode_length_histogram"],
        int(full["capacity_region_cycles"]),
        int(full["modeled_idle_region_cycles"]),
        full_delay,
        full_bet,
    )

    sensitivity = []
    for item in summary["granularities"]:
        size = int(item["region_size_rows"])
        if size in (1, array_rows):
            continue
        sensitivity.append({
            "region_size": f"{size}x{size}",
            "region_count_per_core": int(item["region_count_per_core"]),
            "raw_idle_percent": float(item["modeled_idle_region_percent"]),
            "pe_reference_47_cycle_case": analyze_histogram(
                item["idle_episode_length_histogram"],
                int(item["capacity_region_cycles"]),
                int(item["modeled_idle_region_cycles"]),
                pe_delay,
                pe_bet,
            ),
            "full_sa_reference_469_cycle_case": analyze_histogram(
                item["idle_episode_length_histogram"],
                int(item["capacity_region_cycles"]),
                int(item["modeled_idle_region_cycles"]),
                full_delay,
                full_bet,
            ),
        })

    return {
        "version": 1,
        "source_validation_pass": True,
        "array_rows": array_rows,
        "array_columns": array_columns,
        "reference": {
            "paper": "ReGate: Enabling Power Gating in Neural Processing Units",
            "venue": "MICRO 2025",
            "table": 3,
            "warning": (
                "ReGate parameters are reference sensitivity points, not "
                "measurements of this ONNXim/ASAP7 design."
            ),
        },
        "pe_level": {
            "granularity": "1x1",
            "region_count_per_core": int(pe["region_count_per_core"]),
            "raw_idle_percent": float(pe["modeled_idle_region_percent"]),
            **pe_result,
        },
        "full_sa": {
            "granularity": f"{array_rows}x{array_columns}",
            "region_count_per_core": int(full["region_count_per_core"]),
            "raw_idle_percent": float(full["modeled_idle_region_percent"]),
            **full_result,
        },
        "intermediate_region_sensitivity": sensitivity,
        "interpretation": (
            "Raw idle percent is the zero-overhead theoretical ceiling under "
            "ONNXim's packed stationary-footprint model. Qualifying coverage "
            "applies ReGate's strict delay/BET rule. Upper-bound net static "
            "gain subtracts one BET per accepted episode and assumes the "
            "maximum possible ON-to-OFF leakage reduction; it is not Joules "
            "and excludes controller, switch, isolation, routing, and residual "
            "OFF leakage. Intermediate-region results are sensitivity bounds "
            "only because Table 3 provides no 8x8/16x16/32x32 BETs."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--pe-delay", type=int, default=1)
    parser.add_argument("--pe-bet", type=int, default=47)
    parser.add_argument("--full-delay", type=int, default=10)
    parser.add_argument("--full-bet", type=int, default=469)
    args = parser.parse_args()

    for name in ("pe_delay", "pe_bet", "full_delay", "full_bet"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")

    summary = json.loads(args.summary.read_text())
    result = calculate(
        summary,
        args.pe_delay,
        args.pe_bet,
        args.full_delay,
        args.full_bet,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
