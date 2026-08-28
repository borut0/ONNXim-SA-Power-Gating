#!/usr/bin/env python3
"""Evaluate exact RTL sleep episodes against the frozen SA HSPICE curve.

This intentionally excludes controller energy.  It is a cheap screening step
before workload-specific gate-level controller power characterization.
"""

import argparse
import json
from pathlib import Path

from summarize_controller_net_energy import event_energy_fj, load_curve, load_replay


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("replay_log", type=Path)
    parser.add_argument(
        "--curve", type=Path,
        default=base / "hspice_inputs/m300_duration_energy_tt25c.json")
    args = parser.parse_args()

    curve, points = load_curve(args.curve)
    replays = load_replay(args.replay_log)
    frequency_hz = float(curve["frequency_ghz_for_cycle_mapping"]) * 1e9
    pe_count = int(curve["extrapolation"]["target_pe_count"])
    on_power_w = float(curve["on_reference_power_w_per_pe"])
    recovery_ns = float(curve["event_recovery_window_ns"])

    gross_fj_per_pe = 0.0
    positive = 0
    nonpositive = 0
    below_curve = 0
    durations = []
    for data in replays.values():
        for duration_cycles in data["episodes"]:
            duration_ns = duration_cycles * 1e9 / frequency_hz
            durations.append(duration_cycles)
            if duration_ns < points[0][0]:
                below_curve += 1
            event_fj = event_energy_fj(
                duration_ns, curve, points, short_episode_policy="clamp-min")
            reference_fj = on_power_w * (duration_ns + recovery_ns) * 1e6
            saving_fj = reference_fj - event_fj
            gross_fj_per_pe += saving_fj
            if saving_fj > 0:
                positive += 1
            else:
                nonpositive += 1

    total_cycles = sum(data["summary"]["cycles"] for data in replays.values())
    sleep_cycles = sum(
        data["summary"]["sleep_cycles"] for data in replays.values())
    total_on_static_j = on_power_w * pe_count * total_cycles / frequency_hz
    gross_j = gross_fj_per_pe * pe_count * 1e-15

    output = {
        "core_count": len(replays),
        "total_core_cycles": total_cycles,
        "sleep_cycles": sleep_cycles,
        "sleep_percent_total": 100.0 * sleep_cycles / total_cycles,
        "sleep_episodes": len(durations),
        "average_episode_cycles": (
            sum(durations) / len(durations) if durations else 0.0),
        "minimum_episode_cycles": min(durations) if durations else None,
        "maximum_episode_cycles": max(durations) if durations else None,
        "positive_gross_energy_episodes": positive,
        "nonpositive_gross_energy_episodes": nonpositive,
        "episodes_below_measured_curve": below_curve,
        "gross_sa_savings_j": gross_j,
        "gross_percent_total_on_static_reference": (
            100.0 * gross_j / total_on_static_j if total_on_static_j else None),
        "controller_energy_included": False,
        "net_energy_known": False,
        "method": (
            "Exact synthesized-controller sleep episodes evaluated with the "
            "direct HSPICE duration-energy curve; below-minimum episodes are "
            "conservatively charged the full shortest measured event."),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
