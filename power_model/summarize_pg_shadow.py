#!/usr/bin/env python3
import argparse
import json
import math
from collections import Counter
from pathlib import Path


PREFIX = "SA_IDLE_PREDICTOR "


def percentile(values, percentile_value):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values):
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "avg": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": max(values),
    }


def load_records(log_path):
    records = []
    malformed = 0
    with log_path.open(errors="replace") as source:
        for line in source:
            marker = line.find(PREFIX)
            if marker < 0:
                continue
            payload = line[marker + len(PREFIX):].strip()
            try:
                records.append(json.loads(payload))
            except json.JSONDecodeError:
                malformed += 1
    return records, malformed


def load_curve(curve_path):
    curve = json.loads(curve_path.read_text())
    points = []
    for point in curve["points"]:
        total = (point["event_vdd_energy_fj"] +
                 point["event_control_energy_fj"])
        points.append((float(point["sleep_plateau_ns"]), total))
    points.sort()
    return curve, points


def interpolate_event_energy_fj(duration_ns, curve, points):
    if duration_ns < points[0][0]:
        return None
    if duration_ns >= points[-1][0]:
        x0, y0 = points[-1]
        extra_ns = duration_ns - x0
        return y0 + curve["steady_off_power_w_per_pe"] * extra_ns * 1e6
    else:
        for index in range(1, len(points)):
            if duration_ns <= points[index][0]:
                x0, y0 = points[index - 1]
                x1, y1 = points[index]
                break
    ratio = (duration_ns - x0) / (x1 - x0)
    return y0 + ratio * (y1 - y0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument(
        "--curve", type=Path,
        default=Path(__file__).resolve().parent /
                "hspice_inputs/m300_duration_energy_tt25c.json")
    args = parser.parse_args()

    records, malformed = load_records(args.log)
    curve, points = load_curve(args.curve)
    versions = Counter(record.get("version") for record in records)
    v5 = [record for record in records if record.get("version", 0) >= 5]

    durations = []
    episode_savings_fj_per_pe = []
    unmodeled_short_episodes = 0
    total_episode_count = 0
    intervals_with_episodes = 0
    for record in v5:
        episodes = record.get("pg_shadow_episodes", [])
        if episodes:
            intervals_with_episodes += 1
        for episode in episodes:
            total_episode_count += 1
            duration_ns = float(episode["duration"])
            event_energy = interpolate_event_energy_fj(
                duration_ns, curve, points)
            if event_energy is None:
                unmodeled_short_episodes += 1
                continue
            reference_duration_ns = (
                duration_ns + curve["event_recovery_window_ns"])
            reference_energy = (
                curve["on_reference_power_w_per_pe"] *
                reference_duration_ns * 1e6)
            durations.append(duration_ns)
            episode_savings_fj_per_pe.append(reference_energy - event_energy)

    target_pe_count = curve["extrapolation"]["target_pe_count"]
    total_savings_fj_per_pe = sum(episode_savings_fj_per_pe)
    output = {
        "records": len(records),
        "malformed": malformed,
        "versions": dict(versions),
        "v5_records": len(v5),
        "baseline_safety_violations": sum(
            bool(record.get("safety_violation")) for record in v5),
        "work_drain_safety_violations": sum(
            bool(record.get("work_drain_safety_violation")) for record in v5),
        "retention_violations": sum(
            bool(record.get("pg_shadow_retention_violation")) for record in v5),
        "late_preload_commit_violations": sum(
            bool(record.get("pg_shadow_late_preload_commit")) for record in v5),
        "active_at_sa_activation_violations": sum(
            bool(record.get("pg_shadow_active_at_sa_activation"))
            for record in v5),
        "intervals_with_shadow_episodes": intervals_with_episodes,
        "episode_count": total_episode_count,
        "energy_modeled_episode_count": len(durations),
        "unmodeled_episodes_below_10_cycles": unmodeled_short_episodes,
        "episode_duration_cycles": distribution(durations),
        "positive_energy_episodes": sum(
            saving > 0 for saving in episode_savings_fj_per_pe),
        "negative_energy_episodes": sum(
            saving < 0 for saving in episode_savings_fj_per_pe),
        "total_direct_model_savings_fj_per_pe": total_savings_fj_per_pe,
        "total_extrapolated_128x128_savings_j":
            total_savings_fj_per_pe * target_pe_count * 1e-15,
        "stale_rolling_only_qualified_cycles": sum(
            record.get("pg_shadow_stale_only_qualified_cycles", 0)
            for record in v5),
        "method": {
            "episode_timing": "measurement-only fresh causal certificate",
            "energy_curve": curve["description"],
            "target_scaling": curve["extrapolation"]["basis"],
            "limitations": curve["extrapolation"]["not_included"],
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
