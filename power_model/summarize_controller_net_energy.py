#!/usr/bin/env python3
"""Combine exact RTL sleep episodes, HSPICE event energy, and controller power."""

import argparse
import json
import re
from pathlib import Path


TRACE_PREFIX = "CONTROLLER_REPLAY_TRACE "
EPISODE_PREFIX = "CONTROLLER_SLEEP_EPISODE "
SUMMARY_PREFIX = "CONTROLLER_REPLAY_SUMMARY "
CORE_RE = re.compile(r"core(\d+)")
TOTAL_RE = re.compile(r"^Total\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)")
PAIR_GATE_RE = re.compile(r"^CONTROLLER_PAIR core=(\d+) stage=gate_replay\b")


def load_curve(path):
    curve = json.loads(path.read_text())
    points = sorted(
        (float(p["sleep_plateau_ns"]),
         float(p["event_vdd_energy_fj"]) +
         float(p["event_control_energy_fj"]))
        for p in curve["points"])
    return curve, points


def event_energy_fj(duration_ns, curve, points, short_episode_policy="error"):
    if duration_ns < points[0][0]:
        if short_episode_policy == "clamp-min":
            # Conservative: charge the complete shortest measured event even
            # though the requested OFF plateau is shorter.
            return points[0][1]
        raise ValueError(
            f"sleep episode {duration_ns} ns is below measured curve minimum")
    if duration_ns >= points[-1][0]:
        x0, y0 = points[-1]
        return y0 + (duration_ns - x0) * float(
            curve["steady_off_power_w_per_pe"]) * 1e6
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if duration_ns <= x1:
            return y0 + (duration_ns - x0) * (y1 - y0) / (x1 - x0)
    raise AssertionError("unreachable interpolation interval")


def core_from_text(text):
    match = CORE_RE.search(text)
    if not match:
        raise ValueError(f"cannot determine core from: {text}")
    return int(match.group(1))


def load_replay(path):
    cores = {}
    current = None
    for raw in path.read_text(errors="replace").splitlines():
        pair_match = PAIR_GATE_RE.match(raw)
        if pair_match:
            current = int(pair_match.group(1))
            if current in cores:
                raise ValueError(f"duplicate replay for core {current}")
            cores[current] = {
                "episodes": [], "summary": None,
                "annotation_coverage_complete": False,
            }
        elif raw.startswith(TRACE_PREFIX):
            current = core_from_text(raw[len(TRACE_PREFIX):])
            if current in cores:
                raise ValueError(f"duplicate replay for core {current}")
            cores[current] = {
                "episodes": [], "summary": None,
                "annotation_coverage_complete": False,
            }
        elif raw.startswith(EPISODE_PREFIX):
            if current is None:
                raise ValueError("episode appears before replay trace")
            payload = json.loads(raw[len(EPISODE_PREFIX):])
            cores[current]["episodes"].append(int(payload["duration_cycles"]))
        elif raw.startswith(SUMMARY_PREFIX):
            if current is None:
                raise ValueError("summary appears before replay trace")
            cores[current]["summary"] = json.loads(raw[len(SUMMARY_PREFIX):])
        elif re.match(r"^unannotated\s+0(?:\s|$)", raw):
            if current is None:
                raise ValueError("annotation result appears before replay trace")
            cores[current]["annotation_coverage_complete"] = True

    if not cores:
        raise ValueError("no controller replay records found")
    for core, data in cores.items():
        summary = data["summary"]
        if summary is None:
            raise ValueError(f"missing replay summary for core {core}")
        if summary["safety_violations"] != 0:
            raise ValueError(f"safety violation in core {core}")
        if summary["open_sleep_episode_cycles"] != 0:
            raise ValueError(f"incomplete sleep episode at end of core {core}")
        if summary["completed_sleep_episodes"] != len(data["episodes"]):
            raise ValueError(f"episode-count mismatch in core {core}")
        if summary["sleep_cycles"] != sum(data["episodes"]):
            raise ValueError(f"sleep-cycle mismatch in core {core}")
    return cores


def load_controller_power(report_dir, cores, report_stem):
    powers = {}
    for core in cores:
        prefix = f"{report_stem}_core{core}_tt"
        log = report_dir / f"{prefix}.power.log"
        report = report_dir / f"{prefix}.power.rpt"
        log_text = log.read_text(errors="replace") if log.exists() else ""
        standalone_coverage = bool(re.search(
            r"^unannotated\s+0(?:\s|$)", log_text, re.MULTILINE))
        replay_coverage = bool(cores[core]["annotation_coverage_complete"])
        if not standalone_coverage and not replay_coverage:
            raise ValueError(f"controller core {core} is not 100% annotated")
        totals = [TOTAL_RE.match(line) for line in report.read_text().splitlines()]
        totals = [match for match in totals if match]
        if len(totals) != 1:
            raise ValueError(f"expected one Total row for controller core {core}")
        internal, switching, leakage, total = map(float, totals[0].groups())
        if abs((internal + switching + leakage) - total) > max(1e-12, total * 1e-6):
            raise ValueError(f"power-component mismatch for controller core {core}")
        powers[core] = {
            "internal_w": internal,
            "switching_w": switching,
            "leakage_w": leakage,
            "total_w": total,
            "annotation_source": (
                "standalone_power_log" if standalone_coverage
                else "successful_worker_replay_log"),
        }
    return powers


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("replay_log", type=Path)
    parser.add_argument(
        "--curve", type=Path,
        default=base / "hspice_inputs/m300_duration_energy_tt25c.json")
    parser.add_argument(
        "--power-report-dir", type=Path,
        default=base / "controller_power_reports")
    parser.add_argument(
        "--power-report-stem", default="controller_opt125m",
        help="report filename stem before _coreN_tt")
    parser.add_argument(
        "--short-episode-policy", choices=("error", "clamp-min"),
        default="error",
        help=("handling below the shortest measured HSPICE duration; "
              "clamp-min conservatively charges the full minimum-duration "
              "event energy"))
    args = parser.parse_args()

    curve, points = load_curve(args.curve)
    replays = load_replay(args.replay_log)
    powers = load_controller_power(
        args.power_report_dir, replays, args.power_report_stem)
    frequency_hz = float(curve["frequency_ghz_for_cycle_mapping"]) * 1e9
    pe_count = int(curve["extrapolation"]["target_pe_count"])

    per_core = {}
    gross_sa_savings_j = 0.0
    controller_energy_j = 0.0
    short_episode_count = 0
    for core in sorted(replays):
        summary = replays[core]["summary"]
        episode_savings_fj_per_pe = []
        for duration_cycles in replays[core]["episodes"]:
            duration_ns = duration_cycles * 1e9 / frequency_hz
            if duration_ns < points[0][0]:
                short_episode_count += 1
            event = event_energy_fj(
                duration_ns, curve, points, args.short_episode_policy)
            reference_ns = duration_ns + float(curve["event_recovery_window_ns"])
            reference = float(curve["on_reference_power_w_per_pe"]) * reference_ns * 1e6
            episode_savings_fj_per_pe.append(reference - event)
        core_gross_j = sum(episode_savings_fj_per_pe) * pe_count * 1e-15
        core_controller_j = powers[core]["total_w"] * summary["cycles"] / frequency_hz
        gross_sa_savings_j += core_gross_j
        controller_energy_j += core_controller_j
        per_core[str(core)] = {
            "workload_cycles": summary["cycles"],
            "sleep_episodes": len(replays[core]["episodes"]),
            "sleep_cycles": summary["sleep_cycles"],
            "gross_sa_savings_j": core_gross_j,
            "controller_power_w": powers[core]["total_w"],
            "controller_annotation_source":
                powers[core]["annotation_source"],
            "controller_energy_j": core_controller_j,
            "net_savings_j": core_gross_j - core_controller_j,
        }

    output = {
        "cores": per_core,
        "aggregate": {
            "workload_cycles": sum(v["summary"]["cycles"] for v in replays.values()),
            "sleep_episodes": sum(len(v["episodes"]) for v in replays.values()),
            "sleep_cycles": sum(v["summary"]["sleep_cycles"] for v in replays.values()),
            "gross_sa_power_gating_savings_j": gross_sa_savings_j,
            "controller_energy_j": controller_energy_j,
            "net_savings_j": gross_sa_savings_j - controller_energy_j,
            "controller_fraction_of_gross_savings":
                (controller_energy_j / gross_sa_savings_j
                 if gross_sa_savings_j else None),
            "episodes_below_measured_curve": short_episode_count,
        },
        "method": {
            "sleep_timing": "exact synthesized-controller RTL replay episodes",
            "sa_event_energy": curve["description"],
            "controller_power": "100%-annotated ASAP7 TT gate-level workload activity",
            "controller_power_report_stem": args.power_report_stem,
            "controller_time": "full per-core trace at curve frequency",
            "short_episode_policy": args.short_episode_policy,
            "short_episode_policy_detail": (
                "below-minimum episodes charged the full shortest measured "
                "event energy" if args.short_episode_policy == "clamp-min"
                else "below-minimum episodes rejected"),
            "sa_scaling": curve["extrapolation"]["basis"],
            "limitations": list(curve["extrapolation"]["not_included"]) + [
                "controller clock-tree and routed-interconnect power",
                "delay-dependent controller glitch power",
            ],
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
