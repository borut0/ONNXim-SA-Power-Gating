#!/usr/bin/env python3
"""Validate and aggregate final per-core SA_HYBRID_SUMMARY records."""

import argparse
import json
from pathlib import Path


PREFIX = "SA_HYBRID_SUMMARY "


def load_records(path):
    records = {}
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            marker = line.find(PREFIX)
            if marker < 0:
                continue
            try:
                record = json.loads(line[marker + len(PREFIX):])
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"malformed hybrid record at {path}:{line_number}: {error}"
                ) from error
            records[int(record["core_id"])] = record
    if not records:
        raise ValueError(f"no {PREFIX.strip()} records found in {path}")
    return records


def validate(core_id, record):
    if record.get("version") != 1:
        raise ValueError(f"core {core_id}: unsupported version")
    cycles = int(record["core_cycles"])
    state_sum = sum(
        int(record[field])
        for field in ("on_cycles", "retention_cycles", "off_cycles")
    )
    if state_sum != cycles:
        raise ValueError(f"core {core_id}: power-state cycles do not sum")
    if int(record["low_power_cycles"]) != (
        int(record["retention_cycles"]) + int(record["off_cycles"])
    ):
        raise ValueError(f"core {core_id}: low-power cycle mismatch")
    if int(record["causal_deadline_violations"]) != 0:
        raise ValueError(f"core {core_id}: causal deadline violation")
    energy_fields = (
        "baseline_on_idle_energy_j",
        "mode_idle_energy_j",
        "transition_energy_j",
        "controller_energy_j",
        "net_savings_j",
    )
    if record["energy_known"]:
        if any(record[field] is None for field in energy_fields):
            raise ValueError(f"core {core_id}: incomplete known energy")
        expected = (
            float(record["baseline_on_idle_energy_j"])
            - float(record["mode_idle_energy_j"])
            - float(record["transition_energy_j"])
            - float(record["controller_energy_j"])
        )
        observed = float(record["net_savings_j"])
        tolerance = max(1e-24, abs(expected) * 1e-9)
        if abs(expected - observed) > tolerance:
            raise ValueError(f"core {core_id}: net-energy arithmetic mismatch")
    elif any(record[field] is not None for field in energy_fields):
        raise ValueError(f"core {core_id}: unknown energy must remain null")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    records = load_records(args.log)
    for core_id, record in sorted(records.items()):
        validate(core_id, record)

    def total(field):
        return sum(int(record[field]) for record in records.values())

    core_cycles = total("core_cycles")
    low_power_cycles = total("low_power_cycles")
    energy_known = all(record["energy_known"] for record in records.values())
    result = {
        "validation_pass": True,
        "core_count": len(records),
        "mode": sorted({record["mode"] for record in records.values()}),
        "aggregate": {
            "core_cycles": core_cycles,
            "on_cycles": total("on_cycles"),
            "retention_cycles": total("retention_cycles"),
            "off_cycles": total("off_cycles"),
            "low_power_cycles": low_power_cycles,
            "low_power_percent": (
                100.0 * low_power_cycles / core_cycles if core_cycles else 0.0
            ),
            "retention_episodes": total("retention_episodes"),
            "off_episodes": total("off_episodes"),
            "wake_stall_cycles": total("wake_stall_cycles"),
            "causal_deadline_violations": total(
                "causal_deadline_violations"
            ),
            "energy_known": energy_known,
            "baseline_on_idle_energy_j": (
                sum(float(r["baseline_on_idle_energy_j"]) for r in records.values())
                if energy_known else None
            ),
            "mode_idle_energy_j": (
                sum(float(r["mode_idle_energy_j"]) for r in records.values())
                if energy_known else None
            ),
            "transition_energy_j": (
                sum(float(r["transition_energy_j"]) for r in records.values())
                if energy_known else None
            ),
            "controller_energy_j": (
                sum(float(r["controller_energy_j"]) for r in records.values())
                if energy_known else None
            ),
            "net_savings_j": (
                sum(float(r["net_savings_j"]) for r in records.values())
                if energy_known else None
            ),
        },
        "cores": {str(core): record for core, record in sorted(records.items())},
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
