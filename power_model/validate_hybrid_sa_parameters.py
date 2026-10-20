#!/usr/bin/env python3
"""Preflight a plug-and-play SA hybrid-policy technology parameter file."""

import argparse
import json
from pathlib import Path


STATE_NAMES = ("retention", "off")
ENERGY_FIELDS = ("idle_power_w", "entry_energy_j", "wake_energy_j")


def nonnegative_number(value, field, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    if integer and not isinstance(value, int):
        raise ValueError(f"{field} must be an integer number of cycles")
    if value < 0:
        raise ValueError(f"{field} must be nonnegative")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as stream:
        config = json.load(stream)

    if config.get("version") != 1:
        raise ValueError("version must be 1")
    if config.get("mode", "shadow") not in ("shadow", "closed_loop"):
        raise ValueError("mode must be shadow or closed_loop")
    minimum = nonnegative_number(
        config.get("minimum_causal_bound_cycles", 1),
        "minimum_causal_bound_cycles",
        integer=True,
    )
    if minimum == 0:
        raise ValueError("minimum_causal_bound_cycles must be at least one")

    enabled_count = 0
    effective_bounds = {}
    energy_complete = all(
        config.get(field) is not None
        for field in ("on_idle_power_w", "controller_power_w")
    )
    for field in ("on_idle_power_w", "controller_power_w"):
        if config.get(field) is not None:
            nonnegative_number(config[field], field)

    for name in STATE_NAMES:
        state = config.get(name)
        if not isinstance(state, dict):
            raise ValueError(f"missing object: {name}")
        enabled = state.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError(f"{name}.enabled must be boolean")
        enabled_count += int(enabled)
        entry = nonnegative_number(
            state.get("entry_latency_cycles", 0),
            f"{name}.entry_latency_cycles",
            integer=True,
        )
        wake = nonnegative_number(
            state.get("wake_latency_cycles", 0),
            f"{name}.wake_latency_cycles",
            integer=True,
        )
        break_even = nonnegative_number(
            state.get("break_even_cycles", 0),
            f"{name}.break_even_cycles",
            integer=True,
        )
        required = max(minimum, entry + wake + 1)
        if config.get("apply_break_even_filter", False):
            required = max(required, break_even)
        effective_bounds[name] = required if enabled else None
        for field in ENERGY_FIELDS:
            value = state.get(field)
            if value is not None:
                nonnegative_number(value, f"{name}.{field}")
        if enabled:
            energy_complete = energy_complete and all(
                state.get(field) is not None for field in ENERGY_FIELDS
            )

    if enabled_count == 0:
        raise ValueError("at least one low-power state must be enabled")
    print(json.dumps({
        "validation_pass": True,
        "mode": config.get("mode", "shadow"),
        "control_path_modified": config.get("mode", "shadow") == "closed_loop",
        "apply_break_even_filter": config.get("apply_break_even_filter", False),
        "effective_minimum_bound_cycles": effective_bounds,
        "energy_model_complete": energy_complete,
        "energy_claim_allowed": energy_complete,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
