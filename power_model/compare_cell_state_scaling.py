#!/usr/bin/env python3
"""Compare 16x16 and 32x32 final cell-state population scaling."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def load(path: Path):
    states: Counter[tuple[str, str]] = Counter()
    cells: Counter[str] = Counter()
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["cell"], row["pin_state"])
            count = int(row["count"])
            states[key] += count
            cells[row["cell"]] += count
    return states, cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states16", required=True, type=Path)
    parser.add_argument("--states32", required=True, type=Path)
    args = parser.parse_args()

    state16, cell16 = load(args.states16)
    state32, cell32 = load(args.states32)
    all_states = set(state16) | set(state32)
    all_cells = set(cell16) | set(cell32)
    total16 = sum(state16.values())
    total32 = sum(state32.values())

    # Difference from exact fourfold PE-count scaling.
    l1_fourfold = sum(abs(state32[s] - 4 * state16[s]) for s in all_states)

    # Population-weighted total-variation distance between the two conditional
    # pin-state distributions for each cell type.
    weighted_tv = 0.0
    per_cell_tv = []
    for cell in all_cells:
        keys = [state for state in all_states if state[0] == cell]
        if not cell16[cell] or not cell32[cell]:
            tv = 1.0
        else:
            tv = 0.5 * sum(
                abs(
                    state16[key] / cell16[cell]
                    - state32[key] / cell32[cell]
                )
                for key in keys
            )
        weighted_tv += tv * cell32[cell] / total32
        per_cell_tv.append((tv, cell, cell16[cell], cell32[cell]))

    # Diagnose whether the two-point structural formula is meaningful at the
    # individual pin-state level.  Negative counts make it unusable directly.
    extrapolated = {s: 21 * state32[s] - 20 * state16[s] for s in all_states}
    negative = {s: n for s, n in extrapolated.items() if n < 0}

    summary = {
        "instances16": total16,
        "instances32": total32,
        "states16": len(state16),
        "states32": len(state32),
        "state_union": len(all_states),
        "states_only16": sum(1 for s in all_states if state16[s] and not state32[s]),
        "states_only32": sum(1 for s in all_states if state32[s] and not state16[s]),
        "fourfold_state_L1_over_instances32": l1_fourfold / total32,
        "weighted_state_total_variation": weighted_tv,
        "negative_128_state_counts_from_21x32_minus_20x16": len(negative),
        "negative_128_count_magnitude": -sum(negative.values()),
        "largest_per_cell_state_TV": [
            {
                "cell": cell,
                "tv": tv,
                "count16": count16,
                "count32": count32,
            }
            for tv, cell, count16, count32 in sorted(per_cell_tv, reverse=True)[:10]
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
