#!/usr/bin/env python3
"""Compare an abstract ReGate policy with the unchanged proposed SA replay."""

import argparse
import json
from pathlib import Path

if __package__:
    from .regate_baseline import replay_regate_baseline
    from .summarize_regate_event_driven import aggregate, read_records
else:
    from regate_baseline import replay_regate_baseline
    from summarize_regate_event_driven import aggregate, read_records


def compare(records, *, expected_cores=None, beta=None, mask_policy="delayed"):
    proposed = aggregate(records, expected_cores=expected_cores, beta=beta)
    baseline_cores = [replay_regate_baseline(item, mask_policy=mask_policy,
                                             beta=beta)
                      for item in proposed["per_core"]]
    unresolved_matched = [item for item in baseline_cores
                          if item["matched_two_bank"]["status"] != "feasible"]
    logic_capacity = sum(item["pe_logic_cycles"]["capacity"]
                         for item in baseline_cores)
    bank_capacity = sum(item["matched_two_bank"]["weight_bank_cycles"]["capacity"]
                        for item in baseline_cores) if not unresolved_matched else None
    logic_equivalent = sum(
        item["pe_logic_cycles"]["on"] +
        0.03 * item["pe_logic_cycles"]["off"]
        for item in baseline_cores)
    bank_equivalent = sum(
        item["matched_two_bank"]["weight_bank_cycles"]["on"] +
        item["matched_two_bank"]["weight_bank_cycles"]["w_on"]
        for item in baseline_cores) if not unresolved_matched else None
    matched_ratio = ((1 - beta) * logic_equivalent / logic_capacity +
                     beta * bank_equivalent / bank_capacity
                     if beta is not None and not unresolved_matched else None)
    proposed_ratio = (
        1 - proposed["whole_sa_gross_static_opportunity_percent_on_managed_timeline"]
        / 100 if beta is not None else None)
    literal_conflicts = sum(item["literal_one_bank"]["conflict_count"]
                            for item in baseline_cores)
    literal_energy = None
    proposed_vs_literal = None
    if beta is not None and literal_conflicts == 0:
        literal_energy = sum(
            item["literal_one_bank"]["energy"][
                "normalized_static_energy_ratio_vs_one_bank_always_on"] *
            item["literal_one_bank"]["pe_cycles"]["capacity"]
            for item in baseline_cores) / logic_capacity
        proposed_vs_literal = 100 * (
            1 - (proposed_ratio * 2 / (2 - beta)) / literal_energy)
    return {
        "model": "abstract ReGate comparison on ONNXim analytical SA events",
        "mask_policy": mask_policy,
        "observed_cores": len(records),
        "shared_horizon_cycles": proposed["analytical_shared_horizon_cycles"],
        "beta_two_bank_weight_share": beta,
        "matched_two_bank_regate_style_energy_ratio": matched_ratio,
        "matched_two_bank_status":
            "unavailable: delayed-mask candidate power overlaps a prior live generation in the same bank"
            if unresolved_matched else "feasible",
        "proposed_two_bank_energy_ratio": proposed_ratio,
        "additional_static_saving_percent_vs_matched_two_bank_regate_style":
            100 * (1 - proposed_ratio / matched_ratio)
            if matched_ratio is not None and matched_ratio > 0 else None,
        "literal_one_bank_regate_conflicts": literal_conflicts,
        "literal_one_bank_regate_energy_ratio_vs_one_bank_always_on":
            literal_energy,
        "proposed_additional_static_saving_percent_vs_literal_one_bank_regate":
            proposed_vs_literal,
        "literal_one_bank_comparison_status":
            "unavailable: one-bank generation conflicts require rescheduling or reloads"
            if literal_conflicts else
            "unavailable: beta not provided" if beta is None else
            "gross static comparison on matched analytical horizon; no end-to-end timing claim",
        "performance_penalty_percent": None,
        "performance_status":
            "native non-SA work and memory are not rescheduled; circuit wake costs are not included",
        "proposed_policy_unchanged": True,
        "per_core_regate": baseline_cores,
        "proposed_summary": proposed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--expected-cores", type=int)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--mask-policy", choices=("optimistic", "delayed"),
                        default="delayed")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(read_records(args.log), expected_cores=args.expected_cores,
                     beta=args.beta, mask_policy=args.mask_policy)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.resolve() == args.log.resolve():
            parser.error("--output must not overwrite the raw simulation log")
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
