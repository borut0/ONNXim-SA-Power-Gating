#!/usr/bin/env python3
"""Aggregate and validate compact per-core SA region profiles from a log."""

import argparse
import json
from pathlib import Path


PREFIX = "SA_REGION_PROFILE "


def percentage(numerator, denominator):
    return 0.0 if denominator == 0 else 100.0 * numerator / denominator


def merge_episode_histograms(histograms):
    merged = {}
    for histogram in histograms:
        for item in histogram:
            length = int(item["length_cycles"])
            count = int(item["episode_count"])
            if length <= 0 or count <= 0:
                raise ValueError("idle-episode histograms require positive values")
            merged[length] = merged.get(length, 0) + count
    return [
        {"length_cycles": length, "episode_count": merged[length]}
        for length in sorted(merged)
    ]


def episode_histogram_stats(histogram):
    episode_count = sum(int(item["episode_count"]) for item in histogram)
    episode_cycles = sum(
        int(item["length_cycles"]) * int(item["episode_count"])
        for item in histogram
    )
    return {
        "idle_episode_count": episode_count,
        "idle_episode_cycles": episode_cycles,
        "shortest_idle_episode_cycles": (
            0 if not histogram else int(histogram[0]["length_cycles"])
        ),
        "longest_idle_episode_cycles": (
            0 if not histogram else int(histogram[-1]["length_cycles"])
        ),
        "average_idle_episode_cycles": (
            0.0 if episode_count == 0 else episode_cycles / episode_count
        ),
        "idle_episode_length_histogram": histogram,
    }


def load_records(path):
    records = []
    with path.open("r", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            marker = line.find(PREFIX)
            if marker < 0:
                continue
            payload = line[marker + len(PREFIX):].strip()
            try:
                record = json.loads(payload)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid SA region JSON at {path}:{line_number}: {error}"
                ) from error
            records.append(record)
    if not records:
        raise ValueError(f"no {PREFIX.strip()} records found in {path}")
    return records


def aggregate(records, expected_cores=None):
    by_core = {}
    for record in records:
        core_id = int(record["core_id"])
        if core_id in by_core:
            raise ValueError(f"duplicate SA region profile for core {core_id}")
        by_core[core_id] = record

    reference = records[0]
    dimensions = (reference["array_rows"], reference["array_columns"])
    reference_sizes = [
        (item["region_size_rows"], item["region_size_columns"])
        for item in reference["granularities"]
    ]
    errors = []
    if expected_cores is not None and len(records) != expected_cores:
        errors.append(
            f"expected {expected_cores} cores but found {len(records)}"
        )

    for record in records:
        if (record["array_rows"], record["array_columns"]) != dimensions:
            errors.append("array dimensions differ across cores")
        sizes = [
            (item["region_size_rows"], item["region_size_columns"])
            for item in record["granularities"]
        ]
        if sizes != reference_sizes:
            errors.append("region-size sets differ across cores")
        if not record.get("validation_pass", False):
            errors.append(f"core {record['core_id']} failed source validation")

    totals = {
        "core_cycles": sum(int(item["core_cycles"]) for item in records),
        "sa_instruction_intervals": sum(
            int(item["sa_instruction_intervals"]) for item in records
        ),
        "whole_sa_active_cycles": sum(
            int(item["whole_sa_active_cycles"]) for item in records
        ),
        "whole_sa_idle_cycles": sum(
            int(item["whole_sa_idle_cycles"]) for item in records
        ),
        "whole_sa_committed_cycles": sum(
            int(item["whole_sa_committed_cycles"]) for item in records
        ),
        "whole_sa_uncommitted_cycles": sum(
            int(item["whole_sa_uncommitted_cycles"]) for item in records
        ),
        "invalid_dimension_intervals": sum(
            int(item["invalid_dimension_intervals"]) for item in records
        ),
        "clipped_dimension_intervals": sum(
            int(item["clipped_dimension_intervals"]) for item in records
        ),
    }
    regate_input_presence = [
        "modeled_useful_mac_count" in record for record in records
    ]
    per_core_regate_inputs = []
    if any(regate_input_presence) and not all(regate_input_presence):
        errors.append("ReGate work-count input is missing from some cores")
    elif all(regate_input_presence):
        totals["modeled_useful_mac_count"] = sum(
            int(item["modeled_useful_mac_count"]) for item in records
        )
        per_core_regate_inputs = [
            {
                "core_id": int(item["core_id"]),
                "core_cycles": int(item["core_cycles"]),
                "whole_sa_active_cycles": int(item["whole_sa_active_cycles"]),
                "sa_instruction_intervals": int(item["sa_instruction_intervals"]),
                "modeled_useful_mac_count": int(
                    item["modeled_useful_mac_count"]
                ),
            }
            for item in sorted(records, key=lambda value: int(value["core_id"]))
        ]
    if totals["whole_sa_active_cycles"] + totals["whole_sa_idle_cycles"] != totals["core_cycles"]:
        errors.append("aggregate whole-SA active/idle accounting mismatch")
    if totals["whole_sa_committed_cycles"] + totals["whole_sa_uncommitted_cycles"] != totals["core_cycles"]:
        errors.append("aggregate whole-SA commitment accounting mismatch")

    pe_state_presence = [
        "pe_state_opportunity" in record for record in records
    ]
    pe_state_opportunity = {}
    if any(pe_state_presence) and not all(pe_state_presence):
        errors.append("PE-state opportunity data are missing from some cores")
    elif all(pe_state_presence):
        pe_entries = [record["pe_state_opportunity"] for record in records]
        pe_sum_keys = (
            "completed_intervals",
            "incomplete_intervals",
            "invalid_dimension_intervals",
            "insufficient_duration_intervals",
            "union_active_cycles",
            "active_capacity_pe_cycles",
            "on_compute_pe_cycles",
            "resident_weight_pe_cycles",
            "w_on_candidate_pe_cycles",
            "off_unmapped_candidate_pe_cycles",
            "temporal_idle_unclassified_pe_cycles",
            "total_capacity_pe_cycles",
        )
        pe_sums = {
            key: sum(int(entry[key]) for entry in pe_entries)
            for key in pe_sum_keys
        }
        pe_accounting_exact = (
            pe_sums["union_active_cycles"]
            == totals["whole_sa_active_cycles"]
            and pe_sums["on_compute_pe_cycles"]
            + pe_sums["w_on_candidate_pe_cycles"]
            + pe_sums["off_unmapped_candidate_pe_cycles"]
            == pe_sums["active_capacity_pe_cycles"]
            and pe_sums["active_capacity_pe_cycles"]
            + pe_sums["temporal_idle_unclassified_pe_cycles"]
            == pe_sums["total_capacity_pe_cycles"]
            and all(entry.get("accounting_exact", False)
                    for entry in pe_entries)
        )
        if not pe_accounting_exact:
            errors.append("aggregate PE-state opportunity accounting mismatch")
        pe_state_opportunity = {
            "evidence_class": "derived",
            "model": (
                "packed-footprint weight-stationary structural candidate "
                "accounting"
            ),
            **pe_sums,
            "accounting_exact": pe_accounting_exact,
            "accounting_exact_semantics": (
                "Arithmetic partition exact under the stated origin-packed "
                "footprint and aggregate useful-MAC assumptions; not an "
                "exact functional ReGate state trace."
            ),
            "temporal_idle_classification": "unclassified",
            "not_energy_result": True,
        }

    wave_presence = [
        "regate_analytical_wave_lifetime" in record for record in records
    ]
    regate_analytical_wave_lifetime = {}
    if any(wave_presence) and not all(wave_presence):
        errors.append("analytical wave/lifetime data are missing from some cores")
    elif all(wave_presence):
        wave_entries = [
            record["regate_analytical_wave_lifetime"] for record in records
        ]
        wave_sum_keys = (
            "analytical_on_pe_cycles",
            "simulator_active_capacity_pe_cycles",
            "simulator_active_non_wave_pe_cycles",
            "resident_weight_pe_cycles",
            "pre_start_committed_pe_cycles",
            "analytical_w_on_pe_cycles",
            "structural_off_candidate_pe_cycles",
            "unclassified_pe_cycles",
            "total_capacity_pe_cycles",
            "completed_intervals",
            "incomplete_intervals",
            "invalid_dimension_intervals",
            "wave_exceeds_instruction_intervals",
            "overlapping_wave_intervals",
            "unknown_weight_block_intervals",
            "gemm_without_preload_intervals",
            "weight_block_mismatch_intervals",
            "footprint_mismatch_intervals",
        )
        wave_sums = {
            key: sum(int(entry[key]) for entry in wave_entries)
            for key in wave_sum_keys
        }
        wave_partition_exact = all(
            entry.get("partition_exact_under_model", False)
            for entry in wave_entries
        )
        if wave_partition_exact and (
            wave_sums["analytical_on_pe_cycles"]
            + wave_sums["analytical_w_on_pe_cycles"]
            + wave_sums["structural_off_candidate_pe_cycles"]
            + wave_sums["unclassified_pe_cycles"]
            != wave_sums["total_capacity_pe_cycles"]
        ):
            errors.append("analytical PE-state partition arithmetic mismatch")
        regate_analytical_wave_lifetime = {
            "evidence_class": "derived",
            **wave_sums,
            "per_core": [
                {
                    "core_id": int(record["core_id"]),
                    **{
                        key: record["regate_analytical_wave_lifetime"][key]
                        for key in wave_sum_keys
                    },
                    "wave_overlap_free": record[
                        "regate_analytical_wave_lifetime"
                    ].get("wave_overlap_free", False),
                    "weight_lifetime_complete": record[
                        "regate_analytical_wave_lifetime"
                    ].get("weight_lifetime_complete", False),
                    "partition_exact_under_model": record[
                        "regate_analytical_wave_lifetime"
                    ].get("partition_exact_under_model", False),
                }
                for record in sorted(
                    records, key=lambda item: int(item["core_id"])
                )
            ],
            "wave_overlap_free": all(
                entry.get("wave_overlap_free", False)
                for entry in wave_entries
            ),
            "weight_lifetime_complete": all(
                entry.get("weight_lifetime_complete", False)
                for entry in wave_entries
            ),
            "partition_exact_under_model": wave_partition_exact,
            "off_semantics": (
                "Structural candidate under offline exact-liveness and "
                "atomic-preload assumptions; not a measured/generated "
                "ReGate control signal."
            ),
            "not_energy_result": True,
        }

    whole_episode_presence = [
        "whole_sa_idle_episode_length_histogram" in record
        for record in records
    ]
    whole_episode_stats = {}
    if any(whole_episode_presence) and not all(whole_episode_presence):
        errors.append("whole-SA idle-episode data are missing from some cores")
    elif all(whole_episode_presence):
        whole_episode_stats = episode_histogram_stats(
            merge_episode_histograms([
                record["whole_sa_idle_episode_length_histogram"]
                for record in records
            ])
        )
        if (
            whole_episode_stats["idle_episode_cycles"]
            != totals["whole_sa_idle_cycles"]
        ):
            errors.append("aggregate whole-SA idle-episode accounting mismatch")

    granularities = []
    for index, size in enumerate(reference_sizes):
        entries = [record["granularities"][index] for record in records]
        keys_to_sum = (
            "capacity_region_cycles",
            "modeled_active_region_cycles",
            "modeled_idle_region_cycles",
            "temporally_idle_region_cycles",
            "spatially_idle_region_cycles_during_sa_active",
            "conservative_reserved_region_cycles",
            "conservative_available_region_cycles",
            "conservative_temporally_uncommitted_region_cycles",
            "conservative_spatially_available_region_cycles_during_commitment",
        )
        summed = {
            key: sum(int(entry[key]) for entry in entries)
            for key in keys_to_sum
        }
        accounting_exact = (
            summed["modeled_active_region_cycles"]
            + summed["modeled_idle_region_cycles"]
            == summed["capacity_region_cycles"]
            and summed["modeled_idle_region_cycles"]
            == summed["temporally_idle_region_cycles"]
            + summed["spatially_idle_region_cycles_during_sa_active"]
            and summed["conservative_reserved_region_cycles"]
            + summed["conservative_available_region_cycles"]
            == summed["capacity_region_cycles"]
            and summed["conservative_available_region_cycles"]
            == summed["conservative_temporally_uncommitted_region_cycles"]
            + summed[
                "conservative_spatially_available_region_cycles_during_commitment"
            ]
            and all(entry.get("accounting_exact", False) for entry in entries)
        )
        if not accounting_exact:
            errors.append(f"aggregate accounting failed for region size {size}")
        episode_presence = [
            "idle_episode_length_histogram" in entry for entry in entries
        ]
        episode_stats = {}
        if any(episode_presence) and not all(episode_presence):
            errors.append(
                f"idle-episode data are missing from some cores for region size {size}"
            )
        elif all(episode_presence):
            episode_stats = episode_histogram_stats(
                merge_episode_histograms([
                    entry["idle_episode_length_histogram"]
                    for entry in entries
                ])
            )
            if (
                episode_stats["idle_episode_cycles"]
                != summed["modeled_idle_region_cycles"]
            ):
                errors.append(
                    f"idle-episode accounting failed for region size {size}"
                )
        modeled_active_capacity = (
            summed["modeled_active_region_cycles"]
            + summed["spatially_idle_region_cycles_during_sa_active"]
        )
        committed_capacity = (
            summed["conservative_reserved_region_cycles"]
            + summed[
                "conservative_spatially_available_region_cycles_during_commitment"
            ]
        )
        wave_region_presence = [
            "regate_analytical_wave" in entry for entry in entries
        ]
        wave_region = {}
        if any(wave_region_presence) and not all(wave_region_presence):
            errors.append(
                f"analytical wave data missing for region size {size}"
            )
        elif all(wave_region_presence):
            wave_items = [entry["regate_analytical_wave"] for entry in entries]
            wave_region_keys = (
                "analytical_on_region_cycles",
                "analytical_w_on_region_cycles",
                "structural_off_candidate_region_cycles",
                "preload_committed_unclassified_region_cycles",
                "capacity_region_cycles",
            )
            wave_region_sums = {
                key: sum(int(item[key]) for item in wave_items)
                for key in wave_region_keys
            }
            wave_region_exact = all(
                item.get("partition_exact_under_model", False)
                for item in wave_items
            )
            if wave_region_exact and (
                wave_region_sums["analytical_on_region_cycles"]
                + wave_region_sums["analytical_w_on_region_cycles"]
                + wave_region_sums["structural_off_candidate_region_cycles"]
                + wave_region_sums[
                    "preload_committed_unclassified_region_cycles"
                ]
                != wave_region_sums["capacity_region_cycles"]
            ):
                errors.append(
                    f"analytical state partition failed for region size {size}"
                )
            wave_region = {
                **wave_region_sums,
                "partition_exact_under_model": wave_region_exact,
                "not_energy_result": True,
            }

        granularities.append({
            "region_size_rows": size[0],
            "region_size_columns": size[1],
            "region_rows": entries[0]["region_rows"],
            "region_columns": entries[0]["region_columns"],
            "region_count_per_core": entries[0]["region_count"],
            **summed,
            **episode_stats,
            "modeled_active_region_percent": percentage(
                summed["modeled_active_region_cycles"],
                summed["capacity_region_cycles"],
            ),
            "modeled_idle_region_percent": percentage(
                summed["modeled_idle_region_cycles"],
                summed["capacity_region_cycles"],
            ),
            "spatially_idle_percent_during_sa_active": percentage(
                summed["spatially_idle_region_cycles_during_sa_active"],
                modeled_active_capacity,
            ),
            "conservative_reserved_region_percent": percentage(
                summed["conservative_reserved_region_cycles"],
                summed["capacity_region_cycles"],
            ),
            "conservative_available_region_percent": percentage(
                summed["conservative_available_region_cycles"],
                summed["capacity_region_cycles"],
            ),
            "conservative_spatially_available_percent_during_commitment": percentage(
                summed[
                    "conservative_spatially_available_region_cycles_during_commitment"
                ],
                committed_capacity,
            ),
            "accounting_exact": accounting_exact,
            "regate_analytical_wave": wave_region,
        })

    result = {
        "version": (
            6 if regate_analytical_wave_lifetime else
            (5 if pe_state_opportunity else
            (4 if per_core_regate_inputs else
             (3 if whole_episode_stats else 2)))
        ),
        "source_core_count": len(records),
        "core_ids": sorted(by_core),
        "array_rows": dimensions[0],
        "array_columns": dimensions[1],
        **totals,
        **{
            f"whole_sa_{key}": value
            for key, value in whole_episode_stats.items()
        },
        "whole_sa_active_percent": percentage(
            totals["whole_sa_active_cycles"], totals["core_cycles"]
        ),
        "whole_sa_idle_percent": percentage(
            totals["whole_sa_idle_cycles"], totals["core_cycles"]
        ),
        "granularities": granularities,
        "per_core_regate_inputs": per_core_regate_inputs,
        "pe_state_opportunity": pe_state_opportunity,
        "regate_analytical_wave_lifetime": regate_analytical_wave_lifetime,
        "validation_errors": errors,
        "validation_pass": not errors,
        "interpretation": (
            "Measurement-only packed tile_m-by-tile_k stationary-footprint "
            "occupancy. Modeled spatial idleness uses ONNXim's start-to-finish "
            "SA-active intervals, so the 128x128 modeled-idle result equals "
            "whole-SA idle. A separate conservative issue-to-finish reservation "
            "view is also reported. Neither view is cycle-accurate PE wavefront "
            "activity or an energy-saving claim. Version 3 also retains compact "
            "continuous idle-episode histograms for plug-in break-even analysis. "
            "Version 4 additionally retains per-core architectural MAC counts "
            "for an explicitly derived ReGate-artifact reference model. "
            "Version 5 adds a parameter-free structural candidate partition "
            "of merged SA-active PE-cycle capacity into dense useful-MAC "
            "work, mapped non-MAC candidate capacity, and unmapped candidate "
            "capacity. These are not functional ReGate state traces; temporal "
            "idle remains unclassified because weight lifetime is not "
            "established. Version 6 adds a separate analytical WS wave and "
            "exact weight-block-lifetime model. It keeps issue-to-start "
            "preload commitments unclassified and calls OFF only a structural "
            "candidate under the documented atomic-preload and offline-"
            "liveness assumptions."
        ),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--expected-cores", type=int)
    args = parser.parse_args()
    result = aggregate(load_records(args.log), args.expected_cores)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["validation_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
