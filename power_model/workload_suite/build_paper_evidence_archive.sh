#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
snapshot_root="$repo_root/power_model/paper_evidence_snapshot_2026-08-29"

mkdir -p "$snapshot_root"

copy_tree() {
  local source=$1
  local destination=$2
  if [[ -e "$source" ]]; then
    if [[ -d "$source" ]]; then
      mkdir -p "$destination"
      cp -a "$source/." "$destination/"
    else
      mkdir -p "$(dirname "$destination")"
      cp -a "$source" "$destination"
    fi
  fi
}

# Final derived workload results, audit records, compact simulator summaries,
# paper tables/text, workload manifests, and exact experiment logs.
copy_tree "$repo_root/power_model/workload_suite/results" \
          "$snapshot_root/workload_suite/results"
copy_tree "$repo_root/power_model/workload_suite/audits" \
          "$snapshot_root/workload_suite/audits"
copy_tree "$repo_root/power_model/workload_suite/summaries" \
          "$snapshot_root/workload_suite/summaries"
copy_tree "$repo_root/power_model/workload_suite/logs" \
          "$snapshot_root/workload_suite/logs"
copy_tree "$repo_root/power_model/workload_suite/evidence" \
          "$snapshot_root/workload_suite/evidence"
copy_tree "$repo_root/power_model/workload_suite/manifest.json" \
          "$snapshot_root/workload_suite/manifest.json"
copy_tree "$repo_root/power_model/workload_suite/model_lists" \
          "$snapshot_root/workload_suite/model_lists"

# Gate-level power reports and synthesis/STA reports are retained because they
# are the primary evidence for controller energy, area, timing, and annotation.
copy_tree "$repo_root/power_model/controller_power_reports" \
          "$snapshot_root/controller_power_reports"
find "$repo_root/power_model" -maxdepth 1 -type f \
  \( -name '*synth*.log' -o -name '*equivalence*.log' \
     -o -name '*net_energy*.json' -o -name '*breakdown*.json' \) \
  -exec cp -a {} "$snapshot_root/" \;

# Compact HSPICE curves, summaries, source decks, and technology libraries.
# The complete directory is small enough to retain and provides provenance for
# transition energy, leakage, header sizing, and scaling checks.
copy_tree "$repo_root/power_model/hspice_inputs" \
          "$snapshot_root/hspice_inputs"
copy_tree "$repo_root/power_model/1_2_yosys.v.gz" \
          "$snapshot_root/power_model/1_2_yosys.v.gz"

# Experiment configuration and the exact 32/8 arrival trace.
copy_tree "$repo_root/configs/systolic_ws_128x128_c4_booksim2_big_model.json" \
          "$snapshot_root/configs/systolic_ws_128x128_c4_booksim2_big_model.json"
copy_tree "$repo_root/traces/input_32_8.csv" \
          "$snapshot_root/traces/input_32_8.csv"

# Preserve the scripts/RTL/OpenROAD inputs needed to interpret or regenerate
# the numbers without duplicating multi-gigabyte traces or VCDs.
copy_tree "$repo_root/power_model/rtl" "$snapshot_root/power_model/rtl"
copy_tree "$repo_root/power_model/openroad" "$snapshot_root/power_model/openroad"
find "$repo_root/power_model" -maxdepth 1 -type f \
  \( -name '*.py' -o -name '*.sh' \) \
  -exec cp -a {} "$snapshot_root/power_model/" \;

git -C "$repo_root" rev-parse HEAD > "$snapshot_root/GIT_COMMIT.txt"
git -C "$repo_root" status --short > "$snapshot_root/GIT_STATUS.txt"
git -C "$repo_root" diff -- src include configs power_model scripts \
  > "$snapshot_root/UNCOMMITTED_CHANGES.patch" || true

(
  cd "$snapshot_root"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)

printf 'PAPER_EVIDENCE_SNAPSHOT=%s\n' "$snapshot_root"
du -sh "$snapshot_root"
printf 'FILES='
find "$snapshot_root" -type f | wc -l
