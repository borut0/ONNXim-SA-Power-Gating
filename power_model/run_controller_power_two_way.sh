#!/usr/bin/env bash
set -euo pipefail
trap '' HUP

if (( $# == 0 )); then
  echo "usage: $0 CORE [CORE ...]" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
workload=${CONTROLLER_WORKLOAD:-opt125m}
log_dir="${repo_root}/power_model/controller_power_reports"
combined_log=${CONTROLLER_COMBINED_LOG:-"${repo_root}/power_model/controller_${workload}_two_way_power.log"}
mkdir -p "${log_dir}"

cores=("$@")
logs=()
for ((base = 0; base < ${#cores[@]}; base += 2)); do
  pids=()
  for ((offset = 0; offset < 2 && base + offset < ${#cores[@]}; ++offset)); do
    core=${cores[base + offset]}
    log="${log_dir}/two_way_${workload}_core${core}.driver.log"
    logs+=("${log}")
    echo "CONTROLLER_TWO_WAY core=${core} start=$(date -Is) log=${log}"
    (
      CONTROLLER_SUMMARY_TAG="core${core}" \
        bash "${repo_root}/power_model/run_controller_power_pair.sh" "${core}"
    ) >"${log}" 2>&1 &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done
  if (( status != 0 )); then
    echo "ERROR: a controller power worker failed; inspect ${logs[*]}" >&2
    exit 1
  fi
done

: >"${combined_log}"
for log in "${logs[@]}"; do
  cat "${log}" >>"${combined_log}"
done
echo "CONTROLLER_TWO_WAY_COMPLETE combined_log=${combined_log}"
