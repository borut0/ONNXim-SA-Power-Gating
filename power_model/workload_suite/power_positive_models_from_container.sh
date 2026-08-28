#!/usr/bin/env bash
set -euo pipefail
trap '' HUP

if (( $# < 2 )); then
  echo "usage: $0 CONTAINER MODEL [MODEL ...]" >&2
  exit 2
fi

container=$1
shift
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

for model in "$@"; do
  trace_dir="${repo_root}/power_model/controller_traces/${model}"
  combined_log="${repo_root}/power_model/controller_${model}_sparse_event_power.log"
  net_json="${repo_root}/power_model/controller_traces/${model}/net_energy.json"
  mkdir -p "${trace_dir}"

  echo "POWER_MODEL_START model=${model} time=$(date --iso-8601=seconds)"
  : >"${combined_log}"
  for core in 0 1 2 3; do
    trace="${trace_dir}/sa_controller_core${core}.csv"
    core_log="${combined_log}.core${core}"
    echo "POWER_MODEL_CORE_COPY model=${model} core=${core} time=$(date --iso-8601=seconds)"
    docker cp \
      "${container}:/workspace/ONNXim/power_model/controller_traces/${model}/sa_controller_core${core}.csv" \
      "${trace}"
    if [[ ! -s "${trace}" ]]; then
      echo "ERROR: copied trace is missing or empty: ${trace}" >&2
      exit 1
    fi

    CONTROLLER_VARIANT=clock_gated \
    CONTROLLER_WORKLOAD="${model}" \
    CONTROLLER_FLOW_VARIANT=sparse_event \
    CONTROLLER_REPORT_TAG=sparse_event \
    CONTROLLER_COMBINED_LOG="${core_log}" \
      bash "${repo_root}/power_model/run_controller_power_two_way.sh" "${core}"
    cat "${core_log}" >>"${combined_log}"
    rm -f "${trace}"
    echo "POWER_MODEL_CORE_COMPLETE model=${model} core=${core} time=$(date --iso-8601=seconds)"
  done

  python3 "${repo_root}/power_model/summarize_controller_net_energy.py" \
    "${combined_log}" \
    --curve "${repo_root}/power_model/hspice_inputs/m300_duration_energy_tt25c.json" \
    --power-report-dir "${repo_root}/power_model/controller_power_reports" \
    --power-report-stem "controller_clock_gated_sparse_event_${model}" \
    --short-episode-policy clamp-min \
    > "${net_json}"

  echo "POWER_MODEL_COMPLETE model=${model} net=${net_json}"
done
