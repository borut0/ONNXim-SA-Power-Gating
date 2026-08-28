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
mkdir -p "${repo_root}/power_model/workload_suite/logs"

for model in "$@"; do
  trace_dir="${repo_root}/power_model/controller_traces/${model}"
  capture_dir="${repo_root}/power_model/controller_traces/${model}_replay_capture"
  equivalence_log="${repo_root}/power_model/workload_suite/logs/${model}.equivalence.log"
  replay_log="${trace_dir}/full_replay.log"
  gross_json="${trace_dir}/gross_energy.json"

  mkdir -p "${trace_dir}" "${capture_dir}"

  echo "SCREEN_START model=${model} time=$(date --iso-8601=seconds)"
  : >"${equivalence_log}"
  for core in 0 1 2 3; do
    trace="${trace_dir}/sa_controller_core${core}.csv"
    capture="${capture_dir}/sa_controller_core${core}.replay.log"

    if [[ -s "${capture}" ]]; then
      echo "SCREEN_CORE_SKIP model=${model} core=${core} reason=validated_capture_exists" \
        | tee -a "${equivalence_log}"
      continue
    fi

    echo "SCREEN_CORE_COPY model=${model} core=${core} time=$(date --iso-8601=seconds)" \
      | tee -a "${equivalence_log}"
    docker cp \
      "${container}:/workspace/ONNXim/power_model/controller_traces/${model}/sa_controller_core${core}.csv" \
      "${trace}"
    if [[ ! -s "${trace}" ]]; then
      echo "ERROR: copied trace is missing or empty: ${trace}" >&2
      exit 1
    fi

    CONTROLLER_EQUIVALENCE_CAPTURE_DIR="${capture_dir}" \
      bash "${repo_root}/power_model/compare_controller_clock_gated_replay_two_way.sh" \
        "${trace}" >>"${equivalence_log}" 2>&1
    if [[ ! -s "${capture}" ]]; then
      echo "ERROR: validated replay capture was not produced: ${capture}" >&2
      exit 1
    fi
    rm -f "${trace}"
    echo "SCREEN_CORE_COMPLETE model=${model} core=${core} time=$(date --iso-8601=seconds)" \
      | tee -a "${equivalence_log}"
  done

  capture_count=$(find "${capture_dir}" -maxdepth 1 -type f \
    -name 'sa_controller_core*.replay.log' -size +0c | wc -l)
  if (( capture_count != 4 )); then
    echo "ERROR: expected four validated replay captures for ${model}, found ${capture_count}" >&2
    exit 1
  fi

  cat "${capture_dir}"/*.replay.log >"${replay_log}"
  python3 "${repo_root}/power_model/summarize_controller_gross_sa_energy.py" \
    "${replay_log}" >"${gross_json}"

  echo "SCREEN_COMPLETE model=${model} gross=${gross_json} equivalence=${equivalence_log}"
done
