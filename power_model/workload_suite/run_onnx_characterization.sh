#!/usr/bin/env bash
set -euo pipefail

if (( $# == 0 )); then
  echo "usage: $0 MODEL [MODEL ...]" >&2
  exit 2
fi

root_dir="${ONNXIM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
simulator="${ONNXIM_SIMULATOR:-$root_dir/build/bin/Simulator}"
config="${ONNXIM_CONFIG:-$root_dir/configs/systolic_ws_128x128_c4_booksim2_big_model.json}"

mkdir -p \
  "$root_dir/power_model/workload_suite/logs" \
  "$root_dir/power_model/workload_suite/summaries" \
  "$root_dir/power_model/controller_traces"

for model in "$@"; do
  graph="$root_dir/models/$model/$model.onnx"
  model_list="$root_dir/model_lists/${model}_1.json"
  audit="$root_dir/power_model/workload_suite/audits/$model.json"
  log="$root_dir/power_model/workload_suite/logs/${model}.final.log"
  summary="$root_dir/power_model/workload_suite/summaries/${model}.final.json"
  trace_dir="$root_dir/power_model/controller_traces/$model"

  for required in "$simulator" "$config" "$graph" "$model_list" "$audit"; do
    if [[ ! -s "$required" ]]; then
      echo "ERROR: missing or empty required file: $required" >&2
      exit 1
    fi
  done

  python3 - "$audit" <<'PY'
import json, sys
audit = json.load(open(sys.argv[1]))
if not audit.get("strict_operator_audit_pass", False):
    raise SystemExit(
        "ERROR: strict operator audit failed: "
        + json.dumps(audit.get("unmodeled_operator_counts", {}), sort_keys=True)
    )
PY

  if [[ -e "$trace_dir" ]]; then
    echo "ERROR: refusing to mix/overwrite existing trace directory: $trace_dir" >&2
    exit 1
  fi
  mkdir "$trace_dir"

  echo "WORKLOAD_START model=$model time=$(date --iso-8601=seconds)"
  env \
    -u ONNXSIM_SA_PHASE_MEASURE \
    -u ONNXSIM_SA_PREDICTOR_MEASURE \
    -u ONNXSIM_SA_RETENTION_REASON_MEASURE \
    -u ONNXSIM_MEMORY_STAGE_PROFILE_DETAILED \
    ONNXSIM_COMPONENT_CHARACTERIZATION=1 \
    ONNXSIM_SA_STATE_BREAKDOWN=1 \
    ONNXSIM_SA_CONTROLLER_TRACE_DIR="$trace_dir" \
    "$simulator" \
      --config "$config" \
      --models_list "$model_list" \
      > "$log" 2>&1

  python3 "$root_dir/power_model/summarize_component_characterization.py" \
    "$log" > "$summary"
  python3 - "$summary" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
if not summary.get("validation_pass", False):
    raise SystemExit("ERROR: component-characterization validation failed")
PY
  echo "WORKLOAD_COMPLETE model=$model time=$(date --iso-8601=seconds) summary=$summary traces=$trace_dir"
done
