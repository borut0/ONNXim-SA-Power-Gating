#!/usr/bin/env bash
set -euo pipefail
trap '' HUP

if (( $# == 0 )); then
  echo "usage: $0 TRACE.csv [TRACE.csv ...]" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
work_dir=$(mktemp -d /tmp/controller-two-way-compare.XXXXXX)
cleanup() {
  rm -f "${work_dir}"/compare*.log
  rmdir "${work_dir}" 2>/dev/null || true
}
trap cleanup EXIT

traces=("$@")
logs=()
for ((base = 0; base < ${#traces[@]}; base += 2)); do
  pids=()
  for ((offset = 0; offset < 2 && base + offset < ${#traces[@]}; ++offset)); do
    index=$((base + offset))
    log="${work_dir}/compare${index}.log"
    logs+=("${log}")
    echo "CONTROLLER_TWO_WAY_EQUIVALENCE trace=${traces[index]} start=$(date -Is)"
    bash "${repo_root}/power_model/compare_controller_clock_gated_replay.sh" \
      "${traces[index]}" >"${log}" 2>&1 &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done
  if (( status != 0 )); then
    for log in "${logs[@]}"; do cat "${log}"; done
    echo "CONTROLLER_TWO_WAY_EQUIVALENCE FAIL" >&2
    exit 1
  fi
done

for log in "${logs[@]}"; do cat "${log}"; done
echo "CONTROLLER_TWO_WAY_EQUIVALENCE PASS"
