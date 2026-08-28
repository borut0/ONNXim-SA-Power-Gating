#!/usr/bin/env bash
set -euo pipefail

if (( $# == 0 )); then
  echo "usage: $0 TRACE.csv [TRACE.csv ...]" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
work_dir=$(mktemp -d /tmp/controller-clock-gate-compare.XXXXXX)
trap 'rm -f "${work_dir}/baseline.log" "${work_dir}/gated.log"; rmdir "${work_dir}"' EXIT

bash "${repo_root}/power_model/replay_controller_rtl.sh" "$@" \
  | grep -E '^CONTROLLER_(REPLAY_TRACE|SLEEP_EPISODE|REPLAY_SUMMARY)' \
  > "${work_dir}/baseline.log"
bash "${repo_root}/power_model/replay_controller_clock_gated_rtl.sh" "$@" \
  | grep -E '^CONTROLLER_(REPLAY_TRACE|SLEEP_EPISODE|REPLAY_SUMMARY)' \
  > "${work_dir}/gated.log"

if ! cmp -s "${work_dir}/baseline.log" "${work_dir}/gated.log"; then
  echo "CONTROLLER_CLOCK_GATING_EQUIVALENCE FAIL" >&2
  diff -u "${work_dir}/baseline.log" "${work_dir}/gated.log" | head -100 >&2
  exit 1
fi

if [[ -n "${CONTROLLER_EQUIVALENCE_CAPTURE_DIR:-}" ]]; then
  mkdir -p "${CONTROLLER_EQUIVALENCE_CAPTURE_DIR}"
  trace_base=$(basename "$1" .csv)
  cp "${work_dir}/gated.log" \
    "${CONTROLLER_EQUIVALENCE_CAPTURE_DIR}/${trace_base}.replay.log"
fi

grep '^CONTROLLER_REPLAY_SUMMARY' "${work_dir}/gated.log"
echo "CONTROLLER_CLOCK_GATING_EQUIVALENCE PASS"
