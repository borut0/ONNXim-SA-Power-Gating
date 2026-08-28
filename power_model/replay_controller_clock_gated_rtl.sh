#!/usr/bin/env bash
set -euo pipefail

if (( $# == 0 )); then
  echo "usage: $0 TRACE.csv [TRACE.csv ...]" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
sim=$(mktemp /tmp/tb_sa_idle_predictor_controller_clock_gated_rtl.XXXXXX)
trap 'rm -f "${sim}"' EXIT

iverilog -g2012 -DCLOCK_GATED_CONTROLLER \
  -s tb_sa_idle_predictor_controller_replay \
  -o "${sim}" \
  "${repo_root}/power_model/rtl/controller_clock_cells_sim.sv" \
  "${repo_root}/power_model/rtl/sa_idle_predictor_controller.sv" \
  "${repo_root}/power_model/rtl/sa_idle_predictor_controller_clock_gated.sv" \
  "${repo_root}/power_model/rtl/tb_sa_idle_predictor_controller_replay.sv"

for trace in "$@"; do
  echo "CONTROLLER_REPLAY_TRACE ${trace}"
  vvp "${sim}" "+TRACE=${trace}"
done
