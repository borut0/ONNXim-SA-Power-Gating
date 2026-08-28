#!/usr/bin/env bash
set -euo pipefail

repo_root=${1:-/workspace}
output_root=${2:-"${repo_root}/power_model/activity"}
modes=${POWER_MODES:-"ACTIVE PRELOAD IDLE CLOCK_GATED_IDLE"}

for size in 16 32; do
  sim="/tmp/tb_ws_array_power_${size}"
  iverilog -g2012 -s tb_ws_array_power \
    -Ptb_ws_array_power.ROWS="${size}" \
    -Ptb_ws_array_power.COLS="${size}" \
    -o "${sim}" \
    "${repo_root}/power_model/rtl/ws_pe.sv" \
    "${repo_root}/power_model/rtl/ws_array.sv" \
    "${repo_root}/power_model/rtl/tb_ws_array_power.sv"

  mkdir -p "${output_root}/${size}x${size}"
  for mode in ${modes}; do
    mode_lower=${mode,,}
    vvp "${sim}" \
      +MODE="${mode}" \
      +VCD="${output_root}/${size}x${size}/${mode_lower}.vcd"
  done
done
