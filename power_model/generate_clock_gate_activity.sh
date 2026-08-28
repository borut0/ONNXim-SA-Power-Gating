#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "usage: $0 SYNTH_NETLIST CELL_MODEL [CELL_MODEL ...]" >&2
  exit 2
fi

netlist=$1
shift
cell_models=("$@")
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_dir="${repo_root}/power_model/gate_activity/clock_gate_16x16"
sim=/tmp/tb_ws_array_clock_gate_power
modes=${POWER_MODES:-"ACTIVE IDLE_ON CLOCK_GATED_IDLE ENTRY WAKE"}

iverilog -g2012 -s tb_ws_array_clock_gate_power \
  -o "${sim}" \
  "${cell_models[@]}" \
  "${netlist}" \
  "${repo_root}/power_model/rtl/tb_ws_array_clock_gate_power.sv"

mkdir -p "${output_dir}"
for mode in ${modes}; do
  mode_lower=${mode,,}
  vvp "${sim}" \
    +MODE="${mode}" \
    +VCD="${output_dir}/${mode_lower}.vcd"
done
