#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 )); then
  echo "usage: $0 SIZE SYNTH_NETLIST CELL_MODEL [CELL_MODEL ...]" >&2
  exit 2
fi

size=$1
netlist=$2
shift 2
cell_models=("$@")
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_dir="${repo_root}/power_model/gate_activity/${size}x${size}"
sim="/tmp/tb_ws_array_power_gate_${size}"
modes=${POWER_MODES:-"ACTIVE PRELOAD IDLE CLOCK_GATED_IDLE"}

case "${size}" in
  16) gate_define=POWER_GATE_16 ;;
  32) gate_define=POWER_GATE_32 ;;
  *) echo "SIZE must be 16 or 32" >&2; exit 2 ;;
esac

iverilog -g2012 -s tb_ws_array_power \
  -D"${gate_define}" \
  -Ptb_ws_array_power.ROWS="${size}" \
  -Ptb_ws_array_power.COLS="${size}" \
  -o "${sim}" \
  "${cell_models[@]}" \
  "${netlist}" \
  "${repo_root}/power_model/rtl/tb_ws_array_power.sv"

mkdir -p "${output_dir}"
for mode in ${modes}; do
  mode_lower=${mode,,}
  vvp "${sim}" \
    +MODE="${mode}" \
    +VCD="${output_dir}/${mode_lower}.vcd"
done
