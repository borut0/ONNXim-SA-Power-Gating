#!/usr/bin/env bash
set -euo pipefail

if (( $# < 4 )); then
  echo "usage: $0 TRACE.csv SYNTH_NETLIST.v OUTPUT.vcd CELL_MODEL [CELL_MODEL ...]" >&2
  exit 2
fi

trace=$1
netlist=$2
output_vcd=$3
shift 3
cell_models=("$@")
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
sim=/tmp/tb_sa_idle_predictor_controller_gate

mkdir -p "$(dirname "${output_vcd}")"
iverilog -g2012 -s tb_sa_idle_predictor_controller_replay \
  -o "${sim}" \
  "${cell_models[@]}" \
  "${netlist}" \
  "${repo_root}/power_model/rtl/tb_sa_idle_predictor_controller_replay.sv"

vvp "${sim}" "+TRACE=${trace}" "+VCD=${output_vcd}"
