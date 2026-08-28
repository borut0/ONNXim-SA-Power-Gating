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
sim=$(mktemp /tmp/tb_sa_idle_predictor_controller_clock_gated_gate.XXXXXX)
trap 'rm -f "${sim}"' EXIT

mkdir -p "$(dirname "${output_vcd}")"
iverilog -g2012 -DCLOCK_GATED_CONTROLLER -DCONTROLLER_SHALLOW_GATE_VCD \
  -s tb_sa_idle_predictor_controller_replay \
  -o "${sim}" \
  "${cell_models[@]}" \
  "${netlist}" \
  "${repo_root}/power_model/rtl/tb_sa_idle_predictor_controller_replay.sv"

if [[ "${output_vcd}" == *.gz ]]; then
  command -v gzip >/dev/null || {
    echo "ERROR: gzip is required for compressed VCD output" >&2
    exit 1
  }
  tmp_dir=$(mktemp -d)
  fifo="${tmp_dir}/activity.vcd"
  tmp_gz="${output_vcd}.tmp.$$"
  mkfifo "${fifo}"
  cleanup() {
    rm -f "${fifo}" "${tmp_gz}" "${sim}"
    rmdir "${tmp_dir}" 2>/dev/null || true
  }
  trap cleanup EXIT

  gzip -1 -c < "${fifo}" > "${tmp_gz}" &
  gzip_pid=$!
  vvp "${sim}" "+TRACE=${trace}" "+VCD=${fifo}"
  wait "${gzip_pid}"
  mv "${tmp_gz}" "${output_vcd}"
  rm -f "${fifo}" "${sim}"
  rmdir "${tmp_dir}"
  trap - EXIT
else
  vvp "${sim}" "+TRACE=${trace}" "+VCD=${output_vcd}"
fi
