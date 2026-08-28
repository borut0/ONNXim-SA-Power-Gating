#!/usr/bin/env bash
set -euo pipefail

work_dir="${HSPICE_SA_DIR:-/home/cdsllab/Fenil/hspice_sa}"
generator="$work_dir/generate_connected_power_gate_deck.py"
netlist="$work_dir/ws_array_1x1_yosys.v"
pex="/home/cdsllab/Fenil/pdks/asap7/asap7sc7p5t_28/CDL/xAct3D_extracted/asap7sc7p5t_28_R.sp"
model="/home/cdsllab/Fenil/pdks/asap7/asap7_pdk_r1p7/models/hspice/7nm_TT_160803.pm"
result_dir="$work_dir/duration_sweep_m300"
durations_ns=(5 10 50 100 150 175 200 250 500 1000 2000 3000 5000 8000 15000)

mkdir -p "$result_dir"

for duration_ns in "${durations_ns[@]}"; do
  prefix="$result_dir/sa1_m300_sleep_${duration_ns}ns"
  if [[ -f "$prefix.lis" ]] && grep -q "event_vdd_energy=" "$prefix.lis"; then
    echo "SKIP duration_ns=$duration_ns (completed)"
    continue
  fi

  python3 "$generator" \
    --netlist "$netlist" \
    --pex "$pex" \
    --model "$model" \
    --output "$prefix.sp" \
    --rows 1 --cols 1 --header-m 300 \
    --mode duration-sweep --pattern zero \
    --off-hold-ns "$duration_ns" \
    --wake-hold-ns 5 --post 0 >/dev/null

  echo "RUN duration_ns=$duration_ns"
  /usr/bin/time -v hspice -mt 8 -i "$prefix.sp" -o "$prefix" \
    2> "$prefix.time"
done

summary="$result_dir/duration_sweep_m300.results.txt"
grep -H -Ei \
  'p_on_vdd|vvirtual_at_wake|vvirtual_sleep_min|vvirtual_recovered|wake_90_latency|wake_95_latency|wake_peak_current|sleep_vdd_energy|sleep_control_energy|recovery_vdd_energy|recovery_control_energy|event_vdd_energy|event_control_energy|failed|fatal|error' \
  "$result_dir"/*.lis > "$summary" || true
echo "RESULTS $summary"
