#!/usr/bin/env bash
set -euo pipefail
trap '' HUP

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
orfs_results=${ORFS_RESULTS_ROOT:-/home/fenil/orfs_work/results}
orfs_image=${ORFS_IMAGE:-openroad/orfs:latest}
workload=${CONTROLLER_WORKLOAD:-opt125m}
if [[ ! "${workload}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "ERROR: invalid CONTROLLER_WORKLOAD label" >&2
  exit 2
fi
report_tag=${CONTROLLER_REPORT_TAG:-}
if [[ -n "${report_tag}" && ! "${report_tag}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "ERROR: invalid CONTROLLER_REPORT_TAG label" >&2
  exit 2
fi
summary_tag=${CONTROLLER_SUMMARY_TAG:-}
if [[ -n "${summary_tag}" && ! "${summary_tag}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "ERROR: invalid CONTROLLER_SUMMARY_TAG label" >&2
  exit 2
fi
trace_dir="${repo_root}/power_model/controller_traces/${workload}"
gate_dir="${repo_root}/power_model/gate_activity/controller/${workload}"
report_dir="${repo_root}/power_model/controller_power_reports"
controller_variant=${CONTROLLER_VARIANT:-baseline}
flow_variant=${CONTROLLER_FLOW_VARIANT:-base}
if [[ ! "${flow_variant}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "ERROR: invalid CONTROLLER_FLOW_VARIANT label" >&2
  exit 2
fi
hotspot_count=${CONTROLLER_POWER_HOTSPOT_COUNT:-}
if [[ -n "${hotspot_count}" && ! "${hotspot_count}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: CONTROLLER_POWER_HOTSPOT_COUNT must be a positive integer" >&2
  exit 2
fi
case "${controller_variant}" in
  baseline)
    design=sa_idle_predictor_controller
    gate_generator=generate_controller_gate_activity.sh
    gate_stem=core
    report_stem=controller
    vcd_suffix=vcd
    ;;
  clock_gated)
    design=sa_idle_predictor_controller_clock_gated
    gate_generator=generate_controller_clock_gated_gate_activity.sh
    gate_stem=clock_gated_core
    report_stem=controller_clock_gated
    vcd_suffix=vcd.gz
    ;;
  *)
    echo "ERROR: CONTROLLER_VARIANT must be baseline or clock_gated" >&2
    exit 2
    ;;
esac
if [[ -n "${report_tag}" ]]; then
  report_stem="${report_stem}_${report_tag}"
fi
netlist="${orfs_results}/asap7/${design}/${flow_variant}/1_2_yosys.v"
minimum_free_gib=${CONTROLLER_MIN_FREE_GIB:-40}
if [[ ! "${minimum_free_gib}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: CONTROLLER_MIN_FREE_GIB must be a positive integer" >&2
  exit 2
fi
minimum_free_bytes=$((minimum_free_gib * 1024 * 1024 * 1024))
if (( $# > 0 )); then
  cores=("$@")
else
  cores=(0 1 2 3)
fi

cell_models=(
  "${repo_root}/power_model/asap7_cells/asap7sc7p5t_AO_RVT_TT_201020.v"
  "${repo_root}/power_model/asap7_cells/asap7sc7p5t_INVBUF_RVT_TT_201020.v"
  "${repo_root}/power_model/asap7_cells/asap7sc7p5t_OA_RVT_TT_201020.v"
  "${repo_root}/power_model/asap7_cells/asap7sc7p5t_SEQ_RVT_TT_220101.v"
  "${repo_root}/power_model/asap7_cells/asap7sc7p5t_SIMPLE_RVT_TT_201020.v"
)

for command_name in iverilog vvp docker; do
  command -v "${command_name}" >/dev/null || {
    echo "ERROR: missing ${command_name}" >&2
    exit 1
  }
done
[[ -f "${netlist}" ]] || { echo "ERROR: missing ${netlist}" >&2; exit 1; }
mkdir -p "${gate_dir}" "${report_dir}"
chmod 0777 "${report_dir}"

summary_suffix=""
if [[ -n "${summary_tag}" ]]; then
  summary_suffix="_${summary_tag}"
fi
summary="${report_dir}/${report_stem}_${workload}_all_cores${summary_suffix}_summary.txt"
: > "${summary}"

for core in "${cores[@]}"; do
  trace="${trace_dir}/sa_controller_core${core}.csv"
  vcd="${gate_dir}/${gate_stem}${core}.${vcd_suffix}"
  gate_log="${gate_dir}/${gate_stem}${core}.log"
  prefix="${report_dir}/${report_stem}_${workload}_core${core}_tt"
  power_log="${prefix}.power.log"
  power_rpt="${prefix}.power.rpt"
  unannotated_rpt="${prefix}.unannotated.rpt"
  hotspot_rpt="${prefix}.highest_power_instances.rpt"

  [[ -f "${trace}" ]] || { echo "ERROR: missing ${trace}" >&2; exit 1; }
  free_bytes=$(df --output=avail -B1 "${repo_root}" | tail -1 | tr -d ' ')
  if (( free_bytes < minimum_free_bytes )); then
    echo "ERROR: less than ${minimum_free_gib} GiB free before core ${core}" >&2
    exit 1
  fi

  echo "CONTROLLER_PAIR core=${core} stage=gate_replay start=$(date -Is)"
  gate_start=$SECONDS
  bash "${repo_root}/power_model/${gate_generator}" \
    "${trace}" "${netlist}" "${vcd}" "${cell_models[@]}" \
    2>&1 | tee "${gate_log}"
  gate_seconds=$((SECONDS - gate_start))

  replay_line=$(grep 'CONTROLLER_REPLAY_SUMMARY' "${gate_log}" | tail -1 || true)
  if [[ -z "${replay_line}" || "${replay_line}" != *'"safety_violations":0'* ]]; then
    echo "ERROR: replay failed safety validation for core ${core}; VCD retained" >&2
    exit 1
  fi
  vcd_bytes=$(stat -c %s "${vcd}")

  echo "CONTROLLER_PAIR core=${core} stage=tt_power start=$(date -Is)"
  power_start=$SECONDS
  docker run --rm \
    -v "${repo_root}:/workspace" \
    -v "${orfs_results}:/OpenROAD-flow-scripts/flow/results" \
    "${orfs_image}" \
    sh -lc '
      POWER_LIB_FILES="$(find /OpenROAD-flow-scripts/flow/platforms/asap7/lib/NLDM \
        -type f -name "*RVT_TT_nldm*.lib*" | sort | paste -sd"|" -)" \
      POWER_ODB=/OpenROAD-flow-scripts/flow/results/asap7/'"${design}"'/'"${flow_variant}"'/1_synth.odb \
      POWER_SDC=/OpenROAD-flow-scripts/flow/results/asap7/'"${design}"'/'"${flow_variant}"'/1_synth.sdc \
      POWER_VCD=/workspace/power_model/gate_activity/controller/'"${workload}"'/'"${gate_stem}${core}"'.'"${vcd_suffix}"' \
      POWER_SCOPE=tb_sa_idle_predictor_controller_replay/dut \
      POWER_OUTPUT_PREFIX=/workspace/power_model/controller_power_reports/'"${report_stem}"'_'"${workload}"'_core'"${core}"'_tt \
      POWER_COVERAGE_ACCEPTED=YES \
      POWER_HIGHEST_POWER_INSTANCES='"${hotspot_count}"' \
      /OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad \
        -no_init -exit /workspace/power_model/openroad/power_from_gate_vcd.tcl
    ' 2>&1 | tee "${power_log}"
  power_seconds=$((SECONDS - power_start))

  if ! grep -Eq '^unannotated[[:space:]]+0([[:space:]]|$)' "${power_log}"; then
    echo "ERROR: annotation is not 100% for core ${core}; VCD retained" >&2
    exit 1
  fi
  if [[ ! -s "${unannotated_rpt}" ]] ||
     ! grep -q '^Unannotated pins:' "${unannotated_rpt}"; then
    echo "ERROR: missing annotation report for core ${core}; VCD retained" >&2
    exit 1
  fi
  total_line=$(grep '^Total[[:space:]]' "${power_rpt}" | tail -1 || true)
  if [[ -z "${total_line}" ]]; then
    echo "ERROR: missing total power for core ${core}; VCD retained" >&2
    exit 1
  fi
  if [[ -n "${hotspot_count}" && ! -s "${hotspot_rpt}" ]]; then
    echo "ERROR: missing hotspot report for core ${core}; VCD retained" >&2
    exit 1
  fi

  {
    echo "core=${core}"
    echo "gate_seconds=${gate_seconds}"
    echo "power_seconds=${power_seconds}"
    echo "vcd_bytes=${vcd_bytes}"
    echo "${replay_line}"
    grep -E '^(Annotated|vcd[[:space:]]|unannotated[[:space:]])' \
      "${power_log}" | tail -3
    echo "${total_line}"
    echo
  } | tee -a "${summary}"

  # The large VCD is removed only after replay safety, complete annotation,
  # and a nonempty TT power report have all been verified.
  rm -f "${vcd}"
  echo "CONTROLLER_PAIR core=${core} complete; verified VCD deleted"
done

echo "CONTROLLER_PAIR_COMPLETE summary=${summary}"
