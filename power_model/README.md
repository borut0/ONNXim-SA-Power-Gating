# Systolic-array power test vehicle

The frozen methodology and assumptions are in `SA_POWER_MODEL_SPEC.md`.

The initial RTL target is `ws_array_16x16`, clocked at 1 GHz. Each PE uses a
two-stage, initiation-interval-one MAC pipeline. `tb_ws_pe.sv` checks signed
MAC arithmetic and pipeline drain; the array testbench produces deterministic
preload, active-GEMM, and idle activity in `ws_array_16x16.vcd`.

The host checkout does not currently contain Yosys, Verilator, OpenROAD, or a
standard-cell library. To run the 1-GHz physical experiment, use an OpenROAD
Flow Scripts installation containing the ASAP7 platform:

```sh
make -f /path/to/OpenROAD-flow-scripts/flow/Makefile \
  DESIGN_CONFIG=/absolute/path/to/ONNXim/power_model/openroad/asap7/config.mk
```

The Nangate45 configuration remains available as a reproducible older-node
scaling point; it is not timing-clean at 1 GHz for the current inferred
multiplier.

Before publishing power, simulate the RTL, generate VCD/SAIF, and run
activity-annotated post-synthesis or post-route power analysis. ORFS default
toggle-rate power is not a substitute.

For controller power, set `ONNXSIM_SA_CONTROLLER_TRACE_DIR` on an otherwise
unchanged ONNXim run. This produces change-only `sa_controller_coreN.csv`
files. Validate them with `summarize_controller_trace.py`, then replay each
core separately through the mapped controller using
`generate_controller_gate_activity.sh`. Use scope
`tb_sa_idle_predictor_controller_replay/dut` for the existing OpenROAD VCD
coverage and TT power scripts. Never accept a controller power report with
unannotated instances.

For final net energy, first rerun the fast functional controller replay with
the current replay testbench so that every completed sleep-episode duration is
emitted. Then pass that replay log to `summarize_controller_net_energy.py`.
The script strictly checks replay safety/completeness and 100% controller VCD
annotation before combining the exact RTL episodes, the HSPICE duration-energy
curve, and full-workload controller energy. It does not use the long-idle-only
V5 log as a substitute for full-workload RTL episodes.

The always-clocked controller is retained as the power baseline. The separate
`sa_idle_predictor_controller_clock_gated` wrapper uses explicit ASAP7 ICGs
and local clock buffers; it does not alter the frozen predictor equations. The
resident-descriptor bank is clocked only on descriptor writes, while the
evaluation/countdown bank retains its independently gated clock. Its
clock-enable prerequisite is derived from the controller's existing registered
descriptors. Existing certificate-stage valid bits keep the evaluation clock
running for the cleanup edge that invalidates stale certificates before it
stops; the wrapper therefore adds no always-clocked state. Accept the optimization
only after `compare_controller_clock_gated_replay.sh` reports exact episode-
timeline equivalence for every evaluated core and synthesis confirms exactly
two mapped ICG/buffer pairs with no electrical-limit violations.
