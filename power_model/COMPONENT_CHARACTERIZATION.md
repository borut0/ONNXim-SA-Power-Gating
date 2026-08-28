# Compact component characterization

Set `ONNXSIM_COMPONENT_CHARACTERIZATION=1` for final workload runs. It emits
one final JSON record per core plus one system, NoC, DRAM, and memory-stage
record. Periodic core/NoC/DRAM logs are suppressed in this mode.

## Output records

- `COMPONENT_CORE`: SA/vector active and idle cycles, overlap, neither-active
  cycles, core request-FIFO occupancy, and mutually exclusive core-cycle
  attribution (`compute`, memory/dependency/scheduler stall, `other`).
- `SA_STATE_BREAKDOWN`: mutually exclusive fine-grained SA architectural
  states. `ACTIVE_GEMM_PRELOAD_OPCODE` means the modeled active interval of a
  GEMM_PRELOAD instruction; ONNXim does not expose internal preload-only and
  MAC subcycles.
- `COMPONENT_NOC`: cycles with at least one ONNXim memory packet between
  injection and destination delivery. This is demand occupancy, not router or
  link utilization.
- `COMPONENT_DRAM`: accepted/completed demand transactions, read/write bytes,
  per-channel accepted-to-completion occupancy, and transaction-bandwidth
  utilization using the existing Ramulator wrapper normalization. This is not
  DRAM-command or bank utilization.
- `COMPONENT_MEMORY_STAGES`: request-weighted min/max/average latency for core
  queue, core-to-memory interconnect, DRAM, and memory-to-core interconnect.
- `COMPONENT_SYSTEM`: frequencies, cycle counts, topology, and end-to-end
  packet-flow totals.

Summarize and validate a completed log with:

```sh
python3 power_model/summarize_component_characterization.py RUN.log \
  > RUN.component_summary.json
```

The summary is acceptable only when `validation_pass` is `true`.

## Costly research instrumentation

These experiments are retained for reproducibility but are disabled unless
their explicit environment variable is set:

- `ONNXSIM_SA_PHASE_MEASURE=1`
- `ONNXSIM_SA_PREDICTOR_MEASURE=1`
- `ONNXSIM_SA_RETENTION_REASON_MEASURE=1`
- `ONNXSIM_SA_CONTROLLER_TRACE_DIR=...`
- `ONNXSIM_MEMORY_STAGE_PROFILE_DETAILED=1` (latency histograms/percentiles)

The compact characterization stores only constant-size counters and
request-latency sums/minima/maxima. It does not generate per-cycle traces or
latency-sized histograms.
