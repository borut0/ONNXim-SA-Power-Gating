# Compact component characterization

Set `ONNXSIM_COMPONENT_CHARACTERIZATION=1` for final workload runs. It emits
one final JSON record per core plus one system, NoC, DRAM, and memory-stage
record. Periodic core/NoC/DRAM logs are suppressed in this mode.

## Output records

- `COMPONENT_CORE`: SA/vector active and idle cycles, overlap, neither-active
  cycles, core request-FIFO occupancy, and mutually exclusive core-cycle
  attribution (`compute`, memory/dependency/scheduler stall, `other`), plus
  whole-main-SPAD, whole-accumulator-SPAD, and all-SPAD logical event idleness.
  SPAD reads are recorded only when EX is accepted or MOVOUT requests are
  generated; writes are recorded only by `Sram::fill`. Readiness probes are not
  accesses, and banks/ports/access latency are not inferred.
- `SA_STATE_BREAKDOWN`: mutually exclusive fine-grained SA architectural
  states. `ACTIVE_GEMM_PRELOAD_OPCODE` means the modeled active interval of a
  GEMM_PRELOAD instruction; ONNXim does not expose internal preload-only and
  MAC subcycles.
- `COMPONENT_NOC` (schema v2): model-quiescent cycles. Simple NoC checks
  every input/output buffer and outstanding packet count (unmodeled router/link
  structures are vacuous). BookSim additionally checks traffic-manager source
  and in-flight state, every supported router/VC pipeline and buffer, internal
  plus injection/ejection flit and credit channels, and all adapter boundary
  buffers. Unsupported router implementations make the metric unavailable.
- `COMPONENT_DRAM` (schema v2): Ramulator2 model-quiescent cycles requiring
  zero ONNXim demand outstanding, empty wrapper request/return queues, empty
  pending/active/read/write/priority controller queues, and no command, modeled
  completion transfer, or refresh/maintenance activity in the tick. Simple DRAM
  and Ramulator1 report this metric unavailable. This is not physical data-bus
  or post-command timing-recovery idleness.
- `COMPONENT_MEMORY_STAGES`: request-weighted min/max/average latency for core
  queue, core-to-memory interconnect, DRAM, and memory-to-core interconnect.
- `COMPONENT_SYSTEM`: frequencies, cycle counts, topology, and end-to-end
  packet-flow totals.

Summarize and validate a completed log with:

```sh
python3 power_model/summarize_component_characterization.py RUN.log \
  > RUN.component_summary.json
```

The summary is acceptable only when `validation_pass` is `true`. Current
consumers require schema v2 and reject historical v1 demand-idle fields rather
than falling back to them. Historical logs and summaries remain preserved.

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
