# ONNXim Systolic-Array Power Test-Vehicle Specification

Status: frozen reference specification for architecture-level power modeling.

## Physical reference point

- Flow: OpenROAD Flow Scripts. Nangate45/FreePDK45 is retained as a
  reproducible 45-nm scaling/validation point. The 1-GHz characterization uses
  the ORFS ASAP7 predictive platform if timing closes at its documented
  library corner.
- Report the exact platform, library corner, voltage, and temperature emitted
  by each flow. ASAP7 results must be identified as predictive rather than
  measured silicon data.
- ASAP7 RVT FF is a fast-corner feasibility/sensitivity point, not the nominal
  power result. Use matching RVT TT libraries for nominal area and activity-
  annotated power and RVT SS libraries for worst-case setup timing. Retain FF
  as the fast-corner timing/leakage bound. If the installed platform lacks TT
  or SS, report that limitation and do not relabel FF as nominal.
- Clock target: 1.000 ns (1 GHz), matching ONNXim `core_freq = 1000`.
- Initial implementation: 16x16 array. Target architecture: hierarchical
  128x128 array.
- Report post-synthesis and post-route results separately. Do not frequency-
  scale a timing-failing Nangate45 implementation and present it as a 1-GHz
  physical result, or present either open platform as a proprietary TPU
  process.

## Datapath

- Signed 16-bit activation and weight operands (`precision = 2` bytes).
- Signed 32-bit partial sums/accumulators (`full_precision = 4` bytes).
- Two-stage PE datapath with initiation interval one: stage 1 registers the
  signed 16x16 product and its matching 32-bit incoming partial sum; stage 2
  performs and registers the signed 32-bit addition.
- Activations retain their one-cycle eastward forwarding path. Partial sums
  take two cycles per southward PE hop, so row inputs must use the corresponding
  two-cycle wavefront skew.
- Weight-stationary dataflow: weights shift south during preload and then stay
  in each PE; activations move east and partial sums move south during GEMM.
- PE state consists of the weight and forwarding registers plus the registered
  product, matching partial sum, and product-valid bit.

The internal RTL pipeline is a physical implementation choice for meeting the
characterization clock, not a change to ONNXim's architectural timing model.
Because its initiation interval is one, it is used to characterize power per
ONNXim active/preload/idle cycle while ONNXim retains its original instruction
latency and activity traces. Pipeline fill/drain latency is not imported into
the simulator or counted a second time.

The RTL is a power-characterization test vehicle, not a claim that ONNXim
already specifies this exact circuit. Any later RTL change requires a new
characterization identifier.

## Power modes and boundary

- `ACTIVE_GEMM`: compute enabled; MAC and forwarding registers active.
- `PRELOAD`: weight shift enabled; MAC disabled.
- `IDLE_ON`: array state held. Clock-tree power must be reported explicitly;
  RTL clock-enable muxing must not be mislabeled as physical clock gating.
- `CLOCK_GATED_IDLE_ARRAY`: state held and the array clock input stopped after
  pipeline drain. This measures the gated array only; clock-gate insertion,
  enable generation, controller power, and any ungated clock-tree segment are
  separate overheads.
- `SLEEP`: PE array power-gated; controller and isolation remain always-on.

SRAM, vector units, NoC, DRAM, and the predictor/controller are outside the SA
power domain. CACTI results, if later used, must be reported separately.

Power-gating the current RTL loses PE state. It is therefore legal only when
no weight or partial-sum state is needed after wake (for example, when the next
SA operation reloads weights), unless retention or reload cost is added.

## Characterization experiments

Generate activity traces for identical implemented netlists:

1. weight preload;
2. steady, full-utilization GEMM with deterministic pseudo-random operands;
3. partial row/column utilization points;
4. powered-on idle with state held.

For each experiment report total, internal, switching, and leakage power,
along with area, achieved timing, cell count, and whether clock-tree power is
included. Default toggle-rate power is not acceptable for final results.

RTL VCDs are characterization stimuli only. Yosys optimization and technology
mapping do not preserve sufficient internal RTL names to assume direct VCD-to-
ODB annotation. Final synthesis-stage activity therefore comes from simulation
of the mapped Verilog netlist with the ASAP7 functional cell models. OpenSTA's
`report_activity_annotation` must be recorded for every mode; reject a result
with unexplained unannotated functional pins. Zero-delay mapped-netlist power
must be labeled as excluding delay-dependent glitch and routed-interconnect
power. A later post-route/SDF experiment is required before claiming signoff-
quality absolute dynamic power.

Characterize one PE and at least 16x16 and 32x32 arrays. Use a full
hierarchical 128x128 synthesis/place-and-route if tractable. Otherwise,
extrapolate only after fitting and reporting array-size scaling error.

The minimum checkpoint before attempting 128x128 is 32x32. Compare total and
per-PE area, sequential fraction, cell count, and timing against 16x16. A
64x64 point is useful only when resources permit and the 16-to-32 result is
nonlinear; it is not required when its projected memory demand is already
infeasible and the two measured points exhibit near-linear scaling.

For the current resource-limited study, the measured 16x16 and 32x32
structural points scale within 0.4%, while 32x32 synthesis already consumes
11.57 GB peak RSS. A 64x64 or 128x128 run is therefore not required on this
host. The exact two-point structural and power extrapolation rules are frozen
in `STRUCTURAL_EXTRAPOLATION.md`.

## Transition and controller parameters

Ordinary standard-cell synthesis does not provide power-switch rail ramp or
inrush behavior. Entry/wake latency, transition energy, and powered-off
leakage must come from characterized power switches or explicitly cited,
technology-matched literature and must be swept in sensitivity analysis.

The frozen predictor/controller will be synthesized as a separate always-on
RTL block. Its gate-level activity must be generated from ONNXim event traces.

The controller characterization includes two explicit resident-tile metadata
descriptors rather than treating the simulator's instruction-deque scan as
free hardware.  Each descriptor stores the next-SA presence/opcode/distance,
remaining instruction count, and up to two unique required-operand readiness
and FIFO-tail-ticket records.  Two operands cover the measured OPT-125M GEMM
case; a larger or unrepresentable source set disables power gating.  Tile or
operand changes force immediate wake; new descriptor values are considered
for entry after registration, so they cannot create a stale extension.
The synthesized overhead therefore includes descriptor storage, FIFO-bound
arithmetic, retention checks, threshold comparison, and the ON/OFF policy FSM.
It does not include the baseline instruction storage or SRAM itself.

The synthesized FIFO arithmetic is specialized to the evaluated ONNXim
configuration (`icnt_injection_ports_per_core = 16`, `core_freq = icnt_freq =
1000 MHz`).  For this case the proven dequeue bound reduces exactly to
`(position - 1) >> 4`; no divider or clock-ratio arithmetic is required.
Results are not reused for another port count or clock ratio without a new
constant-ratio implementation and characterization.

The implementation uses 20-bit modulo FIFO tickets and horizons.  This covers
up to 1,048,575 queued requests, versus the measured OPT-125M maximum position
of 101,280.  Queue overflow or ticket ambiguity must deassert
`core_queue_tickets_exact`, which wakes immediately and disables FIFO-based
entry.  While OFF, the controller never recomputes or extends the saved
frontier: it decrements the entry certificate once per core cycle, while any
resident-descriptor change, committed-work appearance, or loss of ticket
exactness wakes immediately.  This is conservative relative to V5 because
later stronger certificates are ignored until after wake.

The entry certificate is implemented as a three-stage pipeline: stage 1
freezes the four operand FIFO bounds, stage 2 registers the per-resident
candidate and future-tile-cap frontier, and stage 3 applies the V5 global
minimum/work-drain rule.  Entry therefore requires at least 170 freshly
certified cycles (166 electrical break-even + one wake-guard cycle + three
pipeline cycles), and the saved countdown is reduced by those three elapsed
cycles before sleep is asserted.  Descriptor writes invalidate every stage.
The immediate wake path is not pipelined and does not depend on
certificate-generation logic.

Controller dynamic power is characterized from workload-derived activity,
not a synthetic toggle factor. Setting `ONNXSIM_SA_CONTROLLER_TRACE_DIR`
enables one change-only CSV per core at the existing predictor recomputation
point. The trace contains the exact synthesized-controller inputs and full
descriptor-write pulses; it is otherwise disabled and does not affect
simulated scheduling or time. Required operands use the same SRAM entry and
MOVIN tail-ticket association as the predictor. An invalid operand whose tail
is no longer strictly ahead of the dequeue frontier, exceeds the 20-bit
window, is unallocated, or has no pending memory is marked unsupported and
therefore cannot authorize sleep.

Each core trace is replayed independently through the mapped controller
netlist. Gate-level VCD annotation coverage must be 100% before ASAP7 TT power
is accepted. Per-core controller energy is integrated over its own trace;
energies are summed across cores. Reset activity is excluded from the VCD.

## ONNXim integration contract

ONNXim will eventually accumulate:

`E = sum(P_mode * time_mode) + N_entry * E_entry + N_wake * E_wake + E_controller`

Active power is selected from characterized utilization points. A wake starts
at least `wake_latency` before the predictor's safe frontier. The controller
may enter sleep only when the certified horizon exceeds the full measured
break-even threshold.
