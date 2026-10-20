# SA PE-state opportunity profile

For the newer opt-in native-clock PE/two-bank replay, see
[`EVENT_DRIVEN_SA.md`](EVENT_DRIVEN_SA.md). The profile described below is the
earlier structural opportunity model; its limitations and status should not
be read as the status of the newer replay.

This profile is an opt-in, measurement-only extension to ONNXim. It reports a
parameter-free structural opportunity split for each core; it does not change
instruction scheduling, SA timing, outputs, or power-state control.

## Evidence classification

- ONNXim core cycles, SA instruction intervals, and tile dimensions are
  `measured` simulator outputs.
- Useful MACs and all PE-cycle opportunity counts are `derived` from those
  outputs using the equations below.
- ReGate power, voltage, entry/wake latency, transition energy, and break-even
  time are not inputs to this profile and remain `pending` unless supplied by
  a separate characterized artifact.
- The split is not a cycle-accurate ReGate `PE_on` control trace. ReGate's
  released HW/Full artifact equation is separately pinned in
  `calculate_regate_artifact_model.py` at commit
  `004dbfccb90cfc0a61bb79bbee3a5770b4b3897b`, source locator
  `trace_util/llm_ops_generator/power_analysis_lib.py:328-356,432-540`.

## Derived model

For an ONNXim weight-stationary instruction with dimensions
`tile_m`, `tile_k`, and `tile_n`:

```text
mapped PEs               = tile_m * tile_k
useful-MAC ON PE-cycles  = tile_m * tile_k * tile_n
```

The mapped stationary-weight footprint is an origin-packed
`tile_m`-by-`tile_k` rectangle. Its residency window is ONNXim's half-open
`[start_cycle, finish_cycle)` interval. Overlapping rectangles are unioned in
physical time before computing capacity, so overlapping instructions never
double-count a PE-cycle.

Across merged SA-active time:

```text
W_on candidate = resident-weight PE-cycles - useful-MAC ON PE-cycles
OFF candidate  = array area * merged active cycles
                 - resident-weight PE-cycles
```

The mapped non-MAC difference is a structural **upper candidate** for `W_on`,
not proof that every such cycle can use that state. Likewise, unmapped
capacity is an `OFF` candidate, not a generated power-control trace. ONNXim
does not model ReGate's row/column zero detection, prefix forwarding,
diagonal `PE_on` propagation, per-row control queues, or actual per-PE state
transitions in this profile.

The implementation rejects the classification when useful-MAC cycles exceed
resident capacity or any accounting invariant fails. It never clips such an
inconsistency into a plausible-looking result.

`accounting_exact: true` means only that the arithmetic partition is exact
under the stated origin-packed footprint and aggregate useful-MAC assumptions.
It does not mean that the result is an exact functional ReGate trace.

Whole-SA idle PE-cycles are reported as
`temporal_idle_unclassified_pe_cycles`. They are not called `W_on` or `OFF`,
because this footprint profile alone does not prove whether resident weights
remain live. The existing causal retention/hybrid policy must classify those
episodes separately. The issue-to-start interval is likewise retained as a
committed/reserved interval and is not silently counted as OFF.

## Version 6 analytical wave and weight-lifetime model

Version 6 retains the conservative version-5 result above and adds a separate
analytical model. The dimension mapping comes from ONNXim's implementation:
`GemmWS.cc` and `ConvWS.cc` emit useful work as
`tile_m * tile_k * tile_n`; the existing profiler maps an origin-packed
`tile_m`-row by `tile_k`-column stationary footprint; and
`SystolicWS::get_inst_compute_cycles` schedules
`core_height + core_width - 2 + max(tile_n,4)` cycles. Consequently, the new
useful wave and ONNXim's full-array scheduled duration are reported separately.

For PE row `r`, column `c`, and streamed wave `b`, the analytical schedule is

```text
local_cycle = r + c + b,
0 <= r < tile_m, 0 <= c < tile_k, 0 <= b < tile_n.
```

Therefore `U(t)` is exactly the coefficient of

```text
(1+x+...+x^(tile_m-1))
* (1+x+...+x^(tile_k-1))
* (1+x+...+x^(tile_n-1)).
```

`useful_macs_at_cycle` evaluates this coefficient by inclusion--exclusion. The
model validates that the wave needs `tile_m + tile_k + tile_n - 2` cycles and
that `sum_t U(t) = tile_m * tile_k * tile_n`. The field
`simulator_active_non_wave_pe_cycles` preserves, rather than hides, the extra
capacity in ONNXim's full scheduled interval.

### Weight lifetime and fail-closed behavior

GEMM and convolution generators attach an interned identity formed from the
exact global weight-address set and mapped footprint. A known `GEMM_PRELOAD`
starts a lifetime; later `GEMM` instructions extend it only when identity and
footprint match. The lifetime ends after the last matched consumer finishes
and can overlap a newer preload while the previous wave drains. Unions prevent
double counting.

ONNXim does not expose wire-by-wire preload movement. The model therefore uses
an explicit **atomic-preload approximation**: the mapped footprint becomes
weight-valid at `start_cycle`. The mapped `[issue_cycle,start_cycle)` footprint
is `preload_committed_unclassified`; it is never silently called OFF or
`W_on`.

Unknown identities, a GEMM without a preceding preload, identity/footprint
mismatches, a wave outside its scheduled interval, or overlapping useful
waves make complete classification fail closed. Unresolved capacity remains
`unclassified`.

For every shared region the priority is:

```text
any useful MAC                                      -> ON
otherwise unresolved preload commitment             -> unclassified
otherwise any compiler-live stationary weight       -> W_on
otherwise                                            -> structural OFF candidate
```

One active PE therefore forces the complete shared region to ON. Structural
OFF is an offline exact-liveness opportunity, not a generated ReGate signal or
a measured low-power state. `partition_exact_under_model: true` means only
that these categories partition the horizon under the stated analytical and
atomic-preload assumptions; it does not establish functional ReGate parity or
energy savings.

## Build and focused validation

The source list is collected by CMake at configure time, so reconfigure once
after adding `src/SaPeStateOpportunity.cc`:

```sh
cmake -S . -B build
cmake --build build -j2 --target Simulator_test
./Simulator_test --gtest_filter=SaPeStateOpportunityTest.*
python3 -B -m unittest power_model.tests.test_calculate_regate_artifact_model
```

## Workload collection

Enable the profiler without enabling any power-state controller:

```sh
ONNXSIM_SA_REGION_PROFILE=1 \
ONNXSIM_SA_REGION_SIZES=8,16,32,64,128 \
./build/bin/Simulator \
  --config ./configs/systolic_ws_128x128_c4_simple_noc_tpuv4.json \
  --model ./example/models_list.json \
  2>&1 | tee sa_pe_state_profile.log
```

The model file used by this example is `example/models_list.json`; replace it
with the specific workload list being evaluated and preserve both that file
and the raw log. The version-6 analytical state accounting is independent of the
requested region-size list; omitting one-PE regions avoids unnecessary
per-region profiling overhead. Aggregate and validate the per-core records
with:

```sh
python3 power_model/summarize_sa_region_profile.py \
  sa_pe_state_profile.log --expected-cores 4 \
  > sa_pe_state_profile_summary.json
```

Only a summary with `validation_pass: true` may be used as derived evidence.
Even then, the candidate counts are opportunity estimates, not energy savings.

## ReGate logical-control shadow approximation

`ONNXSIM_REGATE_FUNCTIONAL_PROFILE=1` enables a separate, measurement-only
logical-control approximation. It does not enact a ReGate controller, enable
the retention controller, alter instruction scheduling, or modify numerical
outputs.

For an origin-packed mapped tile, the structural prefix masks are:

```text
row_on[r] = (r < tile_m)
col_on[c] = (c < tile_k)
```

For each mapped PE `(r,c)`, the idealized diagonal control interval is:

```text
PE_on(r,c) = 1 for
[start_cycle + r + c, start_cycle + r + c + tile_n)
```

The rising edge therefore follows the analytical input wave, and the falling
edge drains diagonally after the final input. Per-PE state priority is
`ON > committed/unclassified > W_on > OFF`. Known weight-block lifetimes
establish `W_on`; structurally unused PEs may become `OFF`. Missing weight
identity, mismatched lifetimes, overlapping useful waves, or invalid timing
fail closed as unclassified.

This is a zero-latency logical-control model. ONNXim exposes the mapped
footprint atomically at instruction start, so it cannot reconstruct
wire-by-wire weight preload. It also does not expose tensor values, so only
structural padding is classified; numerical zero-weight sparsity remains
`pending`.

The source mapping is ReGate Section 4.1 for the three PE states, row/column
prefix controls, and diagonal `PE_on`; Table 3 reports the reference PE
power delay and BET. The pinned public artifact is commit
`004dbfccb90cfc0a61bb79bbee3a5770b4b3897b`,
`trace_util/llm_ops_generator/power_analysis_lib.py:328-356,432-540`.

`ideal_low_power_pe_cycle_exposure_percent` is the derived fraction of
PE-cycle capacity classified as `W_on` or `OFF`. It is **not energy
savings**. The emitted `savings.energy_saving_percent` remains null until
explicit ReGate W_on/OFF power factors and transition costs are supplied by a
separate reference or characterized artifact.

After building and running the focused tests, collect a workload with:

```sh
ONNXSIM_REGATE_FUNCTIONAL_PROFILE=1 \
./build/bin/Simulator \
  --config ./configs/systolic_ws_128x128_c4_simple_noc_tpuv4.json \
  --model ./example/models_list.json \
  2>&1 | tee regate_functional.log
```

The model list above is only an example. Preserve the actual model-list file,
configuration, raw log, and all `REGATE_FUNCTIONAL_PROFILE` records used in
an experiment. A record may be interpreted only when
`validation_pass: true`.

After the run, explicitly select the pinned ReGate HW/Full factor and calculate
a ReGate-only gross static sensitivity estimate:

```sh
python3 power_model/summarize_regate_functional.py \
  regate_functional.log --expected-cores 4 \
  --regate-hw-full-reference \
  > regate_functional_summary.json
```

The public artifact uses 0.03 in its aggregate utilization equation; it does
not define separate `W_on` and `OFF` state-power ratios. This calculator
therefore transfers 0.03 to both shadow states as an explicit cross-model
assumption. The resulting output is labeled `extrapolated`, not
artifact-equivalent. Unclassified cycles are charged at ON power. The
resulting
`gross_static_saving_percent` excludes transition, controller, switch,
isolation, routing, and physical wake/stall costs; net energy saving therefore
remains `pending`.


## Two-bank ReGate-like analytical shadow (version 2)

Set `ONNXSIM_REGATE_TWO_BANK_PROFILE=1` to emit one
`REGATE_TWO_BANK_PROFILE` JSON record per core. This opt-in profile observes
the existing ONNXim instruction trace; it changes neither scheduling nor
numerical outputs. It is separate from the version-1
`REGATE_FUNCTIONAL_PROFILE` record and its calculator.

Each PE has two **logical** weight banks. Every explicit `GEMM_PRELOAD`
creates a distinct preload instance, even if its interned weight-block ID
matches a prior preload. An ordinary `GEMM` is attributed to the latest
earlier matching preload by weight-block ID and `tile_m × tile_k` footprint.
This is an offline, future-aware association for accounting; ONNXim does not
provide a physical bank selector or a causal two-bank controller.

At issue, the selected bank's entire origin-packed footprint is treated as
powered and loading. At instruction start it is treated as ready. Thus
`[issue_cycle, start_cycle)` contributes mapped bank PE-cycles to `W_on`
unless an overlapping activation wave makes a PE `ON`. The separate
`preload_bank_pe_cycles` field counts loading-bank occupancy; it is not
preload write energy. After a bank's last matched consumer finishes, its
weight remains resident and can be replaced by a later preload.

For mapped PE `(r,c)`, an instruction's assumed activation wave is
`[start_cycle+r+c, start_cycle+r+c+tile_n)`. `ON` takes priority over
mapped bank occupancy. With no wave, a PE is `W_on` if one or two banks map
to it, and `OFF` otherwise. The record separates `ON` by zero, one, and
two powered banks, and separates `W_on` by one and two banks. The model
uses the union of each bank's origin-packed forwarding rectangle. This
per-bank route-control interpretation is an architectural assumption; a
single shared row-prefix OR and column-prefix OR could power additional
cross-product PEs for differently shaped bank footprints.
Classifying PEs outside those rectangles as `OFF` also assumes routing
closure or bypass. A physical array that forwards partial sums through
unmapped rows could require those PEs to stay powered; this trace does not
functionally verify that path.

Bank assignment gives empty banks priority, then replaces a bank only after
all its matched instruction consumers finish. The assignment uses knowledge
of later consumers from the completed trace. A third preload while both banks
are still needed produces `two_bank_conflicts > 0`; the trace is incompatible
with this issue-time atomic two-bank model. Per-PE staggered physical loading
could behave differently, but ONNXim does not expose row-arrival events to
test that case. Missing identity, an unmatched GEMM, an invalid wave, a
compute collision, or any bank conflict also fails closed:

```text
validation_pass = false
unclassified = capacity
on = w_on_one_bank = w_on_two_banks = off = 0
```

For a validated record, `unclassified = 0` and
`capacity = on + w_on_one_bank + w_on_two_banks + off`. These are
**derived** logical PE-cycle counts, not circuit power measurements.

The accounting sweeps instruction intervals for each physical PE. Its
runtime therefore grows with array area times the number of SA instructions;
budget this profiling cost for long full-array traces.

Collect a workload with the environment variable above and preserve its
configuration, model list, raw log, and all per-core records. After the user
builds and runs ONNXim in Docker, summarize those records with:

```sh
python3 power_model/summarize_regate_two_bank.py \
  regate_two_bank.log --expected-cores 4 \
  > regate_two_bank_summary.json
```

The summary rejects any invalid core or broken partition. It reports ideal
low-power state exposure, with energy saving `pending`. An optional gross
static sensitivity requires explicit factors for `on_one_bank`,
`on_two_banks`, `w_on_one_bank`, `w_on_two_banks`, and `off`, each
relative to an always-ON PE-cycle, plus `--factor-source`. Such factors are
labeled `assumed`. No factor is supplied by default, and the version-1
ReGate artifact's aggregate 0.03 factor is not applied to this two-bank
model. Net energy remains pending until separate two-bank leakage,
preload write, control, transition, and timing evidence is available.

### Diagnostic two-bank timing audit

The version-2 record now includes a separate `timing_audit` object. Its
numbers are **derived** from the same recorded issue/start/finish cycles,
origin-packed dimensions, and exact latest-prior preload matches used by the
state model. They do not change `two_bank_conflicts`, `validation_pass`,
PE-state counts, or the `pending` energy fields. The audit is populated even
when a two-bank conflict makes all state capacity unclassified. It is
unavailable if an interval is incomplete or invalid, its useful wave exceeds
its instruction, a compute collision occurs, or identity/footprint/ready
matching cannot be established.

For every explicit preload instance, both the preload instruction and every
ordinary GEMM matched to that exact instance are consumers. The two release
candidates are (1) the maximum whole-instruction `finish_cycle`, and
(2) the maximum modeled half-open last-MAC end
`start_cycle + r + c + tile_n` over those consumers. The global version of
(2) takes the maximum across that preload's mapped PEs. The load candidates
are `issue_cycle`, `start_cycle`, and, only for mapped-PE accounting,
`start_cycle + r + c` (the preload's first modeled MAC at that PE).
Intervals are half-open; a release exactly equal to another reservation is
not overlap.


The five named comparisons are:

- `issue_to_whole_finish`: issue reservation, whole-instruction release
  (the existing bank-assignment timing abstraction).
- `start_to_whole_finish`: instantaneous start reservation,
  whole-instruction release.
- `issue_to_pe_last_mac`: issue reservation, modeled last-MAC release;
  the global release is the latest mapped-PE last MAC.
- `start_to_pe_last_mac`: instantaneous start reservation, modeled
  last-MAC release.
- `pe_first_to_pe_last_mac`: instantaneous PE-local first-MAC reservation
  and PE-local last-MAC release; global occupancy is inapplicable.

Each case reports the peak number of distinct live preload instances and the
time with more than two live instances. `global.over_two_cycles` counts
cycles with more than two instances overlapping in time, irrespective of
whether their mapped footprints intersect;
`mapped_pe.over_two_pe_cycles` sums such cycles only at PEs covered by
each instance's checked footprint, and `pes_ever_over_two` counts affected
PEs. The `pe_first_to_pe_last_mac` case has no single global load event, so
its `global` entry is null. A global conflict with zero mapped-PE excess
identifies footprint or timing conservatism, not an implemented per-PE bank
assignment.

The start and PE-first-use reservations are optimistic mathematical bounds:
they assume a weight can appear instantaneously at the chosen time.
`GEMM_PRELOAD` is fused load+compute, and ONNXim does not expose actual
per-PE weight-arrival events, write bandwidth, bank selection, or forwarding
hazards. Consequently, even zero excess in these candidates cannot establish
a feasible physical two-bank schedule. Conversely, start-time reservation may
overstate occupancy for a staggered preload; only the explicit
PE-first-use ideal bound explores that possibility. Physical feasibility,
timing acceptance, and energy remain `pending` until corresponding hardware
or circuit evidence exists.

### Explicit row-wise two-bank diagnostic

The optional `rowwise_feasibility` object is **derived from an assumed row
schedule**, not measured ReGate or ONNXim timing. It does not change the
version-2 `validation_pass`, `two_bank_conflicts`, PE-state counts, or
`pending` energy fields. Without a complete explicit schedule it is
unavailable with a `pending` reason. Set all of:

```sh
ONNXSIM_REGATE_TWO_BANK_PROFILE=1
ONNXSIM_REGATE_ROW_PRELOAD_PROFILE=1
ONNXSIM_REGATE_ROW_PRELOAD_LEAD_CYCLES=<nonnegative-integer>
ONNXSIM_REGATE_ROW_PRELOAD_PERIOD_CYCLES=<positive-integer>
ONNXSIM_REGATE_ROW_PRELOAD_DURATION_CYCLES=<positive-integer>
ONNXSIM_REGATE_ROW_PRELOAD_ORDER=<ascending-or-descending>
```

The first row's hypothesized write begins at recorded preload
`start_cycle - lead`. For PE row `r`, the ordered row index is `r` for
ascending order or `tile_m - 1 - r` for descending order. Its write starts
at `start_cycle - lead + ordered_row_index × period`; weight becomes ready
after `duration` cycles. A write before recorded `issue_cycle` fails the
necessary conditions, never silently clips to issue. Such writes are
counted, but omitted from subsequent sweeps; the emitted readiness,
occupancy, and local-wait fields are then `null` with
`secondary_counts_complete: false`, not misreported as complete zeros.
The model assumes
all mapped columns in one row receive that row's weights simultaneously.
For each mapped PE, the bank remains needed until the later of write
completion and its last modeled MAC across the preload and all exactly
matched later GEMMs; intervals are
half-open, so a replacement at the release cycle does not conflict.

The top-level `preload_feasibility_audit` retains a whole-bank row-stream
check for **every** explicit `GEMM_PRELOAD`. `preload_index` is the stable
issue-order preload index and `instruction_index` is the stable issue-order
SA instruction index. Under the current origin-packed mapping, `rows` is
`tile_m`. Its equations are:

```text
fill_span = (rows - 1) * period + duration
planned_load_begin = start_cycle - fill_span
load_begin = max(issue_cycle, planned_load_begin)
bank_ready = load_begin + fill_span
first_use = start_cycle
required_stall = max(0, bank_ready - first_use)
```

Each event reports those values and an outcome of `no_added_cycle` or
`conflict`; the conflict-only subset is repeated in `offending_events`.
Totals sum the independent fixed-schedule per-PRELOAD stalls and report the
maximum. The sum is not a globally propagated runtime. This audit assumes
optimistically that each scheduled row is already available at the SA input
after ONNXim's issue-time operand-validity checks. ONNXim does not expose
post-issue per-row scratchpad read, assembly, or delivery latency; those are
`pending`, as are shared-bus arbitration, globally consistent bank selection,
forwarding, and circuit timing. It is
therefore a **derived analytical readiness check**, not a measured or fully
physical feasibility claim, and it never changes ONNXim execution.

The separate `rowwise_feasibility.per_row_local_audit` retains the existing
per-row/per-PE necessary diagnostic. Each preload reports its hypothesized
first-row `planned_load_begin`. Each mapped row reports that planned value
plus a causal `load_begin`, `bank_ready`, and the earliest modeled row
`first_use`:

```text
planned_load_begin(row) = start_cycle - lead + ordered_row_index(row) * period
load_begin(row) = max(planned_load_begin(row), issue_cycle,
                      local two-bank availability)
bank_ready(row) = load_begin(row) + duration
first_use(row) = start_cycle + row
required_stall(row) = max(0, bank_ready(row) - first_use(row))
```

The per-row outcome is `no_added_cycle` exactly when the causal
`required_stall_cycles` is zero. A plan reaching before cycle zero is clamped
to the recorded issue for the causal calculation and retains an exact
`load_begin_before_cycle_zero` schedule diagnostic rather than wrapping an
unsigned cycle. `load_begin_before_issue` and
`two_bank_unavailable_at_causal_load` are likewise schedule diagnostics; they
may be absorbed by slack and therefore do not by themselves imply an added
cycle. `bank_not_ready_at_first_use` is the conflict reason when the causal
stall is nonzero. `bank_conflicts` additionally identifies the exact mapped
PE column, earliest bank release, delayed ready cycle, first use, and stall.
The full row list is retained even for passing preloads, while
`offending_events` is the conflict-only row subset.

Rows of one preload delay that instruction together, so its
`required_stall_cycles` is the maximum row stall, not the sum. The whole-core
`total_required_stall_cycles` sums those per-preload maxima;
`max_required_stall_cycles` is the worst preload. The separately named
`sum_row_required_stall_cycles` is event exposure only and must not be read as
execution delay.

This is a **derived analytical shadow** whose row timing inputs are
**assumed**. The local bank waits hold earlier reservations fixed; delayed
rows are not propagated through a globally rescheduled shared row bus. Thus
the row-level `no_added_cycle` outcome is only a per-PE necessary check under
that assumption. It is not a measured stall, a rescheduled execution, or
proof that a shared bus, globally consistent bank selector, forwarding path,
or physical wake/write circuit can realize the schedule. The profiler does
not apply `required_stall_cycles` to ONNXim and does not change any issue,
start, finish, or total-cycle value.

The diagnostic counts mapped **PE-weight** write events, late arrivals,
and per-PE occupancy above two banks. `necessary_conditions_pass` means
only these local checks pass for the supplied architectural hypothesis.
It does not prove shared row-bus bandwidth, globally consistent bank
selection, forwarding, control latency, or function. Local bank-wait and
first-MAC-delay fields hold earlier planned reservations fixed; they are
neither measured ONNXim stalls nor a globally solved rescheduling. The
3×3 test with issues 0/1/2 and starts 5/9/13 exposes an atomic issue-time
conflict but no excess under one assumed staggered schedule. That is a
diagnostic difference, not evidence that the physical schedule exists.

Source boundary: Xue and Huang, [*ReGate* §4.1, "Row/column-wise power
gating"](https://arxiv.org/html/2508.02536), says weights are pushed row
by row, but supplies no ONNXim-compatible row cycle, direction, bank
selector, or replacement controller. ONNXim's issue/start/finish equations
are in `src/SystolicWS.cc` (GEMM/PRELOAD issue branch and
`get_inst_compute_cycles`). The latest-prior identity/footprint matching
and PE-local MAC-release equation are in `derive_two_bank_regate` in
`src/SaPeStateOpportunity.cc`. The four row-write inputs and simultaneous
columns are therefore **assumed**; physical feasibility remains `pending`.

The two-bank summarizer emits aggregate schema version **2** (source record
version remains 2). Its default mode still rejects any core whose legacy
atomic `validation_pass` is false, and reports PE-state exposure only from
fully validated state partitions. A separate `rowwise_feasibility` result is
available only when `--expected-cores` is supplied and **every** core has an
available, complete rowwise record with identical lead, period, duration,
row direction, simultaneous-column rule, equations, and evidence labels.
Event and PE-cycle fields sum across cores; peak and maximum-delay fields
take their maximum; `necessary_conditions_pass` is the AND across cores.
Otherwise that field is explicitly `pending` with a reason—never a partial
aggregate. Physical feasibility, actual stall cycles, and energy stay pending.

The summarizer also independently validates every whole-bank
`preload_feasibility_audit` equation and requires exactly one ordered event per
recorded preload. Across a complete core set it sums event/preload counts and
the independent per-PRELOAD stall totals, takes the maximum stall, and retains
the exact offending events with `core_id`. A missing, partial, mislabeled, or
equation-inconsistent core makes this aggregate `pending`; it is never filled
from the older PE-level counters.

For a trace that fails only the legacy issue-time atomic bank assignment,
use the explicit diagnostic-only mode:

```sh
python3 power_model/summarize_regate_two_bank.py \
  regate_two_bank.log --expected-cores 4 \
  --rowwise-diagnostic-only > regate_rowwise_diagnostic_summary.json
```

This mode accepts a failed legacy core only if its state capacity was fully
fail-closed to `unclassified` and its other validation failures are zero;
only `two_bank_conflicts` or `weight_evicted_before_use_intervals` may be
nonzero. It emits no interpreted PE-state totals, ideal low-power exposure,
gross static saving, or net energy saving: those fields are `null`. It also
rejects power factors. A passing rowwise diagnostic in this mode is still
only a necessary timing check under explicitly assumed architecture, not a
recovered ReGate implementation or a measured stall-free schedule.

### Row-wise logical PE-state accounting (nested version 1)

When the explicit row schedule is supplied, each source version-2 profile now
also emits `rowwise_state_accounting.version: 1`. This is a second,
independent state partition; it does not replace or alter the legacy atomic
`pe_state_cycles`, `validation_pass`, timing audit, ONNXim scheduling,
functional execution, or retention controller. In particular, a trace may
fail the legacy whole-footprint assignment and still pass this conditional
row-wise partition.

For preload instance `p` and mapped PE `(r,c)`, let:

```text
write_start(p,r) = start(p) - lead + ordered_row_index(p,r) * period
write_ready(p,r) = write_start(p,r) + duration
safe_replace(p,r,c) = max(write_ready(p,r),
    max(start(i) + r + c + tile_n(i)) for p's fused preload and
        every exactly matched later GEMM consumer i)
```

The two PE-local logical banks give an empty bank priority. Otherwise a write
may replace a bank only when `safe_replace <= write_start`; equality is a
valid half-open replacement. `safe_replace` does **not** erase or power down
the old weight. A logical bank remains resident from its row write until that
same selected bank is actually overwritten, or until the profile horizon.
Thus an idle mapped resident bank is `W_on`, including after its last MAC.
An unmapped PE with no resident bank is `OFF` under the same routing/bypass
assumption as the legacy model.

For each PE, let `A` be the union of modeled useful-MAC spans, and let `B1`
and `B2` be the spans with exactly one and two resident logical banks. The
disjoint accounting is:

```text
ON_one  = |A intersect B1|
ON_two  = |A intersect B2|
W_on_one = |B1 minus A|
W_on_two = |B2 minus A|
OFF = horizon - ON - W_on_one - W_on_two
capacity = ON + W_on_one + W_on_two + OFF + unclassified
```

`ON` is also split by opcode: `on_preload_wave` is the fused
`GEMM_PRELOAD` activation wave and `on_gemm_wave` is the wave of exactly
matched later `GEMM` consumers. Their sum must equal `ON`. A row write alone
does not make a PE ON. The union of `[write_start, write_ready)` is reported
separately as `preload_write_exposure.pe_cycles`, with its overlap with `ON`
also reported; these are activity exposures, not additional disjoint power
states and not energy.

Zero `unclassified` is emitted only when the trace is complete, dimensions
and waves are valid, weight identity and footprint matching is exact, every
write is at or after issue, every row is ready by its first MAC, PE-local
needed occupancy never exceeds two, local bank assignment succeeds, and all
partition identities above pass. Any missing or failed required check zeros
every classified state and assigns the complete capacity to `unclassified`.
An explicit row schedule with an empty SA trace is the sole vacuous case: its
complete capacity is exactly `OFF`. A nonempty GEMM-only trace still fails
closed because it lacks a matching preload identity.

Aggregate this partition with the separate mode:

```sh
python3 power_model/summarize_regate_two_bank.py \
  regate_two_bank.log --expected-cores 4 \
  --rowwise-state-accounting > regate_rowwise_state_summary.json
```

The output uses aggregate schema version **3**, source-record version **2**,
and nested rowwise-state version **1**. Every core must provide the same
complete assumed row schedule and compatible evidence labels. Incompatible
lead/period/duration/order/equations fail to `pending`; failed timing or state
checks fail the whole aggregate closed to full `unclassified`. The mode
rejects power factors and emits no BET, wake latency, gross saving, or net
saving. Its result is theoretical PE-state exposure under assumed timing,
not measured circuit behavior or proof of physical ReGate feasibility.

## Updated predictor plus identity-independent two-bank shadow

Set both opt-in flags when evaluating the revised method:

```sh
ONNXSIM_SA_UPDATED_PREDICTOR=1 \
ONNXSIM_REGATE_UPDATED_PROFILE=1 \
./build/bin/Simulator ... 2>&1 | tee regate_updated.log
```

The updated profile requires both flags and rejects a simultaneous hybrid SA
configuration, including hybrid shadow mode. These restrictions keep its
measurement-only and unchanged-scheduling provenance accurate. This path
leaves ONNXim functional scheduling unchanged. The legacy
predictor and identity-based two-bank profile remain available when these
flags are absent. The updated predictor evaluates each resident tile's next
SA boundary as:

```text
deadline(tile) = max(instruction-prefix lower bound,
                     operand-readiness lower bound,
                     shared-SA-resource lower bound)
next deadline  = min(deadline(tile 0), deadline(tile 1))
```

### Pending resident-slot policy (design intent; not implemented)

The intended next revision removes the unconditional `current_cycle + 1`
future-tile cap that is currently applied merely because fewer than two tile
slots are occupied. The proposed resident-slot policy is:

* If both tile slots are empty, all PE compute logic may remain `OFF`; both
  weight banks are classified independently from their validity/liveness.
* If both resident tiles have no remaining SA instruction, all PE compute
  logic may remain `OFF`. Weight-bank domains with no live consumer may be
  `OFF`.
* If one slot is occupied and one is empty, only the occupied tile's remaining
  SA work is used for the immediate decision. If it has no remaining SA
  instruction, all PE compute logic may remain `OFF` and dead weight-bank
  domains may be `OFF`.
* If the sole occupied tile's next SA instruction is `GEMM_PRELOAD`, PE logic
  remains `OFF` until required by the later activation wave. The preload's
  target bank domains become `ON` only for loading; replaceable old contents
  need not be retained.
* If the sole occupied tile's next SA instruction is plain `GEMM`, its required
  stationary weights must be preserved. The required weight bank therefore
  enters an appropriate retention state selected from the certified idle
  interval; it must not be discarded. PE compute logic remains independently
  `OFF` until reached by the ReGate-like activation/control wave.

This policy relies on the planned invariant that every newly admitted tile
begins its SA phase with `GEMM_PRELOAD`. Repairing the legacy direct-ONNX
attention path to satisfy that invariant is a separate pending change. Until
the policy, invariant, and wake/stall accounting are implemented together,
this section is a design note and does not describe current simulator behavior.

#### Pending unified per-PE/weight-bank power-domain hierarchy

The selected target design does not introduce a separately charged global
whole-SA `ON/OFF/RETENTION` state. It uses one shared runtime timeline while
keeping PE compute logic and weight storage as independent power domains:

```text
per-PE compute logic:           ON / OFF
weight bank 0 per domain:       ON / SS / DS / OFF
weight bank 1 per domain:       ON / SS / DS / OFF
```

There is no SA-level retention state and no full-SA 469-cycle lockout in this
target design. `SS` and `DS` apply only to independently gated weight-bank
domains. A live stationary weight may remain in `SS` or `DS`, while a dead,
unmapped, or replaceable weight-bank domain may be `OFF`.

PE logic does not all become computationally ON simultaneously. The ReGate-like
activation/control wave wakes only the required PE and forwarding path cycle by
cycle. For mapped dimensions `R`, `C`, and `B`, PE `(r,c)` is required over the
analytical interval
`[start+r+c, start+r+c+B)`. Mapping-derived edge/padding domains outside the
required stationary footprint remain weight-bank `OFF`; required forwarding
PE logic must still wake even when it performs no useful MAC.

`whole_sa_idle` may still be reported as the aggregate condition that every PE
logic domain is `OFF`, but it is not another power state or another copy of
energy saving. Bank energy is accounted separately. The intended unified
runtime state machine must advance every PE-logic state, both weight-bank states
and validity bits, their wake/entry transitions, and any resulting stalls on
the same cycle axis. The current coarse C++ whole-SA controller plus offline
Python bank replay does not yet implement this complete hierarchy; this
subsection records the target design.

Operand readiness is source-specific. A valid source contributes no wait; an
allocated source uses its actual SRAM pending-response count and exact core
FIFO tail ticket; an unallocated source uses the request count and producer
position stored at tile construction. No NoC or DRAM ETA is guessed.

Every SA phase needs an explicit `GEMM_PRELOAD` or an explicitly marked
*analytical* preload boundary. The legacy direct-Attention path retains its
native `GEMM` opcode and timing; its marked boundary is an assumption in the
shadow, not proof that ONNXim performed a physical preload. Such a marker
does not by itself certify a whole-SA OFF decision. Metadata is stored only
at tile start and successive SA boundaries. Each explicit preload is a
separate assumed physical copy.
Following `GEMM` instructions in that tile stream use the latest preload
until the next explicit preload. Weight values and `weight_block_id` are not
needed by this updated approximation.

The separate analytical timeline assumes two logical weight banks per PE,
one shared MAC datapath, one preload lane, and one mapped stationary-weight
row written per cycle with all mapped columns in that row written in
parallel. Preloading one bank may overlap computation from the other bank;
compute waves are serialized. For mapped dimensions `R=tile_m`,
`C=tile_k`, and `B=tile_n`, mapped PE `(r,c)` is ON over:

```text
[analytical_start + r + c, analytical_start + r + c + B)
```

The disjoint priority is `ON > LOADING > W_on > OFF`. A valid result must
satisfy exactly:

```text
capacity = ON + LOADING + W_on + OFF
unclassified = 0
```

Raw preload-write exposure and its overlap with ON are reported separately;
they are not added again to the partition. ON and W_on retain one-/two-live-
bank splits for later circuit calibration. Offline interval coloring assigns
bank 0/1 and reports conflicts. Malformed preload sequencing, invalid mapped
dimensions, a third simultaneously live copy, or a failed partition causes a
fail-closed result with full capacity `unclassified`.

Analytical start/finish disagreements with ONNXim are diagnostic timing
differences, not measured performance penalties. Event-lateness sums can
double-count propagated delay; trace-makespan extension is reported
separately. Neither value stalls or reorders the simulator.
The profile does not report energy savings because ON/W_on/LOADING/OFF power,
transition energy/latency, second-bank overhead, selector/controller cost,
isolation, switch, and routing costs are not supplied.

The opt-in reference calculator adds a **gross analytical-shadow SA energy
opportunity** under an explicit cross-paper scenario. It checks the weight
byte width emitted from the simulation config and groups 128/weight_bytes
stationary weights into one assumed 128-byte power domain in each bank;
this is not a 128-byte weight register per PE. A domain is ON if any member
is loaded or read. A live but idle domain may enter SS or DS; an empty or
discardable domain may enter OFF. Wake guards are charged at ON power. The
offline SS/DS selection maximizes gross leakage saving with perfect future
knowledge, so it is an opportunity estimate rather than a realized updated
predictor policy. The bank ratios ON/SS/DS/OFF = 1/0.94/0.42/0 and wake
delays 0/4/13/16 cycles come from Slumber, Table 1
(https://devashreetrip.github.io/publication/Website_Slumber_ISLPED2020.pdf).
Slumber's wake delays are not treated as energy break-even times. ReGate's
PE OFF ratio 0.03 and PE wake/BET 1/47 cycles are reference inputs from
Section 4 and Table 3 (https://arxiv.org/html/2508.02536v2).

The scenario assumes 50% SA static-energy fraction and 50% weight-bank
share of SA static energy. Both appear in output and can be overridden on
the command line. Its primary baseline is hypothetical **two-bank always
ON** storage plus always-ON PE logic on the same analytical horizon. The
unchanged dynamic fraction remains in the energy denominator. It computes:

```text
gross overall SA opportunity (%) = static_fraction *
    [bank_share * bank_leakage_saving (%) +
     (1-bank_share) * PE_logic_leakage_saving (%)]
```

Bank/PE transition energy, controller, selector, isolation, routing, and
added preload dynamic energy are excluded. This is `extrapolated`, and the
overall realized predictor saving remains `pending`. The numeric opportunity
normalizes baseline and managed static energy to the same analytical shadow
horizon; it does not include extra energy from a change in run duration.
An additional one-bank-always-ON sensitivity includes the cost of adding
the second bank in its numerator and uses a hypothetical one-bank baseline
in its denominator. The 50% weight share is reinterpreted for that baseline;
neither comparison is a measured ONNXim circuit baseline.
The output separately reports
predictor-driven whole-SA **compute-logic** OFF episodes. Those are observed
on ONNXim time, while bank and PE states use the analytical shadow time.
The ordinary reference result keeps those timelines separate. The optional
trace replay below admits an episode only when the original ONNXim and
ungated analytical SA event start/finish times agree through the episode and
its wake guard. A timing mismatch or managed wake stall disables subsequent
predictor projection. Actual end-to-end penalty remains pending because
non-SA dependent work is not rescheduled.

Aggregate a completed run with:

```sh
python3 power_model/summarize_regate_updated.py \
  regate_updated.log --expected-cores 4 \
  --output regate_updated_summary.json
```

To include the explicit reference opportunity and predictor diagnostics:

```sh
python3 power_model/summarize_regate_updated.py \
  regate_updated.log --expected-cores 4 --reference-energy \
  --static-fraction 0.5 --bank-leakage-share 0.5 \
  --output regate_updated_reference_summary.json
```

This requires a new log containing per-event mapped dimensions and
`predictor_reference_whole_sa_logic` episodes. Older logs fail closed for
the reference calculation; ordinary validation remains available.

For the explicitly assumed common-cycle SA scenario, use:

```sh
python3 power_model/summarize_regate_updated.py \
  regate_updated.log --expected-cores 4 --unified-reference \
  --static-fraction 0.5 --bank-leakage-share 0.5 \
  --assume-zero-overheads \
  --output regate_updated_unified_summary.json
```

Without `--assume-zero-overheads` or both explicit transition and overhead
inputs, the unified replay reports state counts and its modeled SA schedule
penalty, while numeric overall energy remains `pending`. Zero overheads are
an explicit **assumed gross scenario**, not a measurement. The transition
input is incremental energy in normalized full-SA static-cycle units per
core; `--overhead-fraction-of-baseline` is relative to the hypothetical
one-bank always-ON total energy. Sweeping those inputs is necessary before
claiming a robust saving.

The replay runs bank 0, bank 1, PE compute logic, and accepted whole-SA
compute-logic OFF episodes on one managed analytical clock. It propagates
whole-SA wake stalls through the shared compute lane, same-stream order, and
bank replacement. Bank SS/DS/OFF and local PE OFF decisions use offline
knowledge of the next analytical access to reserve their 4/13/16 and
one-cycle wake guards; they therefore create no modeled stall in this
scenario. The whole-SA wake occupies ten ON cycles after a projected OFF
episode. Whole-SA OFF is a subset of PE-logic OFF and contributes no second
static saving term when both use the same 0.03 OFF ratio.

The main energy denominator is hypothetical original one-bank always-ON SA
power over the ungated **two-bank analytical schedule**. The managed numerator
charges both banks over the managed schedule. A two-bank always-ON comparator
is also reported. These are `extrapolated` cross-paper reference scenarios;
neither denominator establishes actual original one-bank timing or measured
net circuit energy. The reported penalty is an SA schedule extension only.
The full workload penalty and uncharacterized transition, loading, selector,
controller, and routing energies remain pending.

`power_model.reference_sa_energy.calculate_aligned_overall` is a separate
calculator for a future common-timeline result. Its caller must supply
validated baseline and managed horizons, complete bank and PE-logic mode
counts over the managed horizon, **incremental** transition charges in
ON-equivalent domain/PE cycles above the leakage already charged during
wake-guard ON cycles, and any other overhead as a fraction of baseline energy.
It assumes unchanged workload dynamic energy and the same number of bank
domains in the supplied baseline and managed design.
Whole-SA OFF PE-cycles must be a subset of the PE-logic OFF count and are
not charged a second time. The function accounts for extra managed cycles
and reports a modeled penalty relative to the supplied no-gating baseline.
The optional unified replay supplies a narrower assumed SA scenario from
matching-prefix predictor episodes. It does not prove a rescheduled runtime
predictor or establish measured end-to-end energy.
