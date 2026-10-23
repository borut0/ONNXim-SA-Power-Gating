# Shared functional SA timing comparison

Run the **same model, config, and trace** three times, changing only
`ONNXSIM_SHARED_SA_POLICY` among `always_on`, `regate`, and `ours`. Do not set
the older `ONNXSIM_REGATE_FUNCTIONAL_PROFILE`,
`ONNXSIM_REGATE_TWO_BANK_PROFILE`, `ONNXSIM_REGATE_ROW_PRELOAD_PROFILE`,
`ONNXSIM_REGATE_EVENT_DRIVEN_PROFILE`, `ONNXSIM_REGATE_UPDATED_PROFILE`, or
`ONNXSIM_SA_REGION_PROFILE` at the same time. The simulator emits one
schema-v2 `SHARED_SA_TIMING` record per core. Keep each complete raw log. The
summarizer intentionally rejects older records that do not certify the
47-cycle BET and 48-cycle lockout assumptions.

The new path changes the functional SA instruction completion time using
one-row-per-cycle weight loading and a diagonal mapped PE wave. The previous
analytical event-driven and ReGate baseline scripts remain separate.
`GEMM_PRELOAD` supplies the physical preload boundary in this mode; a plain
`GEMM` marked only as an analytical boundary fails closed. Two banks are
assigned when instructions execute, and a bank remains protected until its
last tile-local consumer has been scheduled and its wave has drained.
The direct ONNX Attention path now emits explicit physical `GEMM_PRELOAD`
instructions at the start of each stationary-weight reuse group. Both direct
ONNX attention and the language-model attention path must still pass the
metadata unit tests and a representative workload run before results are used.

Policy timing is deliberately different only where specified:

- `always_on` has no PE wake or minimum-residency delay;
- `regate` is wave-driven and adds one explicit PE(0,0) wake cycle for every
  first or non-contiguous wave, even after preload or resource waiting. It has
  no 48-cycle lockout;
- `ours` commits every non-contiguous PE(0,0) wave interval to at least 48
  cycles, one cycle beyond ReGate's published 47-cycle PE break-even time.
  Every first or non-contiguous wave pays one explicit wake cycle. If a
  non-contiguous wave's original gap is `g`, its total policy delay is
  `max(1, 48-g)` cycles; only the excess beyond the wake is recorded as
  minimum-OFF lockout. The delay propagates through normal instruction
  completion/dependency timing. One PE(0,0) decision delays the diagonal
  wave; there is no separate 48-cycle stall per PE.

`natural_wave_finish` is the last mapped PE's wave end. ONNXim retires SA
instructions from a FIFO, at most one per cycle, so the new mode may set a
later dependency-visible `compute_finish` to preserve that order. The record
reports the resulting `modeled_fifo_commit_delay_cycles`. This is a
conservative simulator integration delay, not measured hardware latency.
`modeled_bank_wait_cycles` includes core cycles with a blocked preload and
projected waiting for a bank's previous wave to drain; it is diagnostic and
must not be added to the measured end-to-end penalty.

After the three simulator runs:

```sh
python3 power_model/shared_sa_comparison.py \
  --always-on-log /path/to/always_on.log \
  --regate-log /path/to/regate.log \
  --ours-log /path/to/ours.log \
  --expected-cores 4 \
  --output /path/to/comparison.json
```

The output always reports the requested sensitivity sweep
`beta={0.1,0.2,0.3,0.4,0.5,0.6}`, where `beta` is the **assumed** fraction of
always-ON SA static power attributable to both weight banks. No measured beta
is implied. Both policies use the same matched two-bank architecture. The default
ReGate structural mask policy is `delayed`: candidate columns remain powered
until the mapped preload row stream completes. `--mask-policy optimistic`
instead powers only mapped columns from the outset.

The script checks each row write, wave duration, bank reuse, state partition,
and the multiset of SA work in the three runs. It uses each run's maximum
per-core `core_cycles` as its full-simulator horizon for capacity padding, but
checks every event and row write against the emitting core's own local
`core_cycles`; a longer-running peer cannot legalize an event past that local
endpoint. Schema-v2 event delays are recomputed from
`compute_start_before_power_policy` and `modeled_power_policy_delay`, then
reconciled with the emitted wake and lockout counters. It cannot certify that
non-SA work and input traces match; the operator must run identical workload
arguments. On a mismatch or impossible timeline it exits with an error.

The `observed_*_horizon_delta_percent` fields compare the independently
simulated run horizons. They are diagnostics, not an isolated power-policy
performance penalty: a stall can change tile order and shared memory/NoC
contention, and even produce a shorter observed horizon. The
`fixed_sa_order_serial_scenario` field instead replays ReGate's explicit wake
and our 48-cycle PE(0,0) lockout against the always-ON SA event order. It
assumes each inserted delay shifts all subsequent work on that core and its
endpoint by the same amount. It is an **analytical serial scenario**, not a
certified end-to-end causal replay; the logs do not contain the full non-SA
dependency and memory schedule needed to certify one.

Static energy uses each policy's **own observed elapsed time**, then divides
by the always-ON run's energy. ReGate bank residency is reported as
ON/W_ON/OFF: active preload/consumer cycles and newly powered delayed-mask
candidates are ON; a candidate position already holding a retained valid
weight remains W_ON until actively written or used. Every loaded mapped physical weight retains power in
W_ON after its final known consumer, through later overwrites of that same
position and until the shared run horizon. A later smaller preload does not
silently invalidate old mapped positions outside its footprint because the
event log has no per-position invalidation or numerical-zero bitmap. Thus this
is a conservative retained-valid-weight ReGate abstraction; a measured bitmap
could reduce that residency. Cells never loaded and candidates released by
the delayed mask are OFF. ON and W_ON both have normalized leakage 1.
This follows ReGate's [ON/W_on/OFF PE control description](https://arxiv.org/abs/2508.02536)
without granting its controller our last-consumer lookahead. Ours uses ON/DS/OFF with
Slumber's 0.42 normalized DS power extrapolated to
each PE bank. PE OFF leakage uses ReGate's 0.03 ratio. The DS wake is assumed
fully hidden by PE wake; that circuit timing is unverified.

The net result charges ReGate's published 47-cycle PE BET once for every
complete, non-contiguous per-PE low-power episode. Touching waves coalesce and
incur no transition. ReGate short episodes are not clamped: an episode of at
most 47 cycles can reduce its net saving. Our 48-cycle PE(0,0) lockout changes
functional completion time, while transition cost is still counted for every
PE that actually leaves and later re-enters the active wave footprint.
Initial and final low-power tails are not charged a complete round-trip cost.
PE logic ON residency counts MAC cycles; the one-cycle wake/guard belongs to
the BET transition cost and is not also charged as a full ON leakage cycle.

Weight-bank transition energy is currently assumed zero for **both** matched
policies because it is uncharacterized; hidden wake latency does not establish
zero transition energy. Dynamic MAC, controller, selector, and preload energy
also remain outside this result and must not be presented as whole-NPU energy.
