# Abstract ReGate SA comparison

The comparison consumes the existing `REGATE_EVENT_DRIVEN_SOURCE` simulator
log. It reuses the proposed replay's analytical SA clock and does not modify
the proposed dual-bank policy or ONNXim's functional execution:

```sh
python3 power_model/summarize_regate_baseline.py RUN.log \
  --expected-cores 4 --beta 0.1 --mask-policy delayed \
  --output regate_comparison.json
```

`beta` is the assumed share of static power belonging to *both weight banks*
in an always-on two-bank SA. Use the same beta when reporting the proposed
design. The literal one-bank weight share is derived as `beta/(2-beta)` so the
per-bank and logic powers stay consistent. No combined static percentage is
reported when beta is absent.

There are two deliberately distinct comparisons:

* `matched_two_bank`: ReGate-style PE wave gating and ON/W_ON/OFF weight policy
  applied to the proposal's two-bank hardware and exact managed event clock.
  This isolates the effect of changing idle bank storage from W_ON to DS/OFF.
  It is a policy ablation, **not** literal ReGate hardware.
* `literal_one_bank`: one stationary weight register per PE. It reports exact
  ON/W_ON/OFF residency only if preload generations can share that one register
  without a conflict. If two generations overlap, the energy field is null;
  a one-bank instruction reschedule or extra reload is required before a
  literal ReGate saving can be claimed.

The PE wave uses the proposed replay's one-cycle advance signal and mapped
origin-aligned rectangle. ReGate's numerical zero-weight detector is not
reproduced: ONNXim's event trace has dimensions but no actual stationary
weight values. The footprint mask is therefore a **structural approximation**
that sees edge/padded positions but misses unstructured or pruned zero weights.

`--mask-policy delayed` is the default sensitivity scenario: when a mapped
row is loaded, *all columns of that row*
are powered until the mapped row stream ends; unused columns are then OFF.
It uses the existing mapped-row preload interval and does **not** assert that
ReGate physically writes every padded row of the 128x128 array. ReGate's
paper does not specify the exact power timing of zero destination registers
during preload. `--mask-policy optimistic` instead treats the mapping footprint
as known before a destination register is powered. Neither scenario is claimed
to reproduce ReGate's RTL.

The delayed scenario is narrower than the all-128-rows-powered picture in the
discussion: it powers candidate columns only in rows for which the ONNXim event
model has a row-write opportunity. Assuming full-array candidate power during
those same mapped-row cycles would be an upper bound, not a physical 128-row
preload schedule. A literal full-row stream would require a new timing replay.

Both variants let a weight go OFF immediately after its final known consumer.
This is an optimistic retirement assumption for the ReGate-style baseline;
keeping it in W_ON longer would increase its calculated leakage.

The delayed two-bank ablation reports no energy if a previous generation in
the same bank still powers a high column that the new candidate mask would
also power. The simple extra-cycle formula cannot assign that overlap twice.
The optimistic mask mode remains available in that case; a resolved delayed
estimate needs an explicit interval union for the overlapping column.

The main percentages are gross static leakage estimates on a common
analytical SA horizon. Transition energy, actual tensor-value zeros, circuit
wake feasibility, non-SA rescheduling and end-to-end performance remain
unmeasured. A negative additional saving means the proposed design costs more
under the selected power shares and preload scenario.
