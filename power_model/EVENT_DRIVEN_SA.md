# Opt-in stall-aware SA power opportunity

Set `ONNXSIM_REGATE_EVENT_DRIVEN_PROFILE=1` when running `Simulator`. It emits
one `REGATE_EVENT_DRIVEN_SOURCE` JSON record per core without enabling the
legacy whole-SA predictor. After simulation, summarize the saved raw log:

```sh
python3 power_model/summarize_regate_event_driven.py RUN.log \
  --expected-cores 4 --output event_driven_summary.json
```

The result starts from ONNXim's recorded instruction issue, compute-start, and
compute-finish cycles. It inserts analytical SA stalls when a row write, PE
wake guard, prior same-PE MAC, or live bank generation makes the native start
impossible. Later SA events on that core inherit the delay. Native overlapping
windows remain overlapping when their per-PE uses are compatible. The replay
classifies each PE's logic as ON/OFF and each of its two assumed weight banks
as ON/DS/OFF on the adjusted clock.

Across multiple cores, the summarizer first schedules each core, finds the
latest analytical SA finish, and then accounts every core on that same shared
horizon. It reports `analytical_sa_timeline_extension_percent`. This is an SA
timeline scenario. The source trace has no complete non-SA dependency graph,
so `actual_end_to_end_penalty_percent` remains null.

The preload lane, two independently gated banks per PE, one-row-per-cycle
writes aligned immediately ahead of the native diagonal weight-use wave,
diagonal local MAC intervals, one-cycle PE advance signal, and Slumber's
128-byte deep-sleep leakage ratio applied to each PE bank are *architectural
assumptions*, not ONNXim execution facts. Instruction issue is a causality
bound: a required row stream that would begin before issue delays its wave.
Plain-GEMM
preload boundaries and bank IDs may also be assumed by the C++ source shadow.
The shadow currently validates bank assignment using its older serialized
schedule, so it may reject a native trace that another bank allocator could
support. Semantic convolution activation zeros are not treated as missing
stationary weights; only the mapped weight rectangle controls structural OFF.

Without `--beta`, the output reports separate PE-logic and two-bank gross
static-leakage opportunities. `--beta VALUE` supplies an explicitly assumed
baseline fraction of SA static power belonging to the two banks and adds a
combined **gross static opportunity on the managed analytical timeline**.
There is no default beta and no measured whole-SA energy saving. Exact
end-to-end penalty, net energy saving, and physical timing feasibility remain
pending: ONNXim is not functionally rescheduled; ReGate's PE wake reference is
one cycle, whereas Slumber's deep-sleep wake reference is 13 cycles.

State accounting uses closed-form rectangular-wave sums and a 128x128
last-use grid. It does not allocate one interval object for every PE/event
pair, so large OPT traces can be summarized without the earlier memory blowup.
