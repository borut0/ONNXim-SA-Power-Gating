# Non-Retentive SA Power-Gating Snapshot

This snapshot freezes the non-retentive systolic-array power-gating work before
retention-state modeling is introduced.

## Included

- ONNXim instrumentation and causal predictor integration in `src/`
- synthesized controller RTL, clock gating, and replay testbenches in `rtl/`
- OpenROAD synthesis, timing, and activity-based power scripts
- exact final sparse-event controller netlist, synthesized SDC, synthesis/STA
  reports, workload power/annotation reports, and immutable ORFS image digest
  under `openroad/artifacts/sparse_event/`
- HSPICE deck generators and compact measured energy curves
- exact evaluated workload/model-list configurations
- strict ONNX operator audits, component summaries, and available gross/net
  energy JSON results under `workload_suite/`
- representative-layer and full-model OPT-125M configurations used to validate
  the repeated-layer methodology

## Deliberately Excluded from Git

Raw controller CSV traces, VCD/FST activity files, HSPICE listing/deck outputs,
third-party ASAP7 libraries, synthesized netlists, and verbose power logs are
generated artifacts and are excluded by `.gitignore`. They remain local and can
be regenerated with the checked-in scripts and configurations.

The final sparse-event controller netlist is the one deliberate exception: it
is retained as a compact, checksummed reproducibility artifact. Large SA
netlists and binary ODB stage databases remain excluded.

## Host and Docker Consolidation

The stopped ONNXim container was not bind-mounted to the host repository. Its
`src/`, `configs/`, and `scripts/` trees were compared against the host before
this snapshot. The simulator source was identical except for final newlines.
The host contained the newer controller RTL and analysis scripts, so Docker did
not overwrite host source. Small Docker-only audits, workload summaries, model
lists, and final logs were recovered; only compact evidence is versioned.

## Result Completeness

Each directory in `workload_suite/results/` contains the canonical compact files
available at snapshot time. OPT-125M full-model and OPT-66B contain gross-energy
results but no canonical net-energy JSON in the local trace tree at snapshot
time. This absence is preserved rather than filled with an inferred value.

## Boundary of This Snapshot

This version models non-retentive OFF operation: PE weight state is not assumed
to survive an OFF interval. It does not implement a retention state, retention
leakage model, or wake-up-latency/performance model. Those changes must be made
on a later branch and compared against this frozen baseline.
