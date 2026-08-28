# 128x128 structural extrapolation

All values are ASAP7 RVT FF synthesis results for the same two-stage PE RTL.
The independent variable is PE count: 256 for 16x16, 1024 for 32x32, and
16384 for 128x128.

| Metric | 16x16 measured | 32x32 measured | 4x 16x16 | 32x32 deviation |
|---|---:|---:|---:|---:|
| Cells | 533954 | 2127660 | 2135816 | -0.3819% |
| Total area (um^2) | 59614.193700 | 237764.443506 | 238456.774800 | -0.2903% |
| Sequential area (um^2) | 12422.072520 | 49687.152840 | 49688.290080 | -0.0023% |

For each structural metric `Y`, fit the only line through the two measurements:

`Y128 = Y32 + ((16384 - 1024) / (1024 - 256)) * (Y32 - Y16)`

Therefore `Y128 = 21*Y32 - 20*Y16`, giving:

| 128x128 extrapolated metric | Value |
|---|---:|
| Cells | 34001780 |
| Total area | 3800769.439626 um^2 (3.800769 mm^2) |
| Sequential area | 794988.759240 um^2 (0.794989 mm^2) |
| Sequential area fraction | 20.9165% |

The measured 16-to-32 deviations above are empirical scaling residuals, not
statistical confidence intervals. Full 128x128 synthesis is not attempted on
the current host: 32x32 already required 11.57 GB peak RSS, while a linear PE-
count projection for 128x128 is at least 16 times larger.

## Power extrapolation contract

Measure each activity mode independently on both 16x16 and 32x32 using the
same RTL, library corner, clock, activity duration, VCD scope, and power-flow
stage. For each reported power component and mode, use
`P128 = 21*P32 - 20*P16`. Also report the observed deviation
`P32/(4*P16)-1`; do not assume it equals the structural residual. No smaller
third point is needed: it would be farther from 128x128 and would not test
large-array interconnect scaling. If two-point power scaling is materially
nonlinear, report a sensitivity range rather than claim unsupported precision.
