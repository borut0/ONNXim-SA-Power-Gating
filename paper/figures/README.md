# Result-independent paper figures

Each figure is available as editable SVG, paper-ready vector PDF, and PNG preview.

| Figure | Suggested caption | Important qualification |
| --- | --- | --- |
| `01_greensa_pe` | Conceptual GreenSA processing element with independently controlled compute logic and two weight banks. | Logical architecture, not a fabricated transistor-level circuit. The selector and hidden bank wake are model assumptions. |
| `02_policy_timeline` | Illustrative PE and weight-bank states around a preload and activation wave. | Widths are schematic, not measured cycle durations. The unneeded second bank is OFF only in the illustrated example. |
| `03_power_states` | PE and weight-bank power-state transitions in the GreenSA model. | The 3% PE OFF leakage and 42% bank DS leakage are borrowed model parameters; cite ReGate and Slumber, respectively, in the paper. |
| `04_evaluation_flow` | Workload mapping, policy-isolated simulation, validation, and reporting flow. | The comparison uses matched architecture/workload arguments; the reported penalty is an estimate, not a strict end-to-end causal replay. |

These figures contain no workload results. Recheck their labels against the final implementation and paper terminology before submission.
