# Guaranteed SA-Idle Prediction in ONNXim

## Source-grounded architecture review and recommended design

**Status:** design only; no predictor or power controller has been implemented.

**Audit snapshot:** repository commit `fb87ca4`, plus the supplied dirty working tree as inspected on 2026-08-04. The dirty tree already contains profiling and producer-counter changes. Those changes are treated as part of the supplied source state, but are identified below where they are not part of the checked-in baseline.

**Inputs reviewed:**

- ONNXim source, tests, configurations, BookSim2 submodule, and Ramulator2 submodule.
- `/home/fenil/Delete/implementation_plan.md`.
- `/home/fenil/Delete/sa_idle_predictor_algorithm.md`.
- `/home/fenil/College/MTP/extra_work/mam_paper.pdf` (Tripathy et al., *Slumber: Static-Power Management for GPGPU Register Files*, ISLPED 2020).

**Existing-test check:** the configured CTest entry contains a stale
`/workspace/ONNXim` executable path, so the existing test binary was invoked
directly with the repository build library path. It ran all 29 tests: 23
passed and 6 existing ConvWS/GemmWS analytical-error tolerance tests failed.
No failure involved predictor code because no predictor code exists or was
added in this review.

The failing test names were `ResNet18_Conv1`,
`ResNet18_layer1_0_Conv1`, `ResNet18_layer2_0_Conv1`,
`ResNet18_layer2_0_downsample_0`, `ResNet18_layer2_0_Conv2`, and
`GemmWS1x2048x1000Test`.

The labels used throughout are:

- **[PROVEN]**: directly established by the inspected source.
- **[DERIVED]**: follows from multiple proven facts; the derivation is given.
- **[NOT PROVEN]**: the repository does not establish the claim.
- **[WRONG]**: the repository contradicts the claim.

---

## 1. Executive decision

The current full forward-chain predictor should not be implemented.

It contains multiple unsafe timing assumptions, relies on simulator-oracle state, has an incorrect future-Tile-C cutoff, treats already-in-flight memory as if it were a fresh request, and pays for dependency analysis that ONNXim does not need for a safe lower bound.

The recommended design is a **phase-defined, rolling-frontier, certificate-based predictor**:

1. Run at one precisely defined point in `SystolicWS::cycle()`.
2. Use the committed compute-pipeline front directly; do not scan the pipeline for arbitrary bubbles.
3. Give each resident tile a generator-supplied presence bit and small downward-saturating counter: a certified lower bound on the number of not-yet-issued non-SA instructions before its next `GEMM`/`GEMM_PRELOAD`.
4. Examine only the tile's current head instruction. If that head is blocked, use a lower-bound readiness certificate for that head and combine it with the remaining issue-distance counter.
5. Store a persistent `ready_not_before` certificate with each allocated SRAM entry when a producer is actually created or dispatched. Never restart a fresh memory latency from the current time for an already-in-flight request.
6. Bound an unknown Tile C from core capacity and remaining resident-instruction counts. Assume a direct-GEMM incoming tile unless a minimum prefix is explicitly encoded and enforced.
7. Cache an **absolute earliest-use epoch**, not a relative idle length.
8. During a sleep episode, never postpone the original wake deadline in the baseline controller. Earlier deadlines pull wake-up earlier; later predictions are used only after waking, unless a complete extension proof is added later.

This design has no arbitrary instruction scan, no associative producer search through pipelines, no backward dependency graph, and no BookSim/Ramulator introspection at prediction time.

The primary safety condition is:

\[
E_{pred}(c) \le E_{actual}(c)
\]

where `E_pred(c)` is the absolute epoch returned at the chosen decision point and `E_actual(c)` is the first epoch at which the relevant SA power domain could be needed.

Underestimation is safe. Overestimation is unsafe.

---

## 2. Critical conclusions about the existing proposal

| Existing claim or mechanism | Classification | Source-grounded conclusion |
|---|---:|---|
| `_tiles` has at most two entries | **[PROVEN]** | `Core::can_issue()` returns `_tiles.size() < 2`; `src/Core.cc:33-35`. |
| Tile issue is ordinary round-robin | **[WRONG]** | It is sticky/work-conserving. The successful tile remains first (`tile_rr = i`), while a blocked tile is skipped and the other tile is tried in the same pass; `src/Core.cc:102-174`. |
| A blocked compute instruction enters and clogs `_ex_inst_queue` | **[WRONG]** | It enters EX only after `can_issue_compute()` succeeds; `src/Core.cc:149-168`. |
| EX issue at numeric cycle `c` implies dispatch at numeric `c+1` | **[WRONG]** | It is one physical core tick later but still observes stored `_core_cycle == c`, because handlers precede the next `Core::_core_cycle++`; `src/SystolicWS.cc:18-24,153`, `src/Core.cc:99,154-173`. |
| `finish_cycle` is the last active cycle | **[WRONG]** | It is end-exclusive. Retirement occurs before activity sampling when `finish_cycle <= _core_cycle`; `src/Core.cc:461-481`, `src/SystolicWS.cc:96-108`. |
| The committed SA queue may contain arbitrary internal idle gaps | **[WRONG]** | Under current WS scheduling, surviving entries form one contiguous active block after the front starts. Only a leading future-start interval is possible; derivation in Section 4. |
| A new Tile C must wait for a resident tile to finish whenever any tile is resident | **[WRONG]** | With zero or one resident tile, a slot is already free and admission is attempted before every core invocation; `src/Core.cc:33-35`, `src/Simulator.cc:123-148`. |
| Every SA-capable tile begins with mandatory MOVINs | **[WRONG]** as a Core invariant | Built-in production generators have at least two MOVINs before their first GEMM, but tests construct immediate `GEMM_PRELOAD` tiles; `tests/SystolicWsTest.cc:23-35,66-83`. |
| A tile remains resident until every operation originating from it completes | **[NOT PROVEN]** | Removal follows the instruction marked `last_inst`, not an outstanding-work reference count; `src/Core.cc:108-110,176-185,461-507,588-592`. |
| MOVOUT finishes a tile when it issues from the resident deque | **[WRONG]** | The last MOVOUT sets `inst_finished` later, when the store handler creates write requests; `src/Core.cc:557-593`. |
| `N == MOVIN.size == MOVIN.src_addrs.size()` | **[WRONG]** globally | SRAM pending count uses `size`; requests generated use `src_addrs.size()`; `src/Core.cc:527-549`. Existing Conv paths violate equality. |
| `now + fresh_minimum_memory_latency` is safe for pending memory | **[WRONG]** | A response can already be at the NoC boundary and fill after the current core phase. Existing SRAM state has no remaining-latency certificate. Safe fallback is zero remaining delay. |
| `16` is a universal request-injection rate | **[WRONG]** | It is the configured number of injection ports only in the 128x128 target configurations. Default is one, and the rate is per ICNT event, not per core event. |
| Both NoC `+8` terms are architectural constants | **[WRONG]** | They are configuration-specific observations for one-stage fly, one-flit packets. BookSim ignores JSON `icnt_latency`. |
| Configured `dram_latency=10` is the Ramulator2 minimum | **[WRONG]** | That field is not used by the Ramulator2 path. A read can be forwarded from the write buffer in one Ramulator tick. |
| Finding one pipeline producer with matching `(address, buffer)` is sufficient | **[WRONG]** | Repeated writes and mixed memory/compute producers are legal; all pending contributions must finish. |
| IM2COL has zero latency and no dependency effect | **[WRONG]** | It has nominal zero configured latency, but enters the vector pipeline, increments a compute-pending count on its destination, and retires on the following invocation. |
| Full forward chain walking is required | **[WRONG]** | Program-order distance plus the current frontier's certificate is sufficient for a safe rolling lower bound. |
| Pure address-based backward tracing is authoritative | **[WRONG]** | Address reuse, implicit accumulation, in-place vector writes, mixed field meanings, and missing producer IDs prevent an exact general trace. |
| No visible future GEMM permits an infinite idle prediction | **[WRONG]** | A later tile, barrier release, layer, or model can introduce SA work. Infinity requires an explicit global-quiescence certificate. |
| Keeping only a later/extending wake deadline is safe | **[WRONG]** | An earlier new horizon requires an earlier wake. The existing pseudocode does the opposite in the unsafe case. |
| Underestimating actual memory latency is unsafe | **[WRONG]** | For a lower bound, underestimation is the safe direction; overestimating the minimum or remaining latency is unsafe. |
| The stated tiny hardware estimate covers the full proposal | **[NOT PROVEN]** | It omits future instruction storage/read ports, scans, associative address comparisons, scheduler signals, and cache state. |

---

## 3. Fixed timing convention

### 3.1 Recommended predictor point

Define `H(c)` as the point in `SystolicWS::cycle()` with local `_core_cycle == c`:

- after `finish_compute_pipeline()`;
- after `finish_vector_pipeline()`;
- after `handle_ld_inst_queue()`;
- after the EX dispatch attempt;
- after `handle_st_inst_queue()`;
- before activity sampling and before `Core::cycle()` increments `_core_cycle`.

This corresponds to the boundary after `src/SystolicWS.cc:94` and before `:96`.

The predictor must not use raw `_core_cycle` values without this phase definition.

### 3.2 Consequences of this point

To avoid conflating an integer cycle label with a point inside that cycle,
write `U(e)` for the earlier EX-dispatch/power-domain-use phase in the same
invocation as `H(e)`; the store handler lies between them, so only the ordering
`U(e) < H(e)` is intended. Numeric candidates such as `E=e` mean
the ordered phase `U(e)`, not completion sometime at or after `H(e)`. A real
controller should retain an absolute-time or `(epoch, phase)` representation.

**[PROVEN]** Simulator admission has already occurred before `H(c)`; `src/Simulator.cc:123-148`.

**[PROVEN]** After `H(c)`, `Core::cycle()` increments `_core_cycle` and issues at most one resident instruction; `src/Core.cc:91-175`.

**[DERIVED]** A ready resident plain GEMM at the tile head can:

1. enter EX in the issue phase after `H(c)`;
2. dispatch at `U(c+1)`, before `H(c+1)`;
3. start the modeled SA with stored epoch `c+1` when the compute pipeline is empty.

**[DERIVED]** A not-yet-resident arbitrary tile cannot be admitted until the
admission phase preceding `U(c+1)` and `H(c+1)`. If its first instruction is a
ready plain GEMM, it enters EX after `H(c+1)` and dispatches at `U(c+2)`.
Thus, with a free resident slot and no stronger contract, the core-local
unknown-tile candidate has numeric epoch `c+2` at this decision point.

These offsets differ from a predictor placed after `Core::cycle()`. That is why the current documents contain apparently contradictory `+1` results. The implementation must use one phase and test it explicitly.

### 3.3 Absolute physical time for other clock domains

Simulator periods are integer picoseconds:

```text
Tcore = 1,000,000 / core_freq
Ticnt = 1,000,000 / icnt_freq
Tdram = 1,000,000 / dram_freq
```

At a tie, the code executes CORE, then DRAM, then ICNT; `src/Simulator.cc:117-159,284-298`.

Therefore memory certificates should first be computed in absolute event time or `(cycle, phase)` tuples and then converted to the earliest observable `H(c)`. Raw ICNT and DRAM tick counts must not be added as though they were core cycles.

---

## 4. Verified execution model

### 4.1 Tile residency, double buffering, and completion

- **[PROVEN]** Capacity is two resident tiles; `is_accum_tile` is ignored by `Core::can_issue()`; `src/Core.cc:33-35`.
- **[PROVEN]** A new tile alternates the regular-SPAD buffer and flushes it. The accumulator buffer also alternates except for an accumulating tile of the same layer/fused operation, which intentionally reuses it; `src/Core.cc:37-75`.
- **[PROVEN]** Skipped tiles are immediately moved to `_finished_tiles` and never become resident; `src/Core.cc:65-74`.
- **[PROVEN]** A resident tile is erased only when its deque is empty and `inst_finished` is true; at most one resident tile is erased per invocation; `src/Core.cc:176-185`.
- **[PROVEN]** A last GEMM or vector instruction sets `inst_finished` at pipeline retirement; a last MOVOUT sets it in the store handler; `src/Core.cc:461-507,557-593`.
- **[WRONG in the current proposal]** A last MOVIN never sets `inst_finished` in the current source, so the proposed “last MOVIN completes when memory returns” lifecycle is not implemented; `src/Core.cc:511-554,212-234`.

### 4.2 Issue policy and queues

- **[PROVEN]** At most one instruction total is removed from resident tile deques per `Core::cycle()`; successful issue breaks the scan; `src/Core.cc:102-174`.
- **[PROVEN]** The scheduler is sticky, not round-robin: the successful tile remains the first scanned next time (`tile_rr = i`, not `i+1`); `src/Core.cc:169-173`.
- **[PROVEN]** If the first tile is blocked, the second tile is examined during the same scan.
- **[PROVEN]** MOVIN routes to `_ld_inst_queue`; MOVOUT/MOVOUT_POOL routes to `_st_inst_queue`; all other opcodes route to `_ex_inst_queue`; `src/Core.cc:124-168`.
- **[PROVEN]** EX has effective capacity one because issue requires `_ex_inst_queue.empty()`; `src/Core.cc:154-158`.
- **[DERIVED]** LD depth is at most one in normal execution. Its handler removes one unconditionally before the core can issue at most one new instruction; `src/SystolicWS.cc:13-14`, `src/Core.cc:511-554,102-174`.
- **[PROVEN]** ST can accumulate because the front store remains while its SRAM input is invalid, while subsequent issue from the other tile can enqueue more stores; `src/Core.cc:140-148,557-597`.
- **[PROVEN]** A blocked compute instruction remains at the tile head. It does not occupy EX.

### 4.3 Compute and vector pipelines

- **[PROVEN]** Only `GEMM` and `GEMM_PRELOAD` route to `_compute_pipeline`; every other EX opcode routes to `_vector_pipeline`; `src/SystolicWS.cc:23-90`.
- **[PROVEN]** Compute-pipeline capacity is `core_height`; `src/SystolicWS.cc:253-263`.
- **[PROVEN]** Vector capacity is one because non-GEMM issue is rejected while `_vector_pipeline` is nonempty; `src/SystolicWS.cc:264-271`.
- **[PROVEN]** SA and vector work may overlap. The GEMM capacity check does not check vector occupancy, and the vector check does not check compute occupancy; `src/SystolicWS.cc:243-275`.
- **[PROVEN]** Only one compute completion and one vector completion are retired per core invocation; `src/Core.cc:461-507`.

For a GEMM with dimensions summarized by `compute_size`, array height `H`, and width `W`:

\[
L_{gemm}=H+W-2+\max(compute\_size,4)
\]

This is **[PROVEN]** by `src/SystolicWS.cc:277-279`.

With an empty compute pipeline:

```text
plain GEMM:     start = dispatch
GEMM_PRELOAD:   start = dispatch + 2H - 1
finish = start + L_gemm
```

With a nonempty pipeline:

```text
offset = max(back.compute_size, 4)
if new opcode is GEMM_PRELOAD: offset = max(offset, H)
start = max(dispatch, back.start + offset)
finish = start + L_gemm
```

These equations are **[PROVEN]** by `src/SystolicWS.cc:56-80`.

The existing algorithm adds the empty-pipeline `2H-1` preload cost twice in one path. That is **[WRONG]**.

### 4.4 No arbitrary committed SA bubbles

For an existing back entry:

\[
finish_{back}=start_{back}+H+W-2+\max(compute\_size_{back},4)
\]

The next scheduling offset is at most the previous entry's duration, including the `max(offset,H)` preload case. A new instruction can be appended only while the previous entries are resident. The resulting finish times are strictly increasing.

Therefore:

\[
start_{new}\le finish_{back}
\]

**[DERIVED]** Once `_compute_pipeline.front()->start_cycle <= c`, committed activity is contiguous through `_compute_pipeline.back()->finish_cycle`, using end-exclusive intervals. The proposal's example `G1=[10,20), G2=[25,35)` is not reachable under the current WS scheduler.

If the front has not started, the first committed activation is exactly `front.start_cycle`. In current code this leading gap is produced by an empty-pipeline `GEMM_PRELOAD`.

Consequences:

- Do not copy or scan the compute queue for holes.
- When active, `back.finish_cycle` is enough to identify the end of the committed active block.
- When inactive but the queue is nonempty, `front.start_cycle` is the next modeled activation.
- The existing `find_sa_active_region()` can return infinity for a single future-start preload and is therefore unsafe.

Prediction may run speculatively while this block is active, but no sleep can
begin before the current `back.finish_cycle`. Any newly dispatched GEMM while
the block is resident either overlaps/touches it and may extend that tail.
For the simulator, recomputing when the block becomes idle loses no modeled
cycle. A real pipelined predictor may precompute on dirty events and revalidate
at the final retirement, but that is an implementation-latency optimization,
not a different bound.

### 4.5 What “SA active” currently means

**[PROVEN for the simulator metric]** After retirement and dispatch, ONNXim counts the SA active when:

```text
!_compute_pipeline.empty() &&
_compute_pipeline.front()->start_cycle <= _core_cycle
```

See `src/SystolicWS.cc:96-108`.

Thus a scheduled instruction is modeled active on `[start_cycle, finish_cycle)`.

**[NOT PROVEN for physical power gating]** The `2H-1` future-start interval is described in the source as weight-buffer/weight-preload latency. The repository has no power-domain or state-retention model showing that the whole SA may be off during this interval, nor that preloaded weights survive gating. Until that is established:

- the predictor may report the simulator's modeled next active epoch;
- the power controller should conservatively treat a dispatched preload, or live retained weight context, as a current power-domain requirement.

This distinction is essential. A correct simulator-utilization predictor is not automatically a correct whole-array power-gating controller.

---

## 5. SRAM validity and producer semantics

### 5.1 Supplied working-tree state

The supplied dirty tree adds `pending_memory_count`, `pending_compute_count`, and `get_entry_state()` to SRAM. These fields are **[PROVEN for the supplied state]**, but they are not present at commit `fb87ca4` without the working-tree changes.

For each exact SRAM key `(domain, buffer_id, address)`:

- `prefetch()` creates an invalid entry with `remain_req_count = count` and the selected producer-specific count; `src/Sram.cc:58-88`.
- `count_up()` increments total and producer-specific pending counts and invalidates an entry that was valid; `src/Sram.cc:119-133`.
- `fill()` decrements both total and the selected producer count; the entry becomes valid only when total reaches zero, at which point both producer counts must be zero; `src/Sram.cc:91-117`.

Therefore:

\[
remain\_req\_count = pending\_memory\_count + pending\_compute\_count
\]

is **[DERIVED]** for well-formed calls in the supplied tree.

Memory and compute pending states can coexist because dispatch to an already allocated destination performs `count_up(COMPUTE)` even if the allocation is still memory-pending; `src/SystolicWS.cc:25-54`.

If both kinds remain, validity requires both:

\[
R_{entry}\ge \max(R_{last\ memory},R_{last\ compute})
\]

### 5.2 Exact key and source-domain rules

The predictor must mirror Core's exact lookup semantics:

- destination domain is chosen by `dest_addr >= ACCUM_SPAD_BASE`;
- a source uses accumulator SRAM only when `src_from_accum` is true **and** the address is in the accumulator range;
- otherwise the source uses regular SPAD;
- the buffer ID comes from the resident `Tile` for not-yet-issued instructions, because `Instruction.spad_id` and `accum_spad_id` are assigned only when that instruction reaches the tile head.

Refs: `src/Core.cc:107-122,262-297`.

Address ranges are not modeled. SRAM hashes exact base addresses; `src/Sram.cc:18-47`.

### 5.3 Why pipeline producer search is the wrong mechanism

Searching `_compute_pipeline`, `_vector_pipeline`, and EX for the first matching destination is not sufficient:

- repeated GEMMs write the same accumulator address;
- vector operations are frequently in-place;
- multiple compute producers can be pending on one entry;
- memory and compute producers can coexist;
- accumulating tiles can share an accumulator buffer;
- only one completion is retired per pipeline per invocation;
- EX has not yet called `count_up()`, so it is not represented in `pending_compute_count` until dispatch.

The right rule is to update a persistent entry certificate at producer creation/dispatch, not reconstruct producer history by searching later.

### 5.4 Recommended entry certificate

Add one phase-aware absolute-epoch field per allocated entry:

```text
ready_not_before
```

Its invariant is:

```text
if the entry is invalid at H(c), it cannot become valid at any H(e)
with e < ready_not_before, subject to the observed-current-state floor.
```

Rules:

1. Initialize/reset it on every new SRAM allocation or prefetch. On memory batch creation, update it with the certified earliest epoch at which all required responses could have arrived.
2. On compute/vector dispatch, update it with `max(finish_cycle, dispatch_epoch + 1)`; the `+1` handles zero-latency vector instructions because completion checking already occurred in the dispatch invocation.
3. On another producer for the same entry, take `MAX(old, new)` when the old certificate is valid. If old metadata is invalid/missing, overwrite it with the new bound: that newly observed pending producer is itself necessary for validity, so its bound alone is safe.
4. Do not reduce it on an individual fill; producer identity is not retained. Clear it only when the entry becomes valid or is flushed.
5. At prediction time, if the entry is observed invalid at `H(c)`, use at least `c+1`, even if its stored certificate has expired.
6. A batch already in flight before certificate tracking began has no retroactive timing proof. Treat its certificate as missing and retain the `c+1` observed-state fallback until the entry becomes valid or is reallocated.

Keeping an expired maximum while another producer remains is safe because the observed-invalid floor advances to `c+1`.

The pseudocode below uses absolute epochs only. Its field must either cover the
maximum supported uptime or detect an unencodable epoch and mark the
certificate invalid, causing the dynamic `c+1` fallback. A separately designed
countdown representation could be safe only with defined aging, update, and
wrap semantics; it is not interchangeable with the absolute-epoch MAX and
comparison operations below.

---

## 6. Authoritative opcode routing and generation table

The live opcode enum is `src/Common.h:98-118`. `src/Instruction.h:12` contains a stale smaller duplicate and is not included by the active source tree.

Let:

```text
B = vector_process_bit >> 3
V(x) = ceil(x / B)
A(x) = 1                                  when x <= B
     = ceil(x/B) + A(ceil(x/B))           otherwise
```

`V` and `A` come from `src/Core.cc:600-619`.

| Opcode | Generated where | Route and latency | SRAM inputs / output | Can affect a later generated GEMM? |
|---|---|---|---|---|
| `MOVIN` | GemmWS, ConvWS, Attention, BiasGelu, BiasAct, Softmax, SkipLayerNorm, Conv IM2COL, KVCacheConcat | LD queue; memory-dependent | `src_addrs` are DRAM addresses, not SRAM dependencies. Allocates `dest_addr`. | Yes; all built-in first GEMMs consume at least two preceding fresh MOVIN destinations. |
| `MOVOUT` | Most compute/vector generators | ST queue; no compute latency | SRAM input is named `dest_addr`; `src_addrs` are DRAM destinations. | No data feed, but program order and store readiness can delay a future GEMM. |
| `MOVOUT_POOL` | Conventional ConvWS | Same ST route; no pooling compute is modeled | Same as MOVOUT | No data feed. |
| `GEMM` | ConvWS, Attention | SA; `H+W-2+max(compute_size,4)` | All `src_addrs` must be valid; destination gains one compute producer. | Yes; Attention GEMMs feed vector chains and later GEMMs. |
| `GEMM_PRELOAD` | GemmWS, ConvWS, language Attention, tests | SA; same compute duration, with preload scheduling rule | Same generic GEMM SRAM semantics | Yes. |
| `GEMM_WRITE` | Not generated | Falls through to vector path and returns zero latency | Generic compute destination semantics | No generated case. Current routing contradicts its name. |
| `SOFTMAX` | Attention, standalone Softmax, BiasAct mapping | Vector; `2*A(cs)*add_tree_latency*tile_m + V(cs)*(add+exp+mul)` | Generic vector sources/output, often in-place | Yes in ONNX Attention. |
| `LAYERNORM` | SkipLayerNorm | Vector; formula at `src/SystolicWS.cc:286-291` | In-place vector output | No within the same generated tile. |
| `ADD` | BiasGelu, BiasAct, SkipLayerNorm, Attention | Vector; `V(cs)*add_latency` | Generic, often in-place | Yes transitively in language Attention. |
| `MUL` | BiasAct, Attention | Vector; `V(cs)*mul_latency` | Generic, often in-place | Yes for the pre-second-GEMM Attention instance. |
| `MAC` | Attention | Vector; `V(cs)*mac_latency` | Accumulator read/write | Yes transitively. |
| `DIV` | Attention | Vector; `V(cs)*div_latency` | Generic accumulator vector op | Earlier instance feeds later GEMM transitively; final instance does not. |
| `ADDTREE` | Attention | Vector; `A(cs)*add_tree_latency*tile_m` | Accumulator output | Yes transitively. |
| `EXP` | Attention | Vector; `V(cs)*exp_latency` | In-place accumulator output | Yes transitively. |
| `GELU` | BiasGelu, BiasAct mapping | Vector; `V(cs)*gelu_latency` | In-place vector output | No within current generated tile. |
| `SWISH` | BiasAct mapping | Vector; currently uses GELU latency and carries a TODO | In-place vector output | No within current generated tile. |
| `COMP` | BiasAct maps `relu` to it | Vector; `V(cs)` | In-place vector output | No within current generated tile. |
| `IM2COL` | `Conv::im2col_nhwc()` | Vector; absent from switch, logs and returns zero | No `src_addrs`; writes the same destination as preceding MOVIN and calls `count_up(COMPUTE)` | No current GEMM consumer, but it changes validity and vector occupancy. |
| `BAR` | No live instruction generator | If injected as an instruction, generic vector route with zero latency | Generic destination semantics | No. Real barriers are Tile status values handled by Scheduler. |

Latency switch refs: `src/SystolicWS.cc:281-320`. Generator refs include:

- `src/operations/GemmWS.cc:118-296`;
- `src/operations/ConvWS.cc:237-627`;
- `src/operations/Attention.cc:219-645`;
- `src/operations/BiasGelu.cc:74-112`;
- `src/operations/BiasAct.cc:4-9,95-141`;
- `src/operations/Softmax.cc:71-93`;
- `src/operations/SkipLayerNorm.cc:86-123`;
- `src/operations/Conv.cc:146-210`;
- `src/operations/KVCacheConcat.cc:127-158`.

### 6.1 IM2COL correction

**[PROVEN]** IM2COL:

1. has no source addresses and can dispatch before the preceding MOVIN completes;
2. sees the already allocated destination and calls `count_up(COMPUTE)`;
3. enters `_vector_pipeline`;
4. receives `finish_cycle == start_cycle` because the latency switch returns zero;
5. cannot retire until the following SystolicWS invocation because completion checking precedes dispatch.

It is therefore not “one issue cycle plus no dependency effect.” The zero configured latency is real, but its queue and SRAM effects are also real.

### 6.2 Activation correction

`_activation` is not a special execution unit. `BiasAct.cc:4-9` maps:

```text
gelu    -> GELU
relu    -> COMP
swish   -> SWISH
softmax -> SOFTMAX
```

All follow the ordinary vector path. Conv parses an activation string but ConvWS does not emit an activation instruction, so Conv activation behavior is not modeled in the same way.

---

## 7. Generator structures and relevant defects

| Generator/path | Generated structure | SA-capable? |
|---|---|---:|
| GemmWS | Load phase, then a GEMM_PRELOAD block, then optional MOVOUT | Yes |
| Conventional ConvWS (`group == 1`) | Activation MOVIN, weight MOVINs, GEMM/PRELOAD block, optional MOVOUT/POOL | Yes |
| Group ConvWS | IM2COL tiles followed by a so-called matmul path that currently contains MOVIN/MOVOUT but no GEMM | No in current source |
| ONNX fused Attention | One key MOVIN and one value MOVIN for the tile, then per-head query MOVIN, GEMM, SOFTMAX, GEMM, MOVOUT | Yes |
| Language Attention | Tile-level key MOVIN, per-head query MOVINs, tile-level value MOVIN, all QK GEMMs, `7 x heads` vector operations, all SV GEMMs, optional DIV/MUL/MOVOUT | Yes |
| Non-fused Attention | Intended: GemmWS -> BAR -> Softmax -> BAR -> GemmWS -> BAR; current construction is broken as described below | Intended yes; current path may crash |
| BiasGelu | MOVIN, MOVIN, ADD, GELU, MOVOUT | No |
| BiasAct | MOVIN, MOVIN, ADD, activation, optional MUL, MOVOUT | No |
| Softmax | MOVIN, SOFTMAX, MOVOUT | No |
| SkipLayerNorm | MOVIN, MOVIN, LAYERNORM, ADD, MOVOUT | No |
| IM2COL tile | Repeated MOVIN, IM2COL, MOVOUT triplets, then scheduler barrier | No |
| KVCacheConcat | MOVIN/MOVOUT instructions, but every tile is marked `skip` | No executed work |
| Pool/Flatten/Dummy/EmbedLayerNorm/Concat | Empty and/or skipped tiles | No |
| ConvOS/GemmOS | Incomplete/nonfunctioning paths; SystolicOS asserts on cycle | Not a usable target |

### 7.1 Built-in first-GEMM prefix

**[DERIVED for current production generators]** Every built-in SA-capable WS tile with valid positive dimensions has at least two MOVIN instructions before its first GEMM, and that GEMM reads at least two fresh-load destinations:

- GemmWS: `src/operations/GemmWS.cc:118-205,210-250`.
- conventional ConvWS: `src/operations/ConvWS.cc:237-320,325-394`.
- ONNX Attention: `src/operations/Attention.cc:251-301`.
- language Attention: `src/operations/Attention.cc:377-438`.
- non-fused Attention reuses GemmWS tiles.

Core admission flushes the incoming regular-SPAD buffer, so these regular-SPAD operands cannot be inherited as valid stale entries; `src/Core.cc:44-64`.

**[NOT PROVEN architecturally]** `Tile` has no prefix constraint. Tests create immediate source-free preloads. A `+2 MOVIN` Tile-C bound may be used only after one of:

- an admission-time assertion;
- trusted tile metadata validated in debug builds;
- an explicitly scoped “built-in generators only” experiment.

The recommended architecture-wide fallback is prefix distance zero.

### 7.2 Source defects that should not be hidden by the predictor

The audit found existing generator/control defects:

- IM2COL uses `size = dest_set.size()` but creates `src_set.size()` memory requests. Padding can leave phantom pending counts; `src/operations/Conv.cc:187-196`.
- The IM2COL tile repeatedly MOVINs the same `SPAD_BASE` without deallocation; later MOVINs can fail the destination-allocation check.
- Group ConvWS has additional duplicate accumulator allocations and size/source-count mismatches around `src/operations/ConvWS.cc:503-603`; it also initializes local `out_sp_addr` from itself rather than `out_sp_base` at `:563-565`.
- Non-fused Attention move-inserts `key_query`'s `unique_ptr`s, then aliases `attention_score_tiles` to the moved-from `key_query` deque and dereferences/inserts it again. Both references around `src/operations/Attention.cc:615-623` should instead use `attention_score.get_tiles()`; the current path can dereference null moved-from elements.
- Standalone Softmax leaves `compute_size` unset for SOFTMAX (`src/operations/Softmax.cc:79-85`). SkipLayerNorm leaves it unset for both LAYERNORM and ADD (`src/operations/SkipLayerNorm.cc:102-115`). The latency formulas are present, but these generators feed zero/default width into some terms and under-model execution.
- `HalfSplitScheduler::get_tile()` does not implement the same BAR handling as the other schedulers; a BAR can be popped and discarded rather than enforced.
- `finish_vector_pipeline()` checks regular-SPAD allocation with `accum_spad_id` in its diagnostic path but fills with `spad_id`; `src/Core.cc:497-502`.

These are not reasons to make the predictor emulate broken behavior. They require separate assertions/fixes or explicit exclusion from experiments.

---

## 8. Independent memory lower-bound audit

### 8.1 What MOVIN actually does

When `handle_ld_inst_queue()` processes a MOVIN:

- SRAM is allocated with `remain_req_count = front->size`;
- exactly one `MemoryAccess` is appended for each `front->src_addrs` element;
- all those requests are appended to one core FIFO in one invocation;
- all responses call `Sram::fill(..., MEMORY)`;
- validity occurs only when the total pending count reaches zero.

Refs: `src/Core.cc:511-550,212-234`, `src/Sram.cc:91-117`.

Define:

```text
N_gen      = MOVIN.src_addrs.size()
R_required = MOVIN.size
```

The current formula is well-formed only if `N_gen == R_required > 0`. It also
requires `P>0`. The implementation does not enforce these conditions and
existing generator paths violate the request-count equality.

### 8.2 Request FIFO service

At every ICNT event, Simulator iterates over:

```text
P = icnt_injection_ports_per_core
```

and may pop at most one request per port from the same core FIFO; `src/Simulator.cc:161-181`.

If a newly created batch has `Q` not-yet-injected FIFO entries strictly ahead of it and `N_gen` requests, then under ideal full-rate service the batch's last request cannot be injected before service-event offset:

\[
k_{last}=\left\lfloor\frac{Q+N_{gen}-1}{P}\right\rfloor
\]

This is **[DERIVED]** and has the correct boundary behavior:

- `Q=0, N=P`: all can use the first opportunity, so offset zero;
- `Q=0, N=P+1`: the last needs the second opportunity, so offset one.

`Q` is sampled immediately before the first request of the new batch is
appended. It includes queued MOVIN and MOVOUT requests already in the FIFO and
therefore strictly ahead of the batch; it excludes the batch itself and
requests already in BookSim or DRAM. Sampling after enqueue would count some
or all of the new batch twice and could overestimate the lower bound.
`is_full()` and all later contention can increase the actual
service offset; the expression above is only the earliest offset under
maximum-rate FIFO service.

After a packet leaves the core FIFO, BookSim's GNN traffic manager injects at
most one flit per source node per network step; see
`extern/booksim/src/GNNTrafficManager.cpp:319-505`. On the memory side,
Simulator transfers at most one packet per memory port per ICNT event
(`src/Simulator.cc:184-219`), and the Ramulator2 wrapper attempts to accept
only its front queued request before ticking the memory system
(`extern/ramulator2/resources/ndp_wrappers/ramulator2.cc:66-85`). These limits
can further delay traffic. They do not justify postponing a lower bound beyond
the fastest empty-path case.

`P=16` is **[PROVEN only for the target 128x128 configurations]**, such as `configs/systolic_ws_128x128_c4_booksim2_tpuv4.json:102-106`. `SimulationConfig` defaults it to one; `src/SimulationConfig.h:58-63`. The 8x8 configurations generally run ICNT at 2x core frequency and leave the default at one.

### 8.3 The two `+8` terms

For the target BookSim configuration:

- request control packet is 8 bytes;
- read response is `dram_req_size`, 32 bytes in the target JSON;
- BookSim adds a header and uses 64-byte flits;
- the configured topology is one-stage `fly` with `n=1`.

Refs: `src/Interconnect.cc:94,134-172`, `configs/booksim2_configs/fly_c64_m16.icnt:1-18`, `extern/booksim/src/Interconnect.cpp:183-203`.

A no-contention wrapper probe observed eight BookSim `run()` calls in each direction for this configuration. That is **[DERIVED and experimentally checked for this exact configuration]**, not a portable architectural fact. The JSON `icnt_latency` field is used only by `SimpleInterconnect`; BookSim ignores it.

### 8.4 Ramulator2 minimum

The target configuration selects Ramulator2/HBM2. JSON `dram_latency=10` is used only by `SimpleDram`, not this backend; `src/Dram.cc:15-18,136-178`.

Normal HBM2 reads have timing determined by the Ramulator2 model. More importantly, the generic controller forwards a read that matches a buffered write with:

```text
depart = m_clk + 1
```

at `extern/ramulator2/src/dram_controller/impl/generic_dram_controller.cpp:56-69`.

Therefore a row-hit or closed-row latency is not a universal minimum. MOVOUT followed by a same-address MOVIN makes forwarding architecturally relevant. A global minimum must include this fastest path unless absence of matching writes is proven.

### 8.5 Conditional target formula

Under all of the following conditions:

- target 1 GHz core / 1 GHz ICNT / 1.2 GHz DRAM event schedule;
- target one-stage BookSim setup;
- `P=16`;
- one-flit request and response;
- `N_gen == R_required > 0`;
- `Q` sampled at actual batch creation;
- one-tick Ramulator forwarding path that completes in time for the next eligible ICNT transfer in this target schedule;
- the current source's CORE→DRAM→ICNT tie order;

the old expression can be repaired as a bound on the response-fill ICNT phase:

\[
T_{fill\mbox{-}phase,LB}=
T_{create}+1+
\left\lfloor\frac{Q+N_{gen}-1}{16}\right\rfloor+
8+1+8
\]

A one-request batch therefore has conditional fill-phase label
`T_create + 18`. In the tied target schedule, `H(c)` occurs before Core and
the global label increment to `c+1`; the ICNT phase carrying label `e` is
therefore after `H(e-1)` and before `U(e)`/`H(e)`. A fill at label
`T_create+18` is visible to the next dispatch and decision phases with numeric
epoch `T_create+18`. Thus `+18`, not `+19`, is the phase-correct certificate
for this exact schedule.

The `fresh_batch_ready_lower_bound()` helper in Section 12 is defined to
perform this phase conversion and return the next `U`/`H`-visible numeric
epoch, not an unconverted component tick count.

This result is **[DERIVED, conditional]**, not a repository-wide constant. The old `MIN_DRAM_LATENCY` and both `+8`s were presented too broadly.

The first `+1` is not an abstract “issue cost.” It follows from this exact
stored-label and phase order: `handle_ld_inst_queue()` creates requests before
the later `Core::_core_cycle++`, and the tied ICNT phase observes the
incremented global core count. The middle `+1` is an ICNT/core-label interval,
not the direct addition of one 1.2 GHz DRAM tick: in this target event schedule
the forwarded request can complete between adjacent ICNT transfers and enter
the return NoC at the latter. With another predictor point or frequency
relation, either raw label can change.

### 8.6 Portable event-time formula

The portable form is:

```text
t0       = first eligible ICNT service event after creation phase
k_last   = floor((Q + N_gen - 1) / P)
t_inject = t0 + k_last * Ticnt
t_fillLB = earliest serial event path through:
           request-NoC minimum,
           globally fastest DRAM completion,
           response-NoC minimum
t_readyH = first predictor decision phase that can observe that fill
```

The path must be evaluated with the event scheduler's tie order. Independently rounding each component up to core cycles and adding the rounded values can overestimate the earliest path and is unsafe.

### 8.7 `is_full`, congestion, MOVOUTs, and direction of safety

- BookSim input fullness can only delay injection relative to the maximum-rate FIFO bound.
- NoC congestion, DRAM queueing, and channel conflicts normally delay a request.
- MOVOUT requests ahead in the core FIFO delay injection and are included in `Q`.
- MOVOUT traffic can also **accelerate** a later same-address read through Ramulator write forwarding. Thus “all traffic only delays” is false unless the DRAM minimum already includes forwarding.

Ignoring congestion is safe only after choosing the globally fastest possible downstream path.

### 8.8 Already-in-flight memory

For an allocated invalid entry with pending memory:

```text
now + fresh_minimum_round_trip
```

is **[WRONG and unsafe]**.

Counterexample:

1. The final response has already reached the target core's BookSim boundary.
2. The predictor runs during the current core phase.
3. Later in the same outer iteration, Simulator's ICNT phase calls `push_memory_response()`.
4. The entry can be valid by the next core decision point, far earlier than a new round trip.

The current SRAM counters contain no stage, age, or per-request certificate. The only universally safe remaining-delay lower bound from existing state is zero; at `H(c)`, the observed-invalid phase gives only the structural next-observation floor `c+1`.

The proposed persistent creation-time certificate fixes this prospectively,
for batches whose creation the mechanism observes, without inspecting
BookSim/Ramulator internals. It cannot reconstruct a certificate for a batch
that was already in flight when tracking began; that entry must use `c+1`
until it becomes valid or is reallocated.

### 8.9 Multiple virtual future MOVINs

Reusing the current `Q` for a MOVIN created much later is unsafe because the queue may drain. Blindly adding all virtual requests without subtracting maximum service is also unsafe.

For an exact optimistic recurrence, index ICNT service slots globally with `P` slots per event. Let `tail` be the absolute slot index of the last request in real plus previously modeled virtual work. For batch `j`, created before service event `e_j`:

```text
first_j = max(e_j * P, tail + 1)
last_j  = first_j + N_j - 1
inj_event_j = floor(last_j / P)
tail = last_j
```

Equivalently, the residual requests ahead are:

\[
Q_j=\max(0,tail_{prev}+1-e_jP)
\]

Creation after the event's service phase must advance `e_j`.

The inexpensive safe alternative is to set `Q=0` independently for every virtual future MOVIN while retaining its own `floor((N-1)/P)` term. Ignoring earlier virtual traffic is optimistic and safe.

**Recommendation:** do neither at runtime. Do not predict virtual MOVIN batches. Create the entry certificate only when `handle_ld_inst_queue()` creates the real batch. This removes the virtual-queue state entirely and is the best hardware/correctness tradeoff.

---

## 9. Dependency strategy: forward, backward, hybrid, or rolling

### 9.1 Full forward walk

A forward walk correctly observes mandatory program order, but the current proposal makes it do too much:

- scans an unbounded `deque`;
- invents future queue state;
- propagates every producer even if irrelevant;
- needs future instruction storage and read bandwidth;
- can add independent issue and readiness bounds instead of taking their maximum;
- becomes stale on many pipeline/SRAM events.

It is useful as a **validation oracle**, not as the recommended hardware mechanism.

### 9.2 Pure backward address tracing

A pure backward trace is not reliable for every generated instruction:

- MOVIN `src_addrs` are DRAM sources;
- MOVOUT `src_addrs` are DRAM destinations and its SRAM input is `dest_addr`;
- IM2COL has no source addresses despite its in-place destination behavior;
- repeated GEMMs write the same accumulator destination without an explicit previous-destination source edge;
- vector chains frequently overwrite in place;
- multiple live producers can share a destination;
- two accumulating tiles can share an accumulator buffer;
- `dependent_ids` and `dest_id` exist but live generators do not populate a usable dependency graph;
- a future instruction's buffer IDs are not assigned until it reaches the head.

At minimum, identity is `(SPAD/accumulator domain, buffer_id, exact address)`, but even that does not provide version/producer identity.

Backward tracing can safely omit ambiguous dependencies and become optimistic, but then it loses much of the hoped-for tightness.

### 9.3 Recommended rolling-frontier hybrid

The recommended design uses static program-order metadata and dynamic information only at the current frontier:

1. The generator/compiler reverse-scans a tile once and records a downward-saturating lower bound on the number of non-SA instructions before the first SA instruction.
2. Each SA instruction carries the same kind of certified lower bound to the following SA instruction, or the tile command stream provides an equivalent sideband value.
3. Runtime holds one `has_next_sa` bit and one `next_sa_distance` counter per resident tile and decrements a positive finite distance on successful non-SA issue.
4. Runtime examines only the current head's existing issue conditions and readiness certificate.
5. When a necessary producer is actually created (MOVIN batch) or dispatched (compute/vector), its destination certificate is updated.
6. When the frontier advances, prediction rolls forward. There is no attempt to solve the entire future dependency graph in one step.

This is “hybrid” only in the useful sense:

- static metadata captures unavoidable program order;
- dynamic certificates capture committed producer timing;
- no runtime address-chain traversal is performed.

### 9.4 Why long chains do not require arbitrary traversal

Consider:

```text
MOVIN -> vector -> vector -> ... -> GEMM
```

The distance counter supplies one issue-opportunity cost for every certified
prefix instruction. If saturation omitted part of a long prefix, the result is
only earlier and remains safe. Blocking can only delay frontier advancement.
Once a vector dispatches, it updates its output certificate. When the GEMM
becomes the head, its exact current source keys are checked.

For the seven-operation vector block emitted per head by language Attention:

```text
ADDTREE -> ADD -> EXP -> ADDTREE -> MAC -> DIV -> MUL
```

the counter after the last QK GEMM represents a certified lower bound on the
aggregate vector prefix (the exact prefix when it fits).
The generator emits all QK GEMMs first, then all seven-operation per-head
blocks, then all SV GEMMs; this is not a per-head adjacent
`GEMM1 -> vector block -> GEMM2` chain. Global vector serialization and source
dependencies can only delay issue. In-place/repeated writes are handled by
destination certificates at actual dispatch. When the first SV GEMM reaches
the head, its source readiness is read directly.

No chain-depth limit is needed for correctness. There is no repository-wide maximum chain length: language Attention emits `7 * num_heads` vector operations and dimensions/head counts are configuration-dependent.

### 9.5 Irrelevant prefix work

An instruction that does not feed the target GEMM still precedes it in the tile deque and therefore costs at least one issue opportunity. Its execution or memory completion should not be added unless it actually blocks the current frontier or a target source.

This handles the key example: GemmWS/ConvWS load phases may contain loads unrelated to the first GEMM. Their issue slots are mandatory; their memory completion is not.

---

## 10. Two resident tiles and future Tile C

### 10.1 Independent resident-tile relaxation

Let `A_i` be the actual first future SA use attributable to resident tile `i`. If the predictor constructs `L_i <= A_i` independently, then:

\[
\min_i L_i \le \min_i A_i
\]

Shared issue, EX, vector, SRAM, NoC, DRAM, and SA resources normally add
delay. The memory exception is traffic-induced acceleration such as MOVOUT
write forwarding; the independent relaxation is safe only because the
recommended memory certificate already uses the globally fastest downstream
path and does not retain virtual future traffic. Subject to that rule, giving
each tile optimistic independent access is safe.

One subtlety is `GEMM_PRELOAD`: another GEMM can hide part of its empty-pipeline preload delay, so a standalone preload-start estimate is not independently monotone. The recommended predictor avoids the problem by treating an uncommitted SA instruction's **dispatch** as its conservative power-domain-use epoch. It does not add an empty-pipeline preload bonus to a resident-tile candidate.

### 10.2 Earliest capacity for C

At `H(c)`, define `n_i` as a downward-saturating lower bound on the number of
instructions still in resident tile `i`'s deque. It is exact when the counter
has enough width; rounding a larger deque downward only makes admission look
earlier and remains safe.

If fewer than two tiles are resident:

\[
A_{cap}=c+1
\]

because the current admission opportunity has passed and the next is before `H(c+1)`.

If two tiles are resident, even under unrealistically favorable issue and instant completion, tile `i` cannot free a slot before:

\[
F_i=c+\max(1,n_i)
\]

The `1` covers a deque already empty at `H(c)`: erasure still occurs only in the upcoming `Core::cycle()`.

Relaxing global issue competition independently is optimistic, so:

\[
A_{cap}=\min_i F_i
\]

is a safe lower bound on the next admission epoch.

### 10.3 Incoming tile to SA

Let `p_min` be an **enforced** architectural minimum number of non-SA prefix instructions for every admitted SA-capable tile. With no contract, `p_min=0`.

Let admission epoch `A` denote the scheduler/admission phase before `U(A)` and
`H(A)`. Such a tile cannot dispatch its first SA instruction before:

\[
E_C=A+p_{min}+1
\]

Thus the baseline unknown-C candidate is:

\[
E_C=A_{cap}+1
\]

For the current built-in generator set, a validated `p_min=2` would give `E_C=A_cap+3`, but this must not be silently assumed.

### 10.4 Optional scheduler certificate

A safe optional scheduler interface is one atomically read, versioned per-core
certificate:

```text
future_sa_use_not_before
generation / valid
```

Its meaning must be: **no work that is currently nonresident, nor any work that
can become schedulable under the covered generation, can dispatch an SA
instruction before `U(future_sa_use_not_before)`**. It must cover the entire
future admission sequence and both resident slots, not merely the next tile.
Then it is another necessary lower bound and may tighten the baseline as:

\[
E_C=\max(A_{cap}+p_{min}+1, E_{sched-future})
\]

A lone next-tile header is insufficient even when that tile contains an SA.
For example, with zero residents, a long-prefix tile can enter one slot before
`H(c+1)`, while a later direct-GEMM tile enters the other before `H(c+2)` and
dispatches at `U(c+3)`. Replacing the global fallback with the first tile's
large distance would overestimate. Either retain conservative candidates for
every later admission that can overtake it, or use the sequence-wide
certificate above. The latter is the recommended minimal interface.

This is a small epoch/valid/version sideband at the core boundary, although
producing it may require scheduler-side knowledge. Without it, the universal
`c+2` threat with a vacant slot often prevents sleep for wake latencies greater
than two.

However, the certificate is valid only if the scheduler covers model arrivals, layer readiness, barriers, and policy remapping. `Scheduler::empty()` is not such a certificate; it tests only the top-level request queue (`src/scheduler/Scheduler.cc:185`).

### 10.5 Scheduler/barrier facts

- **[PROVEN]** Scheduler `get_tile()` can release a barrier and distribute the next phase synchronously when launched and finished counts match; `src/scheduler/Scheduler.cc:95-137`.
- **[PROVEN]** That `get_tile()` call can refresh/release work yet return `EMPTY`; the side effect itself must change any optional certificate generation.
- **[PROVEN]** `finish_tile()` calls `refresh_status()`, which can make child work executable and enqueue it; `src/Model.cc:174` and scheduler variants.
- **[PROVEN]** Cores are visited sequentially. A `get_tile()` barrier release or finish on an earlier core can make work available to a later core in the same global core loop; `src/Simulator.cc:123-149`.
- **[PROVEN]** `handle_model()` runs before all core admissions and may inject new work; `src/Simulator.cc:89-106`.
- **[WRONG]** The proposal's `scheduler_has_tile_for_core` signal does not exist in the current interface.

The scheduling variants also do not share one simple future-work rule:

| Variant | Relevant behavior |
|---|---|
| Base/Simple | Launches a new layer only when scheduler queues are empty and no active layer remains; `src/scheduler/Scheduler.cc:187-241`. |
| TimeMultiplex | May launch when scheduler queues are empty even while earlier layers remain active; `src/scheduler/Scheduler.cc:381-458`. |
| Dedicated/Partition | Can enqueue work independently per partition/core and overlap active work; `src/scheduler/Scheduler.cc:254-380`. |
| HalfSplit | Refills request queues independently and maps with `core_id % request_queue.size()`; request erasure changes the divisor and can remap every core, even if a particular core's queue did not change; `src/scheduler/Scheduler.cc:459-558`. Its `get_tile()` does not apply the base BAR-release logic. |

This is why a top-level “scheduler empty” bit cannot substitute for a certified
per-core admission horizon.

The core-local baseline remains safe without observing these events because it always assumes a tile is ready at the earliest capacity epoch. If an optional scheduler certificate suppresses that threat, all certificate-changing events become safety-relevant and require an early-notice contract.

---

## 11. Recommended mathematical predictor

### 11.1 Definitions

At the decision point `H(c)`:

- `E` is an absolute lower bound on the next SA power-domain use.
- `has_next_sa_i` states whether resident tile `i` has a not-yet-issued SA instruction; when true, `d_i` is a downward-saturating lower bound on the number of non-SA instructions before it.
- `n_i` is a downward-saturating lower bound on its remaining deque length.
- `B_i` is a lower bound on the earliest decision epoch at which its current head can successfully issue in the following Core issue phase.
- `X` is a lower bound on the epoch at which the existing EX entry can dispatch and free EX.

### 11.2 SRAM readiness helper

For an exact source key `k`:

\[
R(k,c)=
\begin{cases}
c, & valid(k)\\
\max(c+1,certificate(k)), & allocated\ but\ invalid\\
c+1, & unallocated/unknown
\end{cases}
\]

The unallocated fallback is safe at the chosen point: the handler phases have passed, and one Core issue phase cannot both create a required producer and issue the blocked consumer.

If certificate metadata is inconsistent, use `c+1`, not a guessed future latency.

### 11.3 EX release helper

If EX is empty after the dispatch attempt:

\[
X=c
\]

If EX remains occupied, it failed to dispatch at `U(c)` earlier in the
invocation and that failure is observed at `H(c)`, so:

\[
X\ge c+1
\]

It may be tightened using only existing committed state:

- maximum source `R(k,c)`;
- current vector finish when the EX opcode is vector;
- earliest compute slot when it is GEMM.

Unknown conditions fall back to `c+1`.

If the EX opcode is `GEMM`/`GEMM_PRELOAD`, its candidate is `X`, because it is already issued and dispatch is the conservative power-domain-use point.

### 11.4 Current-head issue lower bound

Start with `B_i=c`.

- If the head requires EX and EX is occupied, set `B_i=max(B_i,X)`.
- For each head source, set `B_i=max(B_i,R(source,c))`.
- If the head is a vector op and the vector pipeline is occupied, set `B_i=max(B_i,vector_finish)`.
- Other unavailable resources may be ignored; ignoring a blocker makes the result earlier and remains safe.

The current head need not be a dependency ancestor of the next GEMM. It is still a mandatory program-order predecessor and therefore its issue block is relevant.

### 11.5 Resident candidate

If tile `i` has a next SA instruction:

\[
E_i=B_i+d_i+1
\]

Derivation:

- the current head may issue no earlier than the phase after `H(B_i)`;
- `d_i` prefix instructions, including the current head when `d_i>0`, must be removed in order;
- the target SA instruction then enters EX;
- it dispatches one decision epoch after its tile issue.

For `d_i=0`, this reduces to the optimistic candidate `E_i=B_i+1`. With a
downward-saturated counter, zero does not prove that the SA instruction is
currently at the head; assuming that it is merely makes the lower bound
earlier and remains safe.

Do not add independent issue distance and memory/vector latency. Necessary lower bounds combine with `MAX`; serialized mandatory steps combine through the distance recurrence above.

### 11.6 Committed pipeline candidate

After current retirement/dispatch:

```text
if compute pipeline nonempty and front.start <= c:
    SA is active now; return c

if compute pipeline nonempty and front.start > c:
    modeled next SA activation is exactly front.start
```

No later source can activate the modeled SA before the committed front.

For whole-array gating, a future-start preload is conservatively treated as “domain already in use” until the preload/retention question is resolved.

### 11.7 Final bound

When the compute pipeline is empty and the domain is presently gateable:

\[
E=\min\left(E_{EX-SA},E_0,E_1,E_C\right)
\]

Candidates absent from the current state contribute infinity. `E_C` remains finite unless there is an explicit global `NO_FUTURE_WORK` certificate.

The guaranteed modeled idle length from `H(c)` is:

\[
G(c)=\max(0,E-c)
\]

The inactive epochs are `[c,E)`.

---

## 12. Pseudocode

### 12.1 Static tile metadata generation

```text
annotate_tile(tile):
    next_has_sa = false
    next_gap = 0

    walk instructions backward:
        if opcode is GEMM or GEMM_PRELOAD:
            instruction.has_following_sa = next_has_sa
            if next_has_sa:
                instruction.gap_after_sa = next_gap
            next_has_sa = true
            next_gap = 0
        else if next_has_sa:
            next_gap = saturating_add_down(next_gap, 1)

    tile.has_first_sa = next_has_sa
    tile.first_sa_distance = next_gap
    tile.remaining_instruction_count = saturating_cast_down(instruction_count)

on_tile_admitted(tile):
    tile.has_next_sa = tile.has_first_sa
    tile.next_sa_distance = tile.first_sa_distance
    tile.remaining_instruction_count =
        saturating_cast_down(tile.instructions.size())
```

Only the tile header and SA instructions need gap fields. A field on every instruction is simpler in the simulator but costs more command storage in hardware.

On successful resident issue:

```text
on_tile_instruction_issued(tile, inst):
    if tile.remaining_instruction_count > 0:
        tile.remaining_instruction_count--

    if inst is GEMM or GEMM_PRELOAD:
        tile.has_next_sa = inst.has_following_sa
        if tile.has_next_sa:
            tile.next_sa_distance = inst.gap_after_sa
    else if tile.has_next_sa and tile.next_sa_distance > 0:
        tile.next_sa_distance--
```

The presence bit is distinct from the numeric field. A finite distance that
saturates downward may reach zero before the actual SA instruction reaches
the head. It must remain zero through intervening non-SA issues; zero means
only “no positive prefix is certified,” not “the head is definitely SA.” A
debug oracle may check that the stored distance never exceeds the exact
distance, but must not assert the latter implication.

The presence metadata must have no false negatives: while any SA instruction
remains, `has_next_sa` must be true. A detected header/sideband/parity error
therefore fails safe to `has_next_sa=true, next_sa_distance=0`. A false
positive only creates an earlier/spurious candidate and is safe; a false
negative removes a resident threat and is unsafe.

### 12.2 SRAM certificate update

```text
on_entry_allocated(entry, now_epoch):
    // Called for every successful SRAM prefetch/allocation, regardless of
    // whether the creator is memory or compute.
    if absolute epoch now_epoch + 1 is representable:
        entry.ready_not_before = now_epoch + 1
        entry.certificate_valid = true
    else:
        entry.certificate_valid = false

install_necessary_producer_bound(entry, r):
    if r is not representable:
        keep any existing valid earlier bound; otherwise remain invalid
        return
    if entry.certificate_valid:
        entry.ready_not_before = max(entry.ready_not_before, r)
    else:
        entry.ready_not_before = r
    entry.certificate_valid = true

on_memory_batch_created(entry, movin, now_phase):
    assert(this allocation and batch-creation event were observed)
    now_epoch = epoch_label(now_phase)

    N = movin.src_addrs.size()
    P = validated_config.injection_ports_per_core
    Q_ahead = request_queue.size() // capture BEFORE appending this batch

    if movin.size != N or N == 0 or P == 0:
        entry.certificate_valid = false     // dynamic c+1 fallback
        return

    cert = fresh_batch_ready_lower_bound(movin, Q_ahead, P, now_phase)
    if cert.valid: // exact supported topology/packets/frequencies/phases
        install_necessary_producer_bound(entry, cert.epoch)
    else:
        entry.certificate_valid = false     // never guess a constant

on_compute_dispatched(entry, inst, dispatch_epoch):
    // If dispatch allocated the destination, on_entry_allocated ran first.
    r = max(inst.finish_cycle, dispatch_epoch + 1)
    install_necessary_producer_bound(entry, r)

on_entry_fill(entry):
    update existing pending counts
    if entry becomes valid:
        clear ready_not_before and certificate_valid

on_entry_flush(entry):
    clear all predictor metadata
```

All queue-position, request-count, and epoch arithmetic must use a sufficiently
wide type or downward-saturating operations. In particular, evaluating
`Q+N-1` with `N=0` can underflow and create an unsafe large certificate.

### 12.3 Source readiness

```text
source_ready_lb(source_key, c):
    state = lookup_exact_sram_entry(source_key)

    if state.valid:
        return c

    if !state.allocated:
        return c + 1

    if !state.certificate_valid or certificate is inconsistent:
        return c + 1

    return max(c + 1, state.ready_not_before)
```

### 12.4 EX release

```text
ex_dispatch_lb(c):
    if EX empty:
        return c

    // It already failed the U(c) dispatch attempt observed at H(c).
    x = c + 1

    for each actual EX source:
        x = max(x, source_ready_lb(source, c))

    if EX is vector and vector pipeline nonempty:
        x = max(x, vector_pipeline.front.finish_cycle)

    if EX is GEMM or GEMM_PRELOAD, and compute pipeline full:
        // Optional tightening. Ignoring this blocker is safe.
        x = max(x, earliest_committed_compute_retirement())

    return x
```

### 12.5 Head issue

```text
head_issue_lb(tile, c, ex_lb):
    if tile.instructions.empty():
        return INF

    head = tile.instructions.front
    b = c

    if head routes to EX and EX nonempty:
        b = max(b, ex_lb)

    for each route-specific SRAM operand of head:
        b = max(b, source_ready_lb(source, c))

    if head routes to vector and vector pipeline nonempty:
        b = max(b, vector_pipeline.front.finish_cycle)

    // Other blockers are deliberately ignored.
    return b
```

The operand decoder is opcode-specific: MOVIN has no SRAM operand because its
`src_addrs` are DRAM addresses; MOVOUT/MOVOUT_POOL read the SRAM entry named by
`dest_addr`; EX operations read `src_addrs` in the regular or accumulator
domain selected by their flags and the owning tile's assigned buffer IDs.
Using one generic address-field interpretation would be wrong.

### 12.6 Unknown admission

```text
capacity_admission_lb(c):
    if resident_tile_count < 2:
        return c + 1

    a = INF
    for each of the two resident tiles:
        a = min(a, c + max(1, tile.remaining_instruction_count))
    return a

unknown_tile_sa_lb(c):
    a = capacity_admission_lb(c)

    prefix = enforced_incoming_min_prefix   // zero without a contract
    e = a + prefix + 1

    if scheduler certificate enabled:
        snapshot = scheduler.read_versioned_snapshot()
        if snapshot validly covers every nonresident/future admission
           threat across both slots:
            e = max(e, snapshot.future_sa_use_not_before)

    return e
```

### 12.7 Core predictor

```text
predict_next_sa_use_at_H(c):
    // Called after retire, LD, EX dispatch, and ST handlers.

    if whole_array_gating and live_SA_context_is_not_retention_safe:
        return c

    if compute pipeline nonempty:
        if compute_pipeline.front.start_cycle <= c:
            return c                  // active now

        if whole_array_gating and front is PRELOAD:
            return c                  // conservative unresolved-domain rule

        return compute_pipeline.front.start_cycle

    ex_lb = ex_dispatch_lb(c)
    E = INF

    if EX nonempty and EX opcode is GEMM or GEMM_PRELOAD:
        E = min(E, ex_lb)             // already issued

    for each resident tile i:
        if tile[i].has_next_sa:
            b = head_issue_lb(tile[i], c, ex_lb)
            E_i = b + tile[i].next_sa_distance + 1
            E = min(E, E_i)

    if not explicit_global_no_future_work:
        E = min(E, unknown_tile_sa_lb(c))

    assert(E >= c)
    return E
```

### 12.8 Guaranteed idle query

```text
guaranteed_idle(c):
    E = cached_or_recomputed_absolute_epoch(c)
    if E is INF:
        return INF
    if E <= c:
        return 0
    return E - c
```

Never reuse the beginning of an old idle episode to compute remaining
guarantee. The compare must precede subtraction because `cycle_type` is
unsigned; `max(0, E-c)` as literal C++ can wrap when a cached bound expires.

---

## 13. Event-driven operation

### 13.1 Two distinct meanings of “must update”

The current proposal mixes two concepts:

1. **Safety invalidation:** an event can introduce a use earlier than the cached absolute bound.
2. **Tightness/state maintenance:** an event makes a later/tighter prediction available but cannot violate the old bound.

If `E_cached <= actual_next_use` was proved correctly, normal forward progress cannot make actual use earlier than `E_cached`. A memory response at or after its creation-time certificate, for example, does not invalidate the old bound.

### 13.2 Baseline core-local safety set

With unconditional unknown-Tile-C coverage and a non-extending sleep controller, the hard safety events are:

- the controller's armed wake action reaching its phase-aware service point;
- violation of tile metadata/prefix contract;
- violation of a memory-certificate/configuration assertion;
- any external mechanism that bypasses normal Core issue/EX dispatch ordering.

Scheduler barrier/layer/model events do not invalidate the core-local bound because the predictor already assumes a tile is available at the earliest possible capacity epoch.

### 13.3 Dirty events for a current/tighter cached result

Set `predictor_dirty` or update the affected small state on:

1. tile admission, skip completion, resident removal;
2. successful resident instruction issue/deque pop;
3. EX enqueue or EX dequeue/dispatch;
4. compute-pipeline enqueue and the last retirement that changes active/idle state;
5. vector enqueue/retirement when it changes EX release or a head blocker;
6. actual MOVIN processing and memory-certificate creation;
7. `count_up()` on an SRAM destination;
8. final relevant fill/validity transition;
9. scheduler certificate creation/change/revocation, if the optional interface is enabled.

Coalesce ordinary predictor-state events occurring before `H(c)` into one
recomputation there. The armed wake action is serviced directly by the
controller at its named phase, not converted into predictor dirtiness.
Likewise, a relied-upon scheduler revocation notice cannot wait for `H(c)`
unless its notice proof explicitly covers that delay.

### 13.4 Events that need no standalone recomputation

- nonfinal memory fills when the aggregate certificate is unchanged;
- MOVOUT responses, which only decrement `_waiting_write_reqs`;
- NoC/DRAM congestion changes, because the predictor does not use them to postpone its bound;
- every raw simulator tick;
- every GEMM finish while the committed pipeline remains active and contiguous;
- every future GEMM in the tile stream.

### 13.5 Optional scheduler-certificate safety set

If a scheduler certificate moves `E_C` later, then all events capable of revoking it become hard safety events:

- model/request arrival;
- a layer becoming executable;
- any `get_tile()` refresh or barrier release, including a call that returns `EMPTY`;
- per-core queue transition or head change;
- scheduler policy remapping/partition change, including HalfSplit request erasure and modulo remapping;
- cross-core `get_tile()` and `finish_tile()` effects.

Merely detecting such an event is insufficient when wake latency exceeds the remaining prefix. The certificate contract must guarantee:

\[
wake\_complete(t_{event},W_{time})\preceq U(E_{new\ threat})
\]

or the old wake deadline must already keep the SA awake. A notice that is
needed to satisfy this inequality must reach the controller immediately in
the required phase; merely setting a dirty bit for the next coalesced `H(c)`
recomputation is not sufficient without a proof that the delay still fits.

Generation invalidation/revocation must occur before or atomically with the
scheduler mutation that can create the earlier threat. An atomic snapshot
prevents torn fields but does not by itself prevent a
read-valid/mutate/dispatch time-of-check/time-of-use window.

---

## 14. Controller correctness

Use power-state names, not utilization names:

```text
SA_ON -> SLEEPING -> WAKING_UP -> SA_ON
```

Let:

- `E` be the absolute lower bound on the next power-domain-use phase `U(E)`;
- `W_time` be physical wake latency;
- `deadline_time(E) = time(U(E)) - W_time` be the latest safe wake-start time;
- `deadline_service(E)` be the latest implementable controller service phase no later than that time.

The familiar scalar `D=E-W` is valid only after defining a controller phase
and proving that a wake started at that phase in epoch `D` completes before
`U(E)`. A simulator implementation can service an explicit controller phase
before EX dispatch; an implementation restricted to later phases must round
the deadline to an earlier serviceable point.

Required invariant while sleeping:

\[
E\le E_{actual},\qquad time(wake\_start)\le deadline\_time(E),\qquad
wake\_complete\preceq U(E)
\]

The last comparison is phase-aware: completing wake-up at `H(E)` is too late,
because dispatch/use `U(E)` has already occurred.

### 14.1 Baseline deadline rule

```text
on entering sleep with proof E:
    if E is INF:
        deadline = INF
        enter sleep with no finite timer under the global-quiescence contract
        return
    deadline = checked_deadline_service(E)
    if deadline is unrepresentable or deadline <= now_phase:
        remain SA_ON                 // correctness guard, not break-even
        return
    arm a direct controller wake action at deadline

on recomputation with E_new:
    ready_if_acted_now = earliest_ready_phase(state, now_phase)
        // SA_ON: now; WAKING_UP: scheduled completion;
        // SLEEPING/entry: fastest completion if wake/abort is commanded now

    if E_new is finite and ready_if_acted_now > U(E_new):
        command/continue the fastest available wake (or remain SA_ON)
        flag a proof-timing violation and apply the chosen interlock policy
        return

    new_deadline = INF if E_new is INF else checked_deadline_service(E_new)

    if E_new is finite and new_deadline is unrepresentable:
        command/continue wake immediately (or remain SA_ON)
        return

    if E_new is finite and new_deadline <= now_phase:
        command/continue wake immediately (or remain SA_ON)
        deadline = new_deadline
        return

    if new_deadline < deadline:
        deadline = new_deadline
        re-arm the direct controller wake action

    else:
        keep old deadline       // baseline never extends a sleep episode
```

Compute the physical deadline with checked arithmetic and round toward an
earlier service phase. A numeric implementation must compare before
subtracting because `cycle_type` is unsigned; evaluating `E-W` first can wrap
a short horizon into a huge, unsafe deadline. `INF` also requires explicit
handling rather than ordinary arithmetic.

This rule assumes sleep entry is abortable and cannot prevent an armed wake
from starting. If entry is non-abortable or has an abort latency, that latency
must be included in the guard and `deadline_service()` calculation.

This directly fixes the existing bug. Under a whole-epoch controller mapping,
if old `E=140`, new `E=125`, and `W=5`, the wake deadline must move from 135
to 120, not remain at 135.

The non-extending baseline is deliberately conservative. After the SA wakes, if it is still idle, it may recompute and enter another sleep episode.

### 14.2 Optional extension

Moving a deadline later is safe only after a complete recomputation proves all
current threats, including Tile C. If scheduler lookahead contributed, the
same sequence-wide certificate and its future-arrival
generation/revocation/notice contract must remain valid for the entire
extended interval; scanning current scheduler state cannot prove arbitrary
future arrivals. The new proof becomes the controller's relied-upon
certificate. Add this optimization only after the baseline passes exhaustive
lower-bound assertions.

### 14.3 Late update theorem

If an unforeseen event is first detected at phase/time `t` and can cause use at `E_new` with:

\[
wake\_complete(t,W_{time})\succ U(E_{new})
\]

then zero-performance-loss wake-up is already impossible. Recomputing more quickly cannot repair it.

One of the following must have been true before the event:

- the old deadline kept the SA awake;
- the event had a proven prefix/notice of at least `W_time`;
- admission/dispatch is interlocked with wake-up, accepting possible performance loss;
- the SA was not placed in a non-ready sleep state.

### 14.4 Break-even threshold

Do not yet finalize:

```text
guaranteed_idle >= break_even + wake_latency
```

Whether this double-counts depends on the eventual definition of break-even time and whether it already includes transition energy/time. Predictor correctness requires only the absolute `E`; energy policy is a separate layer.

### 14.5 Existing worked-example failure

The current algorithm's worked example reaches `T=25`, reports a horizon at
26, but reuses an idle start at 8 and treats the interval as 18 cycles. It then
creates a wake deadline of 21, already in the past. Remaining guarantee must
be computed from the current phase; in unsigned code use `E <= now ? 0 :
E-now`, not a subtract-before-MAX expression.

---

## 15. Lower-bound proof sketch

Assume:

1. decision/use/admission/fill/controller phases have the ordering defined in this document;
2. `has_next_sa` has no false negative while an SA instruction remains;
3. `d_i` does not exceed the true remaining non-SA prefix length;
4. stored `n_i` does not exceed the true remaining deque length;
5. SRAM certificate hooks and source keys are atomic, phase-correct, and every stored bound does not exceed earliest possible validity;
6. incoming prefix metadata is zero unless enforced for every future SA-capable tile;
7. optional scheduler certificates satisfy their sequence-wide revocation/notice contract;
8. finite arithmetic is checked or rounds toward an earlier event, and `INF` is handled separately;
9. dispatch is treated as the conservative use point for uncommitted preloads.

Then:

### Lemma 1: resident issue distance

Core removes at most one resident instruction per decision epoch and preserves per-tile order. If the current head cannot issue before `B_i`, then `d_i` prefix instructions and the target SA instruction cannot be issued/dispatch before `B_i+d_i+1`. Other-tile competition can only delay them.

### Lemma 2: current producer readiness

An invalid source cannot become valid before its observed-current-state floor and persistent producer certificate. Every source condition is necessary, so `MAX` of source bounds is a lower bound on the head's issue epoch.

### Lemma 3: EX

An EX entry that failed the current dispatch point cannot dispatch before the next decision epoch. Source and route-capacity conditions can only delay it. An existing SA EX entry therefore cannot use the domain before `X`; a vector EX entry cannot free EX before `X`.

### Lemma 4: committed pipeline

If the front is active, the SA is active now. If it is future-starting, no later appended entry can start before it. Its start is therefore the exact next modeled activation.

### Lemma 5: unknown tile

Admission cannot occur before the next scheduler opportunity and cannot occur while both resident slots remain occupied. The relaxed remaining-count formula can only free a slot too early. After admission, at least `p_min` prefix issues and one SA issue-to-dispatch step are required.

When the optional scheduler certificate is valid,
`A_cap+p_min+1` and `E_sched-future` are both necessary lower bounds on the
same nonresident/future-work threat. Their `MAX` is therefore also a lower
bound. A next-tile-only certificate would not satisfy this premise.

### Theorem

Every possible next SA use belongs to one of:

- a committed pipeline entry;
- the existing EX entry;
- a resident tile's first remaining SA instruction;
- a future admitted tile.

Each candidate is lower-bounded by the corresponding lemma. The minimum of those lower bounds is therefore no greater than the actual earliest use.

---

## 16. Adversarial analysis

| Scenario | Result for current proposal | Exact failure or conclusion | Recommended correction |
|---|---:|---|---|
| Ready GEMM already in EX at a post-Core point | **Unsafe** | Proposal adds numeric `+1`; next physical tick dispatch sees the same stored cycle label. | Fixed `H(c)` point or phase tuple. |
| One future-start PRELOAD in compute queue | **Unsafe** | Existing active-region walk can miss the front and return infinity. | Read `front.start_cycle` directly. |
| Proposed internal SA gap `G1 10-20, G2 25-35` | **Impossible** | Current scheduling makes committed entries overlap/touch before prior finish. | Remove pipeline bubble scan. |
| One resident long vector tile plus queued Tile C | **Unsafe** | Existing C logic waits for resident finish although second slot is free. | With baseline metadata, use C candidate `c+2` at `H(c)`. |
| Two resident tiles, one blocked | Safe only after per-tile fixes | Blocked tile is skipped; other can issue in same pass. Independent optimistic issue remains safe. | Use `min(E_0,E_1)`. |
| Existing invalid entry; final reply at core NoC boundary | **Unsafe** | `now + fresh latency` overestimates remaining time. | Use persistent creation certificate; without it, fallback to `c+1` observed floor. |
| Multiple virtual MOVINs separated by long issue gap | **Unsafe** | Retaining old Q fails to model queue drain and can delay predicted readiness. | No virtual batches, or use service-slot recurrence; Q=0 is cheap/safe. |
| MOVOUT before same-address MOVIN | **Unsafe with row-hit minimum** | Write-buffer forwarding can make DRAM complete in one tick. | Global DRAM minimum includes forwarding. |
| `size != src_addrs.size()` or `N=0` | **Unsafe or nonsensical for the finite formula** | Pending fills and generated responses do not match, or `Q+N-1` underflows. | Require `size == N > 0` and `P>0` for certification; otherwise use the dynamic early fallback and fix the generator/configuration. |
| Memory plus compute pending on same entry | **Unsafe with producer-type selection** | Validity waits for both. | One aggregate MAX certificate plus observed-invalid floor. |
| Two compute producers for same accumulator entry | **Unsafe with first-match lookup** | One producer finish does not establish validity. | Update certificate on every dispatch; never search for one match. |
| Zero-latency IM2COL | Inaccurate | It occupies vector until next invocation and modifies destination pending count. | Treat through normal dispatch/certificate events. |
| Long unrelated MOVIN prefix | Current walk overworks | Its issue slot matters; its completion need not. | Gap counter plus current-head block only. |
| `GEMM -> 7H vector ops -> GEMM` | Full traversal unnecessary | Program order and vector serialization only delay the next frontier. | Rolling counter; destination certificate at each actual dispatch. |
| EX occupied by blocked vector, tile head GEMM ready | Potentially unsafe if EX ignored with a tightened claim | GEMM cannot enter EX until vector dispatch clears it. | Include `X` in head `B_i`, or omit the tightening. |
| Last MOVOUT | Phase error | It completes tile in store handler, not resident issue phase. | Capacity bound uses optimistic remaining count and fixed phase. |
| Tile removal while earlier unrelated work remains | **Not covered by lifecycle assumption** | Core tracks only last instruction. | Always consider committed pipeline independently; add lifecycle assertion later. |
| Barrier/layer release on another core | Unsafe with unguarded scheduler lookahead | It can populate a later core within the same global loop. | Core-local C threat, or scheduler-wide epoch/notice certificate. |
| No resident/visible GEMM | **Unsafe if infinity** | Future models/layers/tiles can arrive. | Infinity only with explicit global quiescence. |
| Truncated scan followed by `+memory_min` | **Unsafe** | The very next unscanned instruction may be a ready GEMM. | Saturated distance rounds down; fallback is the known prefix or now. |
| Earlier recomputed wake deadline | **Unsafe in current controller** | It retains the later old deadline. | Deadline takes the earlier value; baseline never extends. |
| Recompute after wake lead is already lost | Cannot guarantee zero loss | A wake begun now completes after `U(E_new)`. | Prior certificate/notice must prevent this state; otherwise flag violation/interlock. |

The proposed rolling design remains a lower bound in these cases because every unresolved effect is either represented by a persistent certificate or deliberately treated as able to occur at the next legal phase.

---

## 17. Hardware realism and cost

### 17.1 Naturally available state

These inputs correspond to state an implementation already needs or can expose cheaply:

- SA pipeline empty/front start/back finish/capacity;
- vector busy/front finish;
- EX valid/opcode and current sources;
- resident count and current tile heads;
- SRAM valid bits and buffer IDs;
- request FIFO occupancy at actual MOVIN creation;
- core/ICNT/DRAM configuration constants;
- a controller timer and power-state bits.

### 17.2 Extra realistic state

Recommended additions:

| State | Quantity | Purpose |
|---|---:|---|
| `has_next_sa`, `next_sa_distance` | 2 resident valid bits and 2 counters | Presence plus mandatory program-order lower bound; finite saturation never aliases absence |
| `remaining_instruction_count` | 2 zero-sticky downward-saturating counters | Earliest capacity for Tile C |
| `has_first_sa`, `first_sa_distance` | one protected presence bit plus counter per tile header | Initialize the resident frontier without scanning |
| `has_following_sa`, `gap_after_sa` | one protected presence bit plus counter per SA command, or equivalent sideband | Reload the frontier without scanning |
| `ready_not_before`, `certificate_valid` | one absolute epoch plus valid bit per allocated SRAM scoreboard entry | Persistent producer lower bound with explicit fallback |
| cached `E`, wake deadline | 2 phase-aware absolute timers per core | Event-driven prediction/controller |
| dirty/valid/contract flags | a few bits per core | Coalesced update and fail-safe |
| optional scheduler future-use certificate | one phase-aware epoch plus valid/version per core | Avoid universal vacant-slot pessimism only when all future admissions and both slots are covered |

Counter widths must be selected from measured useful horizons. Distance/count
saturation must round lower bounds downward and stay zero-sticky. Absolute
epochs must cover supported uptime or invalidate to the early fallback on an
encoding/wrap hazard. The paper's 10-bit counter is a useful reference point,
not an automatic choice for ONNXim.

The per-entry epoch is **not automatically small**. In the target
`32768 KiB` regular plus `4096 KiB` accumulator configuration, a pathological
minimum-size (`32 B`) allocation pattern permits up to 1,179,648 live
allocation entries across both halves. Even a 32-bit epoch plus one valid bit
would then be about 4.6 MiB per core. Actual generators normally allocate much
larger regions, but a workload histogram—not intuition—is required.

The preferred hardware realization attaches the field to an already bounded
allocation scoreboard. If that scoreboard is not bounded cheaply, use a
small capped pending-entry certificate table and make a miss/overflow return
the dynamic `c+1` fallback; do not add an unbounded CAM. Report both its peak
occupancy and lookup energy before claiming low overhead.

### 17.3 Combinational work per evaluation

Baseline worst-case operations are approximately:

- one committed-pipeline active/start comparison;
- one EX opcode check and small MAX tree over its current operands;
- up to two resident head checks;
- two counter additions for resident candidates;
- one remaining-count MIN for Tile C;
- a final MIN over roughly four or five candidates;
- source metadata reads already adjacent to the issue readiness path.

For generated GEMMs, source count is typically two, but this is not an enforced Core limit. Hardware should cap the supported read count and fall back early if metadata exceeds it.

For `P=16`, the optional queue serialization divide is a shift. A general non-power-of-two `P` needs a small divider/LUT or should use the Q=0 fallback.

### 17.4 Simulator oracle state that is rejected

Do not rely on:

- arbitrary scans of `tile.instructions` at runtime;
- deep inspection of future scheduler queues;
- associative producer searches across all pipeline entries;
- arbitrary BookSim boundary/ejection state;
- Ramulator request-buffer/row-state inspection;
- a C++ hash table/deque as though it were free hardware storage;
- full future dependency graphs inferred from addresses.

These may be used in a validation oracle that checks the hardware-oriented predictor, not in the mechanism being evaluated for overhead.

### 17.5 Why the original cost estimate is incomplete

The original estimate counts a few counters/comparators but not:

- storage/read bandwidth for an unbounded future instruction scan;
- producer-address CAM comparisons;
- per-tile scan caches and invalidation logic;
- scheduler lookahead signals;
- extra SRAM metadata ports;
- scan latency and energy;
- memory-system visibility.

The rolling design makes the hardware estimate credible because its runtime work is bounded by two resident heads and one EX entry.

---

## 18. Comparison with SLUMBER

### 18.1 What the paper actually does

From Tripathy et al., Sections 2–4 and Figures 7–8:

- static compiler analysis supplies the type of a register's next access, read or write;
- deterministic functional-unit latency is used once a compute operation is known;
- memory status is classified as L1 hit, L1 miss/L2 hit, or L2 miss/DRAM using a 2-bit MSHR-status tracker per warp;
- wake-up uses the measured minimum latency of the observed class;
- shallow-sleep wake latency is hidden with warp/instruction lookahead, including up to three future warps and a fetch-unit list;
- the power states are ON, shallow sleep, deep sleep, and OFF;
- the reported wake delays are 0, 4, 13, and 16 cycles respectively;
- power gating is used only where register contents need not be retained, such as a next write or finished allocation.

The stated overhead is not just a 2-bit field:

- `10 bits x 1024` register counters = 1.25 KB;
- 16-byte MSHR tracker;
- 128-byte fetch-unit list;
- muxes, tri-modal switches, and control logic;
- about 4% of register-file size and under 1% of register-file leakage power.

The paper reports about 94% RF leakage saving and 1.2% average performance degradation. It is not a proof of zero late wake-up.

### 18.2 Transferable lesson

SLUMBER waits until a long-latency cause is committed/classified and combines it with compact static metadata. It does not reconstruct an arbitrary future dependency graph.

That lesson motivates the recommended ONNXim design:

- gap metadata replaces deep instruction lookahead;
- creation/dispatch certificates replace later producer reconstruction;
- memory is tightened only after the actual MOVIN batch exists;
- controller decisions use minimum-class latency, not expected latency.

### 18.3 Why ONNXim is different

SLUMBER controls individual register entries associated with warps. ONNXim's target is a shared SA with:

- two resident command streams;
- one shared issue path and EX slot;
- concurrent vector execution;
- accumulated/in-place SRAM producers;
- whole-tile admission/replacement;
- scheduler barriers/layers/models;
- preload/weight-stationary state whose retention under gating is unknown.

Some extra reasoning is therefore necessary. The existing proposal is nevertheless complex in the wrong places: full future walks, pipeline bubble scans, address-chain reconstruction, and memory-system introspection are unnecessary. The genuinely necessary additions are the two-tile minimum, Tile-C threat, fixed phase convention, and power-domain/preload contract.

---

## 19. Direct disposition of the 31 open questions

1. **Why forward-walk instead of start from GEMM?** Neither at runtime. Use static gap metadata plus current-head readiness. Pure backward tracing is not authoritative.
2. **Irrelevant MOVIN/vector paths?** Count their unavoidable issue slots only. Do not add their completion delay unless they block the current head or target source.
3. **Long chains?** Rolling frontier handles arbitrary length; no arbitrary-depth runtime traversal.
4. **Parallel memory requests?** SRAM validity waits for all required fills, so the last completion matters; `MAX`, not `MIN`.
5. **Future-MOVIN formula?** Conditional only. `P`, `Q`, `N`, phase conversion, NoC minima, and global DRAM minimum must be corrected as in Section 8.
6. **Multiple future MOVIN Q?** Model maximum-rate drain with service slots, or safely use Q=0. Baseline predicts only actual batches.
7. **Already-in-flight memory?** Fresh round trip from now is unsafe. Existing-state fallback is zero remaining delay; a persistent creation certificate fixes only batches observed at creation, while legacy in-flight entries retain the fallback.
8. **Mixed producer counters?** Entry readiness is after both; combine committed lower bounds with `MAX`.
9. **Identify compute producer by pipeline search?** No. Repeated/multiple producers invalidate first-match lookup. Update one aggregate certificate at each dispatch.
10. **IM2COL?** Vector-routed, nominal latency zero, retires next invocation, modifies compute-pending state.
11. **`_activation`?** Ordinary GELU/SWISH/COMP/SOFTMAX opcodes.
12. **Independent A/B walks?** MIN is safe when each candidate is a true optimistic lower bound. Avoid standalone preload bonuses.
13. **Can blocked A stop B?** A is skipped and B is tried in the same pass; issue priority is sticky, not round-robin.
14. **Tile replacement while SA executes?** Core does not globally enforce no outstanding earlier work. Always track committed pipeline independently.
15. **Is C completely unknown?** Built-ins have two MOVIN prefixes, but Core/tests do not enforce it. Metadata/assertion is required for a tighter bound.
16. **Inspect Scheduler?** Deep core-side inspection is rejected. An optional compact scheduler `future_sa_use_not_before` interface is useful only if it covers all admissions, both slots, overtaking, and arrivals; a next-tile header alone is unsafe.
17. **Both resident tiles have no GEMM?** Sleep only until the safe C candidate; use remaining resident counts and, if implemented, a sequence-wide future-use certificate rather than a next-tile header.
18. **How far to look?** Only the first possible next use. Reload metadata after it issues.
19. **Run every cycle?** No. Use event-maintained state and one coalesced `H(c)` recomputation when dirty or deciding sleep, except that any relied-upon scheduler revocation notice must meet its phase-aware wake contract immediately.
20. **Does SRAM becoming valid require recomputation?** Not for safety if the original certificate was honored; yes for a tighter/current result or optional extension.
21. **Predict while GEMM executes?** Optional. It offers little in a zero-latency simulator controller and complicates proof; baseline decides when the committed block becomes idle.
22. **Can `back.finish` hide bubbles?** No arbitrary committed internal bubbles exist in current WS scheduling.
23. **What is active?** Simulator: nonempty pipeline with `front.start <= c`, on `[start,finish)`. Physical preload/state need is not proven.
24. **State name COMPUTING?** Replace with `SA_ON`.
25. **Wake deadline bug?** Confirmed. Earlier horizons require earlier deadlines.
26. **MUST_UPDATE too late?** Confirmed. Prior proof/notice must guarantee that a wake begun at the notice phase completes no later than `U(E_new)`; otherwise zero loss is impossible.
27. **Break-even plus wake latency?** Leave unresolved until the energy definition is fixed.
28. **Predictor complexity?** Runtime is bounded to two heads, one EX, and small MIN/MAX/counter logic.
29. **Simulator oracle?** Full deque/scheduler/pipeline/memory scans are rejected; metadata and certificates make the inputs hardware-realistic.
30. **Overengineering vs paper?** Two-tile/admission/power-domain logic is necessary; dependency and memory-oracle complexity is not.
31. **Future energy model?** Keep predictor event/operation counters and separate prediction correctness from the later transition/break-even policy.

---

## 20. Still Not Proven / Requires Experiment or Further Inspection

1. **Physical SA gating boundary.** Does the `2H-1` preload interval require powered array circuitry? Which weight/register state must be retained between PRELOAD and later GEMMs?
2. **Power-state implementation.** Wake latency, sleep-entry latency, transition energy, sleep leakage, retained-state leakage, and allowable clock/power domains are absent from ONNXim.
3. **Portable positive NoC minimum.** The two eight-tick observations are specific to one BookSim configuration and packet shape.
4. **Portable DRAM minimum.** A useful minimum larger than one forwarding tick requires proof that matching buffered writes cannot exist.
5. **Fresh MOVIN well-formedness.** `size == src_addrs.size()` is globally false. Every certified GEMM-feeding subset needs `size == N > 0`, `P>0`, assertions, and tests.
6. **Existing in-flight positive remaining bound.** Current SRAM state cannot provide one. The new creation-time certificate is required.
7. **Instruction-prefix contract.** Built-in generators satisfy two MOVINs before first GEMM, but Core/tests do not.
8. **Tile lifecycle invariant.** Core does not prove that every earlier independent operation finishes before the last instruction retires.
9. **Maximum practical counter widths.** Tile length and Attention chain length are shape-dependent; workload histograms are needed to select saturating widths.
10. **Maximum GEMM/head source count.** Generated cases are small, but Core does not enforce a hardware port bound.
11. **External work-arrival contract.** A hardware system cannot predict an arbitrary new command without a scheduler wake/notice protocol.
12. **Scheduler certificate feasibility.** It needs evaluation across Simple, TimeMultiplex, Dedicated/Partition, HalfSplit, and language scheduling.
13. **HalfSplit barrier semantics.** They differ from other scheduler variants and appear defective.
14. **Group Conv and non-fused Attention correctness.** Existing generator defects should be repaired or excluded before predictor evaluation.
15. **Exact phase conversion tests.** All supported core/ICNT/DRAM frequency ratios and tie cases require executable timing tests.
16. **Energy benefit of tighter queue terms.** Measure whether Q-aware memory certificates materially improve sleep opportunities over Q=0 end-to-end minima.
17. **Predictor overhead.** Area, dynamic energy, metadata SRAM overhead, and timing impact require synthesis/CACTI-style modeling.
18. **Zero-loss empirical validation.** The lower-bound assertion must be checked over all tests/models/configurations and adversarial synthetic tiles before gating is enabled.

---

## 21. Concrete implementation plan — no code in this review

### Phase 0: freeze and test timing semantics

1. Add debug-only named phases for scheduler admission, `U(c)` dispatch/use, `H(c)` decision, ICNT fill, and controller wake-start/completion.
2. Write cycle-exact tests for:
   - tile head → EX → plain GEMM start;
   - source-free immediate GEMM/PRELOAD;
   - future-start PRELOAD;
   - vector zero latency;
   - MOVIN issue → LD processing → request creation;
   - target one-request ordering `H(c+17) < fill(label c+18) < U(c+18) < H(c+18)`;
   - last MOVOUT processing and tile removal;
   - vacancy and two-full-slot Tile C admission.
3. Assert active intervals are `[start,finish)` in tests.

Likely files later: `src/SystolicWS.cc/.h`, `src/Core.cc/.h`, `tests/SystolicWsTest.cc`.

### Phase 1: repair/guard source invariants before prediction

1. Add debug assertions that certified MOVINs satisfy `size == src_addrs.size()`, `size > 0`, and `P>0`.
2. Fix every generator/scheduler defect in Section 7.2 or explicitly exclude the affected path from predictor evaluation.
3. Add an explicit supported-tile contract instead of relying on generator inspection.
4. Resolve/remove the stale duplicate opcode declaration in `src/Instruction.h`.

### Phase 2: add static tile metadata

1. Add separate `has_first_sa` and `first_sa_distance` metadata to each Tile descriptor.
2. Add `has_following_sa` plus `gap_after_sa` to each SA command, or an equivalent compact sideband stream.
3. Add zero-sticky downward-saturating `remaining_instruction_count` and update it on actual deque pop without underflow.
4. Use separate protected presence bits and downward saturation for finite distance counters; a detected metadata error defaults to present/distance zero.
5. In debug builds, recompute presence, distances, and remaining count by scanning the deque at admission and assert that stored numeric fields never exceed the exact values. This scan is validation-only, not predictor cost.

Likely files later: `src/Common.h`, all active tile generators, `src/Core.cc`.

### Phase 3: add producer readiness certificates

1. Add absolute `ready_not_before` and `certificate_valid` fields to the SRAM entry/scoreboard; initialize/reset them on every allocation.
2. Update them at actual MOVIN batch creation and actual compute/vector dispatch; capture Q before batch enqueue.
3. Aggregate valid bounds with MAX; overwrite an invalid bound with a newly known necessary-producer bound; clear on valid/flush.
4. First implement a conservative per-configuration end-to-end fresh minimum with Q=0.
5. Add the actual-Q/P serialization term only behind configuration validation, with an explicit invalid/fallback result for unsupported cases.
6. Express cross-domain timing through absolute event time/phase conversion.
7. Never infer a fresh latency from current pending counts.
8. Measure live invalid-entry occupancy; if a full scoreboard attachment is too large, use a capped certificate table whose miss/overflow selects `c+1`.

Likely files later: `src/Sram.h/.cc`, `src/Core.cc`, `src/SystolicWS.cc`, `src/Simulator` timing helper, configuration validation.

### Phase 4: implement predictor in shadow mode

1. Add a small per-core predictor object/state.
2. Invoke it only at `H(c)` when dirty or when the controller needs a decision.
3. Implement committed, EX, two resident, and unknown-C candidates exactly as specified.
4. Keep power gating disabled.
5. Record:
   - predicted absolute epoch;
   - actual next power-domain-use/dispatch phase and modeled SA start;
   - prediction slack;
   - fallback reason;
   - number/type of comparisons and metadata reads.
6. Assert online/offline that `U(E_pred)` is no later than the actual next
   power-domain-use/dispatch phase for every prediction. An offline
   full-future oracle may be used only for validation.

### Phase 5: optional scheduler certificate

1. Measure baseline lost opportunity caused by `c+2` vacant-slot fallback.
2. If material, add a compact per-core `future_sa_use_not_before` certificate that covers all future admissions and both slots; do not substitute a lone next-tile header.
3. Version the certificate with an epoch/generation number.
4. Revoke/change the generation before or atomically with every model/layer/barrier/`get_tile()`/queue/remap mutation listed in Section 13, with phase-timely controller notice.
5. Test cross-core same-global-loop releases, a long-prefix T1 overtaken through the second slot by direct-GEMM T2, and read-valid/mutate/dispatch TOCTOU cases.

### Phase 6: controller shadow model

1. Implement `SA_ON/SLEEPING/WAKING_UP` without changing execution timing.
2. Use absolute deadlines and the non-extending baseline rule.
3. Assert wake completion before every actual domain-use/dispatch phase.
4. Treat preload/live-weight context as nongateable until the physical contract is resolved.
5. Test horizons `W-1`, `W`, and `W+1`, plus `W=0`, `INF`, expired bounds, unsigned limits, earlier recomputation while SLEEPING and already WAKING_UP, and timer/completion ordering before `U(E)`.
6. Model sleep-entry/abort latency or assert that a scheduled wake can interrupt entry; keep break-even/energy policy separate.

### Phase 7: adversarial and property testing

Required synthetic tests include:

- zero/one/two resident tiles;
- immediate direct GEMM tiles;
- two mandatory MOVIN prefix tiles;
- blocked A with ready B;
- EX GEMM and EX vector cases;
- long vector chains and in-place writes;
- multiple compute producers for one accumulator entry;
- mixed memory/compute pending state;
- final response already at NoC output;
- multiple real and virtual MOVIN queue-drain cases;
- `N=0`, `P=0`, pre-enqueue Q capture, unsupported memory configuration fallback, and the exact target `+18` visibility boundary;
- SRAM allocation/reset/reuse, valid-to-invalid `count_up()`, and missing legacy certificates;
- finite gap saturation to zero versus true no-SA presence, zero-sticky remaining-count saturation, near-overflow arithmetic, and separate `INF` handling;
- route-specific MOVIN/MOVOUT/EX SRAM operand decoding;
- MOVOUT forwarding a later MOVIN read;
- all supported clock ratios;
- future-start PRELOAD;
- tile removal with unrelated committed SA work;
- barriers, layer transitions, model arrivals, and cross-core release.

Use randomized legal tile streams and compare the hardware-oriented predictor against an offline exact next-use oracle.

### Phase 8: energy integration only after correctness

1. Define the actual gateable SA domain and retained state.
2. Obtain ON/idle/sleep/wake power, transition energy, and wake latency.
3. Measure predictor/controller metadata and operation energy.
4. Define break-even precisely.
5. Enable timing-affecting gating only after shadow runs show zero lower-bound violations.

---

## 22. Acceptance criteria before implementation is considered safe

The design should not move from shadow prediction to functional gating until all are true:

- every timing equation is tied to the named admission, `U(c)`, `H(c)`, ICNT-fill, wake-start, and wake-completion phases;
- no prediction violates `E_pred <= actual_next_domain_use` in unit, fuzz, or workload tests;
- direct-GEMM Tile C tests pass with no generator-prefix assumption;
- memory certification either validates its configuration or falls back conservatively;
- already-in-flight memory never receives a restarted fresh latency;
- detected metadata corruption and every overflow/unrepresentable case round toward earlier use or select the explicit early fallback;
- scheduler lookahead, if enabled, has a tested notice/revocation contract;
- preload/weight retention and the actual power domain are defined;
- controller deadlines are absolute and earlier updates always pull wake-up earlier;
- predictor operations/state are counted for the later energy model.

---

## Final recommendation

Implement the rolling-frontier design in shadow mode first. Do not implement the current full forward-chain pseudocode.

The core mechanism should be:

```text
committed front
    MIN existing EX-SA
    MIN tile-A(frontier certificate + static distance)
    MIN tile-B(frontier certificate + static distance)
    MIN future-Tile-C(capacity + enforced prefix)
```

with persistent SRAM `ready_not_before` metadata and phase-aware absolute wake
deadlines.

This is safer, smaller, easier to validate, and closer to SLUMBER's useful principle—small static annotations plus committed runtime classification—while still accounting for the ONNXim-specific two-tile, EX, SRAM, and scheduler structure.
