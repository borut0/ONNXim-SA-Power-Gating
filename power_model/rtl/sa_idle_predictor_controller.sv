module sa_idle_predictor_controller #(
  parameter integer HORIZON_W = 20,
  parameter integer TICKET_W = 20,
  parameter integer INJECTION_PORTS_LOG2 = 4,
  parameter integer BREAK_EVEN_CYCLES = 166,
  parameter integer WAKE_GUARD_CYCLES = 1
) (
  input  logic                    clk,
  input  logic                    control_clk,
  input  logic                    descriptor_clk,
  input  logic                    sample_enable,
  input  logic                    rst_n,

  // Current core/scheduler state at the V5 recomputation point.
  input  logic                    sa_active,
  input  logic                    compute_pipeline_empty,
  input  logic                    compute_front_future_valid,
  input  logic [HORIZON_W-1:0]    compute_front_remaining,
  input  logic                    ex_inst_queue_empty,
  input  logic                    ex_front_is_sa,
  input  logic                    scheduler_core_tile_waiting,
  input  logic [HORIZON_W-1:0]    global_issue_drain_bound,
  input  logic                    core_queue_tickets_exact,
  input  logic [TICKET_W-1:0]     last_dequeued_ticket,

  // A write replaces the complete metadata descriptor for resident slot 0.
  input  logic                    tile0_we,
  input  logic                    tile0_present,
  input  logic                    tile0_candidate_found,
  input  logic                    tile0_candidate_preload,
  input  logic                    tile0_sources_supported,
  input  logic [HORIZON_W-1:0]    tile0_prefix_count,
  input  logic [HORIZON_W-1:0]    tile0_remaining_count,
  input  logic [1:0]              tile0_operand_used,
  input  logic [1:0]              tile0_operand_allocated,
  input  logic [1:0]              tile0_operand_valid,
  input  logic [1:0]              tile0_operand_pending_memory,
  input  logic [TICKET_W-1:0]     tile0_operand0_tail_ticket,
  input  logic [TICKET_W-1:0]     tile0_operand1_tail_ticket,
  input  logic [HORIZON_W-1:0]    tile0_operand0_pending_count,
  input  logic [HORIZON_W-1:0]    tile0_operand1_pending_count,

  // A write replaces the complete metadata descriptor for resident slot 1.
  input  logic                    tile1_we,
  input  logic                    tile1_present,
  input  logic                    tile1_candidate_found,
  input  logic                    tile1_candidate_preload,
  input  logic                    tile1_sources_supported,
  input  logic [HORIZON_W-1:0]    tile1_prefix_count,
  input  logic [HORIZON_W-1:0]    tile1_remaining_count,
  input  logic [1:0]              tile1_operand_used,
  input  logic [1:0]              tile1_operand_allocated,
  input  logic [1:0]              tile1_operand_valid,
  input  logic [1:0]              tile1_operand_pending_memory,
  input  logic [TICKET_W-1:0]     tile1_operand0_tail_ticket,
  input  logic [TICKET_W-1:0]     tile1_operand1_tail_ticket,
  input  logic [HORIZON_W-1:0]    tile1_operand0_pending_count,
  input  logic [HORIZON_W-1:0]    tile1_operand1_pending_count,

  output logic                    sleep_request,

  // Read-only clock-wrapper summaries derived from the controller's existing
  // registered state.  They add no architectural state.
  output logic                    entry_descriptors_eligible,
  output logic                    certificate_pipeline_active
);

  localparam logic [HORIZON_W-1:0] ONE = {{(HORIZON_W-1){1'b0}}, 1'b1};
  localparam logic [HORIZON_W-1:0] ENTRY_REQUIRED =
      BREAK_EVEN_CYCLES + WAKE_GUARD_CYCLES;
  localparam logic [HORIZON_W-1:0] WAKE_GUARD = WAKE_GUARD_CYCLES;
  localparam logic [HORIZON_W-1:0] ENTRY_PIPELINE_CYCLES = 3;
  localparam logic [HORIZON_W-1:0] PIPELINED_ENTRY_REQUIRED =
      BREAK_EVEN_CYCLES + WAKE_GUARD_CYCLES + 3;

  // These registers are the incremental hardware metadata cost.  The tile
  // generator supplies the next-SA descriptor; the core updates it whenever
  // the resident head, operand state, or associated MOVIN tail ticket changes.
  logic                 tile0_present_q;
  logic                 tile0_candidate_found_q;
  logic                 tile0_candidate_preload_q;
  logic                 tile0_sources_supported_q;
  logic [HORIZON_W-1:0] tile0_prefix_count_q;
  logic [HORIZON_W-1:0] tile0_remaining_count_q;
  logic [1:0]           tile0_operand_used_q;
  logic [1:0]           tile0_operand_allocated_q;
  logic [1:0]           tile0_operand_valid_q;
  logic [1:0]           tile0_operand_pending_memory_q;
  logic [TICKET_W-1:0]  tile0_operand0_tail_ticket_q;
  logic [TICKET_W-1:0]  tile0_operand1_tail_ticket_q;

  logic                 tile1_present_q;
  logic                 tile1_candidate_found_q;
  logic                 tile1_candidate_preload_q;
  logic                 tile1_sources_supported_q;
  logic [HORIZON_W-1:0] tile1_prefix_count_q;
  logic [HORIZON_W-1:0] tile1_remaining_count_q;
  logic [1:0]           tile1_operand_used_q;
  logic [1:0]           tile1_operand_allocated_q;
  logic [1:0]           tile1_operand_valid_q;
  logic [1:0]           tile1_operand_pending_memory_q;
  logic [TICKET_W-1:0]  tile1_operand0_tail_ticket_q;
  logic [TICKET_W-1:0]  tile1_operand1_tail_ticket_q;

  // Certificate generation uses a stable registered descriptor.  A write
  // prevents entry and, if already asleep, forces immediate wake.  This
  // removes the long write-through path without allowing stale metadata to
  // extend a sleep episode.
  logic                 tile0_present_e;
  logic                 tile0_candidate_found_e;
  logic                 tile0_candidate_preload_e;
  logic                 tile0_sources_supported_e;
  logic [HORIZON_W-1:0] tile0_prefix_count_e;
  logic [HORIZON_W-1:0] tile0_remaining_count_e;
  logic [1:0]           tile0_operand_used_e;
  logic [1:0]           tile0_operand_allocated_e;
  logic [1:0]           tile0_operand_valid_e;
  logic [1:0]           tile0_operand_pending_memory_e;
  logic [TICKET_W-1:0]  tile0_operand0_tail_ticket_e;
  logic [TICKET_W-1:0]  tile0_operand1_tail_ticket_e;

  logic                 tile1_present_e;
  logic                 tile1_candidate_found_e;
  logic                 tile1_candidate_preload_e;
  logic                 tile1_sources_supported_e;
  logic [HORIZON_W-1:0] tile1_prefix_count_e;
  logic [HORIZON_W-1:0] tile1_remaining_count_e;
  logic [1:0]           tile1_operand_used_e;
  logic [1:0]           tile1_operand_allocated_e;
  logic [1:0]           tile1_operand_valid_e;
  logic [1:0]           tile1_operand_pending_memory_e;
  logic [TICKET_W-1:0]  tile1_operand0_tail_ticket_e;
  logic [TICKET_W-1:0]  tile1_operand1_tail_ticket_e;

  logic [HORIZON_W-1:0] tile0_candidate_remaining;
  logic [HORIZON_W-1:0] tile1_candidate_remaining;
  logic [HORIZON_W-1:0] tile0_retirement_bound;
  logic [HORIZON_W-1:0] tile1_retirement_bound;
  logic [HORIZON_W-1:0] future_tile_cap;
  logic [HORIZON_W-1:0] visible_remaining;
  logic [HORIZON_W-1:0] safe_remaining;
  logic                  visible_valid;
  logic                  now1_cap_binding;
  logic                  retention_safe_preload;
  logic                  response_count_sample_enable;
  logic [HORIZON_W-1:0] fresh_work_drain_remaining;
  logic [HORIZON_W-1:0] sleep_countdown;
  logic                  certificate_stage1_valid;
  logic [HORIZON_W-1:0] tile0_operand0_fifo_stage1;
  logic [HORIZON_W-1:0] tile0_operand1_fifo_stage1;
  logic [HORIZON_W-1:0] tile1_operand0_fifo_stage1;
  logic [HORIZON_W-1:0] tile1_operand1_fifo_stage1;
  logic                  certificate_stage2_valid;
  logic                  certificate_stage2_visible_valid;
  logic [HORIZON_W-1:0] certificate_stage2_visible_remaining;
  logic [HORIZON_W-1:0] certificate_stage2_future_tile_cap;
  logic [HORIZON_W-1:0] stage2_safe_remaining;
  logic                  stage2_now1_cap_binding;
  logic                  stage2_used_work_drain;
  logic [HORIZON_W-1:0] stage2_fresh_work_drain_remaining;
  logic                  certificate_stage3_valid;
  logic                  certificate_stage3_used_work_drain;
  logic [HORIZON_W-1:0] certificate_stage3_remaining;
  logic                  entry_commit;

  assign entry_descriptors_eligible =
      tile0_present_q && tile0_candidate_found_q &&
      tile0_candidate_preload_q &&
      tile1_present_q && tile1_candidate_found_q &&
      tile1_candidate_preload_q;

  // Keeping the gated clock alive while a valid stage exists is sufficient
  // to flush every stale certificate after evaluation becomes ineligible.
  assign certificate_pipeline_active = certificate_stage1_valid ||
      certificate_stage2_valid || certificate_stage3_valid;

  // Operand isolation for the live SRAM completion counters. Their values
  // matter only when constructing a fresh retention-safe certificate. Once
  // asleep, the frozen countdown already models maximum-rate response drain.
  assign response_count_sample_enable = !sleep_request && !sa_active &&
      !tile0_we && !tile1_we && core_queue_tickets_exact &&
      retention_safe_preload;

  // The control domain consumes only a complete registered certificate.  In
  // the baseline testbench control_clk == clk and sample_enable == 1, exactly
  // reproducing the original continuously evaluated controller.  A physical
  // wrapper may pulse the evaluation clock/sample input sparsely without
  // changing this commit condition or the saved countdown.
  assign entry_commit = certificate_stage3_valid &&
      certificate_stage3_remaining >= ENTRY_REQUIRED &&
      (!certificate_stage3_used_work_drain ||
       !scheduler_core_tile_waiting) &&
      !sa_active && !tile0_we && !tile1_we &&
      retention_safe_preload && core_queue_tickets_exact;

  function automatic [HORIZON_W-1:0] min_horizon(
      input [HORIZON_W-1:0] lhs,
      input [HORIZON_W-1:0] rhs);
    begin
      min_horizon = (lhs < rhs) ? lhs : rhs;
    end
  endfunction

  function automatic [HORIZON_W-1:0] max_horizon(
      input [HORIZON_W-1:0] lhs,
      input [HORIZON_W-1:0] rhs);
    begin
      max_horizon = (lhs > rhs) ? lhs : rhs;
    end
  endfunction

  // Source-grounded FIFO certificate:
  // floor((position - 1) / 16).  The right shift is exact for the measured
  // target configuration: 16 injection ports and equal 1-GHz core/ICNT
  // clocks.  A different configuration requires a separately synthesized
  // constant-ratio implementation and must conservatively disable this one.
  // It is emitted only for an invalid, allocated operand with pending memory
  // and an exact tail ticket still ahead of the dequeue frontier.
  function automatic [HORIZON_W-1:0] fifo_bound(
      input                         used,
      input                         allocated,
      input                         operand_valid,
      input                         pending_memory,
      input [TICKET_W-1:0]          tail_ticket);
    logic [TICKET_W-1:0] position;
    logic [TICKET_W-1:0] service_intervals;
    begin
      fifo_bound = {HORIZON_W{1'b0}};
      if (used && core_queue_tickets_exact && allocated &&
          !operand_valid && pending_memory &&
          tail_ticket != last_dequeued_ticket) begin
        position = tail_ticket - last_dequeued_ticket;
        service_intervals =
            (position - 1'b1) >> INJECTION_PORTS_LOG2;
        fifo_bound = service_intervals[HORIZON_W-1:0];
      end
    end
  endfunction

  // Source-grounded response-drain certificate sampled directly from the
  // SRAM entry's existing completion counter. With 16 response ports and
  // equal core/NoC clocks, at most 16 required responses can complete between
  // consecutive controller evaluations.  Therefore floor((N - 1) / 16) is a
  // safe lower bound on the remaining cycles before an invalid operand can
  // become valid.  Saturated counts only shorten this bound.
  function automatic [HORIZON_W-1:0] response_drain_bound(
      input                         used,
      input                         allocated,
      input                         operand_valid,
      input [HORIZON_W-1:0]         pending_count);
    begin
      response_drain_bound = {HORIZON_W{1'b0}};
      if (used && allocated && !operand_valid && pending_count != 0)
        response_drain_bound =
            (pending_count - ONE) >> INJECTION_PORTS_LOG2;
    end
  endfunction

  always_comb begin
    tile0_retirement_bound = ONE;
    tile1_retirement_bound = ONE;

    tile0_present_e = tile0_present_q;
    tile0_candidate_found_e = tile0_candidate_found_q;
    tile0_candidate_preload_e = tile0_candidate_preload_q;
    tile0_sources_supported_e = tile0_sources_supported_q;
    tile0_prefix_count_e = tile0_prefix_count_q;
    tile0_remaining_count_e = tile0_remaining_count_q;
    tile0_operand_used_e = tile0_operand_used_q;
    tile0_operand_allocated_e = tile0_operand_allocated_q;
    tile0_operand_valid_e = tile0_operand_valid_q;
    tile0_operand_pending_memory_e = tile0_operand_pending_memory_q;
    tile0_operand0_tail_ticket_e = tile0_operand0_tail_ticket_q;
    tile0_operand1_tail_ticket_e = tile0_operand1_tail_ticket_q;

    tile1_present_e = tile1_present_q;
    tile1_candidate_found_e = tile1_candidate_found_q;
    tile1_candidate_preload_e = tile1_candidate_preload_q;
    tile1_sources_supported_e = tile1_sources_supported_q;
    tile1_prefix_count_e = tile1_prefix_count_q;
    tile1_remaining_count_e = tile1_remaining_count_q;
    tile1_operand_used_e = tile1_operand_used_q;
    tile1_operand_allocated_e = tile1_operand_allocated_q;
    tile1_operand_valid_e = tile1_operand_valid_q;
    tile1_operand_pending_memory_e = tile1_operand_pending_memory_q;
    tile1_operand0_tail_ticket_e = tile1_operand0_tail_ticket_q;
    tile1_operand1_tail_ticket_e = tile1_operand1_tail_ticket_q;

    tile0_candidate_remaining = tile0_prefix_count_e;
    if (tile0_prefix_count_e == 0) begin
      tile0_candidate_remaining = max_horizon(
          tile0_operand0_fifo_stage1,
          tile0_operand1_fifo_stage1);
    end

    tile1_candidate_remaining = tile1_prefix_count_e;
    if (tile1_prefix_count_e == 0) begin
      tile1_candidate_remaining = max_horizon(
          tile1_operand0_fifo_stage1,
          tile1_operand1_fifo_stage1);
    end

    visible_valid = 1'b0;
    visible_remaining = {HORIZON_W{1'b0}};
    if (compute_front_future_valid) begin
      visible_valid = 1'b1;
      visible_remaining = compute_front_remaining;
    end
    if (ex_front_is_sa) begin
      visible_remaining = {HORIZON_W{1'b0}};
      visible_valid = 1'b1;
    end
    if (tile0_present_e && tile0_candidate_found_e) begin
      if (!visible_valid || tile0_candidate_remaining < visible_remaining)
        visible_remaining = tile0_candidate_remaining;
      visible_valid = 1'b1;
    end
    if (tile1_present_e && tile1_candidate_found_e) begin
      if (!visible_valid || tile1_candidate_remaining < visible_remaining)
        visible_remaining = tile1_candidate_remaining;
      visible_valid = 1'b1;
    end

    if (!(tile0_present_e && tile1_present_e)) begin
      future_tile_cap = ONE;
    end else begin
      tile0_retirement_bound = max_horizon(ONE,
                                           tile0_remaining_count_e);
      if (tile0_candidate_found_e)
        tile0_retirement_bound = max_horizon(
            tile0_retirement_bound, tile0_candidate_remaining);
      tile1_retirement_bound = max_horizon(ONE,
                                           tile1_remaining_count_e);
      if (tile1_candidate_found_e)
        tile1_retirement_bound = max_horizon(
            tile1_retirement_bound, tile1_candidate_remaining);
      future_tile_cap = min_horizon(tile0_retirement_bound,
                                    tile1_retirement_bound);
    end

    safe_remaining = visible_valid
                         ? min_horizon(visible_remaining, future_tile_cap)
                         : future_tile_cap;
    now1_cap_binding = (safe_remaining == ONE) &&
                       (future_tile_cap == ONE);

    fresh_work_drain_remaining = safe_remaining;
    if (now1_cap_binding && !scheduler_core_tile_waiting) begin
      fresh_work_drain_remaining = visible_valid
          ? min_horizon(visible_remaining, global_issue_drain_bound)
          : global_issue_drain_bound;
    end

    retention_safe_preload = compute_pipeline_empty &&
        ex_inst_queue_empty && tile0_present_e && tile1_present_e &&
        tile0_candidate_found_e && tile1_candidate_found_e &&
        tile0_candidate_preload_e && tile1_candidate_preload_e;

    // The registered stage-2 visible/future-cap frontier is reduced to the
    // work-drain frontier in stage 3.  The boolean is retained so entry can
    // reject a newly arrived scheduler tile when this extension was used.
    stage2_safe_remaining = certificate_stage2_visible_valid
        ? min_horizon(certificate_stage2_visible_remaining,
                      certificate_stage2_future_tile_cap)
        : certificate_stage2_future_tile_cap;
    stage2_now1_cap_binding = (stage2_safe_remaining == ONE) &&
                              (certificate_stage2_future_tile_cap == ONE);
    stage2_used_work_drain = stage2_now1_cap_binding &&
                             !scheduler_core_tile_waiting;
    stage2_fresh_work_drain_remaining = stage2_safe_remaining;
    if (stage2_used_work_drain) begin
      stage2_fresh_work_drain_remaining =
          certificate_stage2_visible_valid
              ? min_horizon(certificate_stage2_visible_remaining,
                            global_issue_drain_bound)
              : global_issue_drain_bound;
    end
  end

  // Descriptor metadata changes only on tile write events.  A separate clock
  // permits the physical wrapper to stop this large register bank during
  // per-cycle certificate evaluation without changing its update semantics.
  always_ff @(posedge descriptor_clk or negedge rst_n) begin
    if (!rst_n) begin
      tile0_present_q <= 1'b0;
      tile0_candidate_found_q <= 1'b0;
      tile0_candidate_preload_q <= 1'b0;
      tile0_sources_supported_q <= 1'b0;
      tile0_prefix_count_q <= '0;
      tile0_remaining_count_q <= '0;
      tile0_operand_used_q <= '0;
      tile0_operand_allocated_q <= '0;
      tile0_operand_valid_q <= '0;
      tile0_operand_pending_memory_q <= '0;
      tile0_operand0_tail_ticket_q <= '0;
      tile0_operand1_tail_ticket_q <= '0;

      tile1_present_q <= 1'b0;
      tile1_candidate_found_q <= 1'b0;
      tile1_candidate_preload_q <= 1'b0;
      tile1_sources_supported_q <= 1'b0;
      tile1_prefix_count_q <= '0;
      tile1_remaining_count_q <= '0;
      tile1_operand_used_q <= '0;
      tile1_operand_allocated_q <= '0;
      tile1_operand_valid_q <= '0;
      tile1_operand_pending_memory_q <= '0;
      tile1_operand0_tail_ticket_q <= '0;
      tile1_operand1_tail_ticket_q <= '0;
    end else begin
      if (tile0_we) begin
        tile0_present_q <= tile0_present;
        tile0_candidate_found_q <= tile0_candidate_found;
        tile0_candidate_preload_q <= tile0_candidate_preload;
        tile0_sources_supported_q <= tile0_sources_supported;
        tile0_prefix_count_q <= tile0_prefix_count;
        tile0_remaining_count_q <= tile0_remaining_count;
        tile0_operand_used_q <= tile0_operand_used;
        tile0_operand_allocated_q <= tile0_operand_allocated;
        tile0_operand_valid_q <= tile0_operand_valid;
        tile0_operand_pending_memory_q <= tile0_operand_pending_memory;
        tile0_operand0_tail_ticket_q <= tile0_operand0_tail_ticket;
        tile0_operand1_tail_ticket_q <= tile0_operand1_tail_ticket;
      end
      if (tile1_we) begin
        tile1_present_q <= tile1_present;
        tile1_candidate_found_q <= tile1_candidate_found;
        tile1_candidate_preload_q <= tile1_candidate_preload;
        tile1_sources_supported_q <= tile1_sources_supported;
        tile1_prefix_count_q <= tile1_prefix_count;
        tile1_remaining_count_q <= tile1_remaining_count;
        tile1_operand_used_q <= tile1_operand_used;
        tile1_operand_allocated_q <= tile1_operand_allocated;
        tile1_operand_valid_q <= tile1_operand_valid;
        tile1_operand_pending_memory_q <= tile1_operand_pending_memory;
        tile1_operand0_tail_ticket_q <= tile1_operand0_tail_ticket;
        tile1_operand1_tail_ticket_q <= tile1_operand1_tail_ticket;
      end
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      certificate_stage1_valid <= 1'b0;
      tile0_operand0_fifo_stage1 <= '0;
      tile0_operand1_fifo_stage1 <= '0;
      tile1_operand0_fifo_stage1 <= '0;
      tile1_operand1_fifo_stage1 <= '0;
      certificate_stage2_valid <= 1'b0;
      certificate_stage2_visible_valid <= 1'b0;
      certificate_stage2_visible_remaining <= '0;
      certificate_stage2_future_tile_cap <= '0;
      certificate_stage3_valid <= 1'b0;
      certificate_stage3_used_work_drain <= 1'b0;
      certificate_stage3_remaining <= '0;
    end else if (sleep_request || entry_commit) begin
      // The final enabled evaluation edge consumes the certificate.  No
      // pipeline register needs a clock while the separate control domain is
      // counting down the frozen deadline.
      certificate_stage1_valid <= 1'b0;
      certificate_stage2_valid <= 1'b0;
      certificate_stage3_valid <= 1'b0;
    end else begin
      // Stage 1 freezes one explicitly requested certificate sample.  The
      // always-clocked reference ties sample_enable high; the sparse wrapper
      // pulses it only on source-grounded opportunity events.
      certificate_stage1_valid <= sample_enable &&
                                  !tile0_we && !tile1_we &&
                                  !sa_active &&
                                  compute_pipeline_empty &&
                                  ex_inst_queue_empty &&
                                  core_queue_tickets_exact &&
                                  entry_descriptors_eligible;
        tile0_operand0_fifo_stage1 <= max_horizon(
            fifo_bound(tile0_operand_used_q[0],
                       tile0_operand_allocated_q[0],
                       tile0_operand_valid_q[0],
                       tile0_operand_pending_memory_q[0],
                       tile0_operand0_tail_ticket_q),
            response_drain_bound(tile0_operand_used_q[0],
                                 tile0_operand_allocated_q[0],
                                 tile0_operand_valid_q[0],
                                 response_count_sample_enable
                                     ? tile0_operand0_pending_count : '0));
        tile0_operand1_fifo_stage1 <= max_horizon(
            fifo_bound(tile0_operand_used_q[1],
                       tile0_operand_allocated_q[1],
                       tile0_operand_valid_q[1],
                       tile0_operand_pending_memory_q[1],
                       tile0_operand1_tail_ticket_q),
            response_drain_bound(tile0_operand_used_q[1],
                                 tile0_operand_allocated_q[1],
                                 tile0_operand_valid_q[1],
                                 response_count_sample_enable
                                     ? tile0_operand1_pending_count : '0));
        tile1_operand0_fifo_stage1 <= max_horizon(
            fifo_bound(tile1_operand_used_q[0],
                       tile1_operand_allocated_q[0],
                       tile1_operand_valid_q[0],
                       tile1_operand_pending_memory_q[0],
                       tile1_operand0_tail_ticket_q),
            response_drain_bound(tile1_operand_used_q[0],
                                 tile1_operand_allocated_q[0],
                                 tile1_operand_valid_q[0],
                                 response_count_sample_enable
                                     ? tile1_operand0_pending_count : '0));
        tile1_operand1_fifo_stage1 <= max_horizon(
            fifo_bound(tile1_operand_used_q[1],
                       tile1_operand_allocated_q[1],
                       tile1_operand_valid_q[1],
                       tile1_operand_pending_memory_q[1],
                       tile1_operand1_tail_ticket_q),
            response_drain_bound(tile1_operand_used_q[1],
                                 tile1_operand_allocated_q[1],
                                 tile1_operand_valid_q[1],
                                 response_count_sample_enable
                                     ? tile1_operand1_pending_count : '0));

      // Stage 2 registers the natural boundary between per-resident
      // candidate/FIFO calculation and global minimum/future-tile capping.
      if (certificate_stage1_valid && !tile0_we && !tile1_we &&
          !sa_active && retention_safe_preload &&
          core_queue_tickets_exact) begin
        certificate_stage2_valid <= 1'b1;
        certificate_stage2_visible_valid <= visible_valid;
        certificate_stage2_visible_remaining <= visible_remaining;
        certificate_stage2_future_tile_cap <= future_tile_cap;
      end else begin
        certificate_stage2_valid <= 1'b0;
      end

      // Stage 3 applies the V5 work-drain rule and deducts all three elapsed
      // entry-pipeline cycles.  Invalidating events are checked at every
      // boundary; no stale stage may extend the saved deadline.
      if (certificate_stage2_valid && !tile0_we && !tile1_we &&
          !sa_active && retention_safe_preload &&
          core_queue_tickets_exact &&
          stage2_fresh_work_drain_remaining >=
              PIPELINED_ENTRY_REQUIRED) begin
        certificate_stage3_valid <= 1'b1;
        certificate_stage3_used_work_drain <= stage2_used_work_drain;
        certificate_stage3_remaining <=
            stage2_fresh_work_drain_remaining - ENTRY_PIPELINE_CYCLES;
      end else begin
        certificate_stage3_valid <= 1'b0;
        certificate_stage3_used_work_drain <= 1'b0;
        certificate_stage3_remaining <= '0;
      end
    end
  end

  // Small always-available control domain.  It never recomputes or extends a
  // certificate: it only commits a complete entry, decrements its frozen
  // deadline, and applies the original immediate-wake guards.
  always_ff @(posedge control_clk or negedge rst_n) begin
    if (!rst_n) begin
      sleep_request <= 1'b0;
      sleep_countdown <= '0;
    end else if (sleep_request) begin
      if (sa_active || tile0_we || tile1_we ||
          !compute_pipeline_empty || !ex_inst_queue_empty ||
          !core_queue_tickets_exact || sleep_countdown <= WAKE_GUARD) begin
        sleep_request <= 1'b0;
        sleep_countdown <= '0;
      end else begin
        sleep_countdown <= sleep_countdown - ONE;
      end
    end else begin
      sleep_countdown <= '0;
      if (entry_commit) begin
        sleep_request <= 1'b1;
        sleep_countdown <= certificate_stage3_remaining;
      end
    end
  end

endmodule
