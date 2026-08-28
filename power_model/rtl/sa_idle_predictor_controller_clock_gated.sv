// Sparse-clock physical wrapper around the frozen certificate equations.
// The entry threshold and wake guards remain in sa_idle_predictor_controller;
// this wrapper suppresses certificate arithmetic between source-grounded
// opportunity events while leaving the small countdown domain available.
module sa_idle_predictor_controller_clock_gated #(
  parameter integer HORIZON_W = 20,
  parameter integer TICKET_W = 20,
  parameter integer INJECTION_PORTS_LOG2 = 4,
  parameter integer BREAK_EVEN_CYCLES = 166,
  parameter integer WAKE_GUARD_CYCLES = 1
) (
  input  logic                    clk,
  input  logic                    rst_n,
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

  output logic                    sleep_request
);

  logic entry_descriptors_eligible;
  logic certificate_pipeline_active;
  logic evaluation_eligible;
  logic controller_clock_enable;
  logic descriptor_clock_enable;
  logic evaluation_sample_enable;
  logic eligibility_was_true;
  logic descriptor_write_pending;
  logic sleep_was_requested;
  logic gated_clock_unbuffered;
  logic descriptor_clock_unbuffered;
  logic controller_clk;
  logic descriptor_clk;
  logic controller_tickets_exact;

  // This is exactly the frozen controller's retention-safe prerequisite.
  // The descriptor summary is derived from its existing registered metadata,
  // eliminating duplicate always-clocked wrapper state.
  assign evaluation_eligible = !sa_active && compute_pipeline_empty &&
      ex_inst_queue_empty && core_queue_tickets_exact &&
      entry_descriptors_eligible;

  // Count-only SRAM response updates and dequeue progress can only reduce the
  // existing safe lower bound.  They do not wake this heavy pipeline.  A new
  // sample starts after eligibility rises, after a semantic descriptor write
  // is released, or after the small countdown domain wakes at its deadline.
  assign evaluation_sample_enable = !sleep_request &&
      !tile0_we && !tile1_we && evaluation_eligible &&
      ((!eligibility_was_true) || descriptor_write_pending ||
       (sleep_was_requested && !sleep_request));

  // Four edges are sufficient for stage1, stage2, stage3, and commit.  The
  // registered valid bits keep the ICG open only until that sample drains.
  assign controller_clock_enable = !rst_n || evaluation_sample_enable ||
      certificate_pipeline_active;
  assign descriptor_clock_enable = !rst_n || tile0_we || tile1_we;
  assign controller_tickets_exact = core_queue_tickets_exact &&
      (sleep_request || evaluation_eligible);

  // This small event detector stays on the root clock.  It contains no
  // certificate arithmetic and does not observe monotonic response/dequeue
  // progress.  Holding descriptor_write_pending through a write guarantees
  // the first sample sees the newly registered descriptor on a clean edge.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      eligibility_was_true <= 1'b0;
      descriptor_write_pending <= 1'b0;
      sleep_was_requested <= 1'b0;
    end else begin
      eligibility_was_true <= evaluation_eligible;
      sleep_was_requested <= sleep_request;
      if (tile0_we || tile1_we)
        descriptor_write_pending <= 1'b1;
      else if (evaluation_sample_enable)
        descriptor_write_pending <= 1'b0;
    end
  end

  // Actual ASAP7 integrated clock gate and a characterized high-drive local
  // buffer.  The ICG sees only the buffer input; BUFx24's 1474.56-fF Liberty
  // max-capacitance limit is well above the frozen controller register load.
  // Both cells remain explicit so their area and workload power are counted.
  ICGx2_ASAP7_75t_R controller_evaluation_icg (
    .GCLK(gated_clock_unbuffered),
    .ENA(controller_clock_enable),
    .SE(1'b0),
    .CLK(clk)
  );

  BUFx24_ASAP7_75t_R controller_evaluation_clock_buffer (
    .Y(controller_clk),
    .A(gated_clock_unbuffered)
  );

  // The resident-tile descriptor bank changes only on its explicit write
  // pulses.  Its independent gated clock prevents the metadata flops from
  // receiving evaluation clocks while preserving the same write edges.
  ICGx2_ASAP7_75t_R controller_descriptor_icg (
    .GCLK(descriptor_clock_unbuffered),
    .ENA(descriptor_clock_enable),
    .SE(1'b0),
    .CLK(clk)
  );

  BUFx24_ASAP7_75t_R controller_descriptor_clock_buffer (
    .Y(descriptor_clk),
    .A(descriptor_clock_unbuffered)
  );

  sa_idle_predictor_controller #(
    .HORIZON_W(HORIZON_W),
    .TICKET_W(TICKET_W),
    .INJECTION_PORTS_LOG2(INJECTION_PORTS_LOG2),
    .BREAK_EVEN_CYCLES(BREAK_EVEN_CYCLES),
    .WAKE_GUARD_CYCLES(WAKE_GUARD_CYCLES)
  ) controller (
    .clk(controller_clk),
    .control_clk(clk),
    .sample_enable(evaluation_sample_enable),
    .core_queue_tickets_exact(controller_tickets_exact),
    .*
  );

endmodule
