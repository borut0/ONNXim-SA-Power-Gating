`timescale 1ps/1ps

module tb_sa_idle_predictor_controller;
  logic clk = 1'b0;
  wire descriptor_clk = clk;
  wire control_clk = clk;
  wire sample_enable = 1'b1;
  logic rst_n = 1'b0;
  logic sa_active = 1'b0;
  logic compute_pipeline_empty = 1'b1;
  logic compute_front_future_valid = 1'b0;
  logic [19:0] compute_front_remaining = 0;
  logic ex_inst_queue_empty = 1'b1;
  logic ex_front_is_sa = 1'b0;
  logic scheduler_core_tile_waiting = 1'b1;
  logic [19:0] global_issue_drain_bound = 1;
  logic core_queue_tickets_exact = 1'b1;
  logic [19:0] last_dequeued_ticket = 0;

  logic tile0_we = 1'b0;
  logic tile0_count_we = 1'b0;
  logic tile0_present = 1'b0;
  logic tile0_candidate_found = 1'b0;
  logic tile0_candidate_preload = 1'b0;
  logic tile0_sources_supported = 1'b0;
  logic [19:0] tile0_prefix_count = 0;
  logic [19:0] tile0_remaining_count = 0;
  logic [1:0] tile0_operand_used = 0;
  logic [1:0] tile0_operand_allocated = 0;
  logic [1:0] tile0_operand_valid = 0;
  logic [1:0] tile0_operand_pending_memory = 0;
  logic [19:0] tile0_operand0_tail_ticket = 0;
  logic [19:0] tile0_operand1_tail_ticket = 0;
  logic [19:0] tile0_operand0_pending_count = 0;
  logic [19:0] tile0_operand1_pending_count = 0;

  logic tile1_we = 1'b0;
  logic tile1_count_we = 1'b0;
  logic tile1_present = 1'b0;
  logic tile1_candidate_found = 1'b0;
  logic tile1_candidate_preload = 1'b0;
  logic tile1_sources_supported = 1'b0;
  logic [19:0] tile1_prefix_count = 0;
  logic [19:0] tile1_remaining_count = 0;
  logic [1:0] tile1_operand_used = 0;
  logic [1:0] tile1_operand_allocated = 0;
  logic [1:0] tile1_operand_valid = 0;
  logic [1:0] tile1_operand_pending_memory = 0;
  logic [19:0] tile1_operand0_tail_ticket = 0;
  logic [19:0] tile1_operand1_tail_ticket = 0;
  logic [19:0] tile1_operand0_pending_count = 0;
  logic [19:0] tile1_operand1_pending_count = 0;

  logic retention_safe_preload;
  logic [19:0] fresh_work_drain_remaining;
  logic sleep_request;
  logic entry_descriptors_eligible;
  logic certificate_pipeline_active;

  sa_idle_predictor_controller dut (.*);
  assign retention_safe_preload = dut.retention_safe_preload;
  assign fresh_work_drain_remaining = dut.fresh_work_drain_remaining;

  always #500 clk = ~clk;

  task automatic load_two_preload_tiles(input logic [19:0] tail_ticket);
    begin
      tile0_we = 1'b1;
      tile1_we = 1'b1;
      tile0_present = 1'b1;
      tile1_present = 1'b1;
      tile0_candidate_found = 1'b1;
      tile1_candidate_found = 1'b1;
      tile0_candidate_preload = 1'b1;
      tile1_candidate_preload = 1'b1;
      tile0_sources_supported = 1'b1;
      tile1_sources_supported = 1'b1;
      tile0_remaining_count = 1;
      tile1_remaining_count = 1;
      tile0_operand_used = 2'b11;
      tile1_operand_used = 2'b11;
      tile0_operand_allocated = 2'b11;
      tile1_operand_allocated = 2'b11;
      tile0_operand_pending_memory = 2'b11;
      tile1_operand_pending_memory = 2'b11;
      tile0_operand0_tail_ticket = tail_ticket;
      tile0_operand1_tail_ticket = tail_ticket;
      tile1_operand0_tail_ticket = tail_ticket;
      tile1_operand1_tail_ticket = tail_ticket;
      @(posedge clk);
      #1;
      tile0_we = 1'b0;
      tile1_we = 1'b0;
      repeat (4) @(posedge clk);
      #1;
    end
  endtask

  initial begin
    repeat (2) @(posedge clk);
    rst_n = 1'b1;
    load_two_preload_tiles(2721);

    if (!retention_safe_preload || fresh_work_drain_remaining != 170)
      $fatal(1, "FIFO certificate/retention mismatch");
    if (!sleep_request)
      $fatal(1, "controller did not enter after pipelined certificate");

    repeat (167) @(posedge clk);
    #1;
    if (sleep_request)
      $fatal(1, "saved deadline did not wake at one-cycle guard");

    // A descriptor change while asleep must wake on the same edge.
    load_two_preload_tiles(2721);
    tile0_we = 1'b1;
    tile0_candidate_preload = 1'b0;
    @(posedge clk);
    #1;
    tile0_we = 1'b0;
    if (retention_safe_preload || sleep_request)
      $fatal(1, "descriptor change did not force immediate wake");

    // A descriptor with neither a FIFO nor response-drain certificate fails
    // conservatively even when the resident GEMM itself is still visible.
    tile0_we = 1'b1;
    tile0_candidate_preload = 1'b1;
    tile0_sources_supported = 1'b0;
    tile0_operand_pending_memory = 2'b00;
    tile0_operand0_pending_count = 0;
    tile0_operand1_pending_count = 0;
    @(posedge clk);
    #1;
    tile0_we = 1'b0;
    if (sleep_request)
      $fatal(1, "uncertified source set did not disable gating");

    // A dequeued/unsupported FIFO tail remains certifiable from the exact
    // pending-response count: floor((2721 - 1) / 16) = 170 cycles.
    tile0_we = 1'b1;
    tile1_we = 1'b1;
    tile0_operand0_pending_count = 2721;
    tile0_operand1_pending_count = 2721;
    tile1_operand0_pending_count = 2721;
    tile1_operand1_pending_count = 2721;
    tile1_operand_pending_memory = 2'b00;
    tile1_sources_supported = 1'b0;
    @(posedge clk);
    #1;
    tile0_we = 1'b0;
    tile1_we = 1'b0;
    repeat (4) @(posedge clk);
    #1;
    if (!retention_safe_preload || fresh_work_drain_remaining != 170)
      $fatal(1, "response-drain certificate mismatch");
    if (!sleep_request)
      $fatal(1, "response-drain certificate did not permit entry");

    // Monotonic response progress on the existing SRAM counters does not
    // invalidate the already-frozen worst-case-drain deadline.
    tile0_count_we = 1'b1;
    tile1_count_we = 1'b1;
    tile0_operand0_pending_count = 2705;
    tile0_operand1_pending_count = 2705;
    tile1_operand0_pending_count = 2705;
    tile1_operand1_pending_count = 2705;
    @(posedge clk);
    #1;
    tile0_count_we = 1'b0;
    tile1_count_we = 1'b0;
    if (!sleep_request)
      $fatal(1, "count-only response progress incorrectly forced wake");

    $display("SA_PG_CONTROLLER_TB PASS");
    $finish;
  end
endmodule
