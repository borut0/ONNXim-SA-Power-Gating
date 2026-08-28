`timescale 1ps/1ps

module tb_sa_idle_predictor_controller_replay;
  logic clk = 1'b0;
`ifndef CLOCK_GATED_CONTROLLER
  wire descriptor_clk = clk;
  wire control_clk = clk;
  wire sample_enable = 1'b1;
`endif
  logic rst_n = 1'b0;
  logic sa_active;
  logic compute_pipeline_empty;
  logic compute_front_future_valid;
  logic [19:0] compute_front_remaining;
  logic ex_inst_queue_empty;
  logic ex_front_is_sa;
  logic scheduler_core_tile_waiting;
  logic [19:0] global_issue_drain_bound;
  logic core_queue_tickets_exact;
  logic [19:0] last_dequeued_ticket;

  logic tile0_we;
  logic tile0_count_we;
  logic tile0_present;
  logic tile0_candidate_found;
  logic tile0_candidate_preload;
  logic tile0_sources_supported;
  logic [19:0] tile0_prefix_count;
  logic [19:0] tile0_remaining_count;
  logic [1:0] tile0_operand_used;
  logic [1:0] tile0_operand_allocated;
  logic [1:0] tile0_operand_valid;
  logic [1:0] tile0_operand_pending_memory;
  logic [19:0] tile0_operand0_tail_ticket;
  logic [19:0] tile0_operand1_tail_ticket;
  logic [19:0] tile0_operand0_pending_count;
  logic [19:0] tile0_operand1_pending_count;

  logic tile1_we;
  logic tile1_count_we;
  logic tile1_present;
  logic tile1_candidate_found;
  logic tile1_candidate_preload;
  logic tile1_sources_supported;
  logic [19:0] tile1_prefix_count;
  logic [19:0] tile1_remaining_count;
  logic [1:0] tile1_operand_used;
  logic [1:0] tile1_operand_allocated;
  logic [1:0] tile1_operand_valid;
  logic [1:0] tile1_operand_pending_memory;
  logic [19:0] tile1_operand0_tail_ticket;
  logic [19:0] tile1_operand1_tail_ticket;
  logic [19:0] tile1_operand0_pending_count;
  logic [19:0] tile1_operand1_pending_count;
  logic sleep_request;
  logic entry_descriptors_eligible;
  logic certificate_pipeline_active;

`ifdef CLOCK_GATED_CONTROLLER
  sa_idle_predictor_controller_clock_gated dut (.*);
`else
  sa_idle_predictor_controller dut (.*);
`endif

  integer trace_fd;
  integer scan_count;
  integer header_count;
  reg [8*4096-1:0] header_line;
  string trace_path;
  string vcd_path;
  logic dump_vcd;
  longint unsigned record_cycle;
  integer record_final;
  longint unsigned record_values [0:41];
  longint unsigned current_cycle;
  longint unsigned replay_cycles = 0;
  longint unsigned sleep_cycles = 0;
  longint unsigned sleep_transitions = 0;
  longint unsigned descriptor_write_cycles = 0;
  longint unsigned count_write_cycles = 0;
  longint unsigned safety_violations = 0;
  longint unsigned current_sleep_episode_cycles = 0;
  longint unsigned current_sleep_episode_start_cycle = 0;
  longint unsigned completed_sleep_episodes = 0;
  logic previous_sleep = 1'b0;

  always #500 clk = ~clk;

  task automatic read_record;
    begin
      scan_count = $fscanf(trace_fd,
          "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
          record_cycle, record_final,
          record_values[0], record_values[1], record_values[2],
          record_values[3], record_values[4], record_values[5],
          record_values[6], record_values[7], record_values[8],
          record_values[9], record_values[10], record_values[11],
          record_values[12], record_values[13], record_values[14],
          record_values[15], record_values[16], record_values[17],
          record_values[18], record_values[19], record_values[20],
          record_values[21], record_values[22], record_values[23],
          record_values[24], record_values[25], record_values[26],
          record_values[27], record_values[28], record_values[29],
          record_values[30], record_values[31], record_values[32],
          record_values[33], record_values[34], record_values[35],
          record_values[36], record_values[37], record_values[38],
          record_values[39], record_values[40], record_values[41]);
    end
  endtask

  task automatic apply_record;
    begin
      sa_active = record_values[0];
      compute_pipeline_empty = record_values[1];
      compute_front_future_valid = record_values[2];
      compute_front_remaining = record_values[3];
      ex_inst_queue_empty = record_values[4];
      ex_front_is_sa = record_values[5];
      scheduler_core_tile_waiting = record_values[6];
      global_issue_drain_bound = record_values[7];
      core_queue_tickets_exact = record_values[8];
      last_dequeued_ticket = record_values[9];

      tile0_we = record_values[10];
      tile0_count_we = record_values[11];
      tile0_present = record_values[12];
      tile0_candidate_found = record_values[13];
      tile0_candidate_preload = record_values[14];
      tile0_sources_supported = record_values[15];
      tile0_prefix_count = record_values[16];
      tile0_remaining_count = record_values[17];
      tile0_operand_used = record_values[18];
      tile0_operand_allocated = record_values[19];
      tile0_operand_valid = record_values[20];
      tile0_operand_pending_memory = record_values[21];
      tile0_operand0_tail_ticket = record_values[22];
      tile0_operand1_tail_ticket = record_values[23];
      tile0_operand0_pending_count = record_values[24];
      tile0_operand1_pending_count = record_values[25];

      tile1_we = record_values[26];
      tile1_count_we = record_values[27];
      tile1_present = record_values[28];
      tile1_candidate_found = record_values[29];
      tile1_candidate_preload = record_values[30];
      tile1_sources_supported = record_values[31];
      tile1_prefix_count = record_values[32];
      tile1_remaining_count = record_values[33];
      tile1_operand_used = record_values[34];
      tile1_operand_allocated = record_values[35];
      tile1_operand_valid = record_values[36];
      tile1_operand_pending_memory = record_values[37];
      tile1_operand0_tail_ticket = record_values[38];
      tile1_operand1_tail_ticket = record_values[39];
      tile1_operand0_pending_count = record_values[40];
      tile1_operand1_pending_count = record_values[41];
    end
  endtask

  initial begin
    if (!$value$plusargs("TRACE=%s", trace_path))
      $fatal(1, "missing +TRACE=<controller CSV>");
    dump_vcd = $value$plusargs("VCD=%s", vcd_path);
    trace_fd = $fopen(trace_path, "r");
    if (trace_fd == 0)
      $fatal(1, "cannot open controller trace %s", trace_path);
    header_count = $fgets(header_line, trace_fd);
    if (header_count == 0)
      $fatal(1, "empty controller trace");

    repeat (2) @(posedge clk);
    #1;
    rst_n = 1'b1;
    read_record();
    if (scan_count != 44 || record_final != 0 || record_cycle != 0)
      $fatal(1, "controller trace must begin with a 44-field cycle-0 record");
    apply_record();

    if (dump_vcd) begin
      $dumpfile(vcd_path);
`ifdef CONTROLLER_SHALLOW_GATE_VCD
      // Record DUT signals and immediate mapped-cell pins, while excluding
      // deeper simulation-only state inside the standard-cell models.
      $dumpvars(2, dut);
`else
      $dumpvars(0, dut);
`endif
    end
    current_cycle = 0;

    forever begin
      read_record();
      if (scan_count != 44)
        $fatal(1, "malformed controller trace after cycle %0d",
               current_cycle);
      if (record_cycle < current_cycle)
        $fatal(1, "non-monotonic controller trace cycle");

      while (current_cycle < record_cycle) begin
        @(posedge clk);
        #1;
        replay_cycles++;
        if (sleep_request) begin
          if (!previous_sleep)
            current_sleep_episode_start_cycle = current_cycle;
          sleep_cycles++;
          current_sleep_episode_cycles++;
        end
        if (previous_sleep && !sleep_request) begin
          $display("CONTROLLER_SLEEP_EPISODE {\"start_cycle\":%0d,\"end_cycle\":%0d,\"duration_cycles\":%0d}",
                   current_sleep_episode_start_cycle, current_cycle,
                   current_sleep_episode_cycles);
          completed_sleep_episodes++;
          current_sleep_episode_cycles = 0;
        end
        if (sleep_request != previous_sleep) sleep_transitions++;
        if (sleep_request && sa_active) safety_violations++;
        if (tile0_we || tile1_we) descriptor_write_cycles++;
        if (tile0_count_we || tile1_count_we) count_write_cycles++;
        previous_sleep = sleep_request;
        current_cycle++;
      end

      if (record_final != 0) begin
        break;
      end
      apply_record();
    end

    if (dump_vcd) $dumpoff;
    $fclose(trace_fd);
    if (safety_violations != 0)
      $fatal(1, "controller replay safety violations=%0d",
             safety_violations);
    $display("CONTROLLER_REPLAY_SUMMARY {\"cycles\":%0d,\"sleep_cycles\":%0d,\"sleep_transitions\":%0d,\"completed_sleep_episodes\":%0d,\"open_sleep_episode_cycles\":%0d,\"descriptor_write_cycles\":%0d,\"count_write_cycles\":%0d,\"safety_violations\":%0d}",
             replay_cycles, sleep_cycles, sleep_transitions,
             completed_sleep_episodes, current_sleep_episode_cycles,
             descriptor_write_cycles, count_write_cycles,
             safety_violations);
    $finish;
  end
endmodule
