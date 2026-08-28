`timescale 1ps/1ps

module tb_ws_array_clock_gate_power #(
  parameter integer DATA_W = 16,
  parameter integer ACC_W = 32,
  parameter integer MEASURE_CYCLES = 256,
  parameter integer TRANSITION_SIDE_CYCLES = 8,
  parameter integer CLOCK_PERIOD_PS = 1000
);
  localparam integer ROWS = 16;
  localparam integer COLS = 16;

  logic clk = 1'b0;
  logic rst_n = 1'b0;
  logic sa_idle = 1'b0;
  logic gate_request = 1'b0;
  logic weight_shift_en = 1'b0;
  logic compute_en = 1'b0;
  logic [(ROWS*DATA_W)-1:0] activation_in = '0;
  logic [(COLS*DATA_W)-1:0] weight_in = '0;
  logic [(COLS*ACC_W)-1:0] psum_in = '0;
  logic [(ROWS*DATA_W)-1:0] activation_out;
  logic [(COLS*ACC_W)-1:0] psum_out;
  logic [31:0] lfsr = 32'h1ace_beef;

  string mode;
  string vcd_path;
  integer cycle;

  ws_array_clock_gated_16x16 dut (.*);

  always #(CLOCK_PERIOD_PS/2) clk = ~clk;

  task automatic advance_lfsr;
    begin
      lfsr = {lfsr[30:0],
              lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
    end
  endtask

  task automatic drive_weights;
    integer lane;
    begin
      for (lane = 0; lane < COLS; lane = lane + 1) begin
        advance_lfsr();
        weight_in[(lane*DATA_W) +: DATA_W] = lfsr[DATA_W-1:0];
      end
    end
  endtask

  task automatic drive_active_inputs;
    integer lane;
    begin
      for (lane = 0; lane < ROWS; lane = lane + 1) begin
        advance_lfsr();
        activation_in[(lane*DATA_W) +: DATA_W] = lfsr[DATA_W-1:0];
      end
      for (lane = 0; lane < COLS; lane = lane + 1) begin
        advance_lfsr();
        psum_in[(lane*ACC_W) +: ACC_W] = lfsr;
      end
    end
  endtask

  task automatic preload_cycle;
    begin
      drive_weights();
      @(posedge clk);
    end
  endtask

  task automatic prepare_idle_on;
    begin
      weight_shift_en = 1'b1;
      for (cycle = 0; cycle < ROWS; cycle = cycle + 1)
        preload_cycle();
      @(negedge clk);
      weight_shift_en = 1'b0;
      compute_en = 1'b0;
      sa_idle = 1'b1;
      gate_request = 1'b0;
      repeat (4) @(posedge clk);
    end
  endtask

  initial begin
    if (!$value$plusargs("MODE=%s", mode)) mode = "ACTIVE";
    if (!$value$plusargs("VCD=%s", vcd_path))
      vcd_path = "sa_clock_gate_power.vcd";

    $dumpfile(vcd_path);
    $dumpvars(0, dut);
    $dumpoff;

    repeat (4) @(posedge clk);
    @(negedge clk);
    rst_n = 1'b1;

    if (mode == "ACTIVE") begin
      weight_shift_en = 1'b1;
      for (cycle = 0; cycle < ROWS; cycle = cycle + 1)
        preload_cycle();
      @(negedge clk);
      weight_shift_en = 1'b0;
      compute_en = 1'b1;
      for (cycle = 0; cycle < (2*ROWS + COLS); cycle = cycle + 1) begin
        drive_active_inputs();
        @(posedge clk);
      end
      $dumpon;
      for (cycle = 0; cycle < MEASURE_CYCLES; cycle = cycle + 1) begin
        drive_active_inputs();
        @(posedge clk);
      end
    end else if (mode == "IDLE_ON") begin
      prepare_idle_on();
      @(negedge clk);
      $dumpon;
      #(MEASURE_CYCLES * CLOCK_PERIOD_PS);
    end else if (mode == "CLOCK_GATED_IDLE") begin
      prepare_idle_on();
      @(negedge clk);
      gate_request = 1'b1;
      repeat (2) @(posedge clk);
      @(negedge clk);
      $dumpon;
      #(MEASURE_CYCLES * CLOCK_PERIOD_PS);
    end else if (mode == "ENTRY") begin
      prepare_idle_on();
      @(negedge clk);
      $dumpon;
      #(TRANSITION_SIDE_CYCLES * CLOCK_PERIOD_PS);
      gate_request = 1'b1;
      #(TRANSITION_SIDE_CYCLES * CLOCK_PERIOD_PS);
    end else if (mode == "WAKE") begin
      prepare_idle_on();
      @(negedge clk);
      gate_request = 1'b1;
      repeat (2) @(posedge clk);
      @(negedge clk);
      $dumpon;
      #(TRANSITION_SIDE_CYCLES * CLOCK_PERIOD_PS);
      gate_request = 1'b0;
      #(TRANSITION_SIDE_CYCLES * CLOCK_PERIOD_PS);
    end else begin
      $fatal(1, "unsupported MODE %s", mode);
    end

    $dumpoff;
    repeat (2) @(posedge clk);
    $finish;
  end
endmodule
