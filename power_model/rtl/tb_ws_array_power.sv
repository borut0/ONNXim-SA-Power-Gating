`timescale 1ps/1ps

module tb_ws_array_power #(
  parameter integer ROWS = 16,
  parameter integer COLS = 16,
  parameter integer DATA_W = 16,
  parameter integer ACC_W = 32,
  parameter integer MEASURE_CYCLES = 256
);
  logic clk = 1'b0;
  logic clock_run = 1'b1;
  logic rst_n = 1'b0;
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

`ifdef POWER_GATE_16
  ws_array_16x16 dut (.*);
`elsif POWER_GATE_32
  ws_array_32x32 dut (.*);
`else
  ws_array_power_dut #(
    .ROWS(ROWS), .COLS(COLS), .DATA_W(DATA_W), .ACC_W(ACC_W)
  ) dut (.*);
`endif

  always #500 begin
    if (clock_run) clk = ~clk;
    else clk = 1'b0;
  end

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

  initial begin
    if (!$value$plusargs("MODE=%s", mode)) mode = "ACTIVE";
    if (!$value$plusargs("VCD=%s", vcd_path)) vcd_path = "sa_power.vcd";

    $dumpfile(vcd_path);
    $dumpvars(0, dut);
    $dumpoff;

    repeat (4) @(posedge clk);
    @(negedge clk);
    rst_n = 1'b1;

    if (mode == "PRELOAD") begin
      weight_shift_en = 1'b1;
      for (cycle = 0; cycle < ROWS; cycle = cycle + 1) preload_cycle();
      $dumpon;
      for (cycle = 0; cycle < MEASURE_CYCLES; cycle = cycle + 1)
        preload_cycle();
    end else begin
      weight_shift_en = 1'b1;
      for (cycle = 0; cycle < ROWS; cycle = cycle + 1) preload_cycle();
      @(negedge clk);
      weight_shift_en = 1'b0;

      if (mode == "ACTIVE") begin
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
      end else if (mode == "IDLE") begin
        compute_en = 1'b0;
        repeat (4) @(posedge clk);
        $dumpon;
        repeat (MEASURE_CYCLES) @(posedge clk);
      end else if (mode == "CLOCK_GATED_IDLE") begin
        compute_en = 1'b0;
        repeat (4) @(posedge clk);
        @(negedge clk);
        clock_run = 1'b0;
        $dumpon;
        #(MEASURE_CYCLES * 1000);
        $dumpoff;
        clock_run = 1'b1;
      end else begin
        $fatal(1, "MODE must be ACTIVE, PRELOAD, IDLE, or CLOCK_GATED_IDLE");
      end
    end

    $dumpoff;
    @(posedge clk);
    $finish;
  end
endmodule

// Preserve the same `array` hierarchy used by the synthesized size-specific
// wrappers so `read_vcd -scope tb_ws_array_power.dut` can map RTL activity.
module ws_array_power_dut #(
  parameter integer ROWS = 16,
  parameter integer COLS = 16,
  parameter integer DATA_W = 16,
  parameter integer ACC_W = 32
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         weight_shift_en,
  input  logic                         compute_en,
  input  logic        [(ROWS*DATA_W)-1:0] activation_in,
  input  logic        [(COLS*DATA_W)-1:0] weight_in,
  input  logic        [(COLS*ACC_W)-1:0]  psum_in,
  output logic        [(ROWS*DATA_W)-1:0] activation_out,
  output logic        [(COLS*ACC_W)-1:0]  psum_out
);
  ws_array #(
    .ROWS(ROWS), .COLS(COLS), .DATA_W(DATA_W), .ACC_W(ACC_W)
  ) array (.*);
endmodule
