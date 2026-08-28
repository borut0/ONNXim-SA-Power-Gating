`timescale 1ns/1ps

module tb_ws_array_16x16;
  localparam int ROWS   = 16;
  localparam int COLS   = 16;
  localparam int DATA_W = 16;
  localparam int ACC_W  = 32;

  logic clk = 1'b0;
  logic rst_n = 1'b0;
  logic weight_shift_en = 1'b0;
  logic compute_en = 1'b0;
  logic        [(ROWS*DATA_W)-1:0] activation_in = '0;
  logic        [(COLS*DATA_W)-1:0] weight_in = '0;
  logic        [(COLS*ACC_W)-1:0] psum_in = '0;
  logic        [(ROWS*DATA_W)-1:0] activation_out;
  logic        [(COLS*ACC_W)-1:0] psum_out;
  logic [31:0] lfsr = 32'h1ace_beef;

  ws_array_16x16 dut (.*);

  always #0.5 clk = ~clk;

  task automatic advance_lfsr;
    begin
      lfsr = {lfsr[30:0],
              lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
    end
  endtask

  task automatic drive_inputs;
    integer lane;
    begin
      for (lane = 0; lane < ROWS; lane++) begin
        advance_lfsr();
        activation_in[(lane*DATA_W) +: DATA_W] = lfsr[DATA_W-1:0];
      end
      for (lane = 0; lane < COLS; lane++) begin
        advance_lfsr();
        weight_in[(lane*DATA_W) +: DATA_W] = lfsr[DATA_W-1:0];
        psum_in[(lane*ACC_W) +: ACC_W] = '0;
      end
    end
  endtask

  integer cycle;
  initial begin
    $dumpfile("ws_array_16x16.vcd");
    $dumpvars(0, tb_ws_array_16x16);

    repeat (4) @(posedge clk);
    rst_n = 1'b1;

    weight_shift_en = 1'b1;
    for (cycle = 0; cycle < ROWS; cycle++) begin
      drive_inputs();
      @(posedge clk);
    end
    weight_shift_en = 1'b0;

    compute_en = 1'b1;
    for (cycle = 0; cycle < 256; cycle++) begin
      drive_inputs();
      @(posedge clk);
    end
    compute_en = 1'b0;

    repeat (256) @(posedge clk);
    $finish;
  end

endmodule
