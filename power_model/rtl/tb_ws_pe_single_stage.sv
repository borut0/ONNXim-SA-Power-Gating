`timescale 1ns/1ps

module tb_ws_pe_single_stage;
  logic clk = 1'b0;
  logic rst_n = 1'b0;
  logic weight_shift_en = 1'b0;
  logic compute_en = 1'b0;
  logic [15:0] activation_in = '0;
  logic [15:0] weight_in = '0;
  logic [31:0] psum_in = '0;
  logic [15:0] activation_out;
  logic [15:0] weight_out;
  logic [31:0] psum_out;

  ws_pe_single_stage dut (.*);
  always #0.5 clk = ~clk;

  initial begin
    repeat (2) @(posedge clk);
    rst_n = 1'b1;

    @(negedge clk);
    weight_shift_en = 1'b1;
    weight_in = $signed(-3);
    @(negedge clk);
    weight_shift_en = 1'b0;

    compute_en = 1'b1;
    activation_in = $signed(7);
    psum_in = $signed(100);
    @(posedge clk);
    #0.01;
    if ($signed(activation_out) !== 7 || $signed(psum_out) !== 79)
      $fatal(1, "first single-stage signed MAC failed");

    @(negedge clk);
    activation_in = $signed(-4);
    psum_in = $signed(-20);
    @(posedge clk);
    #0.01;
    if ($signed(activation_out) !== -4 || $signed(psum_out) !== -8)
      $fatal(1, "second single-stage signed MAC failed");

    $display("TB_WS_PE_SINGLE_STAGE PASS");
    $finish;
  end
endmodule
