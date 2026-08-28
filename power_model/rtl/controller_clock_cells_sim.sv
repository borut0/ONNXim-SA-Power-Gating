// Functional-only models for source-level replay. Gate-level replay uses the
// official ASAP7 Verilog models instead.
module ICGx2_ASAP7_75t_R (
  output wire GCLK,
  input  wire ENA,
  input  wire SE,
  input  wire CLK
);
  logic enable_latched;
  always_latch begin
    if (!CLK)
      enable_latched <= ENA || SE;
  end
  assign GCLK = CLK && enable_latched;
endmodule

module BUFx24_ASAP7_75t_R (
  output wire Y,
  input  wire A
);
  assign Y = A;
endmodule
