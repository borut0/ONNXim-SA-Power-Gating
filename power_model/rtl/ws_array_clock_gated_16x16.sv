module ws_array_clock_gated_16x16 (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         sa_idle,
  input  logic                         gate_request,
  input  logic                         weight_shift_en,
  input  logic                         compute_en,
  input  logic        [(16*16)-1:0]    activation_in,
  input  logic        [(16*16)-1:0]    weight_in,
  input  logic        [(16*32)-1:0]    psum_in,
  output logic        [(16*16)-1:0]    activation_out,
  output logic        [(16*32)-1:0]    psum_out
);

  logic array_clock_enable;
  logic array_clk;

  // Keep the array clock running during reset. This is exactly
  // !(rst_n && sa_idle && gate_request).
  (* keep = "yes", dont_touch = "yes" *)
  NAND3x2_ASAP7_75t_R clock_gate_control (
    .A(sa_idle),
    .B(gate_request),
    .C(rst_n),
    .Y(array_clock_enable)
  );

  // ORFS maps this technology-independent interface through ASAP7's
  // CLKGATE_MAP_FILE to the platform clock-gating cell.
  OPENROAD_CLKGATE clock_gate (
    .CK(clk),
    .E(array_clock_enable),
    .GCK(array_clk)
  );

  ws_array_16x16 array (
    .clk(array_clk),
    .rst_n(rst_n),
    .weight_shift_en(weight_shift_en),
    .compute_en(compute_en),
    .activation_in(activation_in),
    .weight_in(weight_in),
    .psum_in(psum_in),
    .activation_out(activation_out),
    .psum_out(psum_out)
  );

endmodule
