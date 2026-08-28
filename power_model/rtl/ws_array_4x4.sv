module ws_array_4x4 (
  input  logic                      clk,
  input  logic                      rst_n,
  input  logic                      weight_shift_en,
  input  logic                      compute_en,
  input  logic [(4*16)-1:0]         activation_in,
  input  logic [(4*16)-1:0]         weight_in,
  input  logic [(4*32)-1:0]         psum_in,
  output logic [(4*16)-1:0]         activation_out,
  output logic [(4*32)-1:0]         psum_out
);

  ws_array #(
    .ROWS(4),
    .COLS(4),
    .DATA_W(16),
    .ACC_W(32)
  ) array (
    .*
  );

endmodule
