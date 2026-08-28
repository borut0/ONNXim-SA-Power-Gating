module ws_array_32x32 (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         weight_shift_en,
  input  logic                         compute_en,
  input  logic        [(32*16)-1:0]    activation_in,
  input  logic        [(32*16)-1:0]    weight_in,
  input  logic        [(32*32)-1:0]    psum_in,
  output logic        [(32*16)-1:0]    activation_out,
  output logic        [(32*32)-1:0]    psum_out
);

  ws_array #(
    .ROWS(32),
    .COLS(32),
    .DATA_W(16),
    .ACC_W(32)
  ) array (
    .*
  );

endmodule
