module ws_array_1x1 (
  input  logic             clk,
  input  logic             rst_n,
  input  logic             weight_shift_en,
  input  logic             compute_en,
  input  logic [15:0]      activation_in,
  input  logic [15:0]      weight_in,
  input  logic [31:0]      psum_in,
  output logic [15:0]      activation_out,
  output logic [31:0]      psum_out
);

  ws_array #(
    .ROWS(1),
    .COLS(1),
    .DATA_W(16),
    .ACC_W(32)
  ) array (
    .*
  );

endmodule
