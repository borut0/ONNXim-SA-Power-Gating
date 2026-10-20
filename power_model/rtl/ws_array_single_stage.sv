module ws_array_single_stage #(
  parameter int ROWS   = 1,
  parameter int COLS   = 1,
  parameter int DATA_W = 16,
  parameter int ACC_W  = 32
) (
  input  logic                           clk,
  input  logic                           rst_n,
  input  logic                           weight_shift_en,
  input  logic                           compute_en,
  input  logic        [(ROWS*DATA_W)-1:0] activation_in,
  input  logic        [(COLS*DATA_W)-1:0] weight_in,
  input  logic        [(COLS*ACC_W)-1:0]  psum_in,
  output logic        [(ROWS*DATA_W)-1:0] activation_out,
  output logic        [(COLS*ACC_W)-1:0]  psum_out
);

  logic [DATA_W-1:0] activation_link [0:ROWS-1][0:COLS];
  logic [DATA_W-1:0] weight_link     [0:ROWS][0:COLS-1];
  logic [ACC_W-1:0]  psum_link       [0:ROWS][0:COLS-1];

  genvar row;
  genvar col;
  generate
    for (row = 0; row < ROWS; row++) begin : gen_row_inputs
      assign activation_link[row][0] =
          activation_in[(row*DATA_W) +: DATA_W];
      assign activation_out[(row*DATA_W) +: DATA_W] =
          activation_link[row][COLS];
    end
    for (col = 0; col < COLS; col++) begin : gen_col_inputs
      assign weight_link[0][col] = weight_in[(col*DATA_W) +: DATA_W];
      assign psum_link[0][col] = psum_in[(col*ACC_W) +: ACC_W];
      assign psum_out[(col*ACC_W) +: ACC_W] = psum_link[ROWS][col];
    end
    for (row = 0; row < ROWS; row++) begin : gen_rows
      for (col = 0; col < COLS; col++) begin : gen_cols
        ws_pe_single_stage #(
          .DATA_W(DATA_W),
          .ACC_W(ACC_W)
        ) pe (
          .clk(clk),
          .rst_n(rst_n),
          .weight_shift_en(weight_shift_en),
          .compute_en(compute_en),
          .activation_in(activation_link[row][col]),
          .weight_in(weight_link[row][col]),
          .psum_in(psum_link[row][col]),
          .activation_out(activation_link[row][col+1]),
          .weight_out(weight_link[row+1][col]),
          .psum_out(psum_link[row+1][col])
        );
      end
    end
  endgenerate

endmodule
