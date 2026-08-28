module ws_pe #(
  parameter int DATA_W = 16,
  parameter int ACC_W  = 32
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         weight_shift_en,
  input  logic                         compute_en,
  input  logic        [DATA_W-1:0]     activation_in,
  input  logic        [DATA_W-1:0]     weight_in,
  input  logic        [ACC_W-1:0]      psum_in,
  output logic        [DATA_W-1:0]     activation_out,
  output logic        [DATA_W-1:0]     weight_out,
  output logic        [ACC_W-1:0]      psum_out
);

  logic signed [DATA_W-1:0]       weight_reg;
  logic signed [(2*DATA_W)-1:0]   product_reg;
  logic signed [ACC_W-1:0]        psum_reg;
  logic                           product_valid_reg;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      weight_reg     <= '0;
      activation_out <= '0;
      weight_out     <= '0;
      psum_out       <= '0;
      product_reg    <= '0;
      psum_reg       <= '0;
      product_valid_reg <= 1'b0;
    end else begin
      product_valid_reg <= compute_en;

      if (weight_shift_en) begin
        weight_reg <= weight_in;
        weight_out <= weight_in;
      end

      if (compute_en) begin
        activation_out <= activation_in;
        product_reg <= $signed(activation_in) * $signed(weight_reg);
        psum_reg <= psum_in;
      end

      if (product_valid_reg) begin
        psum_out <= $signed(psum_reg) + $signed(product_reg);
      end
    end
  end

endmodule
