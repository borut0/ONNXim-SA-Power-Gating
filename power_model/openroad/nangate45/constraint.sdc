current_design ws_array_16x16

set clk_period 1.000
create_clock -name core_clock -period $clk_period [get_ports clk]

set non_clock_inputs [all_inputs -no_clocks]
set_input_delay 0.200 -clock core_clock $non_clock_inputs
set_output_delay 0.200 -clock core_clock [all_outputs]
