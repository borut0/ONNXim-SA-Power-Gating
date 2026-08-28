current_design ws_array_128x128

set clk_period 1000.000
create_clock -name core_clock -period $clk_period [get_ports clk]

set non_clock_inputs [all_inputs -no_clocks]
set_input_delay 200.000 -clock core_clock $non_clock_inputs
set_output_delay 200.000 -clock core_clock [all_outputs]
