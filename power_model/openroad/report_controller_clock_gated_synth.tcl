proc required_env {name} {
  if {![info exists ::env($name)] || $::env($name) eq ""} {
    error "required environment variable $name is unset"
  }
  return $::env($name)
}

set odb [required_env CONTROLLER_ODB]
set sdc [required_env CONTROLLER_SDC]
set liberty_files [split [required_env CONTROLLER_LIB_FILES] "|"]

foreach liberty $liberty_files {
  read_liberty $liberty
}
read_db $odb
read_sdc $sdc

set icgs [get_cells -hierarchical -filter "ref_name == ICGx2_ASAP7_75t_R"]
set clock_buffers \
  [get_cells -hierarchical -filter "ref_name == BUFx24_ASAP7_75t_R"]
if {[llength $icgs] != 2} {
  error "expected exactly two mapped ICGx2 cells, found [llength $icgs]"
}
if {[llength $clock_buffers] != 2} {
  error "expected exactly two mapped BUFx24 cells, found [llength $clock_buffers]"
}

puts "CONTROLLER_CLOCK_GATED_SYNTH_REPORT_BEGIN"
puts "mapped_icg_count [llength $icgs]"
puts "mapped_clock_buffer_count [llength $clock_buffers]"
report_units
report_design_area
report_worst_slack -max
report_clock_min_period
report_checks -path_delay max -group_path_count 1 -endpoint_path_count 1 \
  -format full_clock_expanded -digits 4
report_check_types -max_capacitance -max_slew -violators
puts "CONTROLLER_CLOCK_GATED_SYNTH_REPORT_END"
