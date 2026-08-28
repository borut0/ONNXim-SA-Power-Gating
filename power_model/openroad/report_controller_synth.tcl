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

puts "CONTROLLER_SYNTH_REPORT_BEGIN"
report_units
report_design_area
report_worst_slack -max
report_clock_min_period
report_checks -path_delay max -group_path_count 1 -endpoint_path_count 1 \
  -format full_clock_expanded -digits 4
puts "CONTROLLER_SYNTH_REPORT_END"
