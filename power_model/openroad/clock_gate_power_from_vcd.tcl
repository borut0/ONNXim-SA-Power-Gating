proc required_env {name} {
  if {![info exists ::env($name)] || $::env($name) eq ""} {
    error "required environment variable $name is unset"
  }
  return $::env($name)
}

set odb [required_env POWER_ODB]
set sdc [required_env POWER_SDC]
set vcd [required_env POWER_VCD]
set scope [required_env POWER_SCOPE]
set output_prefix [required_env POWER_OUTPUT_PREFIX]
set liberty_files [split [required_env POWER_LIB_FILES] "|"]
if {[required_env POWER_COVERAGE_ACCEPTED] ne "YES"} {
  error "POWER_COVERAGE_ACCEPTED must be YES after reviewing annotation coverage"
}

foreach liberty $liberty_files {
  read_liberty $liberty
}
read_db $odb
read_sdc $sdc
read_vcd -scope $scope $vcd

set controller [get_cells -quiet clock_gate_control]
set clock_gate [get_cells -quiet -hierarchical -filter "ref_name =~ ICG*"]
if {[llength $controller] != 1 || [llength $clock_gate] != 1} {
  error "clock_gate_control and one mapped ASAP7 ICG must each exist exactly once"
}

report_units
report_activity_annotation
report_activity_annotation -report_unannotated > "${output_prefix}.unannotated.rpt"
report_power -digits 9 > "${output_prefix}.total.rpt"
report_power -instances $controller -digits 9 > "${output_prefix}.controller.rpt"
report_power -instances $clock_gate -digits 9 > "${output_prefix}.icg.rpt"
report_power -digits 9
