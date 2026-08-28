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

foreach liberty $liberty_files {
  read_liberty $liberty
}
read_db $odb
read_sdc $sdc
read_vcd -scope $scope $vcd

report_units
report_activity_annotation
report_activity_annotation -report_unannotated > "${output_prefix}.unannotated.rpt"

puts "POWER_COVERAGE_ONLY no power report was generated"
