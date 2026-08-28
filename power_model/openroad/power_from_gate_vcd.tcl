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
set clock_stopped false
if {[info exists ::env(POWER_CLOCK_STOPPED)]} {
  if {$::env(POWER_CLOCK_STOPPED) eq "YES"} {
    set clock_stopped true
  } elseif {$::env(POWER_CLOCK_STOPPED) ne "NO" &&
            $::env(POWER_CLOCK_STOPPED) ne ""} {
    error "POWER_CLOCK_STOPPED must be YES or NO"
  }
}
if {[required_env POWER_COVERAGE_ACCEPTED] ne "YES"} {
  error "POWER_COVERAGE_ACCEPTED must be YES after reviewing annotation coverage"
}

foreach liberty $liberty_files {
  read_liberty $liberty
}
read_db $odb
read_sdc $sdc
if {$clock_stopped} {
  set stopped_clock_port [required_env POWER_CLOCK_PORT]
  set stopped_clock_name [required_env POWER_CLOCK_NAME]
  set stopped_clock [get_clocks $stopped_clock_name]
  set stopped_clock_pins \
    [all_registers -clock $stopped_clock -clock_pins]
  if {[llength $stopped_clock_pins] == 0} {
    error "no sequential clock pins found for $stopped_clock_name"
  }
}
read_vcd -scope $scope $vcd
if {$clock_stopped} {
  # read_vcd requires an SDC clock. Remove its clock classification only after
  # VCD activities have been imported, then override the stopped clock source
  # and every sequential clock sink to exclude initialization edges.
  delete_clock $stopped_clock
  set_power_activity -input_ports [get_ports $stopped_clock_port] \
    -density 0.0 -duty 0.0
  set_power_activity -pins $stopped_clock_pins -density 0.0 -duty 0.0
}

report_units
report_activity_annotation
report_activity_annotation -report_unannotated > "${output_prefix}.unannotated.rpt"
report_power -digits 9 > "${output_prefix}.power.rpt"
if {[info exists ::env(POWER_HIGHEST_POWER_INSTANCES)]
    && $::env(POWER_HIGHEST_POWER_INSTANCES) ne ""} {
  set highest_count $::env(POWER_HIGHEST_POWER_INSTANCES)
  if {![string is integer -strict $highest_count] || $highest_count <= 0} {
    error "POWER_HIGHEST_POWER_INSTANCES must be a positive integer"
  }
  # This OpenSTA build lacks the highest_power_instances helper and its
  # internal leaf iterator. Emit the standard hierarchical cell collection;
  # mapped leaf rows are ranked and truncated offline.
  report_power -instances [get_cells -hierarchical *] -digits 9 \
    > "${output_prefix}.highest_power_instances.rpt"
}
report_power -digits 9
