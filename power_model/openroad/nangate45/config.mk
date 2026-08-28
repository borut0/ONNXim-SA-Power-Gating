export DESIGN_NAME = ws_array_16x16
export PLATFORM = nangate45

export SA_POWER_ROOT ?= $(abspath $(dir $(DESIGN_CONFIG))/../..)
export VERILOG_FILES = \
  $(SA_POWER_ROOT)/rtl/ws_pe.sv \
  $(SA_POWER_ROOT)/rtl/ws_array.sv \
  $(SA_POWER_ROOT)/rtl/ws_array_16x16.sv
export SDC_FILE = $(SA_POWER_ROOT)/openroad/nangate45/constraint.sdc

export CORE_UTILIZATION ?= 50
export PLACE_DENSITY_LB_ADDON ?= 0.15
export SYNTH_REPEATABLE_BUILD ?= 1
