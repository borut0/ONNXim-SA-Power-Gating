export DESIGN_NAME = sa_idle_predictor_controller_clock_gated
export PLATFORM = asap7

export SA_POWER_ROOT ?= $(abspath $(dir $(DESIGN_CONFIG))/../..)
export VERILOG_FILES = \
  $(SA_POWER_ROOT)/rtl/sa_idle_predictor_controller.sv \
  $(SA_POWER_ROOT)/rtl/sa_idle_predictor_controller_clock_gated.sv
export SDC_FILE = $(SA_POWER_ROOT)/openroad/asap7/constraint_pg_controller_clock_gated.sdc

export CORE_UTILIZATION ?= 50
export SYNTH_REPEATABLE_BUILD ?= 1
