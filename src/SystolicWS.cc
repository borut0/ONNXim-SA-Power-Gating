#include "SystolicWS.h"
#include "SaPeStateOpportunity.h"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {

const char* opcode_name(Opcode opcode) {
  switch (opcode) {
    case Opcode::MOVIN: return "MOVIN";
    case Opcode::MOVOUT: return "MOVOUT";
    case Opcode::MOVOUT_POOL: return "MOVOUT_POOL";
    case Opcode::GEMM_PRELOAD: return "GEMM_PRELOAD";
    case Opcode::GEMM: return "GEMM";
    case Opcode::GEMM_WRITE: return "GEMM_WRITE";
    case Opcode::COMP: return "COMP";
    case Opcode::IM2COL: return "IM2COL";
    case Opcode::SOFTMAX: return "SOFTMAX";
    case Opcode::LAYERNORM: return "LAYERNORM";
    case Opcode::ADD: return "ADD";
    case Opcode::MUL: return "MUL";
    case Opcode::MAC: return "MAC";
    case Opcode::DIV: return "DIV";
    case Opcode::ADDTREE: return "ADDTREE";
    case Opcode::EXP: return "EXP";
    case Opcode::GELU: return "GELU";
    case Opcode::SWISH: return "SWISH";
    case Opcode::BAR: return "BAR";
  }
  return "UNKNOWN";
}

}  // namespace

SystolicWS::SystolicWS(uint32_t id, SimulationConfig config)
    : Core(id, config) {
  const char* component_measure =
      std::getenv("ONNXSIM_COMPONENT_CHARACTERIZATION");
  const bool component_enabled =
      component_measure != nullptr && component_measure[0] != '\0' &&
      std::string(component_measure) != "0";
  const char* state_measure =
      std::getenv("ONNXSIM_SA_STATE_BREAKDOWN");
  _sa_state_measure =
      component_enabled ||
      (state_measure != nullptr && state_measure[0] != '\0' &&
       std::string(state_measure) != "0");

  const char* phase_measure =
      std::getenv("ONNXSIM_SA_PHASE_MEASURE");
  _phase_measure_enabled =
      phase_measure != nullptr && phase_measure[0] != '\0' &&
      std::string(phase_measure) != "0";

  const char* predictor_measure =
      std::getenv("ONNXSIM_SA_PREDICTOR_MEASURE");
  _predictor_measure_enabled =
      predictor_measure != nullptr && predictor_measure[0] != '\0' &&
      std::string(predictor_measure) != "0";
  const char* updated_predictor =
      std::getenv("ONNXSIM_SA_UPDATED_PREDICTOR");
  _updated_predictor_enabled =
      updated_predictor != nullptr && updated_predictor[0] != '\0' &&
      std::string(updated_predictor) != "0";
  _predictor_measure_enabled =
      _predictor_measure_enabled || _updated_predictor_enabled;

  const char* reason_measure =
      std::getenv("ONNXSIM_SA_RETENTION_REASON_MEASURE");
  _retention_reason_measure =
      reason_measure != nullptr && reason_measure[0] != '\0' &&
      std::string(reason_measure) != "0";

  const char* retention_shadow =
      std::getenv("ONNXSIM_SA_RETENTION_SHADOW");
  _retention_shadow_enabled =
      retention_shadow != nullptr && retention_shadow[0] != '\0' &&
      std::string(retention_shadow) != "0";
  if ((_retention_reason_measure || _retention_shadow_enabled ||
       _predictor_measure_enabled || _updated_predictor_enabled) &&
      _config.core_freq > 0 &&
      _config.icnt_freq > 0 &&
      _config.icnt_injection_ports_per_core > 0) {
    uint64_t core_period = 1000000 / _config.core_freq;
    uint64_t icnt_period = 1000000 / _config.icnt_freq;
    uint64_t maximum_icnt_events_per_core_cycle =
        icnt_period == 0 ? 0 :
        (core_period + icnt_period - 1) / icnt_period;
    if (core_period > 0) {
      _response_drain_ports_per_core_cycle_bound =
          maximum_icnt_events_per_core_cycle *
          _config.icnt_injection_ports_per_core;
    }
  }

  const char* scheduler_opportunity_measure =
      std::getenv("ONNXSIM_SA_SCHEDULER_OPPORTUNITY_MEASURE");
  _scheduler_opportunity_measure =
      scheduler_opportunity_measure != nullptr &&
      scheduler_opportunity_measure[0] != '\0' &&
      std::string(scheduler_opportunity_measure) != "0";
  const char* scheduler_unlock_measure =
      std::getenv("ONNXSIM_SA_SCHEDULER_UNLOCK_MEASURE");
  _scheduler_unlock_measure =
      scheduler_unlock_measure != nullptr &&
      scheduler_unlock_measure[0] != '\0' &&
      std::string(scheduler_unlock_measure) != "0";
  _scheduler_opportunity_measure =
      _scheduler_opportunity_measure || _scheduler_unlock_measure;

  initialize_hybrid_policy();
  initialize_sa_event_trace();
  initialize_regate_functional_profile();
  initialize_regate_two_bank_profile();
  initialize_updated_regate_profile();
  initialize_sa_region_profile();

  const char* trace_dir = std::getenv("ONNXSIM_SA_CONTROLLER_TRACE_DIR");
  if (trace_dir == nullptr || trace_dir[0] == '\0') {
    return;
  }

  std::filesystem::path directory(trace_dir);
  std::filesystem::create_directories(directory);
  std::filesystem::path trace_path =
      directory / ("sa_controller_core" + std::to_string(_id) + ".csv");
  _controller_trace.open(trace_path, std::ios::out | std::ios::trunc);
  if (!_controller_trace.is_open()) {
    throw std::runtime_error("cannot open SA controller trace: " +
                             trace_path.string());
  }
  _controller_trace
      << "cycle,final,sa_active,compute_pipeline_empty,"
         "compute_front_future_valid,compute_front_remaining,"
         "ex_inst_queue_empty,ex_front_is_sa,scheduler_core_tile_waiting,"
         "global_issue_drain_bound,core_queue_tickets_exact,"
         "last_dequeued_ticket,"
         "tile0_we,tile0_count_we,tile0_present,tile0_candidate_found,"
         "tile0_candidate_preload,tile0_sources_supported,"
         "tile0_prefix_count,tile0_remaining_count,tile0_operand_used,"
         "tile0_operand_allocated,tile0_operand_valid,"
         "tile0_operand_pending_memory,tile0_operand0_tail_ticket,"
         "tile0_operand1_tail_ticket,tile0_operand0_pending_count,"
         "tile0_operand1_pending_count,"
         "tile1_we,tile1_count_we,tile1_present,tile1_candidate_found,"
         "tile1_candidate_preload,tile1_sources_supported,"
         "tile1_prefix_count,tile1_remaining_count,tile1_operand_used,"
         "tile1_operand_allocated,tile1_operand_valid,"
         "tile1_operand_pending_memory,tile1_operand0_tail_ticket,"
         "tile1_operand1_tail_ticket,tile1_operand0_pending_count,"
         "tile1_operand1_pending_count\n";
}

SystolicWS::~SystolicWS() {
  emit_regate_functional_profile();
  emit_regate_two_bank_profile();
  emit_updated_regate_profile();
  emit_sa_region_profile();
  emit_hybrid_summary();
  flush_instruction_stall();
  if (_sa_event_trace.is_open()) {
    write_sa_event(json{{"event", "TRACE_END"},
                        {"cycle", _core_cycle},
                        {"core_id", _id}});
    _sa_event_trace.close();
  }
  if (!_controller_trace.is_open()) {
    return;
  }
  if (_controller_trace_have_inputs) {
    ControllerTraceInputs final_inputs = _controller_trace_last_inputs;
    final_inputs[10] = 0;  // tile0_we
    final_inputs[11] = 0;  // tile0_count_we
    final_inputs[26] = 0;  // tile1_we
    final_inputs[27] = 0;  // tile1_count_we
    write_controller_trace_record(_core_cycle, true, final_inputs);
  }
}

void SystolicWS::issue(std::unique_ptr<Tile> tile) {
  Tile* admitted_tile = tile.get();
  Core::issue(std::move(tile));

  if (_sa_event_trace.is_open()) {
    flush_instruction_stall();
    write_sa_event(json{{"event", "TILE_ADMITTED"},
                        {"cycle", _core_cycle},
                        {"core_id", _id},
                        {"tile_id", admitted_tile->measurement_tile_id},
                        {"layer_id", admitted_tile->layer_id},
                        {"fused_op_id", admitted_tile->fused_op_id},
                        {"operation", admitted_tile->optype},
                        {"skip", admitted_tile->skip},
                        {"instruction_count",
                         admitted_tile->instructions.size()},
                        {"sa_descriptor_count",
                         admitted_tile->sa_descriptors.size()}});
  }

  if (_phase_measure_enabled && _phase_measure) {
    bool contains_gemm = false;
    for (const std::unique_ptr<Instruction>& inst :
         admitted_tile->instructions) {
      if (inst->opcode == Opcode::GEMM ||
          inst->opcode == Opcode::GEMM_PRELOAD) {
        contains_gemm = true;
        break;
      }
    }

    PhaseAdmission admission;
    admission.tile_id = admitted_tile->measurement_tile_id;
    admission.admission_cycle = _core_cycle;
    admission.layer_id = admitted_tile->layer_id;
    admission.operation = admitted_tile->optype;
    admission.skip = admitted_tile->skip;
    admission.contains_gemm = contains_gemm;
    admission.resident =
        find_resident_tile(admitted_tile->measurement_tile_id) != nullptr;
    if (!admission.resident && admission.skip) {
      admission.retirement_set = true;
      admission.retirement_cycle = _core_cycle;
    }
    _phase_admissions.push_back(std::move(admission));

    if (!_phase_t2_set) {
      _phase_t2 = _core_cycle;
      _phase_t2_set = true;
      _phase_next_tile_id = admitted_tile->measurement_tile_id;
    }
    observe_phase_events();
  }
}

void SystolicWS::push_memory_response(MemoryAccess* response) {
  if (_phase_measure_enabled && _phase_measure) {
    observe_required_response(*response);
  }
  Core::push_memory_response(response);
}

void SystolicWS::cycle() {
  if (_phase_measure_enabled) {
    observe_phase_events();
  }

    /*
  Compute unit
  */
  finish_compute_pipeline();
  /* Checking Vector compute pipeline */
  finish_vector_pipeline();
  /* LD in struction queue */
  handle_ld_inst_queue();
  classify_phase_t3();

  bool ex_ready = false;

  if (!_ex_inst_queue.empty()) {
    ex_ready = can_issue_compute(_ex_inst_queue.front());
  }

  /* EX instruction queue */
  if (!_ex_inst_queue.empty() && ex_ready) { // execution dependecy check
    std::unique_ptr<Instruction> front = std::move(_ex_inst_queue.front());
    record_compute_operand_reads(*front);
    if (front->dest_addr >= ACCUM_SPAD_BASE) {
      if (_acc_spad.check_allocated(front->dest_addr, front->accum_spad_id)) {
        _acc_spad.count_up(front->dest_addr, front->accum_spad_id,
                       Sram::PendingProducer::COMPUTE);
      } else {
        int ret = _acc_spad.prefetch(front->dest_addr, front->accum_spad_id, front->size,
                                    front->zero_init? front->size : 1,
                                    Sram::PendingProducer::COMPUTE);
        if (!ret) {
          spdlog::error("Destination allocated: {} Size remain: {}", _acc_spad.check_allocated(front->dest_addr, front->accum_spad_id), _acc_spad.check_remain(front->size, front->accum_spad_id));
          spdlog::error("instruction panic opcode: {:x}, addr: {:x}, size: {} B", (int)front->opcode, front->dest_addr, front->size*_config.dram_req_size);
          _acc_spad.print_all(front->accum_spad_id);
          std::exit(EXIT_FAILURE);
        }
      }
    } else {
      if (_spad.check_allocated(front->dest_addr, front->spad_id)) {
        _spad.count_up(front->dest_addr, front->spad_id,
                     Sram::PendingProducer::COMPUTE);
      } else {
        int ret = _spad.prefetch(front->dest_addr, front->spad_id, front->size,
                                  front->zero_init? front->size : 1,
                                  Sram::PendingProducer::COMPUTE);
        if (!ret) {
          spdlog::error("Destination allocated: {} Size remain: {}", _spad.check_allocated(front->dest_addr, front->spad_id), _spad.check_remain(front->size, front->spad_id));
          spdlog::error("instruction panic opcode: {:x}, addr: {:x}, size: {} B", (int)front->opcode, front->dest_addr, front->size*_config.dram_req_size);
          _spad.print_all(front->spad_id);
          std::exit(EXIT_FAILURE);
        }
      }
    }
    if (front->opcode == Opcode::GEMM || front->opcode == Opcode::GEMM_PRELOAD) {
      if (!_compute_pipeline.empty()) {
        /* Preload can be hided */
        uint32_t offset = _compute_pipeline.back()->compute_size;
        offset = MAX(offset, 4);
        if (front->opcode == Opcode::GEMM_PRELOAD) {
          // State mul-pre
          offset = MAX(offset, _config.core_config[_id].core_height);
          _stat_systolic_preload_issue_count++;
        }
        if (_compute_pipeline.back()->start_cycle+offset < _core_cycle) {
          front->start_cycle = _core_cycle;
          _stat_systolic_bubble_cycle += (_core_cycle - _compute_pipeline.back()->start_cycle+offset);
        } else
          front->start_cycle = _compute_pipeline.back()->start_cycle+offset;
      } else {
        front->start_cycle = _core_cycle;
        /* Preload weight to systolic array*/
        if (front->opcode == Opcode::GEMM_PRELOAD) {
          /* Weight preload  from buffer latecny + WEight preload latency */
          front->start_cycle += _config.core_config[_id].core_height + _config.core_config[_id].core_height - 1;
          _stat_systolic_preload_issue_count++;
        }
      }
      front->finish_cycle = front->start_cycle + get_inst_compute_cycles(front);
      record_sa_region_interval(*front);
      if (_hybrid_parameters.enabled &&
          front->opcode == Opcode::GEMM_PRELOAD) {
        // The preload itself establishes the complete PE weight state before
        // any later ordinary GEMM can consume it.
        _hybrid_pe_state_valid = true;
      }
      _compute_pipeline.push(std::move(front));
      _stat_systolic_inst_issue_count++;
    } else {  // vector unit compute
      front->start_cycle = _core_cycle;
      front->finish_cycle =
          front->start_cycle +
          get_vector_compute_cycles(front);  // Setting IC as 1 (Might need to modify)
      _vector_pipeline.push(std::move(front));
    }
    _ex_inst_queue.pop();
  }

  /* ST in struction queue */
  handle_st_inst_queue();

  bool is_running = running();
  bool is_compute_busy = false;
  bool is_vector_busy = false;

  if (!_compute_pipeline.empty() && _compute_pipeline.front()->start_cycle <= _core_cycle)
    is_compute_busy = true;

  if (!_vector_pipeline.empty() && _vector_pipeline.front()->start_cycle <= _core_cycle)
    is_vector_busy = true;

  const bool sa_committed_start_delay =
      !is_compute_busy && !_compute_pipeline.empty() &&
      _compute_pipeline.front()->start_cycle > _core_cycle;
  const Opcode sa_front_opcode =
      _compute_pipeline.empty() ? Opcode::BAR
                                : _compute_pipeline.front()->opcode;

  if (_hybrid_parameters.enabled) {
    update_hybrid_policy(is_compute_busy);
  }

  if (_sa_event_trace.is_open()) {
    trace_sa_transition(is_compute_busy);
  }

  if (_controller_trace.is_open()) {
    trace_controller_inputs(is_compute_busy);
  }
  if (_retention_reason_measure) {
    measure_retention_source_reasons(is_compute_busy);
  }
  if (_retention_shadow_enabled) {
    update_retention_shadow(is_compute_busy);
  }
  if (_scheduler_opportunity_measure) {
    measure_scheduler_opportunity(is_compute_busy);
  }
  if (_predictor_measure_enabled) {
    update_idle_predictor(is_compute_busy);
  }

  if (is_compute_busy){
    _stat_systolic_active_cycle++;
  }
  if (is_vector_busy){
    _stat_vec_compute_cycle++;
  }

  if (is_compute_busy || is_vector_busy){
    _stat_compute_cycle++;
  }
  if (_component_characterization && is_compute_busy && is_vector_busy) {
    _stat_sa_vector_overlap_cycle++;
  }

  if (_request_queue.empty())
    _stat_memory_idle_cycle++;

  if (!is_running)
  {
    _stat_idle_cycle++;
  }

  if (_phase_measure_enabled) {
    bool sa_idle = !is_compute_busy;
    if (sa_idle && !_phase_sa_idle) {
      _phase_sa_idle = true;
      if (!has_resident_gemm_candidate()) {
        begin_phase_measurement(is_vector_busy);
      }
    } else if (!sa_idle && _phase_sa_idle) {
      if (_phase_measure) {
        complete_phase_measurement();
      }
      _phase_sa_idle = false;
    }
  }

  Core::cycle();
  if (_sa_event_trace.is_open()) {
    trace_instruction_flow_events();
  }
  if (_phase_measure_enabled) {
    observe_phase_events();
  }

  CoreAttributionCategory attribution_category = CoreAttributionCategory::OTHER;
  if (_component_characterization || _sa_state_measure) {
    if (is_compute_busy || is_vector_busy) {
      attribution_category = CoreAttributionCategory::COMPUTE;
    } else if (_last_core_cycle_issued_instruction) {
      attribution_category = CoreAttributionCategory::OTHER;
    } else {
      bool memory_wait = false;
      bool dependency_wait = false;
      bool scheduler_wait = false;
      bool other_wait = false;

      auto record_wait = [&](OperandWaitReason reason) {
        switch (reason) {
          case OperandWaitReason::NONE:
            break;
          case OperandWaitReason::MEMORY:
            memory_wait = true;
            break;
          case OperandWaitReason::DEPENDENCY:
            dependency_wait = true;
            break;
          case OperandWaitReason::UNKNOWN:
            other_wait = true;
            break;
        }
      };

      if (!_ex_inst_queue.empty()) {
        switch (_last_compute_block_reason) {
          case ComputeBlockReason::OPERAND_WAIT:
            record_wait(get_operand_wait_reason(*_ex_inst_queue.front()));
            break;
          case ComputeBlockReason::PIPELINE_FULL:
          case ComputeBlockReason::VECTOR_BUSY:
          case ComputeBlockReason::NONE:
            other_wait = true;
            break;
        }
      }

      if (!_st_inst_queue.empty()) {
        record_wait(get_store_wait_reason(*_st_inst_queue.front()));
      }

      if (_waiting_write_reqs != 0) {
        memory_wait = true;
      }

      memory_wait = memory_wait || _last_core_cycle_memory_block;
      dependency_wait = dependency_wait || _last_core_cycle_dependency_block;
      other_wait = other_wait || _last_core_cycle_unknown_block;

      if (_last_core_cycle_compute_issue_block &&
          _last_compute_block_reason != ComputeBlockReason::OPERAND_WAIT) {
        other_wait = true;
      }

      if (_scheduler_issue_state ==
              Core::SchedulerIssueState::NO_TILE_AVAILABLE &&
          !running()) {
        scheduler_wait = true;
      }

      if (_last_core_cycle_had_pending_instruction &&
          !memory_wait && !dependency_wait && !scheduler_wait &&
          !other_wait) {
        other_wait = true;
      }

      int exact_categories = (memory_wait ? 1 : 0) +
                             (dependency_wait ? 1 : 0) +
                             (scheduler_wait ? 1 : 0) +
                             (other_wait ? 1 : 0);

      if (exact_categories == 1) {
        if (memory_wait) {
          attribution_category = CoreAttributionCategory::MEMORY_STALL;
        } else if (dependency_wait) {
          attribution_category = CoreAttributionCategory::DEPENDENCY_STALL;
        } else if (scheduler_wait) {
          attribution_category = CoreAttributionCategory::SCHEDULER_STALL;
        } else {
          attribution_category = CoreAttributionCategory::OTHER;
        }
      }
    }
  }

  if (_sa_state_measure) {
    SaArchitecturalState state = SaArchitecturalState::IDLE_OTHER;
    if (is_compute_busy) {
      if (sa_front_opcode == Opcode::GEMM) {
        state = SaArchitecturalState::ACTIVE_GEMM;
      } else if (sa_front_opcode == Opcode::GEMM_PRELOAD) {
        state = SaArchitecturalState::ACTIVE_GEMM_PRELOAD;
      } else {
        state = SaArchitecturalState::ACTIVE_UNKNOWN;
      }
    } else if (sa_committed_start_delay) {
      if (sa_front_opcode == Opcode::GEMM) {
        state = SaArchitecturalState::IDLE_COMMITTED_GEMM_START_DELAY;
      } else if (sa_front_opcode == Opcode::GEMM_PRELOAD) {
        state = SaArchitecturalState::IDLE_COMMITTED_GEMM_PRELOAD_START_DELAY;
      }
    } else if (is_vector_busy) {
      state = SaArchitecturalState::IDLE_VECTOR_ACTIVE;
    } else if (_last_core_cycle_issued_instruction) {
      state = SaArchitecturalState::IDLE_INSTRUCTION_DISPATCH;
    } else {
      switch (attribution_category) {
        case CoreAttributionCategory::MEMORY_STALL:
          state = SaArchitecturalState::IDLE_MEMORY_STALL;
          break;
        case CoreAttributionCategory::DEPENDENCY_STALL:
          state = SaArchitecturalState::IDLE_DEPENDENCY_STALL;
          break;
        case CoreAttributionCategory::SCHEDULER_STALL:
          state = SaArchitecturalState::IDLE_SCHEDULER_STALL;
          break;
        case CoreAttributionCategory::COMPUTE:
        case CoreAttributionCategory::OTHER:
          state = SaArchitecturalState::IDLE_OTHER;
          break;
      }
    }
    _sa_state_cycles[static_cast<size_t>(state)]++;
  }

  if (_component_characterization) {
    attribute_core_cycle(attribution_category);
  }
}

void SystolicWS::initialize_sa_event_trace() {
  const char* trace_dir = std::getenv("ONNXSIM_SA_EVENT_TRACE_DIR");
  if (trace_dir == nullptr || trace_dir[0] == '\0') {
    return;
  }
  std::filesystem::path directory(trace_dir);
  std::filesystem::create_directories(directory);
  std::filesystem::path trace_path =
      directory / ("sa_events_core" + std::to_string(_id) + ".jsonl");
  _sa_event_trace.open(trace_path, std::ios::out | std::ios::trunc);
  if (!_sa_event_trace.is_open()) {
    throw std::runtime_error("cannot open SA event trace: " +
                             trace_path.string());
  }
  write_sa_event(json{{"event", "TRACE_START"},
                      {"version", 1},
                      {"cycle", _core_cycle},
                      {"core_id", _id},
                      {"semantics",
                       "causal instruction/state events; repeated identical "
                       "instruction stalls are run-length encoded"}});
}

void SystolicWS::write_sa_event(json event) {
  if (!_sa_event_trace.is_open()) {
    return;
  }
  _sa_event_trace << event.dump() << '\n';
}

const char* SystolicWS::instruction_block_reason_name(
    InstructionBlockReason reason) const {
  switch (reason) {
    case InstructionBlockReason::NONE: return "NONE";
    case InstructionBlockReason::MEMORY: return "MEMORY";
    case InstructionBlockReason::DEPENDENCY: return "DEPENDENCY";
    case InstructionBlockReason::EXECUTION_QUEUE_BUSY:
      return "EXECUTION_QUEUE_BUSY";
    case InstructionBlockReason::SA_PIPELINE_FULL:
      return "SA_PIPELINE_FULL";
    case InstructionBlockReason::VECTOR_BUSY: return "VECTOR_BUSY";
    case InstructionBlockReason::POWER_STATE_WAKE:
      return "POWER_STATE_WAKE";
    case InstructionBlockReason::UNKNOWN: return "UNKNOWN";
  }
  return "UNKNOWN";
}

void SystolicWS::flush_instruction_stall() {
  if (!_pending_instruction_stall.active) {
    return;
  }
  const InstructionFlowEvent& event = _pending_instruction_stall.event;
  write_sa_event(json{
      {"event", "INSTRUCTION_STALL"},
      {"start_cycle", event.cycle},
      {"end_cycle", _pending_instruction_stall.end_cycle},
      {"occurrences", _pending_instruction_stall.occurrences},
      {"core_id", _id},
      {"tile_id", event.tile_id},
      {"layer_id", event.layer_id},
      {"fused_op_id", event.fused_op_id},
      {"operation", event.operation},
      {"instruction_ordinal", event.instruction_ordinal},
      {"opcode", opcode_name(event.opcode)},
      {"reason", instruction_block_reason_name(event.block_reason)},
      {"next_sa_prefix", event.next_sa_prefix_before},
      {"weight_block_known", event.weight_block_known},
      {"weight_block_id", event.weight_block_id}});
  _pending_instruction_stall = PendingInstructionStall{};
}

void SystolicWS::trace_instruction_flow_events() {
  for (const InstructionFlowEvent& event :
       _last_instruction_flow_events) {
    if (!event.issued) {
      bool same = _pending_instruction_stall.active &&
          _pending_instruction_stall.event.tile_id == event.tile_id &&
          _pending_instruction_stall.event.instruction_ordinal ==
              event.instruction_ordinal &&
          _pending_instruction_stall.event.opcode == event.opcode &&
          _pending_instruction_stall.event.block_reason ==
              event.block_reason &&
          _pending_instruction_stall.end_cycle + 1 == event.cycle;
      if (!same) {
        flush_instruction_stall();
        _pending_instruction_stall.active = true;
        _pending_instruction_stall.event = event;
        _pending_instruction_stall.occurrences = 0;
      }
      _pending_instruction_stall.end_cycle = event.cycle;
      _pending_instruction_stall.occurrences++;
      continue;
    }

    flush_instruction_stall();
    write_sa_event(json{
        {"event", "INSTRUCTION_ISSUED"},
        {"cycle", event.cycle},
        {"core_id", _id},
        {"tile_id", event.tile_id},
        {"layer_id", event.layer_id},
        {"fused_op_id", event.fused_op_id},
        {"operation", event.operation},
        {"instruction_ordinal", event.instruction_ordinal},
        {"opcode", opcode_name(event.opcode)},
        {"previous_issued_tile_id", event.previous_issued_tile_id},
        {"tile_switch", event.tile_switch},
        {"next_sa_prefix_before", event.next_sa_prefix_before},
        {"weight_block_known", event.weight_block_known},
        {"weight_block_id", event.weight_block_id}});
  }
  for (uint64_t tile_id : _last_retired_tile_ids) {
    flush_instruction_stall();
    write_sa_event(json{{"event", "TILE_RETIRED"},
                        {"cycle", _core_cycle},
                        {"core_id", _id},
                        {"tile_id", tile_id}});
  }
}

void SystolicWS::trace_sa_transition(bool sa_active) {
  if (sa_active && !_event_trace_previous_sa_active) {
    flush_instruction_stall();
    const Instruction* front = _compute_pipeline.empty()
        ? nullptr : _compute_pipeline.front().get();
    _event_trace_active_opcode = front ? front->opcode : Opcode::BAR;
    _event_trace_active_tile_id = front ? front->measurement_tile_id : 0;
    _event_trace_active_weight_block_id =
        front ? front->sa_weight_block_id : 0;
    _event_trace_active_weight_known =
        front ? front->sa_weight_block_known : false;
    write_sa_event(json{
        {"event", "SA_START"},
        {"cycle", _core_cycle},
        {"core_id", _id},
        {"tile_id", _event_trace_active_tile_id},
        {"opcode", opcode_name(_event_trace_active_opcode)},
        {"weight_block_known", _event_trace_active_weight_known},
        {"weight_block_id", _event_trace_active_weight_block_id}});
  } else if (!sa_active && _event_trace_previous_sa_active) {
    flush_instruction_stall();
    write_sa_event(json{
        {"event", "SA_FINISH"},
        {"cycle", _core_cycle},
        {"core_id", _id},
        {"tile_id", _event_trace_active_tile_id},
        {"opcode", opcode_name(_event_trace_active_opcode)},
        {"weight_block_known", _event_trace_active_weight_known},
        {"weight_block_id", _event_trace_active_weight_block_id}});
  }
  _event_trace_previous_sa_active = sa_active;
}

const char* SystolicWS::hybrid_state_name(HybridSaState state) const {
  switch (state) {
    case HybridSaState::ON: return "ON";
    case HybridSaState::RETENTION: return "RETENTION";
    case HybridSaState::OFF: return "OFF";
    case HybridSaState::COUNT: break;
  }
  return "UNKNOWN";
}

const SystolicWS::HybridStateParameters&
SystolicWS::hybrid_state_parameters(HybridSaState state) const {
  if (state == HybridSaState::RETENTION) {
    return _hybrid_parameters.retention;
  }
  if (state == HybridSaState::OFF) {
    return _hybrid_parameters.off;
  }
  throw std::logic_error("ON has no low-power-state parameters");
}

void SystolicWS::initialize_hybrid_policy() {
  const char* path = std::getenv("ONNXSIM_SA_HYBRID_CONFIG");
  if (path == nullptr || path[0] == '\0' || std::string(path) == "0") {
    return;
  }

  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::runtime_error("cannot open hybrid SA config: " +
                             std::string(path));
  }
  json config;
  input >> config;
  if (!config.is_object() || !config.contains("version") ||
      config.at("version").get<uint32_t>() != 1) {
    throw std::runtime_error("hybrid SA config requires version 1");
  }

  auto get_nonnegative_cycles = [&](const json& object,
                                    const char* key,
                                    cycle_type fallback) {
    if (!object.contains(key)) {
      return fallback;
    }
    if (!object.at(key).is_number_unsigned() &&
        !object.at(key).is_number_integer()) {
      throw std::runtime_error(std::string("hybrid field must be an integer: ") +
                               key);
    }
    int64_t value = object.at(key).get<int64_t>();
    if (value < 0) {
      throw std::runtime_error(std::string("hybrid field must be nonnegative: ") +
                               key);
    }
    return static_cast<cycle_type>(value);
  };
  auto get_optional_nonnegative = [&](const json& object,
                                      const char* key) -> std::optional<double> {
    if (!object.contains(key) || object.at(key).is_null()) {
      return std::nullopt;
    }
    if (!object.at(key).is_number()) {
      throw std::runtime_error(std::string("hybrid field must be numeric: ") +
                               key);
    }
    double value = object.at(key).get<double>();
    if (value < 0.0) {
      throw std::runtime_error(std::string("hybrid field must be nonnegative: ") +
                               key);
    }
    return value;
  };
  auto parse_state = [&](const char* name) {
    if (!config.contains(name) || !config.at(name).is_object()) {
      throw std::runtime_error(std::string("hybrid config missing object: ") +
                               name);
    }
    const json& object = config.at(name);
    HybridStateParameters result;
    result.enabled = object.value("enabled", false);
    result.entry_latency_cycles =
        get_nonnegative_cycles(object, "entry_latency_cycles", 0);
    result.wake_latency_cycles =
        get_nonnegative_cycles(object, "wake_latency_cycles", 0);
    result.break_even_cycles =
        get_nonnegative_cycles(object, "break_even_cycles", 0);
    result.idle_power_w = get_optional_nonnegative(object, "idle_power_w");
    result.entry_energy_j =
        get_optional_nonnegative(object, "entry_energy_j");
    result.wake_energy_j =
        get_optional_nonnegative(object, "wake_energy_j");
    return result;
  };

  std::string mode = config.value("mode", std::string("shadow"));
  if (mode != "shadow" && mode != "closed_loop") {
    throw std::runtime_error(
        "hybrid SA mode must be 'shadow' or 'closed_loop'");
  }
  _hybrid_parameters.enabled = true;
  _hybrid_parameters.closed_loop = mode == "closed_loop";
  _hybrid_parameters.apply_break_even_filter =
      config.value("apply_break_even_filter", false);
  _hybrid_parameters.prefer_off_when_safe =
      config.value("prefer_off_when_safe", true);
  _hybrid_parameters.minimum_causal_bound_cycles =
      get_nonnegative_cycles(config, "minimum_causal_bound_cycles", 1);
  if (_hybrid_parameters.minimum_causal_bound_cycles == 0) {
    throw std::runtime_error(
        "minimum_causal_bound_cycles must be at least one");
  }
  _hybrid_parameters.on_idle_power_w =
      get_optional_nonnegative(config, "on_idle_power_w");
  _hybrid_parameters.controller_power_w =
      get_optional_nonnegative(config, "controller_power_w");
  _hybrid_parameters.retention = parse_state("retention");
  _hybrid_parameters.off = parse_state("off");
  _hybrid_parameters.source_path = path;
  if (!_hybrid_parameters.retention.enabled &&
      !_hybrid_parameters.off.enabled) {
    throw std::runtime_error(
        "hybrid SA config enables neither retention nor off");
  }

  auto state_energy_complete = [](const HybridStateParameters& state) {
    return !state.enabled ||
        (state.idle_power_w.has_value() &&
         state.entry_energy_j.has_value() &&
         state.wake_energy_j.has_value());
  };
  _hybrid_energy_complete =
      _config.core_freq > 0 &&
      _hybrid_parameters.on_idle_power_w.has_value() &&
      _hybrid_parameters.controller_power_w.has_value() &&
      state_energy_complete(_hybrid_parameters.retention) &&
      state_energy_complete(_hybrid_parameters.off);
}

cycle_type SystolicWS::hybrid_required_bound(HybridSaState state) const {
  const HybridStateParameters& parameters =
      hybrid_state_parameters(state);
  cycle_type required =
      _hybrid_parameters.minimum_causal_bound_cycles;
  required = MAX(required,
                 parameters.entry_latency_cycles +
                     parameters.wake_latency_cycles + 1);
  if (_hybrid_parameters.apply_break_even_filter) {
    required = MAX(required, parameters.break_even_cycles);
  }
  return required;
}

void SystolicWS::set_hybrid_state(HybridSaState state,
                                  const char* reason) {
  if (_hybrid_state == state) {
    return;
  }
  HybridSaState previous = _hybrid_state;
  _hybrid_state = state;
  if (state == HybridSaState::OFF) {
    _hybrid_pe_state_valid = false;
  }
  if (state == HybridSaState::RETENTION || state == HybridSaState::OFF) {
    _hybrid_episode_count[static_cast<size_t>(state)]++;
  }
  write_sa_event(json{{"event", "POWER_STATE"},
                      {"cycle", _core_cycle},
                      {"core_id", _id},
                      {"from", hybrid_state_name(previous)},
                      {"to", hybrid_state_name(state)},
                      {"reason", reason},
                      {"deadline", _hybrid_deadline}});
}

void SystolicWS::begin_hybrid_entry(HybridSaState target,
                                    cycle_type safe_deadline) {
  assert(target == HybridSaState::RETENTION ||
         target == HybridSaState::OFF);
  const HybridStateParameters& parameters =
      hybrid_state_parameters(target);
  _hybrid_deadline = safe_deadline;
  if (parameters.entry_energy_j.has_value()) {
    _hybrid_transition_energy_j += *parameters.entry_energy_j;
  }
  if (parameters.entry_latency_cycles == 0) {
    set_hybrid_state(target, "CAUSAL_BOUND");
    return;
  }
  _hybrid_entry_in_progress = true;
  _hybrid_entry_target = target;
  _hybrid_entry_complete_cycle =
      _core_cycle + parameters.entry_latency_cycles;
  write_sa_event(json{{"event", "POWER_ENTRY_START"},
                      {"cycle", _core_cycle},
                      {"core_id", _id},
                      {"target", hybrid_state_name(target)},
                      {"complete_cycle", _hybrid_entry_complete_cycle},
                      {"deadline", _hybrid_deadline}});
}

void SystolicWS::begin_hybrid_wake(const char* reason) {
  if (_hybrid_state == HybridSaState::ON) {
    if (_hybrid_entry_in_progress) {
      _hybrid_entry_in_progress = false;
      _hybrid_entry_cancel_count++;
      write_sa_event(json{{"event", "POWER_ENTRY_CANCEL"},
                          {"cycle", _core_cycle},
                          {"core_id", _id},
                          {"target", hybrid_state_name(_hybrid_entry_target)},
                          {"reason", reason}});
    }
    return;
  }
  if (_hybrid_wake_in_progress) {
    return;
  }
  HybridSaState source = _hybrid_state;
  const HybridStateParameters& parameters =
      hybrid_state_parameters(source);
  if (parameters.wake_energy_j.has_value()) {
    _hybrid_transition_energy_j += *parameters.wake_energy_j;
  }
  _hybrid_immediate_wake_count++;
  if (parameters.wake_latency_cycles == 0) {
    set_hybrid_state(HybridSaState::ON, reason);
    _hybrid_deadline = 0;
    return;
  }
  _hybrid_wake_in_progress = true;
  _hybrid_wake_complete_cycle =
      _core_cycle + parameters.wake_latency_cycles;
  write_sa_event(json{{"event", "POWER_WAKE_START"},
                      {"cycle", _core_cycle},
                      {"core_id", _id},
                      {"from", hybrid_state_name(source)},
                      {"reason", reason},
                      {"complete_cycle", _hybrid_wake_complete_cycle}});
  // Wake-transition cycles are conservatively charged as ON rather than as
  // useful low-power residency. The issue path remains blocked until the
  // configured completion cycle.
  set_hybrid_state(HybridSaState::ON, "WAKE_TRANSITION");
}

void SystolicWS::account_hybrid_cycle(bool sa_active) {
  _hybrid_state_cycles[static_cast<size_t>(_hybrid_state)]++;
  if (_config.core_freq == 0) {
    return;
  }
  long double cycle_seconds =
      1.0L / (static_cast<long double>(_config.core_freq) * 1.0e6L);
  if (!sa_active && _hybrid_parameters.on_idle_power_w.has_value()) {
    _hybrid_baseline_idle_energy_j +=
        *_hybrid_parameters.on_idle_power_w * cycle_seconds;
  }
  if (_hybrid_parameters.controller_power_w.has_value()) {
    _hybrid_controller_energy_j +=
        *_hybrid_parameters.controller_power_w * cycle_seconds;
  }
  if (sa_active) {
    return;
  }
  if (_hybrid_state == HybridSaState::ON) {
    if (_hybrid_parameters.on_idle_power_w.has_value()) {
      _hybrid_mode_energy_j +=
          *_hybrid_parameters.on_idle_power_w * cycle_seconds;
    }
  } else {
    const HybridStateParameters& parameters =
        hybrid_state_parameters(_hybrid_state);
    if (parameters.idle_power_w.has_value()) {
      _hybrid_mode_energy_j += *parameters.idle_power_w * cycle_seconds;
    }
  }
}

void SystolicWS::update_hybrid_policy(bool sa_active) {
  // Committed work is unsafe only for an OFF entry: OFF destroys the PE
  // state and preload activity may already have started.  Retention preserves
  // that state, so a future committed start may itself provide the causal
  // wake deadline.  Applying the OFF restriction to retention unnecessarily
  // discards causally bounded idle cycles.
  bool committed_work_invalidates_entry =
      _hybrid_entry_target == HybridSaState::OFF &&
      (!_compute_pipeline.empty() || !_ex_inst_queue.empty());
  if (_hybrid_entry_in_progress &&
      (sa_active || _hybrid_wake_requested ||
       committed_work_invalidates_entry)) {
    begin_hybrid_wake("ENTRY_INVALIDATED");
  }
  if (_hybrid_entry_in_progress &&
      _core_cycle >= _hybrid_entry_complete_cycle) {
    HybridSaState target = _hybrid_entry_target;
    _hybrid_entry_in_progress = false;
    set_hybrid_state(target, "ENTRY_COMPLETE");
  }
  if (_hybrid_wake_in_progress &&
      _core_cycle >= _hybrid_wake_complete_cycle) {
    _hybrid_wake_in_progress = false;
    set_hybrid_state(HybridSaState::ON, "WAKE_COMPLETE");
    _hybrid_deadline = 0;
  }

  // A deadline denotes the first cycle in which SA use may occur.  Process a
  // due wake before examining activity in that cycle; otherwise a zero-cycle
  // wake is incorrectly reported as late when SA_START occurs exactly at the
  // certified deadline.  Positive-latency wakes are still initiated early by
  // the same guard in the idle path below.
  if (_hybrid_state != HybridSaState::ON &&
      !_hybrid_wake_in_progress && _hybrid_deadline != 0) {
    cycle_type wake_latency =
        hybrid_state_parameters(_hybrid_state).wake_latency_cycles;
    if (_core_cycle + wake_latency >= _hybrid_deadline) {
      begin_hybrid_wake("WAKE_GUARD");
    }
  }

  if (sa_active) {
    if (_hybrid_state != HybridSaState::ON ||
        _hybrid_wake_in_progress) {
      _hybrid_deadline_violation_count++;
      _hybrid_wake_in_progress = false;
      set_hybrid_state(HybridSaState::ON, "OBSERVED_SA_ACTIVE");
    }
    _hybrid_entry_in_progress = false;
    _hybrid_wake_requested = false;
    account_hybrid_cycle(true);
    return;
  }

  if (_hybrid_wake_requested) {
    begin_hybrid_wake("SA_ISSUE_REQUEST");
    _hybrid_wake_requested = false;
  }

  PredictorSnapshot snapshot = compute_idle_predictor_snapshot();
  cycle_type safe_horizon = snapshot.safe_horizon;
  bool now1_cap_binding =
      snapshot.future_cap_limited &&
      snapshot.safe_horizon == _core_cycle + 1 &&
      snapshot.future_tile_cap == _core_cycle + 1;
  if (now1_cap_binding && !_scheduler_core_tile_waiting) {
    cycle_type issue_drain_cap =
        _core_cycle + _scheduler_global_issue_drain_bound;
    safe_horizon = snapshot.visible
        ? MIN(snapshot.visible_horizon, issue_drain_cap)
        : issue_drain_cap;
  }
  cycle_type remaining = safe_horizon > _core_cycle
      ? safe_horizon - _core_cycle : 0;

  if (snapshot.unknown) {
    _hybrid_unknown_bound_cycles++;
  }

  if (_hybrid_state != HybridSaState::ON) {
    if (snapshot.unknown || remaining == 0) {
      begin_hybrid_wake("BOUND_INVALID");
    } else {
      _hybrid_deadline = _hybrid_deadline == 0
          ? safe_horizon : MIN(_hybrid_deadline, safe_horizon);
      cycle_type wake_latency =
          hybrid_state_parameters(_hybrid_state).wake_latency_cycles;
      if (_core_cycle + wake_latency >= _hybrid_deadline) {
        begin_hybrid_wake("WAKE_GUARD");
      }
    }
    account_hybrid_cycle(false);
    return;
  }

  // Retention and OFF intentionally have different entry rules.  A positive
  // causal bound is sufficient for retention because PE state is preserved.
  // OFF eligibility below remains guarded by retention_safe_preload, which
  // requires empty committed structures and PRELOAD candidates for both
  // resident tiles.
  if (_hybrid_entry_in_progress || _hybrid_wake_in_progress ||
      snapshot.unknown || remaining == 0) {
    account_hybrid_cycle(false);
    return;
  }

  bool off_eligible = _hybrid_parameters.off.enabled &&
      snapshot.retention_safe_preload &&
      remaining >= hybrid_required_bound(HybridSaState::OFF);
  bool retention_eligible = _hybrid_parameters.retention.enabled &&
      remaining >= hybrid_required_bound(HybridSaState::RETENTION);
  if (_hybrid_parameters.off.enabled &&
      !snapshot.retention_safe_preload) {
    _hybrid_off_rejected_state_reuse++;
  }

  HybridSaState target = HybridSaState::ON;
  if (off_eligible &&
      (_hybrid_parameters.prefer_off_when_safe ||
       !retention_eligible)) {
    target = HybridSaState::OFF;
  } else if (retention_eligible) {
    target = HybridSaState::RETENTION;
  } else if (off_eligible) {
    target = HybridSaState::OFF;
  } else {
    _hybrid_insufficient_bound_cycles++;
  }
  if (target != HybridSaState::ON) {
    begin_hybrid_entry(target, safe_horizon);
  }
  account_hybrid_cycle(false);
}

void SystolicWS::emit_hybrid_summary() {
  if (!_hybrid_parameters.enabled || _hybrid_summary_emitted) {
    return;
  }
  _hybrid_summary_emitted = true;
  uint64_t low_power_cycles =
      _hybrid_state_cycles[static_cast<size_t>(HybridSaState::RETENTION)] +
      _hybrid_state_cycles[static_cast<size_t>(HybridSaState::OFF)];
  long double actual_energy = _hybrid_mode_energy_j +
      _hybrid_transition_energy_j + _hybrid_controller_energy_j;
  long double net_savings =
      _hybrid_baseline_idle_energy_j - actual_energy;
  const json record{
      {"version", 1},
      {"core_id", _id},
      {"configuration", _hybrid_parameters.source_path},
      {"mode", _hybrid_parameters.closed_loop ? "closed_loop" : "shadow"},
      {"control_path_modified", _hybrid_parameters.closed_loop},
      {"core_cycles", _core_cycle},
      {"on_cycles", _hybrid_state_cycles[
          static_cast<size_t>(HybridSaState::ON)]},
      {"retention_cycles", _hybrid_state_cycles[
          static_cast<size_t>(HybridSaState::RETENTION)]},
      {"off_cycles", _hybrid_state_cycles[
          static_cast<size_t>(HybridSaState::OFF)]},
      {"low_power_cycles", low_power_cycles},
      {"retention_episodes", _hybrid_episode_count[
          static_cast<size_t>(HybridSaState::RETENTION)]},
      {"off_episodes", _hybrid_episode_count[
          static_cast<size_t>(HybridSaState::OFF)]},
      {"entry_cancellations", _hybrid_entry_cancel_count},
      {"wake_events", _hybrid_immediate_wake_count},
      {"causal_deadline_violations", _hybrid_deadline_violation_count},
      {"wake_stall_cycles", _hybrid_wake_stall_cycles},
      {"off_rejected_state_reuse_observations",
       _hybrid_off_rejected_state_reuse},
      {"unknown_bound_cycles", _hybrid_unknown_bound_cycles},
      {"insufficient_bound_cycles", _hybrid_insufficient_bound_cycles},
      {"energy_known", _hybrid_energy_complete},
      {"baseline_on_idle_energy_j",
       _hybrid_energy_complete
           ? json(static_cast<double>(_hybrid_baseline_idle_energy_j))
           : json(nullptr)},
      {"mode_idle_energy_j",
       _hybrid_energy_complete
           ? json(static_cast<double>(_hybrid_mode_energy_j))
           : json(nullptr)},
      {"transition_energy_j",
       _hybrid_energy_complete
           ? json(static_cast<double>(_hybrid_transition_energy_j))
           : json(nullptr)},
      {"controller_energy_j",
       _hybrid_energy_complete
           ? json(static_cast<double>(_hybrid_controller_energy_j))
           : json(nullptr)},
      {"net_savings_j",
       _hybrid_energy_complete
           ? json(static_cast<double>(net_savings))
           : json(nullptr)},
      {"assumptions",
       {{"technology_values_are_user_supplied", true},
        {"break_even_filter_applied",
         _hybrid_parameters.apply_break_even_filter},
        {"off_requires_all_visible_resident_candidates_to_preload", true},
        {"retention_preserves_ordinary_gemm_state", true}}}};
  spdlog::info("SA_HYBRID_SUMMARY {}", record.dump());
}

bool SystolicWS::has_resident_gemm_candidate() const {
  if (!_ex_inst_queue.empty()) {
    Opcode opcode = _ex_inst_queue.front()->opcode;
    if (opcode == Opcode::GEMM || opcode == Opcode::GEMM_PRELOAD) {
      return true;
    }
  }

  for (const std::unique_ptr<Tile>& tile : _tiles) {
    for (const std::unique_ptr<Instruction>& inst : tile->instructions) {
      if (inst->opcode == Opcode::GEMM ||
          inst->opcode == Opcode::GEMM_PRELOAD) {
        return true;
      }
    }
  }
  return false;
}

Tile* SystolicWS::find_resident_tile(uint64_t tile_id) const {
  for (const std::unique_ptr<Tile>& resident : _tiles) {
    if (resident->measurement_tile_id == tile_id) {
      return resident.get();
    }
  }
  return nullptr;
}

void SystolicWS::begin_phase_measurement(bool vector_active) {
  _phase_measure = true;
  _phase_t0 = _core_cycle;
  _phase_t1_set = false;
  _phase_t2_set = false;
  _phase_t3_set = false;
  _phase_t3_classified = false;
  _phase_t1 = 0;
  _phase_t2 = 0;
  _phase_t3 = 0;
  _phase_old_tile_id =
      _tiles.size() == 1 ? _tiles.front()->measurement_tile_id : 0;
  _phase_next_tile_id = 0;
  _phase_pre_t0_committed = !_compute_pipeline.empty();
  _phase_admissions.clear();
  _phase_resident_count = _tiles.size();
  _phase_remaining_known = _tiles.size() == 1;
  _phase_remaining =
      _phase_remaining_known ? _tiles.front()->instructions.size() : 0;
  _phase_final_opcode_known =
      _tiles.size() == 1 && !_tiles.front()->instructions.empty();
  if (_phase_final_opcode_known) {
    _phase_final_opcode = _tiles.front()->instructions.back()->opcode;
  }
  _phase_vector_active = vector_active;
  _phase_core_tile_waiting = _scheduler_core_tile_waiting;
  _phase_t3_wait = PhaseWaitClass::UNKNOWN_OTHER;
}

void SystolicWS::observe_phase_events() {
  if (!_phase_measure) {
    return;
  }

  if (!_phase_t1_set && _phase_old_tile_id != 0 &&
      find_resident_tile(_phase_old_tile_id) == nullptr) {
    _phase_t1 = _core_cycle;
    _phase_t1_set = true;
  }

  for (PhaseAdmission& admission : _phase_admissions) {
    Tile* tile = find_resident_tile(admission.tile_id);
    if (admission.resident && !admission.retirement_set && tile == nullptr) {
      admission.retirement_set = true;
      admission.retirement_cycle = _core_cycle;
      admission.resident = false;
      continue;
    }

    if (admission.first_gemm_set || tile == nullptr ||
        tile->instructions.empty()) {
      continue;
    }

    const Instruction& front = *tile->instructions.front();
    if (front.opcode == Opcode::GEMM ||
        front.opcode == Opcode::GEMM_PRELOAD) {
      admission.first_gemm_set = true;
      admission.first_gemm_cycle = _core_cycle;
      if (admission.tile_id == _phase_next_tile_id && !_phase_t3_set) {
        _phase_t3 = _core_cycle;
        _phase_t3_set = true;
      }
    }
  }
}

void SystolicWS::classify_phase_t3() {
  if (!_phase_measure) {
    return;
  }

  for (PhaseAdmission& admission : _phase_admissions) {
    if (!admission.first_gemm_set || admission.wait_class_set) {
      continue;
    }

    Tile* tile = find_resident_tile(admission.tile_id);
    if (tile == nullptr || tile->instructions.empty()) {
      continue;
    }

    const Instruction& front = *tile->instructions.front();
    if (front.opcode != Opcode::GEMM &&
        front.opcode != Opcode::GEMM_PRELOAD) {
      continue;
    }

    admission.wait_class = classify_gemm_operands(front, *tile);
    admission.wait_class_set = true;
    capture_response_drain(front, *tile, admission);
    freeze_required_request_cohort(admission);
    if (admission.tile_id == _phase_next_tile_id) {
      _phase_t3_wait = admission.wait_class;
      _phase_t3_classified = true;
    }
  }
}

void SystolicWS::complete_phase_measurement() {
  cycle_type t4 = _core_cycle;
  cycle_type idle_length = t4 - _phase_t0;

  if (idle_length >= SA_PHASE_MIN_IDLE) {
    json record{{"version", 2},
                {"core_id", _id},
                {"T0", _phase_t0},
                {"T1", nullptr},
                {"T2", nullptr},
                {"T3", nullptr},
                {"T4", t4},
                {"actual_idle_length", idle_length},
                {"resident_count", _phase_resident_count},
                {"remaining", nullptr},
                {"final_opcode", nullptr},
                {"vector_active", _phase_vector_active},
                {"core_tile_waiting", _phase_core_tile_waiting},
                {"T3_wait", nullptr},
                {"admissions", json::array()},
                {"first_gemms", json::array()},
                {"retirements", json::array()},
                {"causing_tile_id", nullptr},
                {"causing_opcode", nullptr},
                {"response_ports",
                 _config.icnt_injection_ports_per_core},
                {"request_dequeue_ports",
                 _config.icnt_injection_ports_per_core},
                {"dram_channels", _config.dram_channels}};

    if (_phase_t1_set) record["T1"] = _phase_t1;
    if (_phase_t2_set) record["T2"] = _phase_t2;
    if (_phase_t3_set) record["T3"] = _phase_t3;
    if (_phase_remaining_known) record["remaining"] = _phase_remaining;
    if (_phase_final_opcode_known) {
      record["final_opcode"] = opcode_name(_phase_final_opcode);
    }
    if (_phase_t3_classified) {
      switch (_phase_t3_wait) {
        case PhaseWaitClass::READY:
          record["T3_wait"] = "READY";
          break;
        case PhaseWaitClass::MEMORY_WAIT:
          record["T3_wait"] = "MEMORY_WAIT";
          break;
        case PhaseWaitClass::COMPUTE_DEPENDENCY:
          record["T3_wait"] = "COMPUTE_DEPENDENCY";
          break;
        case PhaseWaitClass::UNKNOWN_OTHER:
          record["T3_wait"] = "UNKNOWN/OTHER";
          break;
      }
    }

    auto wait_class_name = [](PhaseWaitClass wait_class) {
      switch (wait_class) {
        case PhaseWaitClass::READY: return "READY";
        case PhaseWaitClass::MEMORY_WAIT: return "MEMORY_WAIT";
        case PhaseWaitClass::COMPUTE_DEPENDENCY:
          return "COMPUTE_DEPENDENCY";
        case PhaseWaitClass::UNKNOWN_OTHER: return "UNKNOWN/OTHER";
      }
      return "UNKNOWN/OTHER";
    };

    for (const PhaseAdmission& admission : _phase_admissions) {
      record["admissions"].push_back(
          json{{"tile_id", admission.tile_id},
               {"admission_cycle", admission.admission_cycle},
               {"layer_id", admission.layer_id},
               {"operation", admission.operation},
               {"skip", admission.skip},
               {"contains_gemm", admission.contains_gemm}});
      if (admission.first_gemm_set) {
        json first_gemm{{"tile_id", admission.tile_id},
                        {"cycle", admission.first_gemm_cycle},
                        {"wait_class",
                         admission.wait_class_set
                             ? json(wait_class_name(admission.wait_class))
                             : json(nullptr)},
                        {"response_drain_cycle", nullptr},
                        {"pending_memory_by_required_source", json::array()},
                        {"total_unique_required_pending_memory", nullptr},
                        {"request_cohort_frozen", false},
                        {"request_cohort_exact", false},
                        {"request_cohort_matched", nullptr},
                        {"request_cohort_completed", nullptr},
                        {"request_cohort_stage_counts", nullptr},
                        {"core_request_queue_depth", nullptr},
                        {"fifo_position_exact", false},
                        {"tailmost_required_fifo_position_1based", nullptr},
                        {"channel_counts_exact", false},
                        {"required_not_yet_dram_by_channel", nullptr},
                        {"last_required_dram_enter_cycle", nullptr},
                        {"last_required_dram_exit_cycle", nullptr},
                        {"last_required_core_arrival_cycle", nullptr}};
        if (admission.response_drain_set) {
          first_gemm["response_drain_cycle"] =
              admission.response_drain_cycle;
          first_gemm["total_unique_required_pending_memory"] =
              admission.total_unique_required_pending_memory;
          for (const PhaseRequiredSource& source :
               admission.required_sources) {
            first_gemm["pending_memory_by_required_source"].push_back(
                json{{"domain", source.accum ? "accum" : "spad"},
                     {"buffer", source.buffer_id},
                     {"address", source.address},
                     {"pending_memory", source.pending_memory}});
          }
          first_gemm["request_cohort_frozen"] =
              admission.request_cohort_frozen;
          first_gemm["request_cohort_exact"] =
              admission.request_cohort_exact;
          first_gemm["request_cohort_matched"] =
              admission.request_cohort_matched;
          first_gemm["request_cohort_completed"] =
              admission.request_cohort_completed;
          first_gemm["request_cohort_stage_counts"] =
              json{{"CORE_REQUEST_QUEUE", admission.cohort_core_queue},
                   {"ICNT_TO_MEMORY", admission.cohort_icnt_to_memory},
                   {"DRAM", admission.cohort_dram},
                   {"ICNT_TO_CORE", admission.cohort_icnt_to_core}};
          first_gemm["core_request_queue_depth"] =
              admission.core_request_queue_depth;
          first_gemm["fifo_position_exact"] =
              admission.fifo_position_exact;
          if (admission.fifo_position_exact) {
            first_gemm["tailmost_required_fifo_position_1based"] =
                admission.tailmost_required_fifo_position_1based;
          }
          first_gemm["channel_counts_exact"] =
              admission.channel_counts_exact;
          if (admission.channel_counts_exact) {
            first_gemm["required_not_yet_dram_by_channel"] =
                admission.required_not_yet_dram_by_channel;
          }
          if (admission.request_cohort_exact &&
              admission.request_cohort_matched > 0 &&
              admission.request_cohort_completed ==
                  admission.request_cohort_matched) {
            first_gemm["last_required_dram_enter_cycle"] =
                admission.last_required_dram_enter_cycle;
            first_gemm["last_required_dram_exit_cycle"] =
                admission.last_required_dram_exit_cycle;
            first_gemm["last_required_core_arrival_cycle"] =
                admission.last_required_core_arrival_cycle;
          }
        }
        record["first_gemms"].push_back(std::move(first_gemm));
      }
      if (admission.retirement_set) {
        record["retirements"].push_back(
            json{{"tile_id", admission.tile_id},
                 {"retirement_cycle", admission.retirement_cycle}});
      }
    }

    if (!_compute_pipeline.empty()) {
      record["causing_opcode"] =
          opcode_name(_compute_pipeline.front()->opcode);
      if (_phase_pre_t0_committed) {
        record["causing_tile_id"] = "pre_T0";
      } else if (_compute_pipeline.front()->measurement_tile_id != 0) {
        record["causing_tile_id"] =
            _compute_pipeline.front()->measurement_tile_id;
      }
    }

    spdlog::info("SA_PHASE_MEASURE {}", record.dump());
  }

  _phase_measure = false;
  _phase_old_tile_id = 0;
  _phase_next_tile_id = 0;
  _phase_admissions.clear();
}

SystolicWS::PhaseWaitClass SystolicWS::classify_gemm_operands(
    const Instruction& inst, const Tile& tile) const {
  bool saw_memory = false;
  bool saw_compute = false;
  bool saw_unknown = false;

  for (addr_type address : inst.src_addrs) {
    bool accum = inst.src_from_accum && address >= ACCUM_SPAD_BASE;
    const Sram& buffer = accum ? _acc_spad : _spad;
    int buffer_id = accum ? tile.accum_spad_id : tile.spad_id;
    Sram::EntryState state = buffer.get_entry_state(address, buffer_id);

    if (state.allocated && state.valid) {
      continue;
    }
    if (!state.allocated) {
      saw_unknown = true;
      continue;
    }
    if (state.pending_memory_count > 0 &&
        state.pending_compute_count == 0) {
      saw_memory = true;
    } else if (state.pending_compute_count > 0 &&
               state.pending_memory_count == 0) {
      saw_compute = true;
    } else {
      saw_unknown = true;
    }
  }

  if (saw_unknown || (saw_memory && saw_compute)) {
    return PhaseWaitClass::UNKNOWN_OTHER;
  }
  if (saw_memory) {
    return PhaseWaitClass::MEMORY_WAIT;
  }
  if (saw_compute) {
    return PhaseWaitClass::COMPUTE_DEPENDENCY;
  }
  return PhaseWaitClass::READY;
}

void SystolicWS::capture_response_drain(const Instruction& inst,
                                        const Tile& tile,
                                        PhaseAdmission& admission) const {
  admission.response_drain_set = true;
  admission.response_drain_cycle = _core_cycle;
  admission.total_unique_required_pending_memory = 0;
  admission.required_sources.clear();

  for (addr_type address : inst.src_addrs) {
    bool accum = inst.src_from_accum && address >= ACCUM_SPAD_BASE;
    int buffer_id = accum ? tile.accum_spad_id : tile.spad_id;

    bool duplicate = false;
    for (const PhaseRequiredSource& source : admission.required_sources) {
      if (source.accum == accum && source.buffer_id == buffer_id &&
          source.address == address) {
        duplicate = true;
        break;
      }
    }
    if (duplicate) {
      continue;
    }

    const Sram& buffer = accum ? _acc_spad : _spad;
    Sram::EntryState state = buffer.get_entry_state(address, buffer_id);
    admission.required_sources.push_back(
        PhaseRequiredSource{.accum = accum,
                            .buffer_id = buffer_id,
                            .address = address,
                            .pending_memory =
                                state.pending_memory_count});
    admission.total_unique_required_pending_memory +=
        state.pending_memory_count;
  }
}

void SystolicWS::freeze_required_request_cohort(
    PhaseAdmission& admission) const {
  admission.request_cohort_frozen = true;
  admission.request_cohort_exact = false;
  admission.request_cohort_matched = 0;
  admission.request_cohort_completed = 0;
  admission.cohort_core_queue = 0;
  admission.cohort_icnt_to_memory = 0;
  admission.cohort_dram = 0;
  admission.cohort_icnt_to_core = 0;
  admission.core_request_queue_depth = _request_queue.size();
  admission.fifo_position_exact = false;
  admission.tailmost_required_fifo_position_1based = 0;
  admission.channel_counts_exact = false;
  admission.required_not_yet_dram_by_channel.assign(
      _config.dram_channels, 0);
  admission.request_cohort_ids.clear();

  bool channel_mapping_valid =
      _config.dram_channels > 0 && _config.dram_req_size > 0;

  for (const auto& [request_id, access] : _outstanding_memory_accesses) {
    if (access->write || access->core_id != _id) {
      continue;
    }

    bool required = false;
    for (const PhaseRequiredSource& source : admission.required_sources) {
      if (access->spad_address == source.address &&
          access->buffer_id == source.buffer_id &&
          (access->spad_address >= ACCUM_SPAD_BASE) == source.accum) {
        required = true;
        break;
      }
    }
    if (!required) {
      continue;
    }

    admission.request_cohort_ids.insert(request_id);
    admission.request_cohort_matched++;
    bool not_yet_in_dram =
        access->stage == MemoryStage::CORE_REQUEST_QUEUE ||
        access->stage == MemoryStage::ICNT_TO_MEMORY;
    if (not_yet_in_dram && channel_mapping_valid) {
      uint32_t channel = Dram::map_channel_id(
          access->dram_address, _config.dram_req_size,
          _config.dram_channels);
      if (channel < admission.required_not_yet_dram_by_channel.size()) {
        admission.required_not_yet_dram_by_channel[channel]++;
      } else {
        channel_mapping_valid = false;
      }
    }
    switch (access->stage) {
      case MemoryStage::CORE_REQUEST_QUEUE:
        admission.cohort_core_queue++;
        break;
      case MemoryStage::ICNT_TO_MEMORY:
        admission.cohort_icnt_to_memory++;
        break;
      case MemoryStage::DRAM:
        admission.cohort_dram++;
        break;
      case MemoryStage::ICNT_TO_CORE:
        admission.cohort_icnt_to_core++;
        break;
      case MemoryStage::DONE:
        break;
    }
  }

  admission.request_cohort_exact =
      admission.request_cohort_matched ==
      admission.total_unique_required_pending_memory;
  if (!admission.request_cohort_exact) {
    admission.request_cohort_ids.clear();
    return;
  }

  size_t channel_count_sum = 0;
  for (size_t count : admission.required_not_yet_dram_by_channel) {
    channel_count_sum += count;
  }
  admission.channel_counts_exact =
      channel_mapping_valid &&
      channel_count_sum ==
          admission.cohort_core_queue + admission.cohort_icnt_to_memory;

  std::queue<MemoryAccess*> queue_snapshot = _request_queue;
  size_t position = 0;
  size_t required_in_queue = 0;
  while (!queue_snapshot.empty()) {
    position++;
    MemoryAccess* access = queue_snapshot.front();
    queue_snapshot.pop();
    if (admission.request_cohort_ids.find(access->id) !=
        admission.request_cohort_ids.end()) {
      required_in_queue++;
      admission.tailmost_required_fifo_position_1based = position;
    }
  }

  admission.fifo_position_exact =
      required_in_queue > 0 &&
      required_in_queue == admission.cohort_core_queue;
}

void SystolicWS::observe_required_response(const MemoryAccess& response) {
  if (!_phase_measure || response.write) {
    return;
  }

  for (PhaseAdmission& admission : _phase_admissions) {
    if (!admission.request_cohort_exact ||
        admission.request_cohort_ids.find(response.id) ==
            admission.request_cohort_ids.end()) {
      continue;
    }

    cycle_type dram_enter_cycle =
        response.start_cycle + response.core_queue_cycles +
        response.icnt_to_mem_cycles;
    cycle_type dram_exit_cycle =
        dram_enter_cycle + response.dram_cycles;
    admission.last_required_dram_enter_cycle =
        MAX(admission.last_required_dram_enter_cycle, dram_enter_cycle);
    admission.last_required_dram_exit_cycle =
        MAX(admission.last_required_dram_exit_cycle, dram_exit_cycle);
    admission.last_required_core_arrival_cycle =
        MAX(admission.last_required_core_arrival_cycle, _core_cycle);
    admission.request_cohort_completed++;
  }
}

SystolicWS::ControllerDescriptor SystolicWS::controller_descriptor(
    const Tile* tile) const {
  constexpr uint64_t FIELD_MAX = (uint64_t{1} << 20) - 1;
  ControllerDescriptor descriptor{};
  if (tile == nullptr) {
    return descriptor;
  }

  descriptor[0] = 1;  // present
  descriptor[5] = MIN(static_cast<uint64_t>(tile->instructions.size()),
                      FIELD_MAX);

  const Instruction* candidate = tile->next_sa_instruction;
  uint64_t prefix = tile->next_sa_prefix_count;
  if (candidate == nullptr) {
    assert(!tile->next_sa_candidate_found);
    return descriptor;
  }
  assert(tile->next_sa_candidate_found);
  assert(candidate->opcode == tile->next_sa_opcode);

  descriptor[1] = 1;  // candidate_found
  descriptor[2] = candidate->opcode == Opcode::GEMM_PRELOAD;
  descriptor[3] = 1;  // sources_supported unless disproved below
  descriptor[4] = MIN(prefix, FIELD_MAX);

  struct SourceKey {
    bool accum;
    int buffer_id;
    addr_type address;
  };
  std::vector<SourceKey> unique_sources;
  for (addr_type address : candidate->src_addrs) {
    bool accum = candidate->src_from_accum &&
                 address >= ACCUM_SPAD_BASE;
    int buffer_id = accum ? tile->accum_spad_id : tile->spad_id;
    bool duplicate = false;
    for (const SourceKey& source : unique_sources) {
      if (source.accum == accum && source.buffer_id == buffer_id &&
          source.address == address) {
        duplicate = true;
        break;
      }
    }
    if (!duplicate) {
      unique_sources.push_back(SourceKey{accum, buffer_id, address});
    }
  }

  if (unique_sources.size() > 2) {
    descriptor[3] = 0;
    return descriptor;
  }

  for (size_t i = 0; i < unique_sources.size(); ++i) {
    const SourceKey& source = unique_sources[i];
    const Sram& buffer = source.accum ? _acc_spad : _spad;
    Sram::EntryState state =
        buffer.get_entry_state(source.address, source.buffer_id);
    uint64_t bit = uint64_t{1} << i;
    descriptor[6] |= bit;  // operand_used
    if (state.allocated) descriptor[7] |= bit;
    if (state.valid) descriptor[8] |= bit;
    descriptor[12 + i] =
        MIN(static_cast<uint64_t>(state.pending_memory_count), FIELD_MAX);
    if (state.valid) {
      continue;
    }

    // The RTL modulo subtraction is safe only for a still-queued tail ticket
    // strictly ahead of the dequeue frontier and within its 20-bit window.
    // Any other invalid state is represented as unsupported, never guessed.
    if (!state.allocated || state.pending_memory_count == 0 ||
        state.memory_tail_core_queue_ticket <=
            _last_dequeued_core_queue_ticket) {
      descriptor[3] = 0;
      continue;
    }
    uint64_t position = state.memory_tail_core_queue_ticket -
                        _last_dequeued_core_queue_ticket;
    if (position > FIELD_MAX) {
      descriptor[3] = 0;
      continue;
    }

    descriptor[9] |= bit;  // FIFO-certifiable pending memory
    descriptor[10 + i] = state.memory_tail_core_queue_ticket & FIELD_MAX;
  }

  return descriptor;
}

void SystolicWS::write_controller_trace_record(
    cycle_type cycle, bool final,
    const ControllerTraceInputs& inputs) {
  _controller_trace << cycle << ',' << (final ? 1 : 0);
  for (uint64_t value : inputs) {
    _controller_trace << ',' << value;
  }
  _controller_trace << '\n';
}

void SystolicWS::trace_controller_inputs(bool sa_active) {
  if (!_controller_trace.is_open()) {
    return;
  }

  constexpr uint64_t FIELD_MAX = (uint64_t{1} << 20) - 1;
  auto saturate = [](uint64_t value) {
    constexpr uint64_t MAX_VALUE = (uint64_t{1} << 20) - 1;
    return MIN(value, MAX_VALUE);
  };

  ControllerDescriptor tile0 =
      controller_descriptor(_tiles.size() > 0 ? _tiles[0].get() : nullptr);
  ControllerDescriptor tile1 =
      controller_descriptor(_tiles.size() > 1 ? _tiles[1].get() : nullptr);
  // Fields 0..11 describe tile/candidate/source semantics and FIFO metadata.
  // Fields 12..13 expose the existing SRAM completion counters directly.
  // Their monotonic progress is traced separately and does not invalidate an
  // already saved deadline, which already assumes maximum-rate drain.
  auto semantic_changed = [](const ControllerDescriptor& current,
                             const ControllerDescriptor& previous) {
    return !std::equal(current.begin(), current.begin() + 12,
                       previous.begin());
  };
  auto count_changed = [](const ControllerDescriptor& current,
                          const ControllerDescriptor& previous) {
    return current[12] != previous[12] ||
           current[13] != previous[13];
  };
  bool tile0_we = semantic_changed(tile0, _controller_trace_descriptors[0]);
  bool tile1_we = semantic_changed(tile1, _controller_trace_descriptors[1]);
  bool tile0_count_we = !tile0_we &&
      count_changed(tile0, _controller_trace_descriptors[0]);
  bool tile1_count_we = !tile1_we &&
      count_changed(tile1, _controller_trace_descriptors[1]);

  ControllerTraceInputs inputs{};
  size_t field = 0;
  bool compute_front_future_valid =
      !_compute_pipeline.empty() &&
      _compute_pipeline.front()->start_cycle > _core_cycle;
  uint64_t compute_front_remaining = compute_front_future_valid
      ? _compute_pipeline.front()->start_cycle - _core_cycle
      : 0;
  bool ex_front_is_sa = false;
  if (!_ex_inst_queue.empty()) {
    Opcode opcode = _ex_inst_queue.front()->opcode;
    ex_front_is_sa =
        opcode == Opcode::GEMM || opcode == Opcode::GEMM_PRELOAD;
  }
  bool specialized_fifo_configuration =
      _config.icnt_injection_ports_per_core == 16 &&
      _config.core_freq == _config.icnt_freq;

  inputs[field++] = sa_active;
  inputs[field++] = _compute_pipeline.empty();
  inputs[field++] = compute_front_future_valid;
  inputs[field++] = saturate(compute_front_remaining);
  inputs[field++] = _ex_inst_queue.empty();
  inputs[field++] = ex_front_is_sa;
  inputs[field++] = _scheduler_core_tile_waiting;
  inputs[field++] = saturate(_scheduler_global_issue_drain_bound);
  inputs[field++] = _core_queue_tickets_exact &&
                    specialized_fifo_configuration;
  inputs[field++] = _last_dequeued_core_queue_ticket & FIELD_MAX;

  inputs[field++] = tile0_we;
  inputs[field++] = tile0_count_we;
  for (uint64_t value : tile0) inputs[field++] = value;
  inputs[field++] = tile1_we;
  inputs[field++] = tile1_count_we;
  for (uint64_t value : tile1) inputs[field++] = value;

  if (field != inputs.size()) {
    throw std::logic_error("SA controller trace field-count mismatch");
  }

  if (!_controller_trace_have_inputs ||
      inputs != _controller_trace_last_inputs) {
    write_controller_trace_record(_core_cycle, false, inputs);
    _controller_trace_last_inputs = inputs;
    _controller_trace_have_inputs = true;
  }
  _controller_trace_descriptors[0] = tile0;
  _controller_trace_descriptors[1] = tile1;
}

SystolicWS::PredictorCandidate SystolicWS::predictor_tile_candidate(
    const Tile& tile) const {
  PredictorCandidate candidate;
  if (!tile.next_sa_candidate_found) {
    assert(tile.next_sa_instruction == nullptr);
    return candidate;
  }

  assert(tile.next_sa_instruction != nullptr);
  candidate.found = true;
  candidate.opcode = tile.next_sa_opcode;
  if (tile.next_sa_prefix_count >
      std::numeric_limits<cycle_type>::max() - _core_cycle) {
    candidate.unknown = true;
    return candidate;
  }
  candidate.prefix_horizon = _core_cycle + tile.next_sa_prefix_count;
  candidate.operand_horizon = _core_cycle;
  candidate.resource_horizon = _core_cycle;
  candidate.horizon = candidate.prefix_horizon;
  if (_updated_predictor_enabled) {
    if (tile.next_sa_descriptor_index >= tile.sa_descriptors.size()) {
      candidate.unknown = true;
      return candidate;
    }
    bool operand_certified = false;
    bool operand_unknown = false;
    candidate.operand_horizon = updated_predictor_operand_horizon(
        tile, tile.sa_descriptors[tile.next_sa_descriptor_index],
        operand_certified, operand_unknown);
    if (!_compute_pipeline.empty()) {
      // The native WS scheduler pipelines from the last committed wave's
      // start, not from the front wave's finish.
      const Instruction& back = *_compute_pipeline.back();
      uint64_t offset = std::max<uint64_t>(back.compute_size, 4);
      if (candidate.opcode == Opcode::GEMM_PRELOAD) {
        offset = std::max<uint64_t>(
            offset, _config.core_config[_id].core_height);
      }
      if (offset > std::numeric_limits<cycle_type>::max() -
                       back.start_cycle) {
        candidate.unknown = true;
      } else {
        candidate.resource_horizon = std::max(
            _core_cycle, back.start_cycle + offset);
      }
    }
    candidate.horizon = std::max(
        {candidate.prefix_horizon, candidate.operand_horizon,
         candidate.resource_horizon});
    candidate.fifo_certified = operand_certified;
    candidate.unknown = candidate.unknown || operand_unknown;
    return candidate;
  }
  if (tile.next_sa_prefix_count == 0) {
    bool fifo_certified = false;
    bool unknown = false;
    cycle_type fifo_horizon = predictor_fifo_horizon(
        *tile.next_sa_instruction, tile.spad_id, tile.accum_spad_id,
        fifo_certified, unknown);
    candidate.horizon = MAX(candidate.horizon, fifo_horizon);
    candidate.fifo_certified = fifo_certified;
    candidate.unknown = unknown;
  }
  return candidate;
}

cycle_type SystolicWS::updated_predictor_operand_horizon(
    const Tile& tile, const SaInstructionDescriptor& descriptor,
    bool& certified, bool& unknown) const {
  certified = false;
  unknown = false;
  cycle_type horizon = _core_cycle;
  if (descriptor.instruction == nullptr || !_core_queue_tickets_exact ||
      !tile.issued_planned_memory_requests_valid) {
    unknown = true;
    return horizon;
  }

  auto extend = [&](uint64_t left, uint64_t right,
                    cycle_type& result) {
    if (right > std::numeric_limits<cycle_type>::max() - left) {
      unknown = true;
      return false;
    }
    result = left + right;
    return true;
  };
  auto injection_delay = [&](uint64_t position,
                             cycle_type& delay) {
    const uint64_t ports = _config.icnt_injection_ports_per_core;
    if (position == 0 || ports == 0 || _config.core_freq == 0 ||
        _config.icnt_freq == 0) {
      unknown = true;
      return false;
    }
    const uint64_t core_period = 1000000 / _config.core_freq;
    const uint64_t icnt_period = 1000000 / _config.icnt_freq;
    if (core_period == 0 || icnt_period == 0) {
      unknown = true;
      return false;
    }
    const uint64_t events = 1 + (position - 1) / ports;
    const unsigned __int128 scaled =
        static_cast<unsigned __int128>(events - 1) * icnt_period;
    const unsigned __int128 cycles = scaled / core_period;
    if (cycles > std::numeric_limits<cycle_type>::max()) {
      unknown = true;
      return false;
    }
    delay = static_cast<cycle_type>(cycles);
    return true;
  };

  const Instruction& inst = *descriptor.instruction;
  std::set<addr_type> seen;
  const size_t current_ordinal = tile.instructions.empty()
      ? descriptor.instruction_ordinal
      : tile.instructions.front()->tile_instruction_ordinal;
  for (addr_type address : inst.src_addrs) {
    if (!seen.insert(address).second) {
      continue;
    }
    const bool accum = inst.src_from_accum &&
                       address >= ACCUM_SPAD_BASE;
    const int buffer_id = accum ? tile.accum_spad_id : tile.spad_id;
    const Sram& buffer = accum ? _acc_spad : _spad;
    const Sram::EntryState state =
        buffer.get_entry_state(address, buffer_id);
    if (state.allocated && state.valid) {
      continue;
    }
    if (state.allocated) {
      if (state.pending_compute_count != 0) {
        unknown = true;
      }
      if (state.pending_memory_count != 0) {
        if (_response_drain_ports_per_core_cycle_bound == 0) {
          unknown = true;
        } else {
          const cycle_type response_cycles =
              (state.pending_memory_count - 1) /
              _response_drain_ports_per_core_cycle_bound;
          cycle_type bound;
          if (extend(_core_cycle, response_cycles, bound)) {
            horizon = std::max(horizon, bound);
            certified = true;
          }
        }
        if (state.memory_tail_core_queue_ticket >
            _last_dequeued_core_queue_ticket) {
          const uint64_t position =
              state.memory_tail_core_queue_ticket -
              _last_dequeued_core_queue_ticket;
          cycle_type delay, bound;
          if (injection_delay(position, delay) &&
              extend(_core_cycle, delay, bound)) {
            horizon = std::max(horizon, bound);
            certified = true;
          }
        }
      } else if (!state.valid && state.pending_compute_count == 0) {
        unknown = true;
      }
      continue;
    }

    const auto planned = std::find_if(
        descriptor.planned_memory_sources.begin(),
        descriptor.planned_memory_sources.end(),
        [address](const auto& source) {
          return source.address == address;
        });
    if (planned == descriptor.planned_memory_sources.end() || accum) {
      unknown = true;
      continue;
    }
    if (planned->request_count == 0 ||
        planned->cumulative_request_count_through_producer <
            tile.issued_planned_memory_requests) {
      unknown = true;
      continue;
    }
    // Already-queued requests, the staged load, and every as-yet-unissued
    // MOVIN through this producer precede its final required request.
    uint64_t position = _request_queue.size();
    if (!_ld_inst_queue.empty() &&
        _ld_inst_queue.front()->opcode == Opcode::MOVIN) {
      if (!extend(position, _ld_inst_queue.front()->src_addrs.size(),
                  position)) continue;
    }
    const uint64_t remaining =
        planned->cumulative_request_count_through_producer -
        tile.issued_planned_memory_requests;
    if (!extend(position, remaining, position)) continue;
    cycle_type delay, fifo_bound;
    if (injection_delay(position, delay) &&
        extend(_core_cycle, delay, fifo_bound)) {
      horizon = std::max(horizon, fifo_bound);
      certified = true;
    }
    // A tile instruction issues in Core::cycle; the staged load is consumed
    // by SystolicWS on the following cycle. Include both issue stages.
    cycle_type issue_bound;
    if (planned->producer_instruction_ordinal >= current_ordinal) {
      const uint64_t ordinal_gap =
          planned->producer_instruction_ordinal - current_ordinal;
      if (extend(_core_cycle, ordinal_gap, issue_bound) &&
          extend(issue_bound, 2, issue_bound)) {
        horizon = std::max(horizon, issue_bound);
        certified = true;
      }
    }
  }
  return horizon;
}

cycle_type SystolicWS::predictor_fifo_horizon(
    const Instruction& inst, int spad_id, int accum_spad_id,
    bool& certified, bool& unknown) const {
  certified = false;
  unknown = false;
  cycle_type horizon = _core_cycle;
  std::vector<PhaseRequiredSource> unique_sources;

  if (!_core_queue_tickets_exact) {
    unknown = true;
    return horizon;
  }

  for (addr_type address : inst.src_addrs) {
    bool accum = inst.src_from_accum && address >= ACCUM_SPAD_BASE;
    int buffer_id = accum ? accum_spad_id : spad_id;

    bool duplicate = false;
    for (const PhaseRequiredSource& source : unique_sources) {
      if (source.accum == accum && source.buffer_id == buffer_id &&
          source.address == address) {
        duplicate = true;
        break;
      }
    }
    if (duplicate) {
      continue;
    }
    unique_sources.push_back(
        PhaseRequiredSource{.accum = accum,
                            .buffer_id = buffer_id,
                            .address = address});

    const Sram& buffer = accum ? _acc_spad : _spad;
    Sram::EntryState state = buffer.get_entry_state(address, buffer_id);
    if (state.allocated && state.valid) {
      continue;
    }
    if (!state.allocated) {
      unknown = true;
      continue;
    }

    if (state.pending_memory_count > 0) {
      bool have_response_bound =
          _response_drain_ports_per_core_cycle_bound > 0;
      if (have_response_bound) {
        cycle_type response_cycles =
            (state.pending_memory_count - 1) /
            _response_drain_ports_per_core_cycle_bound;
        horizon = MAX(horizon, _core_cycle + response_cycles);
        certified = true;
      }

      uint64_t tail_ticket = state.memory_tail_core_queue_ticket;
      if (tail_ticket > _last_dequeued_core_queue_ticket) {
        uint64_t position =
            tail_ticket - _last_dequeued_core_queue_ticket;
        uint64_t ports = _config.icnt_injection_ports_per_core;
        if (ports == 0 || _config.core_freq == 0 ||
            _config.icnt_freq == 0) {
          unknown = true;
          continue;
        }

        cycle_type core_period = 1000000 / _config.core_freq;
        cycle_type icnt_period = 1000000 / _config.icnt_freq;
        if (core_period == 0 || icnt_period == 0) {
          unknown = true;
          continue;
        }

        uint64_t service_events = (position + ports - 1) / ports;
        cycle_type safe_cycles =
            ((service_events > 0 ? service_events - 1 : 0) *
             icnt_period) /
            core_period;
        horizon = MAX(horizon, _core_cycle + safe_cycles);
        certified = true;
      } else if (!have_response_bound) {
        // Once the final required request has left the core FIFO, the
        // outstanding-response count is the remaining causal certificate.
        // It is unknown only when the evaluated configuration provides no
        // conservative response-service bound.
        unknown = true;
      }
    }

    if (!state.valid && state.pending_memory_count == 0) {
      unknown = true;
    }
  }

  return horizon;
}

cycle_type SystolicWS::predictor_future_tile_cap(
    const std::vector<PredictorCandidate>& tile_candidates,
    bool& unknown) const {
  unknown = false;

  // The scheduler has already attempted admission for this core cycle.
  // With a free resident slot, an unknown tile can be admitted next cycle.
  if (_tiles.size() < 2) {
    if (_core_cycle == std::numeric_limits<cycle_type>::max()) {
      unknown = true;
      return _core_cycle;
    }
    return _core_cycle + 1;
  }
  if (_tiles.size() != 2 || tile_candidates.size() != _tiles.size()) {
    unknown = true;
    return _core_cycle;
  }

  cycle_type cap = std::numeric_limits<cycle_type>::max();
  for (size_t i = 0; i < _tiles.size(); ++i) {
    size_t remaining = _tiles[i]->instructions.size();
    const size_t steps = MAX(static_cast<size_t>(1), remaining);
    if (steps > std::numeric_limits<cycle_type>::max() - _core_cycle) {
      unknown = true;
      return _core_cycle;
    }
    cycle_type retirement_lower_bound = _core_cycle + steps;
    if (tile_candidates[i].found) {
      retirement_lower_bound =
          MAX(retirement_lower_bound, tile_candidates[i].horizon);
    }
    cap = MIN(cap, retirement_lower_bound);
  }
  return cap;
}

SystolicWS::PredictorSnapshot
SystolicWS::compute_idle_predictor_snapshot() const {
  const cycle_type inf = std::numeric_limits<cycle_type>::max();
  PredictorSnapshot snapshot;
  snapshot.safe_horizon = _core_cycle;
  snapshot.visible_horizon = inf;

  bool selected_fifo = false;
  auto consider_candidate = [&](cycle_type horizon, bool fifo) {
    if (horizon < snapshot.visible_horizon) {
      snapshot.visible_horizon = horizon;
      selected_fifo = fifo;
    } else if (horizon == snapshot.visible_horizon) {
      selected_fifo = selected_fifo && fifo;
    }
  };

  if (!_compute_pipeline.empty() &&
      _compute_pipeline.front()->start_cycle > _core_cycle) {
    consider_candidate(_compute_pipeline.front()->start_cycle, false);
  }

  if (!_ex_inst_queue.empty()) {
    Opcode opcode = _ex_inst_queue.front()->opcode;
    if (opcode == Opcode::GEMM || opcode == Opcode::GEMM_PRELOAD) {
      consider_candidate(_core_cycle, false);
    }
  }

  std::vector<PredictorCandidate> tile_candidates;
  tile_candidates.reserve(_tiles.size());
  for (const std::unique_ptr<Tile>& tile : _tiles) {
    PredictorCandidate candidate = predictor_tile_candidate(*tile);
    tile_candidates.push_back(candidate);
    snapshot.unknown = snapshot.unknown || candidate.unknown;
    if (candidate.found) {
      consider_candidate(candidate.horizon,
                         candidate.fifo_certified);
    }
  }

  // A non-retentive SA may be switched off only when every instruction that
  // can become the next resident SA user reloads the array state.  Requiring
  // two resident tiles prevents a newly admitted, unknown tile from racing
  // these candidates.  Already-committed work is excluded because preload
  // activity may already have begun even when its modeled SA start is later.
  snapshot.retention_safe_preload =
      _compute_pipeline.empty() && _ex_inst_queue.empty() &&
      _tiles.size() == 2 && tile_candidates.size() == 2;
  if (snapshot.retention_safe_preload) {
    for (const PredictorCandidate& candidate : tile_candidates) {
      if (!candidate.found) {
        // Even with two resident slots, retiring a tile without another
        // known SA candidate can expose an unknown scheduler tile.
        snapshot.retention_safe_preload = false;
        continue;
      }
      if (candidate.opcode != Opcode::GEMM_PRELOAD) {
        snapshot.retention_safe_preload = false;
      }
    }
  }

  snapshot.visible = snapshot.visible_horizon != inf;
  bool future_unknown = false;
  snapshot.future_tile_cap =
      predictor_future_tile_cap(tile_candidates, future_unknown);
  snapshot.unknown = snapshot.unknown || future_unknown;

  if (!snapshot.visible) {
    snapshot.safe_horizon = snapshot.future_tile_cap;
    snapshot.future_cap_limited = true;
  } else if (snapshot.future_tile_cap <= snapshot.visible_horizon) {
    snapshot.safe_horizon = snapshot.future_tile_cap;
    snapshot.future_cap_limited = true;
  } else {
    snapshot.safe_horizon = snapshot.visible_horizon;
    snapshot.fifo_limited = selected_fifo;
  }

  if (snapshot.safe_horizon < _core_cycle) {
    snapshot.safe_horizon = _core_cycle;
    snapshot.unknown = true;
  }
  return snapshot;
}

void SystolicWS::close_retention_shadow_episode(bool right_censored) {
  if (!_retention_shadow_active) {
    return;
  }
  assert(_retention_shadow_current_episode_cycles > 0);
  assert(_retention_shadow_entry_bound_cycles > 0);
  cycle_type duration = static_cast<cycle_type>(
      _retention_shadow_current_episode_cycles);
  _retention_shadow_episode_count++;
  _retention_shadow_longest_episode = MAX(
      _retention_shadow_longest_episode, duration);
  _retention_shadow_duration_histogram[duration]++;
  _retention_shadow_entry_bound_histogram[
      _retention_shadow_entry_bound_cycles]++;
  if (right_censored) {
    _retention_shadow_right_censored_episodes++;
  }
  for (size_t index = 0;
       index < RETENTION_ENTRY_THRESHOLDS.size(); ++index) {
    cycle_type threshold = RETENTION_ENTRY_THRESHOLDS[index];
    if (_retention_shadow_entry_bound_cycles < threshold) {
      continue;
    }
    _retention_shadow_threshold_episode_count[index]++;
    _retention_shadow_threshold_cycle_count[index] += duration;
    if (duration < threshold) {
      _retention_shadow_threshold_early_exit_count[index]++;
    }
    if (right_censored) {
      _retention_shadow_threshold_right_censored_count[index]++;
    }
  }
  _retention_shadow_active = false;
  _retention_shadow_wake_cycle = 0;
  _retention_shadow_entry_bound_cycles = 0;
  _retention_shadow_current_episode_cycles = 0;
}

void SystolicWS::update_retention_shadow(bool sa_active) {
  // `sa_active` is derived from the real compute-pipeline start/finish state.
  // It is therefore the confirmation boundary: retention is never entered
  // during nominal or stalled SA execution.
  if (sa_active) {
    if (_retention_shadow_active) {
      if (_core_cycle < _retention_shadow_wake_cycle) {
        _retention_shadow_causal_deadline_violations++;
      }
      _retention_shadow_immediate_wake_events++;
      close_retention_shadow_episode(false);
    }
    return;
  }

  _retention_shadow_sa_idle_cycles++;

  if (_retention_shadow_active &&
      _core_cycle >= _retention_shadow_wake_cycle) {
    close_retention_shadow_episode(false);
  }

  PredictorSnapshot snapshot = compute_idle_predictor_snapshot();
  cycle_type work_drain_horizon = snapshot.safe_horizon;
  bool now1_cap_binding =
      snapshot.future_cap_limited &&
      snapshot.safe_horizon == _core_cycle + 1 &&
      snapshot.future_tile_cap == _core_cycle + 1;
  if (now1_cap_binding && !_scheduler_core_tile_waiting) {
    cycle_type issue_drain_cap =
        _core_cycle + _scheduler_global_issue_drain_bound;
    work_drain_horizon = snapshot.visible
        ? MIN(snapshot.visible_horizon, issue_drain_cap)
        : issue_drain_cap;
  }

  cycle_type remaining = work_drain_horizon > _core_cycle
      ? work_drain_horizon - _core_cycle
      : 0;
  // Unlike non-retentive OFF, retention preserves the PE state required by
  // an ordinary GEMM.  Therefore eligibility depends on a positive causal
  // next-use bound, not on both resident candidates being PRELOADs (or even
  // already being visible).  `future_tile_cap` bounds an unknown admission.
  bool eligible = !snapshot.unknown && remaining > 0;

  if (_retention_shadow_active) {
    if (!eligible) {
      _retention_shadow_immediate_wake_events++;
      close_retention_shadow_episode(false);
      return;
    }
    // A fresh causal observation may shorten, but never extend, the frozen
    // wake deadline selected at entry.
    _retention_shadow_wake_cycle = MIN(
        _retention_shadow_wake_cycle, _core_cycle + remaining);
  } else if (eligible) {
    _retention_shadow_active = true;
    _retention_shadow_wake_cycle = _core_cycle + remaining;
    _retention_shadow_entry_bound_cycles = remaining;
    _retention_shadow_current_episode_cycles = 0;

    bool any_candidate = false;
    bool all_candidates_preload = true;
    bool any_state_reuse = false;
    for (const std::unique_ptr<Tile>& tile : _tiles) {
      if (!tile->next_sa_candidate_found) {
        continue;
      }
      any_candidate = true;
      all_candidates_preload = all_candidates_preload &&
          tile->next_sa_opcode == Opcode::GEMM_PRELOAD;
      any_state_reuse = any_state_reuse ||
          tile->next_sa_opcode == Opcode::GEMM;
    }
    if (any_state_reuse) {
      _retention_shadow_reuse_episode_count++;
    } else if (any_candidate && all_candidates_preload) {
      _retention_shadow_preload_only_episode_count++;
    } else {
      _retention_shadow_unknown_future_episode_count++;
    }
  }

  if (_retention_shadow_active) {
    _retention_shadow_cycles++;
    _retention_shadow_current_episode_cycles++;
  }
}

void SystolicWS::measure_retention_source_reasons(bool sa_active) {
  if (!_retention_reason_measure) {
    return;
  }
  if (sa_active || !_compute_pipeline.empty() ||
      !_ex_inst_queue.empty() || _tiles.size() != 2) {
    if (sa_active && _response_drain_shadow_sleeping) {
      _response_drain_shadow_safety_violations++;
    }
    close_response_drain_shadow_episode(_core_cycle);
    return;
  }

  constexpr uint64_t FIELD_MAX = (uint64_t{1} << 20) - 1;
  struct SourceKey {
    bool accum;
    int buffer_id;
    addr_type address;
  };

  std::array<const Instruction*, 2> candidates{};
  for (size_t tile_index = 0; tile_index < _tiles.size(); ++tile_index) {
    for (const std::unique_ptr<Instruction>& inst :
         _tiles[tile_index]->instructions) {
      if (inst->opcode == Opcode::GEMM ||
          inst->opcode == Opcode::GEMM_PRELOAD) {
        candidates[tile_index] = inst.get();
        break;
      }
    }
    if (candidates[tile_index] == nullptr ||
        candidates[tile_index]->opcode != Opcode::GEMM_PRELOAD) {
      close_response_drain_shadow_episode(_core_cycle);
      return;
    }
  }

  _retention_reason_context_cycles++;
  bool sources_supported = true;
  bool more_than_two_sources = false;
  uint64_t cycle_reason_mask = 0;
  std::vector<SourceKey> all_unique_sources;
  std::array<cycle_type, 2> candidate_response_bounds{};

  auto add_reason = [&](RetentionSourceReason reason) {
    size_t index = static_cast<size_t>(reason);
    _retention_reason_source_counts[index]++;
    cycle_reason_mask |= uint64_t{1} << index;
  };

  for (size_t tile_index = 0; tile_index < _tiles.size(); ++tile_index) {
    const Tile& tile = *_tiles[tile_index];
    const Instruction& candidate = *candidates[tile_index];
    std::vector<SourceKey> tile_unique_sources;

    for (addr_type address : candidate.src_addrs) {
      bool accum = candidate.src_from_accum && address >= ACCUM_SPAD_BASE;
      int buffer_id = accum ? tile.accum_spad_id : tile.spad_id;
      SourceKey key{accum, buffer_id, address};
      auto same_source = [&](const SourceKey& source) {
        return source.accum == key.accum &&
               source.buffer_id == key.buffer_id &&
               source.address == key.address;
      };
      if (std::find_if(tile_unique_sources.begin(),
                       tile_unique_sources.end(), same_source) ==
          tile_unique_sources.end()) {
        tile_unique_sources.push_back(key);
      }
      if (std::find_if(all_unique_sources.begin(),
                       all_unique_sources.end(), same_source) ==
          all_unique_sources.end()) {
        all_unique_sources.push_back(key);
      }
    }

    if (tile_unique_sources.size() > 2) {
      sources_supported = false;
      more_than_two_sources = true;
      _retention_reason_more_than_two_sources_tiles++;
    }

    if (_response_drain_ports_per_core_cycle_bound > 0) {
      for (const SourceKey& source : tile_unique_sources) {
        const Sram& buffer = source.accum ? _acc_spad : _spad;
        Sram::EntryState state =
            buffer.get_entry_state(source.address, source.buffer_id);
        if (state.allocated && !state.valid &&
            state.pending_memory_count > 0) {
          cycle_type operand_bound =
              (state.pending_memory_count - 1) /
              _response_drain_ports_per_core_cycle_bound;
          candidate_response_bounds[tile_index] =
              MAX(candidate_response_bounds[tile_index], operand_bound);
        }
      }
    }
  }

  if (more_than_two_sources) {
    _retention_reason_more_than_two_sources_cycles++;
  }

  for (const SourceKey& source : all_unique_sources) {
    _retention_reason_unique_source_observations++;
    const Sram& buffer = source.accum ? _acc_spad : _spad;
    Sram::EntryState state =
        buffer.get_entry_state(source.address, source.buffer_id);

    if (!state.allocated) {
      sources_supported = false;
      add_reason(RetentionSourceReason::UNALLOCATED);
      continue;
    }
    if (state.valid) {
      add_reason(RetentionSourceReason::READY);
      continue;
    }

    if (state.pending_memory_count > 0) {
      if (!_core_queue_tickets_exact) {
        sources_supported = false;
        add_reason(RetentionSourceReason::CORE_QUEUE_TICKETS_INEXACT);
      } else if (state.memory_tail_core_queue_ticket <=
                 _last_dequeued_core_queue_ticket) {
        sources_supported = false;
        add_reason(RetentionSourceReason::PENDING_MEMORY_TAIL_DEQUEUED);
      } else {
        uint64_t position = state.memory_tail_core_queue_ticket -
                            _last_dequeued_core_queue_ticket;
        if (position > FIELD_MAX) {
          sources_supported = false;
          add_reason(RetentionSourceReason::TICKET_WINDOW_OVERFLOW);
        } else {
          add_reason(RetentionSourceReason::PENDING_MEMORY_TAIL_QUEUED);
        }
      }
    } else {
      sources_supported = false;
    }
    if (state.pending_compute_count > 0) {
      // This may coexist with pending memory.  It is diagnostically relevant,
      // but a still-queued required memory tail remains a valid lower bound.
      add_reason(RetentionSourceReason::PENDING_COMPUTE);
    } else if (state.pending_memory_count == 0) {
      add_reason(RetentionSourceReason::INVALID_NO_PENDING);
    }
  }

  for (size_t index = 0; index < _retention_reason_cycle_counts.size();
       ++index) {
    if ((cycle_reason_mask & (uint64_t{1} << index)) != 0) {
      _retention_reason_cycle_counts[index]++;
    }
  }
  if (sources_supported) {
    _retention_reason_supported_cycles++;
  } else {
    _retention_reason_unsupported_cycles++;
  }

  if (_response_drain_ports_per_core_cycle_bound == 0) {
    close_response_drain_shadow_episode(_core_cycle);
    return;
  }

  cycle_type next_sa_bound =
      MIN(candidate_response_bounds[0], candidate_response_bounds[1]);
  _response_drain_evaluated_cycles++;
  if (next_sa_bound == 0) {
    _response_drain_zero_bound_cycles++;
  }
  _response_drain_max_bound =
      MAX(_response_drain_max_bound, next_sa_bound);
  for (size_t index = 0; index < RESPONSE_DRAIN_THRESHOLDS.size(); ++index) {
    if (next_sa_bound >= RESPONSE_DRAIN_THRESHOLDS[index]) {
      _response_drain_threshold_cycles[index]++;
    }
  }

  constexpr cycle_type ENTRY_REQUIRED =
      POWER_GATE_BREAK_EVEN_CYCLES + POWER_GATE_WAKE_GUARD_CYCLES;
  if (!_response_drain_shadow_sleeping) {
    if (next_sa_bound >= ENTRY_REQUIRED) {
      _response_drain_shadow_sleeping = true;
      _response_drain_shadow_start = _core_cycle;
    }
  } else if (next_sa_bound <= POWER_GATE_WAKE_GUARD_CYCLES) {
    close_response_drain_shadow_episode(_core_cycle);
  }
}

void SystolicWS::close_response_drain_shadow_episode(
    cycle_type end_cycle) {
  if (!_response_drain_shadow_sleeping) {
    return;
  }
  cycle_type duration = end_cycle - _response_drain_shadow_start;
  _response_drain_shadow_episode_count++;
  _response_drain_shadow_sleep_cycles += duration;
  _response_drain_shadow_longest_episode =
      MAX(_response_drain_shadow_longest_episode, duration);
  _response_drain_shadow_sleeping = false;
}

void SystolicWS::close_scheduler_opportunity_qualified_run(
    cycle_type end_cycle) {
  if (!_scheduler_opportunity_qualified_run_active) {
    return;
  }
  cycle_type duration =
      end_cycle - _scheduler_opportunity_qualified_run_start;
  _scheduler_opportunity_qualified_run_count++;
  _scheduler_opportunity_qualified_run_cycles += duration;
  _scheduler_opportunity_longest_qualified_run =
      MAX(_scheduler_opportunity_longest_qualified_run, duration);
  _scheduler_opportunity_qualified_run_active = false;
}

void SystolicWS::close_scheduler_opportunity_context(
    cycle_type end_cycle, bool censored) {
  if (!_scheduler_opportunity_context_active) {
    return;
  }

  close_scheduler_opportunity_qualified_run(end_cycle);
  if (_scheduler_unlock_measure) {
    close_scheduler_unlock_shadow(_scheduler_issue_shadow, end_cycle,
                                  censored);
    close_scheduler_unlock_shadow(_scheduler_enhanced_shadow, end_cycle,
                                  censored);
  }
  cycle_type duration =
      end_cycle - _scheduler_opportunity_context_start;
  _scheduler_opportunity_interval_count++;
  _scheduler_opportunity_longest_interval =
      MAX(_scheduler_opportunity_longest_interval, duration);
  if (_scheduler_opportunity_context_had_certificate) {
    _scheduler_opportunity_certified_interval_count++;
  }

  constexpr cycle_type PIPELINED_ENTRY_REQUIRED =
      POWER_GATE_BREAK_EVEN_CYCLES +
      POWER_GATE_WAKE_GUARD_CYCLES + 3;
  if (duration >= PIPELINED_ENTRY_REQUIRED) {
    _scheduler_opportunity_oracle_long_interval_count++;
    _scheduler_opportunity_oracle_long_interval_cycles += duration;
  }

  if (censored) {
    _scheduler_opportunity_censored_intervals++;
  } else if (_scheduler_opportunity_context_frontier > end_cycle) {
    // The causal certificate promised that scheduler admission could not
    // occur before this frontier.  Reaching a non-censored end earlier would
    // invalidate the lower-bound construction.
    _scheduler_opportunity_certificate_violations++;
  }
  if (!censored && _scheduler_unlock_context_frontier > end_cycle) {
    _scheduler_unlock_certificate_violations++;
  }

  _scheduler_opportunity_context_active = false;
  _scheduler_opportunity_context_had_certificate = false;
  _scheduler_opportunity_context_frontier = 0;
  _scheduler_opportunity_context_max_bound = 0;
  _scheduler_unlock_context_frontier = 0;
}

void SystolicWS::close_scheduler_unlock_shadow(
    SchedulerUnlockShadow& shadow, cycle_type end_cycle,
    bool censored_context) {
  shadow.pipeline_active = false;
  shadow.pipeline_cycles = 0;
  shadow.pipeline_frontier = 0;
  if (!shadow.sleeping) {
    return;
  }

  cycle_type duration = end_cycle - shadow.sleep_start;
  shadow.episode_count++;
  shadow.sleep_cycles += duration;
  shadow.longest_episode = MAX(shadow.longest_episode, duration);
  if (censored_context) {
    shadow.censored_episodes++;
  } else if (end_cycle < shadow.wake_cycle) {
    shadow.early_context_end_violations++;
  }
  shadow.sleeping = false;
  shadow.sleep_start = 0;
  shadow.wake_cycle = 0;
}

void SystolicWS::advance_scheduler_unlock_shadow(
    SchedulerUnlockShadow& shadow, cycle_type bound) {
  constexpr cycle_type ENTRY_PIPELINE_CYCLES = 3;
  constexpr cycle_type ENTRY_REQUIRED =
      POWER_GATE_BREAK_EVEN_CYCLES +
      POWER_GATE_WAKE_GUARD_CYCLES + ENTRY_PIPELINE_CYCLES;

  if (shadow.sleeping) {
    if (_core_cycle < shadow.wake_cycle) {
      return;
    }
    close_scheduler_unlock_shadow(shadow, _core_cycle, false);
  }

  if (shadow.pipeline_active) {
    shadow.pipeline_cycles++;
    if (shadow.pipeline_cycles >= ENTRY_PIPELINE_CYCLES) {
      cycle_type remaining =
          shadow.pipeline_frontier > _core_cycle
              ? shadow.pipeline_frontier - _core_cycle
              : 0;
      if (remaining >= POWER_GATE_BREAK_EVEN_CYCLES +
                           POWER_GATE_WAKE_GUARD_CYCLES) {
        shadow.sleeping = true;
        shadow.sleep_start = _core_cycle;
        shadow.wake_cycle =
            shadow.pipeline_frontier - POWER_GATE_WAKE_GUARD_CYCLES;
      } else {
        shadow.expired_before_entry++;
      }
      shadow.pipeline_active = false;
      shadow.pipeline_cycles = 0;
      shadow.pipeline_frontier = 0;
    }
  }

  if (!shadow.sleeping && !shadow.pipeline_active &&
      bound >= ENTRY_REQUIRED) {
    shadow.pipeline_active = true;
    shadow.pipeline_cycles = 1;
    shadow.pipeline_frontier = _core_cycle + bound;
    shadow.entry_attempts++;
  }
}

void SystolicWS::measure_scheduler_opportunity(bool sa_active) {
  // This exactly targets an IF-like admission gap: the SA and the complete
  // core are drained, no per-core tile is waiting, and the scheduler tried
  // but could not admit a tile this cycle.  The global bound is non-trivial
  // only for the simple scheduler when its global tile queue is empty.  It is
  // the maximum remaining instruction count on any core; since a core issues
  // at most one instruction per cycle, it is a causal lower bound on the
  // earliest current-layer completion and hence on next-layer admission.
  bool core_fully_drained = !sa_active && !Core::running();
  bool context =
      core_fully_drained && !_scheduler_core_tile_waiting &&
      _scheduler_issue_state == Core::SchedulerIssueState::NO_TILE_AVAILABLE;

  if (!context) {
    bool censored =
        core_fully_drained &&
        _scheduler_issue_state ==
            Core::SchedulerIssueState::NO_SCHEDULED_REQUEST;
    close_scheduler_opportunity_context(_core_cycle, censored);
    return;
  }

  if (!_scheduler_opportunity_context_active) {
    _scheduler_opportunity_context_active = true;
    _scheduler_opportunity_context_start = _core_cycle;
    _scheduler_opportunity_context_frontier = _core_cycle;
    _scheduler_opportunity_context_max_bound = 0;
    _scheduler_opportunity_context_had_certificate = false;
    _scheduler_unlock_context_frontier = _core_cycle;
  }

  _scheduler_opportunity_context_cycles++;
  cycle_type bound = _scheduler_global_issue_drain_bound;
  _scheduler_opportunity_max_bound =
      MAX(_scheduler_opportunity_max_bound, bound);
  _scheduler_opportunity_context_max_bound =
      MAX(_scheduler_opportunity_context_max_bound, bound);

  // A value of one is the deliberately conservative default and therefore
  // does not constitute a scheduler-derived certificate.
  bool certified = bound > 1;
  if (certified) {
    _scheduler_opportunity_certified_cycles++;
    _scheduler_opportunity_context_had_certificate = true;
    _scheduler_opportunity_context_frontier =
        MAX(_scheduler_opportunity_context_frontier,
            _core_cycle + bound);
  }

  constexpr cycle_type PIPELINED_ENTRY_REQUIRED =
      POWER_GATE_BREAK_EVEN_CYCLES +
      POWER_GATE_WAKE_GUARD_CYCLES + 3;
  bool entry_qualified = certified && bound >= PIPELINED_ENTRY_REQUIRED;
  if (entry_qualified) {
    _scheduler_opportunity_entry_qualified_cycles++;
    if (!_scheduler_opportunity_qualified_run_active) {
      _scheduler_opportunity_qualified_run_active = true;
      _scheduler_opportunity_qualified_run_start = _core_cycle;
    }
  } else {
    close_scheduler_opportunity_qualified_run(_core_cycle);
  }

  if (_scheduler_unlock_measure) {
    cycle_type enhanced_bound = _scheduler_global_enhanced_drain_bound;
    _scheduler_unlock_max_enhanced_bound =
        MAX(_scheduler_unlock_max_enhanced_bound, enhanced_bound);
    _scheduler_unlock_max_compute_bound = MAX(
        _scheduler_unlock_max_compute_bound,
        _scheduler_global_committed_compute_bound);
    _scheduler_unlock_max_vector_bound = MAX(
        _scheduler_unlock_max_vector_bound,
        _scheduler_global_committed_vector_bound);
    _scheduler_unlock_max_memory_bound = MAX(
        _scheduler_unlock_max_memory_bound,
        _scheduler_global_memory_injection_bound);

    if (enhanced_bound > 1) {
      _scheduler_unlock_certified_cycles++;
      _scheduler_unlock_context_frontier = MAX(
          _scheduler_unlock_context_frontier,
          _core_cycle + enhanced_bound);
    }
    if (enhanced_bound >= PIPELINED_ENTRY_REQUIRED) {
      _scheduler_unlock_entry_qualified_cycles++;
    }
    if (enhanced_bound > bound) {
      _scheduler_unlock_improved_cycles++;
    }
    if (_scheduler_global_committed_compute_bound == enhanced_bound &&
        enhanced_bound > bound) {
      _scheduler_unlock_compute_binding_cycles++;
    }
    if (_scheduler_global_committed_vector_bound == enhanced_bound &&
        enhanced_bound > bound) {
      _scheduler_unlock_vector_binding_cycles++;
    }
    if (_scheduler_global_memory_injection_bound == enhanced_bound &&
        enhanced_bound > bound) {
      _scheduler_unlock_memory_binding_cycles++;
    }

    advance_scheduler_unlock_shadow(_scheduler_issue_shadow, bound);
    advance_scheduler_unlock_shadow(_scheduler_enhanced_shadow,
                                    enhanced_bound);
  }
}

void SystolicWS::update_idle_predictor(bool sa_active) {
  constexpr cycle_type kReferenceWholeSaWakeCycles = 10;
  constexpr cycle_type kReferenceWholeSaBreakEvenCycles = 469;
  if (sa_active) {
    if (_legacy_updated_regate_enabled) {
      if (_reference_sa_logic_off) {
        _reference_sa_logic_off_episodes.emplace_back(
            _reference_sa_logic_off_start, _core_cycle);
        _reference_sa_logic_off = false;
        _reference_sa_logic_wake_ready =
            _core_cycle + kReferenceWholeSaWakeCycles;
      }
      if (_reference_sa_logic_wake_ready > _core_cycle) {
        ++_reference_sa_logic_late_wake_events;
        _reference_sa_logic_max_late_wake_cycles = MAX(
            _reference_sa_logic_max_late_wake_cycles,
            _reference_sa_logic_wake_ready - _core_cycle);
      }
      _reference_sa_logic_wake_ready = 0;
    }
    if (_predictor_sa_idle) {
      complete_idle_predictor_interval();
    }
    _predictor_sa_idle = false;
    return;
  }

  PredictorSnapshot snapshot = compute_idle_predictor_snapshot();
  cycle_type lookahead = snapshot.safe_horizon - _core_cycle;

  if (!_predictor_sa_idle) {
    _predictor_sa_idle = true;
    _predictor_idle_start = _core_cycle;
    _predictor_initial_horizon = snapshot.safe_horizon;
    _predictor_frontier = snapshot.safe_horizon;
    _predictor_last_horizon = snapshot.safe_horizon;
    _predictor_max_lookahead = lookahead;
    _predictor_max_lookahead_cycle = _core_cycle;
    _predictor_max_fifo_lookahead =
        snapshot.fifo_limited ? lookahead : 0;
    _predictor_max_causal_remaining = 0;
    _predictor_causal_qualified_cycles.fill(0);
    _predictor_causal_current_runs.fill(0);
    _predictor_causal_longest_runs.fill(0);
    _predictor_now1_cap_binding_count = 0;
    _predictor_now1_queue_empty_count = 0;
    _predictor_now1_queue_front_skip_count = 0;
    _predictor_now1_queue_front_no_gemm_count = 0;
    _predictor_now1_queue_front_gemm_count = 0;
    _predictor_now1_front_prefix_histogram.clear();
    _predictor_work_drain_frontier = snapshot.safe_horizon;
    _predictor_work_drain_max_remaining = 0;
    _predictor_work_drain_max_issue_bound = 0;
    _predictor_work_drain_qualified_cycles.fill(0);
    _predictor_work_drain_current_runs.fill(0);
    _predictor_work_drain_longest_runs.fill(0);
    _predictor_recomputations = 0;
    _predictor_frontier_extensions = 0;
    _predictor_fifo_limited_count = 0;
    _predictor_future_cap_limited_count = 0;
    _predictor_no_visible_count = 0;
    _predictor_unknown_count = 0;
    _pg_shadow_sleeping = false;
    _pg_shadow_episode_start = 0;
    _pg_shadow_episode_entry_remaining = 0;
    _pg_shadow_episode_min_remaining = 0;
    _pg_shadow_max_fresh_remaining = 0;
    _pg_shadow_retention_safe_cycles = 0;
    _pg_shadow_stale_only_qualified_cycles = 0;
    _pg_shadow_late_preload_commit = false;
    _pg_shadow_episodes.clear();
  } else {
    if (snapshot.safe_horizon > _predictor_frontier) {
      _predictor_frontier = snapshot.safe_horizon;
      _predictor_frontier_extensions++;
    }
    _predictor_last_horizon = snapshot.safe_horizon;
    if (lookahead > _predictor_max_lookahead) {
      _predictor_max_lookahead = lookahead;
      _predictor_max_lookahead_cycle = _core_cycle;
    }
    if (snapshot.fifo_limited) {
      _predictor_max_fifo_lookahead =
          MAX(_predictor_max_fifo_lookahead, lookahead);
    }
  }

  cycle_type causal_remaining =
      _predictor_frontier > _core_cycle
          ? _predictor_frontier - _core_cycle
          : 0;
  _predictor_max_causal_remaining =
      MAX(_predictor_max_causal_remaining, causal_remaining);
  for (size_t i = 0; i < PREDICTOR_CAUSAL_THRESHOLDS.size(); i++) {
    if (causal_remaining >= PREDICTOR_CAUSAL_THRESHOLDS[i]) {
      _predictor_causal_qualified_cycles[i]++;
      _predictor_causal_current_runs[i]++;
      _predictor_causal_longest_runs[i] =
          MAX(_predictor_causal_longest_runs[i],
              _predictor_causal_current_runs[i]);
    } else {
      _predictor_causal_current_runs[i] = 0;
    }
  }

  bool now1_cap_binding =
      snapshot.future_cap_limited &&
      snapshot.safe_horizon == _core_cycle + 1 &&
      snapshot.future_tile_cap == _core_cycle + 1;
  cycle_type work_drain_horizon = snapshot.safe_horizon;
  if (now1_cap_binding) {
    _predictor_now1_cap_binding_count++;
    if (!_scheduler_core_tile_waiting) {
      _predictor_now1_queue_empty_count++;
    } else if (_scheduler_front_tile_skip) {
      _predictor_now1_queue_front_skip_count++;
    } else if (!_scheduler_front_tile_contains_gemm) {
      _predictor_now1_queue_front_no_gemm_count++;
    } else {
      _predictor_now1_queue_front_gemm_count++;
      _predictor_now1_front_prefix_histogram[
          _scheduler_front_tile_pre_gemm_instruction_count]++;
    }

    if (!_scheduler_core_tile_waiting) {
      _predictor_work_drain_max_issue_bound =
          MAX(_predictor_work_drain_max_issue_bound,
              _scheduler_global_issue_drain_bound);
      cycle_type work_drain_cap =
          _core_cycle + _scheduler_global_issue_drain_bound;
      work_drain_horizon = snapshot.visible
                               ? MIN(snapshot.visible_horizon, work_drain_cap)
                               : work_drain_cap;
    }
  }

  _predictor_work_drain_frontier =
      MAX(_predictor_work_drain_frontier, work_drain_horizon);
  cycle_type work_drain_remaining =
      _predictor_work_drain_frontier > _core_cycle
          ? _predictor_work_drain_frontier - _core_cycle
          : 0;
  _predictor_work_drain_max_remaining =
      MAX(_predictor_work_drain_max_remaining, work_drain_remaining);
  for (size_t i = 0; i < PREDICTOR_CAUSAL_THRESHOLDS.size(); i++) {
    if (work_drain_remaining >= PREDICTOR_CAUSAL_THRESHOLDS[i]) {
      _predictor_work_drain_qualified_cycles[i]++;
      _predictor_work_drain_current_runs[i]++;
      _predictor_work_drain_longest_runs[i] =
          MAX(_predictor_work_drain_longest_runs[i],
              _predictor_work_drain_current_runs[i]);
    } else {
      _predictor_work_drain_current_runs[i] = 0;
    }
  }

  // Shadow-only non-retentive power-gating policy.  It deliberately uses the
  // fresh per-cycle horizon, not the monotonic measurement frontier: a newly
  // visible earlier contender must be allowed to shorten the certificate.
  if (_pg_shadow_sleeping && !_compute_pipeline.empty()) {
    _pg_shadow_late_preload_commit = true;
  }
  cycle_type fresh_work_drain_remaining =
      work_drain_horizon > _core_cycle
          ? work_drain_horizon - _core_cycle
          : 0;
  if (_legacy_updated_regate_enabled) {
    if (_reference_sa_logic_off) {
      if (fresh_work_drain_remaining <= kReferenceWholeSaWakeCycles) {
        _reference_sa_logic_off_episodes.emplace_back(
            _reference_sa_logic_off_start, _core_cycle);
        _reference_sa_logic_off = false;
        _reference_sa_logic_wake_ready =
            _core_cycle + kReferenceWholeSaWakeCycles;
      }
    } else if (_reference_sa_logic_wake_ready <= _core_cycle &&
               fresh_work_drain_remaining >=
                   kReferenceWholeSaBreakEvenCycles +
                       kReferenceWholeSaWakeCycles) {
      _reference_sa_logic_off = true;
      _reference_sa_logic_off_start = _core_cycle;
    }
  }
  _pg_shadow_max_fresh_remaining =
      MAX(_pg_shadow_max_fresh_remaining, fresh_work_drain_remaining);
  if (snapshot.retention_safe_preload) {
    _pg_shadow_retention_safe_cycles++;
  }
  if (work_drain_remaining >=
          POWER_GATE_BREAK_EVEN_CYCLES + POWER_GATE_WAKE_GUARD_CYCLES &&
      fresh_work_drain_remaining <
          POWER_GATE_BREAK_EVEN_CYCLES + POWER_GATE_WAKE_GUARD_CYCLES) {
    _pg_shadow_stale_only_qualified_cycles++;
  }

  if (!_pg_shadow_sleeping) {
    if (snapshot.retention_safe_preload &&
        fresh_work_drain_remaining >=
            POWER_GATE_BREAK_EVEN_CYCLES +
                POWER_GATE_WAKE_GUARD_CYCLES) {
      _pg_shadow_sleeping = true;
      _pg_shadow_episode_start = _core_cycle;
      _pg_shadow_episode_entry_remaining = fresh_work_drain_remaining;
      _pg_shadow_episode_min_remaining = fresh_work_drain_remaining;
    }
  } else {
    _pg_shadow_episode_min_remaining =
        MIN(_pg_shadow_episode_min_remaining, fresh_work_drain_remaining);
    if (!snapshot.retention_safe_preload ||
        fresh_work_drain_remaining <= POWER_GATE_WAKE_GUARD_CYCLES) {
      close_power_gate_shadow_episode(_core_cycle);
    }
  }

  _predictor_recomputations++;
  if (snapshot.fifo_limited) _predictor_fifo_limited_count++;
  if (snapshot.future_cap_limited) {
    _predictor_future_cap_limited_count++;
  }
  if (!snapshot.visible) _predictor_no_visible_count++;
  if (snapshot.unknown) _predictor_unknown_count++;
}

void SystolicWS::close_power_gate_shadow_episode(cycle_type end_cycle) {
  if (!_pg_shadow_sleeping) {
    return;
  }
  _pg_shadow_episodes.push_back(PowerGateShadowEpisode{
      .start_offset = _pg_shadow_episode_start - _predictor_idle_start,
      .duration = end_cycle - _pg_shadow_episode_start,
      .entry_fresh_remaining = _pg_shadow_episode_entry_remaining,
      .minimum_fresh_remaining = _pg_shadow_episode_min_remaining});
  _pg_shadow_sleeping = false;
}

void SystolicWS::complete_idle_predictor_interval() {
  bool shadow_active_at_sa_activation = _pg_shadow_sleeping;
  close_power_gate_shadow_episode(_core_cycle);
  cycle_type actual_next_sa_cycle = _core_cycle;
  cycle_type actual_idle_length =
      actual_next_sa_cycle - _predictor_idle_start;

  if (actual_idle_length >= SA_PHASE_MIN_IDLE) {
    cycle_type initial_lookahead =
        _predictor_initial_horizon - _predictor_idle_start;
    cycle_type frontier_from_start =
        _predictor_frontier - _predictor_idle_start;
    std::vector<cycle_type> causal_thresholds(
        PREDICTOR_CAUSAL_THRESHOLDS.begin(),
        PREDICTOR_CAUSAL_THRESHOLDS.end());
    std::vector<uint64_t> causal_qualified_cycles(
        _predictor_causal_qualified_cycles.begin(),
        _predictor_causal_qualified_cycles.end());
    std::vector<uint64_t> causal_longest_runs(
        _predictor_causal_longest_runs.begin(),
        _predictor_causal_longest_runs.end());
    std::vector<uint64_t> work_drain_qualified_cycles(
        _predictor_work_drain_qualified_cycles.begin(),
        _predictor_work_drain_qualified_cycles.end());
    std::vector<uint64_t> work_drain_longest_runs(
        _predictor_work_drain_longest_runs.begin(),
        _predictor_work_drain_longest_runs.end());
    json front_prefix_histogram = json::array();
    for (const auto& [prefix, cycles] :
         _predictor_now1_front_prefix_histogram) {
      front_prefix_histogram.push_back(
          json{{"pre_gemm_instructions", prefix}, {"cycles", cycles}});
    }
    json shadow_episodes = json::array();
    uint64_t shadow_total_cycles = 0;
    cycle_type shadow_longest_episode = 0;
    for (const PowerGateShadowEpisode& episode : _pg_shadow_episodes) {
      shadow_episodes.push_back(
          json{{"start_offset", episode.start_offset},
               {"duration", episode.duration},
               {"entry_fresh_remaining",
                episode.entry_fresh_remaining},
               {"minimum_fresh_remaining",
                episode.minimum_fresh_remaining}});
      shadow_total_cycles += episode.duration;
      shadow_longest_episode =
          MAX(shadow_longest_episode, episode.duration);
    }
    bool actual_next_opcode_known = !_compute_pipeline.empty();
    Opcode actual_next_opcode =
        actual_next_opcode_known ? _compute_pipeline.front()->opcode
                                 : Opcode::BAR;
    bool shadow_retention_violation =
        !_pg_shadow_episodes.empty() &&
        (!actual_next_opcode_known ||
         actual_next_opcode != Opcode::GEMM_PRELOAD);
    json record{{"version", 5},
                {"core_id", _id},
                {"idle_start", _predictor_idle_start},
                {"actual_next_sa_cycle", actual_next_sa_cycle},
                {"actual_idle_length", actual_idle_length},
                {"initial_safe_horizon", _predictor_initial_horizon},
                {"initial_safe_lookahead", initial_lookahead},
                {"rolling_frontier", _predictor_frontier},
                {"frontier_from_idle_start", frontier_from_start},
                {"last_safe_horizon", _predictor_last_horizon},
                {"max_safe_lookahead", _predictor_max_lookahead},
                {"max_safe_lookahead_cycle",
                 _predictor_max_lookahead_cycle},
                {"max_fifo_limited_lookahead",
                 _predictor_max_fifo_lookahead},
                {"max_causal_remaining_lookahead",
                 _predictor_max_causal_remaining},
                {"causal_thresholds", causal_thresholds},
                {"causal_qualified_cycles", causal_qualified_cycles},
                {"causal_longest_runs", causal_longest_runs},
                {"future_cap_now1_binding_cycles",
                 _predictor_now1_cap_binding_count},
                {"future_cap_now1_queue_empty_cycles",
                 _predictor_now1_queue_empty_count},
                {"future_cap_now1_queue_front_skip_cycles",
                 _predictor_now1_queue_front_skip_count},
                {"future_cap_now1_queue_front_no_gemm_cycles",
                 _predictor_now1_queue_front_no_gemm_count},
                {"future_cap_now1_queue_front_gemm_cycles",
                 _predictor_now1_queue_front_gemm_count},
                {"future_cap_now1_front_prefix_histogram",
                 front_prefix_histogram},
                {"work_drain_rolling_frontier",
                 _predictor_work_drain_frontier},
                {"work_drain_max_causal_remaining_lookahead",
                 _predictor_work_drain_max_remaining},
                {"work_drain_max_issue_bound",
                 _predictor_work_drain_max_issue_bound},
                {"work_drain_causal_qualified_cycles",
                 work_drain_qualified_cycles},
                {"work_drain_causal_longest_runs",
                 work_drain_longest_runs},
                {"work_drain_safety_violation",
                 _predictor_work_drain_frontier > actual_next_sa_cycle},
                {"recomputations", _predictor_recomputations},
                {"frontier_extensions", _predictor_frontier_extensions},
                {"fifo_limited_recomputations",
                 _predictor_fifo_limited_count},
                {"future_cap_limited_recomputations",
                 _predictor_future_cap_limited_count},
                {"no_visible_gemm_recomputations",
                 _predictor_no_visible_count},
                {"unknown_recomputations", _predictor_unknown_count},
                {"pg_shadow_break_even_cycles",
                 POWER_GATE_BREAK_EVEN_CYCLES},
                {"pg_shadow_threshold_provenance",
                 "sensitivity assumption from a one-PE partial-collapse "
                 "reference, not a calibrated full-SA two-bank break-even"},
                {"pg_shadow_wake_guard_cycles",
                 POWER_GATE_WAKE_GUARD_CYCLES},
                {"pg_shadow_entry_required_cycles",
                 POWER_GATE_BREAK_EVEN_CYCLES +
                     POWER_GATE_WAKE_GUARD_CYCLES},
                {"pg_shadow_max_fresh_remaining",
                 _pg_shadow_max_fresh_remaining},
                {"pg_shadow_retention_safe_cycles",
                 _pg_shadow_retention_safe_cycles},
                {"pg_shadow_stale_only_qualified_cycles",
                 _pg_shadow_stale_only_qualified_cycles},
                {"pg_shadow_episode_count", _pg_shadow_episodes.size()},
                {"pg_shadow_total_sleep_cycles", shadow_total_cycles},
                {"pg_shadow_longest_episode", shadow_longest_episode},
                {"pg_shadow_episodes", shadow_episodes},
                {"pg_shadow_active_at_sa_activation",
                 shadow_active_at_sa_activation},
                {"pg_shadow_late_preload_commit",
                 _pg_shadow_late_preload_commit},
                {"actual_next_sa_opcode_known", actual_next_opcode_known},
                {"actual_next_sa_opcode",
                 actual_next_opcode_known
                     ? opcode_name(actual_next_opcode)
                     : "UNKNOWN"},
                {"pg_shadow_retention_violation",
                 shadow_retention_violation},
                {"safety_violation",
                 _predictor_frontier > actual_next_sa_cycle}};
    spdlog::info("SA_IDLE_PREDICTOR {}", record.dump());
  }
}

bool SystolicWS::can_issue_compute(std::unique_ptr<Instruction>& inst)
{
    _last_compute_block_reason = ComputeBlockReason::NONE;

    if (!Core::can_issue_compute(inst)) {
        _last_compute_block_reason =
            ComputeBlockReason::OPERAND_WAIT;
        return false;
    }

    const bool sa_instruction =
        inst->opcode == Opcode::GEMM ||
        inst->opcode == Opcode::GEMM_PRELOAD;
    if (_hybrid_parameters.enabled && inst->opcode == Opcode::GEMM &&
        !_hybrid_pe_state_valid) {
      throw std::logic_error(
          "ordinary GEMM reached issue after OFF destroyed PE state");
    }
    if (_hybrid_parameters.enabled &&
        _hybrid_parameters.closed_loop && sa_instruction &&
        (_hybrid_state != HybridSaState::ON ||
         _hybrid_entry_in_progress || _hybrid_wake_in_progress)) {
      if (_hybrid_state == HybridSaState::ON &&
          _hybrid_entry_in_progress) {
        begin_hybrid_wake("SA_ISSUE_DURING_ENTRY");
      } else if (_hybrid_state != HybridSaState::ON) {
        begin_hybrid_wake("SA_ISSUE_REQUEST");
      }
      if (_hybrid_state != HybridSaState::ON ||
          _hybrid_wake_in_progress) {
        _hybrid_wake_requested = true;
        _hybrid_wake_stall_cycles++;
        _last_compute_block_reason =
            ComputeBlockReason::POWER_STATE_WAKE;
        return false;
      }
    }

    if (sa_instruction)
    {
        if (_compute_pipeline.size() >=
            _config.core_config[_id].core_height)
        {
            _last_compute_block_reason =
                ComputeBlockReason::PIPELINE_FULL;
            return false;
        }
    }
    else
    {
        if (!_vector_pipeline.empty())
        {
            _last_compute_block_reason =
                ComputeBlockReason::VECTOR_BUSY;
            return false;
        }
    }

    return true;
}

Core::InstructionBlockReason
SystolicWS::specialized_compute_block_reason() const {
  switch (_last_compute_block_reason) {
    case ComputeBlockReason::PIPELINE_FULL:
      return InstructionBlockReason::SA_PIPELINE_FULL;
    case ComputeBlockReason::VECTOR_BUSY:
      return InstructionBlockReason::VECTOR_BUSY;
    case ComputeBlockReason::POWER_STATE_WAKE:
      return InstructionBlockReason::POWER_STATE_WAKE;
    case ComputeBlockReason::OPERAND_WAIT:
    case ComputeBlockReason::NONE:
      return InstructionBlockReason::UNKNOWN;
  }
  return InstructionBlockReason::UNKNOWN;
}

cycle_type SystolicWS::get_inst_compute_cycles(std::unique_ptr<Instruction>& inst) {
  return _config.core_config[_id].core_height + _config.core_config[_id].core_width - 2 + MAX(inst->compute_size, 4);
}

cycle_type SystolicWS::get_vector_compute_cycles(std::unique_ptr<Instruction>& inst) {
  cycle_type vec_op_iter = calculate_vector_op_iterations(inst->compute_size);
  cycle_type add_tree_iter = calculate_add_tree_iterations(inst->compute_size);
  cycle_type add_tree, scalar_ops, vector_ops;
  switch (inst->opcode) {
    case Opcode::LAYERNORM:
      add_tree = 2 * add_tree_iter * _config.core_config[_id].add_tree_latency;
      scalar_ops = 2 * _config.core_config[_id].scalar_mul_latency + _config.core_config[_id].scalar_sqrt_latency;
      // 1 addition, 1 subtraction, 1 division, 2 multiplication.
      vector_ops = vec_op_iter * (2 * _config.core_config[_id].add_latency + 3 * _config.core_config[_id].mul_latency) * inst->tile_m;
      return add_tree + scalar_ops + vector_ops;
    case Opcode::SOFTMAX:
      // 1 add tree, 1 compare tree
      add_tree = 2 * add_tree_iter * _config.core_config[_id].add_tree_latency * inst->tile_m;
      vector_ops =
        vec_op_iter * (_config.core_config[_id].add_latency + _config.core_config[_id].exp_latency + _config.core_config[_id].mul_latency);
      return add_tree + vector_ops;
    case Opcode::ADD:
      return vec_op_iter * _config.core_config[_id].add_latency;
    case Opcode::MUL:
      return vec_op_iter * _config.core_config[_id].mul_latency;
    case Opcode::MAC:
      return vec_op_iter * _config.core_config[_id].mac_latency;
    case Opcode::SWISH: //TODO: Implement SWISH
    case Opcode::GELU:
      return vec_op_iter * _config.core_config[_id].gelu_latency;
    case Opcode::COMP:
      return vec_op_iter * 1;
    case Opcode::ADDTREE:
      return add_tree_iter * _config.core_config[_id].add_tree_latency * inst->tile_m;
    case Opcode::DIV:
      return vec_op_iter * _config.core_config[_id].div_latency;
    case Opcode::EXP:
      return vec_op_iter * _config.core_config[_id].exp_latency;
    
  }
  spdlog::info("not configured operation. {}", inst->id);
  // assert(0);
  return 0;
}

void SystolicWS::initialize_sa_region_profile() {
  const char* enabled = std::getenv("ONNXSIM_SA_REGION_PROFILE");
  _sa_region_profile_enabled =
      enabled != nullptr && enabled[0] != '\0' &&
      std::string(enabled) != "0";
  if (!_sa_region_profile_enabled) {
    return;
  }

  const uint32_t array_rows = _config.core_config[_id].core_height;
  const uint32_t array_columns = _config.core_config[_id].core_width;
  if (array_rows == 0 || array_columns == 0) {
    throw std::runtime_error(
        "SA region profiling requires nonzero array dimensions");
  }

  std::set<uint32_t> unique_sizes;
  const char* requested = std::getenv("ONNXSIM_SA_REGION_SIZES");
  if (requested == nullptr || requested[0] == '\0') {
    unique_sizes = {8, 16, 32, 64, 128};
  } else {
    std::stringstream stream(requested);
    std::string token;
    while (std::getline(stream, token, ',')) {
      token.erase(std::remove_if(token.begin(), token.end(),
                                 [](unsigned char value) {
                                   return std::isspace(value);
                                 }),
                  token.end());
      if (token.empty()) {
        throw std::runtime_error(
            "ONNXSIM_SA_REGION_SIZES contains an empty item");
      }
      size_t parsed = 0;
      unsigned long value = 0;
      try {
        value = std::stoul(token, &parsed);
      } catch (const std::exception&) {
        throw std::runtime_error(
            "invalid ONNXSIM_SA_REGION_SIZES item: " + token);
      }
      if (parsed != token.size() || value == 0 ||
          value > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(
            "invalid ONNXSIM_SA_REGION_SIZES item: " + token);
      }
      unique_sizes.insert(static_cast<uint32_t>(value));
    }
  }

  const uint32_t maximum_dimension = MAX(array_rows, array_columns);
  for (uint32_t size : unique_sizes) {
    if (size <= maximum_dimension) {
      _sa_region_sizes.push_back(size);
    }
  }
  // Always include the whole-array granularity, including rectangular arrays.
  if (unique_sizes.count(maximum_dimension) == 0) {
    _sa_region_sizes.push_back(maximum_dimension);
  }
  std::sort(_sa_region_sizes.begin(), _sa_region_sizes.end());
  _sa_region_sizes.erase(
      std::unique(_sa_region_sizes.begin(), _sa_region_sizes.end()),
      _sa_region_sizes.end());
}

void SystolicWS::initialize_regate_functional_profile() {
  const char* enabled =
      std::getenv("ONNXSIM_REGATE_FUNCTIONAL_PROFILE");
  _regate_functional_enabled =
      enabled != nullptr && enabled[0] != '\0' &&
      std::string(enabled) != "0";
  if (_regate_functional_enabled &&
      (_config.core_config[_id].core_height == 0 ||
       _config.core_config[_id].core_width == 0)) {
    throw std::runtime_error(
        "ReGate functional profiling requires nonzero array dimensions");
  }
}


void SystolicWS::initialize_regate_two_bank_profile() {
  const char* enabled =
      std::getenv("ONNXSIM_REGATE_TWO_BANK_PROFILE");
  _regate_two_bank_enabled =
      enabled != nullptr && enabled[0] != '\0' &&
      std::string(enabled) != "0";
  if (_regate_two_bank_enabled &&
      (_config.core_config[_id].core_height == 0 ||
       _config.core_config[_id].core_width == 0)) {
    throw std::runtime_error(
        "two-bank ReGate profiling requires nonzero array dimensions");
  }
}

void SystolicWS::initialize_updated_regate_profile() {
  const char* enabled = std::getenv("ONNXSIM_REGATE_UPDATED_PROFILE");
  _legacy_updated_regate_enabled =
      enabled != nullptr && enabled[0] != '\0' &&
      std::string(enabled) != "0";
  const char* event_driven =
      std::getenv("ONNXSIM_REGATE_EVENT_DRIVEN_PROFILE");
  _event_driven_regate_enabled =
      event_driven != nullptr && event_driven[0] != '\0' &&
      std::string(event_driven) != "0";
  _updated_regate_enabled =
      _legacy_updated_regate_enabled || _event_driven_regate_enabled;
  if (_legacy_updated_regate_enabled && !_updated_predictor_enabled) {
    throw std::runtime_error(
        "updated ReGate profile requires ONNXSIM_SA_UPDATED_PREDICTOR=1");
  }
  if (_updated_regate_enabled && _hybrid_parameters.enabled) {
    throw std::runtime_error(
        "updated ReGate profile cannot run with hybrid SA control, "
        "including hybrid shadow mode, because hybrid state may affect "
        "ordinary GEMM issue");
  }
  if (_updated_regate_enabled &&
      (_config.core_config[_id].core_height == 0 ||
       _config.core_config[_id].core_width == 0)) {
    throw std::runtime_error(
        "updated ReGate profiling requires nonzero array dimensions");
  }
}

void SystolicWS::record_sa_region_interval(
    const Instruction& instruction) {
  if (!_sa_region_profile_enabled && !_regate_functional_enabled &&
      !_regate_two_bank_enabled && !_updated_regate_enabled) {
    return;
  }
  if (instruction.opcode != Opcode::GEMM &&
      instruction.opcode != Opcode::GEMM_PRELOAD) {
    throw std::logic_error(
        "non-SA instruction passed to SA region profiler");
  }
  if (instruction.finish_cycle < instruction.start_cycle) {
    throw std::logic_error("SA interval finishes before it starts");
  }
  _sa_region_intervals.push_back(SaRegionInterval{
      .stream_id = instruction.measurement_tile_id,
      .issue_cycle = _core_cycle,
      .start_cycle = instruction.start_cycle,
      .finish_cycle = instruction.finish_cycle,
      .tile_m = instruction.tile_m,
      .tile_k = instruction.tile_k,
      .tile_n = instruction.tile_n,
      .opcode = instruction.opcode,
      .weight_block_id = instruction.sa_weight_block_id,
      .weight_block_known = instruction.sa_weight_block_known,
      .preload_sequence = instruction.sa_preload_sequence,
      .analytical_preload_boundary =
          instruction.sa_analytical_preload_boundary});
}

void SystolicWS::emit_regate_functional_profile() {
  if (!_regate_functional_enabled || _regate_functional_emitted) {
    return;
  }
  _regate_functional_emitted = true;

  const uint32_t rows = _config.core_config[_id].core_height;
  const uint32_t columns = _config.core_config[_id].core_width;
  std::vector<sa_pe_state::WaveInterval> intervals;
  intervals.reserve(_sa_region_intervals.size());
  for (const SaRegionInterval& interval : _sa_region_intervals) {
    intervals.push_back(sa_pe_state::WaveInterval{
        .issue_cycle = interval.issue_cycle,
        .start_cycle = interval.start_cycle,
        .finish_cycle = interval.finish_cycle,
        .tile_m = interval.tile_m,
        .tile_k = interval.tile_k,
        .tile_n = interval.tile_n,
        .opcode = interval.opcode == Opcode::GEMM_PRELOAD
            ? sa_pe_state::WaveOpcode::PRELOAD
            : sa_pe_state::WaveOpcode::GEMM,
        .weight_block_id = interval.weight_block_id,
        .weight_block_known = interval.weight_block_known});
  }

  const auto result = sa_pe_state::derive_regate_functional(
      intervals, rows, columns, _core_cycle);
  json classes = json::array();
  for (const auto& item : result.control_classes) {
    classes.push_back(json{
        {"tile_m", item.tile_m},
        {"tile_k", item.tile_k},
        {"tile_n", item.tile_n},
        {"occurrences", item.occurrences},
        {"row_on_prefix_length", item.row_on_prefix_length},
        {"col_on_prefix_length", item.col_on_prefix_length},
        {"pe_on_wave_cycles", item.pe_on_wave_cycles},
        {"on_pe_cycles_per_instruction",
         item.on_pe_cycles_per_instruction},
        {"isolated_formula_pe_on_assert_events_per_instruction",
         item.pe_on_assert_events_per_instruction},
        {"isolated_formula_pe_on_deassert_events_per_instruction",
         item.pe_on_deassert_events_per_instruction}});
  }

  const uint64_t low_power_pe_cycles =
      result.w_on_pe_cycles + result.off_pe_cycles;
  const json exposure_percent =
      result.state_partition_exact_under_model &&
              result.capacity_pe_cycles != 0
          ? json(100.0 * static_cast<double>(low_power_pe_cycles) /
                 static_cast<double>(result.capacity_pe_cycles))
          : json(nullptr);
  const uint64_t observed_intervals = _sa_region_intervals.size();
  const json assert_events = result.control_event_counts_valid
      ? json(result.pe_on_assert_events)
      : json(nullptr);
  const json deassert_events = result.control_event_counts_valid
      ? json(result.pe_on_deassert_events)
      : json(nullptr);

  json record{
      {"version", 1},
      {"core_id", _id},
      {"model", "ReGate logical-control shadow approximation"},
      {"evidence_class", "derived"},
      {"evidence_inputs",
       "ONNXim instruction issue/start/finish cycles, mapped dimensions, "
       "opcode, and weight-block address identity"},
      {"measurement_only", true},
      {"control_path_modified", false},
      {"outputs_and_timing_unchanged", true},
      {"regate_only", true},
      {"combined_with_retention", false},
      {"array_rows", rows},
      {"array_columns", columns},
      {"core_cycles", _core_cycle},
      {"sa_instruction_intervals", observed_intervals},
      {"completed_intervals", result.completed_intervals},
      {"incomplete_intervals", result.incomplete_intervals},
      {"control_schedule_valid", result.control_schedule_valid},
      {"control_event_counts_valid",
       result.control_event_counts_valid},
      {"invalid_dimension_intervals",
       result.invalid_dimension_intervals},
      {"wave_exceeds_instruction_intervals",
       result.wave_exceeds_instruction_intervals},
      {"overlapping_wave_intervals",
       result.overlapping_wave_intervals},
      {"unknown_weight_block_intervals",
       result.unknown_weight_block_intervals},
      {"gemm_without_preload_intervals",
       result.gemm_without_preload_intervals},
      {"weight_block_mismatch_intervals",
       result.weight_block_mismatch_intervals},
      {"footprint_mismatch_intervals",
       result.footprint_mismatch_intervals},
      {"weight_lifetime_complete",
       result.weight_lifetime_complete},
      {"pe_state_cycles",
       json{{"capacity", result.capacity_pe_cycles},
            {"on", result.on_pe_cycles},
            {"w_on", result.w_on_pe_cycles},
            {"off", result.off_pe_cycles},
            {"unclassified", result.unclassified_pe_cycles},
            {"partition_exact_under_model",
             result.state_partition_exact_under_model}}},
      {"ideal_low_power_pe_cycle_exposure_percent", exposure_percent},
      {"pe_on_assert_events", assert_events},
      {"pe_on_deassert_events", deassert_events},
      {"control_classes", classes},
      {"savings",
       json{{"scope", "ReGate only"},
            {"energy_saving_status", "pending"},
            {"energy_saving_percent", nullptr},
            {"reason",
             "PE-state exposure is derived, but W_on/OFF circuit power, "
             "entry/wake energy, and physical transition timing are not "
             "inputs to this profile"}}},
      {"semantics",
       json{{"structural_masks",
             "row_on[r]=(r<tile_m), col_on[c]=(c<tile_k); origin-packed "
             "prefixes are derived from mapped dimensions"},
            {"pe_on",
             "for mapped PE(r,c), PE_on is high on "
             "[start_cycle+r+c,start_cycle+r+c+tile_n)"},
            {"drain",
             "PE_on deasserts after the final input at each PE and drains "
             "diagonally across rows and columns"},
            {"state_priority",
             "ON > committed/unclassified > W_on > OFF"},
            {"wake_model",
             "ideal zero-latency assertion; physical wake timing is pending"},
            {"atomic_preload",
             "the mapped weight footprint becomes valid atomically at the "
             "ONNXim instruction start cycle"},
            {"weight_value_sparsity",
             "pending: current metadata provides mapped dimensions and "
             "addresses, not tensor values"}}},
      {"reference",
       json{{"paper",
             "ReGate: Enabling Power Gating in Neural Processing Units"},
            {"paper_locator", "Section 4.1 and Table 3"},
            {"artifact_commit",
             "004dbfccb90cfc0a61bb79bbee3a5770b4b3897b"},
            {"artifact_locator",
             "trace_util/llm_ops_generator/"
             "power_analysis_lib.py:328-356,432-540"}}},
      {"assumptions",
       json::array({
           "mapped tile footprints are origin-packed",
           "tile dimensions identify structural padding only",
           "local useful-wave cycle is row+column+input_wave",
           "weight preload is atomic at ONNXim instruction start",
           "logical PE_on assertion has zero physical wake latency"})},
      {"validation_pass",
       result.control_schedule_valid &&
           result.state_partition_exact_under_model}};
  spdlog::info("REGATE_FUNCTIONAL_PROFILE {}", record.dump());
}


void SystolicWS::emit_regate_two_bank_profile() {
  if (!_regate_two_bank_enabled || _regate_two_bank_emitted) {
    return;
  }
  _regate_two_bank_emitted = true;

  const uint32_t rows = _config.core_config[_id].core_height;
  const uint32_t columns = _config.core_config[_id].core_width;
  std::vector<sa_pe_state::WaveInterval> intervals;
  intervals.reserve(_sa_region_intervals.size());
  for (const SaRegionInterval& interval : _sa_region_intervals) {
    intervals.push_back(sa_pe_state::WaveInterval{
        .issue_cycle = interval.issue_cycle,
        .start_cycle = interval.start_cycle,
        .finish_cycle = interval.finish_cycle,
        .tile_m = interval.tile_m,
        .tile_k = interval.tile_k,
        .tile_n = interval.tile_n,
        .opcode = interval.opcode == Opcode::GEMM_PRELOAD
            ? sa_pe_state::WaveOpcode::PRELOAD
            : sa_pe_state::WaveOpcode::GEMM,
        .weight_block_id = interval.weight_block_id,
        .weight_block_known = interval.weight_block_known});
  }
  const char* rowwise_flag =
      std::getenv("ONNXSIM_REGATE_ROW_PRELOAD_PROFILE");
  const bool rowwise_requested = rowwise_flag != nullptr &&
      rowwise_flag[0] != '\0' && std::string(rowwise_flag) != "0";
  auto parse_rowwise_cycles = [](const char* name)
      -> std::optional<uint64_t> {
    const char* raw = std::getenv(name);
    if (!raw || !raw[0]) return std::nullopt;
    uint64_t value = 0;
    const char* end = raw + std::char_traits<char>::length(raw);
    const auto parsed = std::from_chars(raw, end, value);
    if (parsed.ec != std::errc{} || parsed.ptr != end) {
      return std::nullopt;
    }
    return value;
  };
  std::optional<sa_pe_state::RowWriteTiming> row_write_timing;
  if (rowwise_requested) {
    const auto lead = parse_rowwise_cycles(
        "ONNXSIM_REGATE_ROW_PRELOAD_LEAD_CYCLES");
    const auto period = parse_rowwise_cycles(
        "ONNXSIM_REGATE_ROW_PRELOAD_PERIOD_CYCLES");
    const auto duration = parse_rowwise_cycles(
        "ONNXSIM_REGATE_ROW_PRELOAD_DURATION_CYCLES");
    const char* raw_order =
        std::getenv("ONNXSIM_REGATE_ROW_PRELOAD_ORDER");
    const std::string order = raw_order ? raw_order : "";
    if (lead && period && duration && *period != 0 && *duration != 0 &&
        (order == "ascending" || order == "descending")) {
      row_write_timing.emplace(sa_pe_state::RowWriteTiming{
          *lead, *period, *duration,
          order == "ascending"
              ? sa_pe_state::RowWriteOrder::ASCENDING_PE_ROW
              : sa_pe_state::RowWriteOrder::DESCENDING_PE_ROW});
    }
  }
  const auto result = sa_pe_state::derive_two_bank_regate(
      intervals, rows, columns, _core_cycle,
      row_write_timing ? &*row_write_timing : nullptr);
  const auto timing_case_json =
      [](const sa_pe_state::TwoBankTimingCase& timing_case) {
        return json{
            {"global",
             timing_case.global_applicable
                 ? json{{"peak_live_preloads",
                         timing_case.global_peak_live_preloads},
                        {"over_two_cycles",
                         timing_case.global_over_two_cycles}}
                 : json(nullptr)},
            {"mapped_pe",
             json{{"peak_live_preloads",
                   timing_case.mapped_pe_peak_live_preloads},
                  {"over_two_pe_cycles",
                   timing_case.mapped_pe_over_two_pe_cycles},
                  {"pes_ever_over_two",
                   timing_case.mapped_pes_ever_over_two}}}};
      };
  const auto& timing = result.timing_audit;
  const json timing_audit = timing.available
      ? json{{"available", true},
             {"evidence_class", "derived"},
             {"physical_feasibility_proven", false},
             {"matched_gemm_consumers", timing.matched_gemm_consumers},
             {"issue_to_whole_finish",
              timing_case_json(timing.issue_to_whole_finish)},
             {"start_to_whole_finish",
              timing_case_json(timing.start_to_whole_finish)},
             {"issue_to_pe_last_mac",
              timing_case_json(timing.issue_to_pe_last_mac)},
             {"start_to_pe_last_mac",
              timing_case_json(timing.start_to_pe_last_mac)},
             {"pe_first_to_pe_last_mac",
              timing_case_json(timing.pe_first_to_pe_last_mac)},
             {"interpretation",
              "Counterfactual overlap of exact matched preload instances. "
              "Start-time and PE-first-use reservation assume zero-duration "
              "loading; they do not establish a physical preload schedule."}}
      : json{{"available", false},
             {"reason",
              "A complete valid wave trace with known, exactly matched "
              "weight identities and footprints is required."}};

  const auto& feasibility = result.preload_feasibility_audit;
  json preload_feasibility_events = json::array();
  json preload_feasibility_offenders = json::array();
  for (const auto& event : feasibility.events) {
    const json event_json{
        {"preload_index", event.preload_index},
        {"instruction_index", event.instruction_index},
        {"weight_block_id", event.weight_block_id},
        {"issue_cycle", event.issue_cycle},
        {"start_cycle", event.start_cycle},
        {"rows", event.rows},
        {"columns", event.columns},
        {"fill_span_cycles", event.fill_span_cycles},
        {"planned_load_begin",
         event.planned_load_begin_available
             ? json(event.planned_load_begin_cycle)
             : json(nullptr)},
        {"load_begin", event.load_begin_cycle},
        {"bank_ready", event.bank_ready_cycle},
        {"first_use", event.first_use_cycle},
        {"required_stall_cycles", event.required_stall_cycles},
        {"outcome", event.no_added_cycle ? "no_added_cycle"
                                          : "conflict"}};
    preload_feasibility_events.push_back(event_json);
    if (!event.no_added_cycle) {
      preload_feasibility_offenders.push_back(event_json);
    }
  }
  const json preload_feasibility_audit = !rowwise_requested
      ? json{{"available", false},
             {"evidence_class", "pending"},
             {"reason", "Opt in and explicitly supply row period and row "
                        "write duration; the rowwise profile flag and all "
                        "of its explicit inputs are required."}}
      : !row_write_timing
      ? json{{"available", false},
             {"evidence_class", "pending"},
             {"reason", "Row timing inputs are missing or invalid."}}
      : !feasibility.available
      ? json{{"available", false},
             {"evidence_class", "pending"},
             {"reason", "A complete valid trace with known, exactly "
                        "matched preload identities and footprints is "
                        "required."}}
      : json{{"version", 1},
             {"available", true},
             {"evidence_class", "derived"},
             {"timing_input_evidence_class", "assumed"},
             {"physical_feasibility_proven", false},
             {"input_arrival_assumption",
              "optimistic: each scheduled row is already available at the "
              "SA input after issue-time operand validity; per-row "
              "scratchpad read, assembly, and delivery latency beyond issue "
              "is not exposed"},
             {"fill_span_equation",
              "(rows - 1) * row_start_period_cycles + "
              "row_write_duration_cycles"},
             {"planned_load_begin_equation",
              "start_cycle - fill_span_cycles"},
             {"load_begin_equation",
              "max(issue_cycle, planned_load_begin)"},
             {"bank_ready_equation",
              "load_begin + fill_span_cycles"},
             {"first_use_equation", "start_cycle"},
             {"required_stall_equation",
              "max(0, bank_ready - first_use)"},
             {"audited_preloads", feasibility.audited_preloads},
             {"no_added_cycle_preloads",
              feasibility.no_added_cycle_preloads},
             {"conflicting_preloads", feasibility.conflicting_preloads},
             {"total_required_stall_cycles",
              feasibility.total_required_stall_cycles},
             {"max_required_stall_cycles",
              feasibility.max_required_stall_cycles},
             {"total_interpretation",
              "sum of independent fixed-schedule per-PRELOAD stalls; not a "
              "globally propagated runtime or an ONNXim cycle change"},
             {"events", preload_feasibility_events},
             {"offending_events", preload_feasibility_offenders},
             {"limitations",
              "Passing proves only whole-bank row-stream readiness under "
              "the assumed at-SA-input timing. Shared-bus arbitration, "
              "post-issue scratchpad row delivery, global bank selection, "
              "forwarding, and circuit timing remain unproven."}};

  const auto& rowwise = result.rowwise_feasibility;
  const auto complete_rowwise_count = [&](uint64_t value) -> json {
    return rowwise.secondary_counts_complete ? json(value) : json(nullptr);
  };
  json preload_readiness_audit{
      {"version", 1},
      {"available", rowwise.available},
      {"evidence_class", rowwise.available ? json("derived")
                                            : json("pending")},
      {"timing_input_evidence_class", "assumed"},
      {"physical_feasibility_proven", false},
      {"first_use_equation", "start_cycle + pe_row"},
      {"required_stall_equation",
       "max(0, bank_ready - first_use)"},
      {"outcome_scope",
       "Per-PE necessary shadow with causal issue and local two-bank "
       "availability constraints. It does not propagate a delayed row "
       "through a shared row bus or prove globally consistent bank indices."},
      {"audited_preloads", rowwise.audited_preloads},
      {"audited_preload_rows", rowwise.audited_preload_rows},
      {"outcome",
       rowwise.conflicting_preloads == 0 ? "no_added_cycle"
                                         : "conflict"},
      {"no_added_cycle_preloads", rowwise.no_added_cycle_preloads},
      {"conflicting_preloads", rowwise.conflicting_preloads},
      {"conflicting_preload_rows", rowwise.conflicting_preload_rows},
      {"schedule_diagnostic_preloads",
       rowwise.schedule_diagnostic_preloads},
      {"schedule_diagnostic_rows", rowwise.schedule_diagnostic_rows},
      {"sum_row_required_stall_cycles",
       rowwise.sum_row_required_stall_cycles},
      {"total_required_stall_cycles",
       rowwise.total_required_stall_cycles},
      {"max_required_stall_cycles", rowwise.max_required_stall_cycles},
      {"preloads", json::array()},
      {"offending_events", json::array()},
      {"bank_conflicts", json::array()}};
  for (const auto& preload : rowwise.preload_audit) {
    json preload_json{
        {"preload_index", preload.preload_index},
        {"instruction_index", preload.instruction_index},
        {"weight_block_id", preload.weight_block_id},
        {"issue_cycle", preload.issue_cycle},
        {"start_cycle", preload.start_cycle},
        {"tile_m", preload.tile_m},
        {"tile_k", preload.tile_k},
        {"tile_n", preload.tile_n},
        {"planned_load_begin",
         preload.planned_load_begin_available
             ? json(preload.planned_load_begin_cycle)
             : json(nullptr)},
        {"load_begin", preload.load_begin_cycle},
        {"outcome", preload.no_added_cycle ? "no_added_cycle"
                                             : "conflict"},
        {"conflicting_rows", preload.conflicting_rows},
        {"sum_row_required_stall_cycles",
         preload.sum_row_required_stall_cycles},
        {"required_stall_cycles", preload.required_stall_cycles},
        {"rows", json::array()},
        {"offending_events", json::array()},
        {"bank_conflicts", json::array()}};
    for (const auto& row : preload.rows) {
      json diagnostics = json::array();
      if (!row.planned_load_begin_available) {
        diagnostics.push_back("load_begin_before_cycle_zero");
      } else if (row.load_before_issue) {
        diagnostics.push_back("load_begin_before_issue");
      }
      if (row.max_local_bank_wait_cycles != 0) {
        diagnostics.push_back("two_bank_unavailable_at_causal_load");
      }
      json conflict_reasons = json::array();
      if (row.required_stall_cycles != 0) {
        conflict_reasons.push_back("bank_not_ready_at_first_use");
      }
      json row_json{
          {"preload_index", preload.preload_index},
          {"pe_row", row.pe_row},
          {"ordered_row_index", row.ordered_row_index},
          {"planned_load_begin",
           row.planned_load_begin_available
               ? json(row.planned_load_begin_cycle)
               : json(nullptr)},
          {"load_begin",
           row.load_begin_cycle},
          {"bank_ready", row.bank_ready_cycle},
          {"first_use", row.first_use_cycle},
          {"outcome", row.no_added_cycle ? "no_added_cycle"
                                           : "conflict"},
          {"required_stall_cycles", row.required_stall_cycles},
          {"max_local_bank_wait_cycles",
           row.max_local_bank_wait_cycles},
          {"schedule_diagnostics", diagnostics},
          {"conflict_reasons", conflict_reasons}};
      preload_json["rows"].push_back(row_json);
      if (!row.no_added_cycle) {
        preload_json["offending_events"].push_back(row_json);
        preload_readiness_audit["offending_events"].push_back(row_json);
      }
    }
    for (const auto& conflict : preload.bank_conflicts) {
      const json conflict_json{
          {"preload_index", conflict.preload_index},
          {"pe_row", conflict.pe_row},
          {"pe_column", conflict.pe_column},
          {"planned_load_begin", conflict.planned_load_begin_cycle},
          {"earliest_bank_release",
           conflict.earliest_bank_release_cycle},
          {"load_begin", conflict.load_begin_cycle},
          {"bank_ready", conflict.bank_ready_cycle},
          {"first_use", conflict.first_use_cycle},
          {"required_stall_cycles", conflict.required_stall_cycles}};
      preload_json["bank_conflicts"].push_back(conflict_json);
      preload_readiness_audit["bank_conflicts"].push_back(conflict_json);
    }
    preload_readiness_audit["preloads"].push_back(preload_json);
  }
  const json rowwise_feasibility = !rowwise_requested
      ? json{{"available", false},
             {"evidence_class", "pending"},
             {"reason", "Opt in and explicitly supply first-row lead, "
                        "row period, row write duration, and row order."}}
      : !row_write_timing
      ? json{{"available", false},
             {"evidence_class", "pending"},
             {"reason", "Lead, period, and duration must be valid unsigned "
                        "integers; period and duration must be positive; "
                        "order must be ascending or descending."}}
      : !rowwise.available
      ? json{{"available", false},
             {"evidence_class", "pending"},
             {"reason", "At least one explicit preload and a complete "
                        "valid wave trace with known, exactly matched weight "
                        "identities and footprints are required."}}
      : json{{"available", true},
             {"evidence_class", "derived"},
             {"timing_input_evidence_class", "assumed"},
             {"first_row_lead_cycles_before_start",
              row_write_timing->first_row_lead_cycles_before_start},
             {"row_start_period_cycles",
              row_write_timing->row_start_period_cycles},
             {"row_write_duration_cycles",
              row_write_timing->row_write_duration_cycles},
             {"row_order",
              row_write_timing->row_order ==
                      sa_pe_state::RowWriteOrder::ASCENDING_PE_ROW
                  ? "ascending"
                  : "descending"},
             {"row_columns_written_simultaneously_assumption", true},
             {"row_write_start_equation",
              "start_cycle - first_row_lead_cycles_before_start + "
              "ordered_row_index * row_start_period_cycles"},
             {"row_ready_equation",
              "row_write_start + row_write_duration_cycles"},
             {"bank_release_equation",
              "max(row_ready, last local modeled MAC end across the "
              "preload and its exactly matched later GEMMs)"},
             {"necessary_conditions_pass",
              rowwise.necessary_conditions_pass},
             {"secondary_counts_complete",
              rowwise.secondary_counts_complete},
             {"secondary_counts_status",
              rowwise.secondary_counts_complete
                  ? "complete under assumed schedule"
                  : "pending: pre-issue writes were skipped; occupancy, "
                    "readiness, and local-wait counts are partial"},
             {"physical_feasibility_proven", false},
             {"mapped_pe_weight_write_events",
              rowwise.mapped_pe_weight_write_events},
             {"row_write_before_issue_events",
              rowwise.row_write_before_issue_events},
             {"row_ready_after_first_mac_events",
              complete_rowwise_count(
                  rowwise.row_ready_after_first_mac_events)},
             {"max_row_ready_lateness_cycles",
              complete_rowwise_count(
                  rowwise.max_row_ready_lateness_cycles)},
             {"mapped_pe_peak_live_preloads",
              complete_rowwise_count(
                  rowwise.mapped_pe_peak_live_preloads)},
             {"mapped_pe_over_two_pe_cycles",
              complete_rowwise_count(
                  rowwise.mapped_pe_over_two_pe_cycles)},
             {"mapped_pes_ever_over_two",
              complete_rowwise_count(
                  rowwise.mapped_pes_ever_over_two)},
             {"locally_blocked_pe_weight_write_events",
              complete_rowwise_count(
                  rowwise.locally_blocked_pe_weight_write_events)},
             {"max_local_bank_wait_cycles",
              complete_rowwise_count(
                  rowwise.max_local_bank_wait_cycles)},
             {"max_local_first_mac_delay_cycles",
              complete_rowwise_count(
                  rowwise.max_local_first_mac_delay_cycles)},
             {"per_row_local_audit", preload_readiness_audit},
             {"stall_interpretation",
              "Local no-stall conflict diagnostic with earlier planned "
              "reservations held fixed. Neither an actual ONNXim stall "
              "measurement nor a globally rescheduled latency."},
             {"limitations",
              "Passing checks are necessary only: no shared row-bus "
              "bandwidth, consistent bank selector, forwarding, control "
              "latency, or functional correctness is verified."}};

  const auto& row_state = result.rowwise_state_accounting;
  const json rowwise_state_accounting{
      {"version", 1},
      {"available", row_state.available},
      {"evidence_class",
       row_state.available ? json("derived") : json("pending")},
      {"timing_input_evidence_class", "assumed"},
      {"validation_pass", row_state.validation_pass},
      {"required_checks",
       json{{"complete_known_matched_trace", row_state.available},
            {"rowwise_timing_and_two_bank_checks",
             row_state.available
                 ? json(rowwise.necessary_conditions_pass)
                 : json(nullptr)},
            {"state_partition", row_state.partition_check_pass}}},
      {"failure_policy",
       "Any unavailable or failed required check assigns the complete "
       "rowwise capacity to unclassified and zeros every classified state."},
      {"classification_convention",
       "Only modeled MAC waves are ON. An assumed row write is separate "
       "preload-write exposure; without a MAC, its live logical bank is "
       "W_on. GEMM_PRELOAD contributes its own fused activation wave to "
       "on_preload_wave, and exactly matched later GEMMs contribute to "
       "on_gemm_wave."},
      {"pe_state_cycles",
       json{{"capacity", row_state.capacity_pe_cycles},
            {"on", row_state.on_pe_cycles},
            {"on_preload_wave", row_state.on_preload_wave_pe_cycles},
            {"on_gemm_wave", row_state.on_gemm_wave_pe_cycles},
            {"on_zero_banks", row_state.on_zero_bank_pe_cycles},
            {"on_one_bank", row_state.on_one_bank_pe_cycles},
            {"on_two_banks", row_state.on_two_bank_pe_cycles},
            {"w_on_one_bank", row_state.w_on_one_bank_pe_cycles},
            {"w_on_two_banks", row_state.w_on_two_bank_pe_cycles},
            {"off", row_state.off_pe_cycles},
            {"unclassified", row_state.unclassified_pe_cycles}}},
      {"preload_write_exposure",
       json{{"evidence_class",
             row_state.validation_pass ? "derived" : "pending"},
            {"timing_input_evidence_class", "assumed"},
            {"pe_cycles",
             row_state.validation_pass
                 ? json(row_state.preload_write_pe_cycles)
                 : json(nullptr)},
            {"while_on_pe_cycles",
             row_state.validation_pass
                 ? json(row_state.preload_write_while_on_pe_cycles)
                 : json(nullptr)},
            {"disjoint_from_state_partition", false}}},
      {"physical_feasibility_proven", false},
      {"actual_stall_cycles", nullptr},
      {"gross_static_saving_percent", nullptr},
      {"net_energy_saving_percent", nullptr},
      {"limitations",
       "The complete logical partition is conditional on the supplied "
       "row schedule and checked trace identities/footprints. It does not "
       "prove row-bus bandwidth, a causal bank selector, forwarding, "
       "functional correctness, circuit state power, or transition timing."}};

  const json record{
      {"version", 2},
      {"core_id", _id},
      {"model", "two-bank ReGate-like analytical shadow"},
      {"evidence_class", "derived"},
      {"evidence_inputs",
       "ONNXim instruction issue/start/finish cycles, mapped dimensions, "
       "opcode, and interned weight-block address identity"},
      {"measurement_only", true},
      {"control_path_modified", false},
      {"regate_only", true},
      {"combined_with_retention", false},
      {"two_bank_assumption", true},
      {"atomic_preload_assumption", true},
      {"array_rows", rows},
      {"array_columns", columns},
      {"core_cycles", _core_cycle},
      {"sa_instruction_intervals", intervals.size()},
      {"preload_events", result.preload_events},
      {"gemm_events", result.gemm_events},
      {"preload_bank_pe_cycles", result.preload_bank_pe_cycles},
      {"validation_pass", result.validation_pass},
      {"validation",
       json{{"incomplete_intervals", result.incomplete_intervals},
            {"invalid_intervals", result.invalid_intervals},
            {"unknown_weight_intervals",
             result.unknown_weight_intervals},
            {"gemm_without_weight_intervals",
             result.gemm_without_weight_intervals},
            {"footprint_mismatch_intervals",
             result.footprint_mismatch_intervals},
            {"weight_not_ready_intervals",
             result.weight_not_ready_intervals},
            {"weight_evicted_before_use_intervals",
             result.weight_evicted_before_use_intervals},
            {"wave_exceeds_instruction_intervals",
             result.wave_exceeds_instruction_intervals},
            {"compute_collisions", result.compute_collisions},
            {"two_bank_conflicts", result.two_bank_conflicts}}},
      {"timing_audit", timing_audit},
      {"preload_feasibility_audit", preload_feasibility_audit},
      {"rowwise_feasibility", rowwise_feasibility},
      {"rowwise_state_accounting", rowwise_state_accounting},
      {"pe_state_cycles",
       json{{"capacity", result.capacity_pe_cycles},
            {"on", result.on_pe_cycles},
            {"on_zero_banks", result.on_zero_bank_pe_cycles},
            {"on_one_bank", result.on_one_bank_pe_cycles},
            {"on_two_banks", result.on_two_bank_pe_cycles},
            {"w_on_one_bank", result.w_on_one_bank_pe_cycles},
            {"w_on_two_banks", result.w_on_two_bank_pe_cycles},
            {"off", result.off_pe_cycles},
            {"unclassified", result.unclassified_pe_cycles}}},
      {"gross_static_saving_percent", nullptr},
      {"net_energy_saving_percent", nullptr},
      {"energy_status",
       "pending: separate ON/W_on leakage for one and two powered banks, "
       "OFF leakage, preload activity, control and transition energy are "
       "required for circuit-calibrated savings"},
      {"assumptions",
       json::array({
           "two logical weight banks per PE; ONNXim does not select a bank",
           "each explicit GEMM_PRELOAD creates a new preload instance",
           "a GEMM uses the latest prior matching preload by identity and footprint",
           "a bank is replaceable after all matched instruction consumers finish",
           "resident weights remain mapped until that bank is overwritten",
           "the whole mapped footprint powers at issue; it becomes ready at start",
           "mapped footprints are origin-packed structural rectangles",
           "per-bank origin-packed forwarding rectangles are unioned",
           "bank assignment is offline and uses later consumers in the trace",
           "PE(r,c) is ON during [start+r+c,start+r+c+tile_n)",
           "ON has priority over mapped loading and resident bank states",
           "logical state changes have no modeled transition latency"})},
      {"limitations",
       "This instruction-level approximation does not reconstruct per-PE "
       "weight arrival or prove a physical two-bank ReGate implementation. "
       "Schedules incompatible with issue-time atomic two-bank occupancy "
       "fail closed with full capacity unclassified; staggered physical "
       "loading may differ. OFF outside mapped rectangles assumes routing "
       "closure or bypass and is not functionally verified for every array."}};
  spdlog::info("REGATE_TWO_BANK_PROFILE {}", record.dump());
}

void SystolicWS::emit_updated_regate_profile() {
  if (!_updated_regate_enabled || _updated_regate_emitted) {
    return;
  }
  _updated_regate_emitted = true;
  const uint32_t rows = _config.core_config[_id].core_height;
  const uint32_t columns = _config.core_config[_id].core_width;
  std::vector<sa_pe_state::UpdatedShadowInterval> intervals;
  intervals.reserve(_sa_region_intervals.size());
  uint64_t assumed_preload_boundary_events = 0;
  for (const SaRegionInterval& interval : _sa_region_intervals) {
    const bool analytical_boundary =
        interval.opcode == Opcode::GEMM &&
        interval.analytical_preload_boundary;
    if (analytical_boundary) ++assumed_preload_boundary_events;
    intervals.push_back(sa_pe_state::UpdatedShadowInterval{
        .stream_id = interval.stream_id,
        .issue_cycle = interval.issue_cycle,
        .simulator_start_cycle = interval.start_cycle,
        .simulator_finish_cycle = interval.finish_cycle,
        .tile_m = interval.tile_m,
        .tile_k = interval.tile_k,
        .tile_n = interval.tile_n,
        .opcode = interval.opcode == Opcode::GEMM_PRELOAD ||
                          analytical_boundary
            ? sa_pe_state::WaveOpcode::PRELOAD
            : sa_pe_state::WaveOpcode::GEMM,
        .preload_sequence = interval.preload_sequence});
  }
  const auto result = sa_pe_state::derive_updated_two_bank_shadow(
      intervals, rows, columns, _core_cycle);
  json events = json::array();
  for (const auto& event : result.events) {
    const SaRegionInterval& source =
        _sa_region_intervals.at(event.input_index);
    events.push_back(json{
        {"input_index", event.input_index},
        {"tile_stream_id", event.stream_id},
        {"preload_instance", event.preload_instance},
        {"opcode", source.opcode == Opcode::GEMM_PRELOAD
                       ? "GEMM_PRELOAD" : "GEMM"},
        {"analytical_role", event.opcode == sa_pe_state::WaveOpcode::PRELOAD
                                ? "preload_boundary" : "gemm_consumer"},
        {"analytical_preload_boundary_assumed",
         source.opcode == Opcode::GEMM &&
             source.analytical_preload_boundary},
        {"assumed_bank", event.bank},
        {"tile_m", source.tile_m},
        {"tile_k", source.tile_k},
        {"tile_n", source.tile_n},
        {"issue_cycle", event.issue_cycle},
        {"loading_start_cycle", event.loading_start_cycle},
        {"loading_finish_cycle", event.loading_finish_cycle},
        {"analytical_compute_start", event.compute_start_cycle},
        {"analytical_compute_finish", event.compute_finish_cycle},
        {"onnxim_compute_start", event.simulator_start_cycle},
        {"onnxim_compute_finish", event.simulator_finish_cycle},
        {"event_start_lateness_cycles", event.start_lateness_cycles},
        {"event_finish_lateness_cycles", event.finish_lateness_cycles}});
  }
  json reference_off_episodes = json::array();
  uint64_t reference_off_cycles = 0;
  for (const auto& [begin, end] : _reference_sa_logic_off_episodes) {
    reference_off_episodes.push_back(
        json{{"start_cycle", begin}, {"end_cycle", end}});
    reference_off_cycles += end - begin;
  }
  if (_reference_sa_logic_off) {
    reference_off_episodes.push_back(
        json{{"start_cycle", _reference_sa_logic_off_start},
             {"end_cycle", _core_cycle}});
    reference_off_cycles += _core_cycle - _reference_sa_logic_off_start;
  }
  const json record{
      {"version", 1},
      {"core_id", _id},
      {"model", "updated predictor plus two-bank ReGate-like analytical shadow"},
      {"measurement_only", true},
      {"control_path_modified", false},
      {"functional_scheduling_unchanged", true},
      {"timing_evidence_class", "assumed"},
      {"state_accounting_evidence_class", "derived"},
      {"energy_saving_status", "pending"},
      {"energy_saving_percent", nullptr},
      {"array_rows", rows},
      {"array_columns", columns},
      {"weight_bytes", _config.precision},
      {"onnxim_horizon_cycles", _core_cycle},
      {"analytical_horizon_cycles", result.analytical_horizon_cycles},
      {"preload_events", result.preload_events},
      {"assumed_preload_boundary_events", assumed_preload_boundary_events},
      {"analytical_boundary_provenance",
       "Native GEMM markers are treated as assumed preload boundaries only "
       "inside this analytical shadow; ONNXim opcode and timing are unchanged. "
       "They do not certify a physical weight reload or OFF safety."},
      {"gemm_events", result.gemm_events},
      {"validation_pass", result.validation_pass},
      {"validation",
       json{{"preload_first_valid", result.preload_first_valid},
            {"bank_assignment_valid", result.bank_assignment_valid},
            {"state_partition_exact", result.state_partition_exact},
            {"invalid_intervals", result.invalid_intervals},
            {"sequence_errors", result.sequence_errors},
            {"footprint_mismatch_intervals",
             result.footprint_mismatch_intervals},
            {"bank_assignment_conflicts",
             result.bank_assignment_conflicts}}},
      {"pe_state_cycles",
       json{{"capacity", result.capacity_pe_cycles},
            {"on", result.on_pe_cycles},
            {"on_zero_banks", result.on_zero_bank_pe_cycles},
            {"on_one_bank", result.on_one_bank_pe_cycles},
            {"on_two_banks", result.on_two_bank_pe_cycles},
            {"w_on", result.w_on_pe_cycles},
            {"w_on_one_bank", result.w_on_one_bank_pe_cycles},
            {"w_on_two_banks", result.w_on_two_bank_pe_cycles},
            {"loading", result.loading_pe_cycles},
            {"off", result.off_pe_cycles},
            {"unclassified", result.unclassified_pe_cycles}}},
      {"preload_write_exposure",
       json{{"pe_cycles", result.preload_write_pe_cycles},
            {"while_on_pe_cycles",
             result.preload_write_while_on_pe_cycles},
            {"disjoint_from_state_partition", false}}},
      {"timing_disagreement",
       json{{"interpretation",
             "Schedule-aware analytical comparison only; no timing is "
             "injected into ONNXim. Event-lateness sums are not an "
             "aggregate trace-latency penalty."},
            {"sum_event_start_lateness_cycles",
             result.sum_event_start_lateness_cycles},
            {"max_event_start_lateness_cycles",
             result.max_event_start_lateness_cycles},
            {"sum_event_finish_lateness_cycles",
             result.sum_event_finish_lateness_cycles},
            {"max_event_finish_lateness_cycles",
             result.max_event_finish_lateness_cycles},
            {"simulator_trace_makespan_cycles",
             result.simulator_trace_makespan_cycles},
            {"analytical_trace_makespan_cycles",
             result.analytical_trace_makespan_cycles},
            {"trace_makespan_extension_cycles",
             result.trace_makespan_extension_cycles}}},
      {"predictor_reference_whole_sa_logic",
       json{{"policy_evidence_class", "extrapolated"},
            {"predictor_input", "fresh work-drain safe horizon per idle cycle"},
            {"domain", "compute logic only; weight banks independent"},
            {"break_even_cycles", 469},
            {"wake_latency_cycles", 10},
            {"off_cycles", reference_off_cycles},
            {"off_episodes", reference_off_episodes},
            {"late_wake_events", _reference_sa_logic_late_wake_events},
            {"max_local_wake_lateness_cycles",
             _reference_sa_logic_max_late_wake_cycles},
            {"actual_stall_cycles", nullptr},
            {"timing_modified", false}}},
      {"assumptions",
       json::array({
           "two logical weight banks per PE and one shared MAC datapath",
           "every explicit GEMM_PRELOAD, or native GEMM with an analytical boundary marker, creates an assumed distinct weight copy",
           "ordinary GEMM consumes the latest preload in the same tile stream",
           "offline bank assignment selects the free bank with the earliest feasible preload load start",
           "one mapped stationary-weight row is written per cycle and all mapped columns in that row are written in parallel",
           "one preload lane may overlap the other bank's compute wave",
           "compute waves are serialized",
           "PE(r,c) is ON on [start+r+c,start+r+c+tile_n)",
           "state priority is ON over LOADING over W_on over OFF",
           "edge footprints use tile_m by tile_k; semantic convolution activation padding is not treated as absent stationary weights"})},
      {"energy_parameters_required",
       json::array({
           "ON/W_on/LOADING/OFF power with one and two live banks",
           "entry, wake, and bank-switch energy",
           "entry, wake, preload, and bank-switch latency",
           "second-bank, selector, controller, isolation, switch, and routing overhead"})},
      {"power_gating_break_even_status",
       "pending full-SA two-bank circuit calibration; a one-PE partial-collapse "
       "reference threshold is sensitivity input only, not a validated break-even"},
      {"events", events}};
  if (_legacy_updated_regate_enabled) {
    spdlog::info("REGATE_UPDATED_PROFILE {}", record.dump());
  }
  if (_event_driven_regate_enabled) {
    const json source_record{
        {"version", 1},
        {"core_id", _id},
        {"model", "event-driven unified PE and two-bank analytical source"},
        {"measurement_only", true},
        {"functional_scheduling_unchanged", true},
        {"legacy_whole_sa_predictor_used", false},
        {"event_driven_replay_required", true},
        {"array_rows", rows},
        {"array_columns", columns},
        {"weight_bytes", _config.precision},
        {"onnxim_horizon_cycles", _core_cycle},
        {"analytical_horizon_cycles", result.analytical_horizon_cycles},
        {"validation_pass", result.validation_pass},
        {"validation", record["validation"]},
        {"preload_boundary_provenance",
         "Explicit GEMM_PRELOAD is an ONNXim opcode. A native GEMM with "
         "an analytical boundary marker is only an assumed preload in "
         "this source trace, not a physical weight movement."},
        {"source_bank_assignment_assumptions",
         json::array({
             "bank IDs come from the older serialized analytical shadow and may fail its validation",
             "one origin-aligned tile_m-by-tile_k weight rectangle",
             "native ONNXim compute start/finish are separately recorded and must anchor the new replay",
             "shadow analytical compute times and row-write times are not native ONNXim timing"})},
        {"events", events}};
    spdlog::info("REGATE_EVENT_DRIVEN_SOURCE {}", source_record.dump());
  }
}

void SystolicWS::emit_sa_region_profile() {
  if (!_sa_region_profile_enabled || _sa_region_profile_emitted) {
    return;
  }
  _sa_region_profile_emitted = true;

  const uint32_t array_rows = _config.core_config[_id].core_height;
  const uint32_t array_columns = _config.core_config[_id].core_width;

  auto merge_spans = [&](std::vector<std::pair<cycle_type, cycle_type>> spans) {
    for (auto& span : spans) {
      span.first = MIN(span.first, _core_cycle);
      span.second = MIN(span.second, _core_cycle);
    }
    spans.erase(std::remove_if(spans.begin(), spans.end(),
                               [](const auto& span) {
                                 return span.second <= span.first;
                               }),
                spans.end());
    std::sort(spans.begin(), spans.end());
    if (spans.empty()) {
      return spans;
    }
    std::vector<std::pair<cycle_type, cycle_type>> merged;
    merged.reserve(spans.size());
    cycle_type start = spans.front().first;
    cycle_type finish = spans.front().second;
    for (size_t index = 1; index < spans.size(); ++index) {
      if (spans[index].first <= finish) {
        finish = MAX(finish, spans[index].second);
      } else {
        merged.emplace_back(start, finish);
        start = spans[index].first;
        finish = spans[index].second;
      }
    }
    merged.emplace_back(start, finish);
    return merged;
  };

  auto merged_cycles = [](const auto& spans) {
    uint64_t total = 0;
    for (const auto& span : spans) {
      total += span.second - span.first;
    }
    return total;
  };

  auto accumulate_idle_episode_histogram = [this](
      const auto& active_spans, std::map<uint64_t, uint64_t>& histogram) {
    cycle_type cursor = 0;
    for (const auto& span : active_spans) {
      if (cursor < span.first) {
        histogram[span.first - cursor]++;
      }
      cursor = MAX(cursor, span.second);
    }
    if (cursor < _core_cycle) {
      histogram[_core_cycle - cursor]++;
    }
  };

  auto histogram_json = [](const std::map<uint64_t, uint64_t>& histogram) {
    json result = json::array();
    for (const auto& [length, count] : histogram) {
      result.push_back(
          json{{"length_cycles", length}, {"episode_count", count}});
    }
    return result;
  };

  auto histogram_episode_count = [](const auto& histogram) {
    uint64_t count = 0;
    for (const auto& [length, occurrences] : histogram) {
      (void)length;
      count += occurrences;
    }
    return count;
  };

  auto histogram_cycle_count = [](const auto& histogram) {
    uint64_t cycles = 0;
    for (const auto& [length, occurrences] : histogram) {
      cycles += length * occurrences;
    }
    return cycles;
  };

  std::vector<std::pair<cycle_type, cycle_type>> whole_active_spans;
  std::vector<std::pair<cycle_type, cycle_type>> whole_committed_spans;
  whole_active_spans.reserve(_sa_region_intervals.size());
  whole_committed_spans.reserve(_sa_region_intervals.size());
  uint64_t invalid_dimension_intervals = 0;
  uint64_t clipped_dimension_intervals = 0;
  uint64_t modeled_useful_mac_count = 0;
  for (const SaRegionInterval& interval : _sa_region_intervals) {
    whole_active_spans.emplace_back(interval.start_cycle,
                                    interval.finish_cycle);
    whole_committed_spans.emplace_back(interval.issue_cycle,
                                       interval.finish_cycle);
    if (interval.tile_m == 0 || interval.tile_k == 0 ||
        interval.tile_n == 0) {
      invalid_dimension_intervals++;
    }
    if (interval.tile_m > array_rows || interval.tile_k > array_columns) {
      clipped_dimension_intervals++;
    }
    modeled_useful_mac_count +=
        static_cast<uint64_t>(interval.tile_m) * interval.tile_k *
        interval.tile_n;
  }
  const auto merged_whole_active_spans =
      merge_spans(std::move(whole_active_spans));
  const uint64_t whole_sa_active_cycles =
      merged_cycles(merged_whole_active_spans);
  const uint64_t whole_sa_idle_cycles =
      _core_cycle - whole_sa_active_cycles;
  const auto merged_whole_committed_spans =
      merge_spans(std::move(whole_committed_spans));
  const uint64_t whole_sa_committed_cycles =
      merged_cycles(merged_whole_committed_spans);
  const uint64_t whole_sa_uncommitted_cycles =
      _core_cycle - whole_sa_committed_cycles;
  std::map<uint64_t, uint64_t> whole_sa_idle_episode_histogram;
  accumulate_idle_episode_histogram(
      merged_whole_active_spans, whole_sa_idle_episode_histogram);
  const uint64_t whole_sa_idle_episode_count =
      histogram_episode_count(whole_sa_idle_episode_histogram);
  const uint64_t whole_sa_idle_episode_cycles =
      histogram_cycle_count(whole_sa_idle_episode_histogram);

  std::vector<sa_pe_state::Interval> pe_state_intervals;
  pe_state_intervals.reserve(_sa_region_intervals.size());
  for (const SaRegionInterval& interval : _sa_region_intervals) {
    pe_state_intervals.push_back(sa_pe_state::Interval{
        .start_cycle = interval.start_cycle,
        .finish_cycle = interval.finish_cycle,
        .tile_m = interval.tile_m,
        .tile_k = interval.tile_k,
        .tile_n = interval.tile_n});
  }
  const sa_pe_state::Result pe_state = sa_pe_state::derive(
      pe_state_intervals, array_rows, array_columns, _core_cycle,
      whole_sa_active_cycles);

  std::vector<sa_pe_state::WaveInterval> wave_intervals;
  wave_intervals.reserve(_sa_region_intervals.size());
  for (const SaRegionInterval& interval : _sa_region_intervals) {
    wave_intervals.push_back(sa_pe_state::WaveInterval{
        .issue_cycle = interval.issue_cycle,
        .start_cycle = interval.start_cycle,
        .finish_cycle = interval.finish_cycle,
        .tile_m = interval.tile_m,
        .tile_k = interval.tile_k,
        .tile_n = interval.tile_n,
        .opcode = interval.opcode == Opcode::GEMM_PRELOAD
            ? sa_pe_state::WaveOpcode::PRELOAD
            : sa_pe_state::WaveOpcode::GEMM,
        .weight_block_id = interval.weight_block_id,
        .weight_block_known = interval.weight_block_known});
  }
  const sa_pe_state::WaveLifetimeResult wave_lifetime =
      sa_pe_state::derive_wave_lifetime(
          wave_intervals, array_rows, array_columns, _core_cycle,
          whole_sa_active_cycles);
  json weight_lifetimes = json::array();
  for (const auto& lifetime : wave_lifetime.weight_lifetimes) {
    weight_lifetimes.push_back(json{
        {"start_cycle", lifetime.start_cycle},
        {"finish_cycle", lifetime.finish_cycle},
        {"rows", lifetime.rows},
        {"columns", lifetime.columns},
        {"weight_block_id", lifetime.weight_block_id},
        {"consumer_instructions", lifetime.consumer_instructions}});
  }

  json pe_state_interval_classes = json::array();
  for (const auto& item : pe_state.interval_classes) {
    pe_state_interval_classes.push_back(json{
        {"duration_cycles", item.duration_cycles},
        {"tile_m", item.tile_m},
        {"tile_k", item.tile_k},
        {"tile_n", item.tile_n},
        {"occurrences", item.occurrences},
        {"on_compute_pe_cycles_per_interval",
         item.on_compute_pe_cycles_per_interval},
        {"resident_weight_pe_cycles_per_interval",
         item.resident_weight_pe_cycles_per_interval},
        {"w_on_candidate_pe_cycles_per_interval",
         item.w_on_candidate_pe_cycles_per_interval},
        {"off_unmapped_candidate_pe_cycles_per_interval",
         item.off_unmapped_candidate_pe_cycles_per_interval}});
  }

  json granularities = json::array();
  bool all_accounting_exact = true;
  for (uint32_t region_size : _sa_region_sizes) {
    const sa_pe_state::RegionStateResult analytical_region =
        sa_pe_state::derive_region_states(
            wave_intervals, wave_lifetime, array_rows, array_columns,
            region_size, region_size, _core_cycle);
    const uint32_t region_rows = ceil_div(array_rows, region_size);
    const uint32_t region_columns = ceil_div(array_columns, region_size);
    const uint64_t region_count =
        static_cast<uint64_t>(region_rows) * region_columns;
    std::vector<uint64_t> per_region_active_cycles;
    std::vector<uint64_t> per_region_reserved_cycles;
    per_region_active_cycles.reserve(region_count);
    per_region_reserved_cycles.reserve(region_count);
    uint64_t active_region_cycles = 0;
    uint64_t reserved_region_cycles = 0;
    std::map<uint64_t, uint64_t> idle_episode_histogram;

    for (uint32_t region_row = 0; region_row < region_rows; ++region_row) {
      const uint32_t row_start = region_row * region_size;
      for (uint32_t region_column = 0;
           region_column < region_columns; ++region_column) {
        const uint32_t column_start = region_column * region_size;
        std::vector<std::pair<cycle_type, cycle_type>> active_spans;
        std::vector<std::pair<cycle_type, cycle_type>> reserved_spans;
        for (const SaRegionInterval& interval : _sa_region_intervals) {
          const uint32_t used_rows = MIN(interval.tile_m, array_rows);
          const uint32_t used_columns = MIN(interval.tile_k, array_columns);
          if (used_rows != 0 && used_columns != 0 &&
              row_start < used_rows && column_start < used_columns) {
            active_spans.emplace_back(interval.start_cycle,
                                      interval.finish_cycle);
            reserved_spans.emplace_back(interval.issue_cycle,
                                        interval.finish_cycle);
          }
        }
        const auto merged_active_spans =
            merge_spans(std::move(active_spans));
        const auto merged_reserved_spans =
            merge_spans(std::move(reserved_spans));
        const uint64_t region_active =
            merged_cycles(merged_active_spans);
        const uint64_t region_reserved =
            merged_cycles(merged_reserved_spans);
        accumulate_idle_episode_histogram(
            merged_active_spans, idle_episode_histogram);
        per_region_active_cycles.push_back(region_active);
        per_region_reserved_cycles.push_back(region_reserved);
        active_region_cycles += region_active;
        reserved_region_cycles += region_reserved;
      }
    }

    const uint64_t capacity_region_cycles = region_count * _core_cycle;
    const uint64_t temporal_idle_region_cycles =
        region_count * whole_sa_idle_cycles;
    const uint64_t whole_active_capacity =
        region_count * whole_sa_active_cycles;
    const uint64_t spatial_idle_region_cycles =
        whole_active_capacity - active_region_cycles;
    const uint64_t total_idle_region_cycles =
        capacity_region_cycles - active_region_cycles;
    const uint64_t conservative_available_region_cycles =
        capacity_region_cycles - reserved_region_cycles;
    const uint64_t conservative_temporally_uncommitted_region_cycles =
        region_count * whole_sa_uncommitted_cycles;
    const uint64_t committed_capacity =
        region_count * whole_sa_committed_cycles;
    const uint64_t conservative_spatially_available_region_cycles =
        committed_capacity - reserved_region_cycles;
    const uint64_t idle_episode_count =
        histogram_episode_count(idle_episode_histogram);
    const uint64_t idle_episode_cycles =
        histogram_cycle_count(idle_episode_histogram);
    const bool accounting_exact =
        total_idle_region_cycles ==
            temporal_idle_region_cycles + spatial_idle_region_cycles &&
        idle_episode_cycles == total_idle_region_cycles &&
        active_region_cycles <= whole_active_capacity &&
        conservative_available_region_cycles ==
            conservative_temporally_uncommitted_region_cycles +
                conservative_spatially_available_region_cycles &&
        reserved_region_cycles <= committed_capacity;
    all_accounting_exact = all_accounting_exact && accounting_exact;

    auto percent = [](uint64_t numerator, uint64_t denominator) {
      return denominator == 0
          ? 0.0
          : 100.0 * static_cast<double>(numerator) /
                static_cast<double>(denominator);
    };
    granularities.push_back(json{
        {"region_size_rows", region_size},
        {"region_size_columns", region_size},
        {"region_rows", region_rows},
        {"region_columns", region_columns},
        {"region_count", region_count},
        {"capacity_region_cycles", capacity_region_cycles},
        {"modeled_active_region_cycles", active_region_cycles},
        {"modeled_idle_region_cycles", total_idle_region_cycles},
        {"modeled_active_region_percent",
         percent(active_region_cycles, capacity_region_cycles)},
        {"modeled_idle_region_percent",
         percent(total_idle_region_cycles, capacity_region_cycles)},
        {"temporally_idle_region_cycles", temporal_idle_region_cycles},
        {"spatially_idle_region_cycles_during_sa_active",
         spatial_idle_region_cycles},
        {"spatially_idle_percent_during_sa_active",
         percent(spatial_idle_region_cycles, whole_active_capacity)},
        {"per_region_modeled_active_cycles", per_region_active_cycles},
        {"idle_episode_count", idle_episode_count},
        {"idle_episode_cycles", idle_episode_cycles},
        {"shortest_idle_episode_cycles",
         idle_episode_histogram.empty()
             ? 0
             : idle_episode_histogram.begin()->first},
        {"longest_idle_episode_cycles",
         idle_episode_histogram.empty()
             ? 0
             : idle_episode_histogram.rbegin()->first},
        {"average_idle_episode_cycles",
         idle_episode_count == 0
             ? 0.0
             : static_cast<double>(idle_episode_cycles) /
                   static_cast<double>(idle_episode_count)},
        {"idle_episode_length_histogram",
         histogram_json(idle_episode_histogram)},
        {"conservative_reserved_region_cycles", reserved_region_cycles},
        {"conservative_available_region_cycles",
         conservative_available_region_cycles},
        {"conservative_available_region_percent",
         percent(conservative_available_region_cycles,
                 capacity_region_cycles)},
        {"conservative_temporally_uncommitted_region_cycles",
         conservative_temporally_uncommitted_region_cycles},
        {"conservative_spatially_available_region_cycles_during_commitment",
         conservative_spatially_available_region_cycles},
        {"conservative_spatially_available_percent_during_commitment",
         percent(conservative_spatially_available_region_cycles,
                 committed_capacity)},
        {"per_region_conservative_reserved_cycles",
         per_region_reserved_cycles},
        {"regate_analytical_wave",
         json{
             {"evidence_class", "derived"},
             {"power_domain_rule",
              "ON if any PE in the region has a useful MAC; otherwise W_on "
              "if any compiler-live stationary weight intersects the region; "
              "otherwise structural OFF candidate"},
             {"analytical_on_region_cycles",
              analytical_region.analytical_on_region_cycles},
             {"analytical_w_on_region_cycles",
              analytical_region.analytical_w_on_region_cycles},
             {"structural_off_candidate_region_cycles",
              analytical_region.analytical_off_region_cycles},
             {"preload_committed_unclassified_region_cycles",
              analytical_region.unclassified_region_cycles},
             {"capacity_region_cycles",
              analytical_region.capacity_region_cycles},
             {"per_region_on_cycles",
              analytical_region.per_region_on_cycles},
             {"per_region_w_on_cycles",
              analytical_region.per_region_w_on_cycles},
             {"per_region_structural_off_candidate_cycles",
              analytical_region.per_region_off_cycles},
             {"partition_exact_under_model",
              analytical_region.state_accounting_exact}}},
        {"accounting_exact", accounting_exact}});
  }

  const bool baseline_active_match =
      whole_sa_active_cycles == _stat_tot_systolic_active_cycle;
  const bool validation_pass =
      invalid_dimension_intervals == 0 &&
      clipped_dimension_intervals == 0 && baseline_active_match &&
      whole_sa_idle_episode_cycles == whole_sa_idle_cycles &&
      all_accounting_exact && pe_state.accounting_exact;
  json record{
      {"version", 6},
      {"core_id", _id},
      {"measurement_only", true},
      {"control_path_modified", false},
      {"array_rows", array_rows},
      {"array_columns", array_columns},
      {"core_cycles", _core_cycle},
      {"sa_instruction_intervals", _sa_region_intervals.size()},
      {"modeled_useful_mac_count", modeled_useful_mac_count},
      {"invalid_dimension_intervals", invalid_dimension_intervals},
      {"clipped_dimension_intervals", clipped_dimension_intervals},
      {"whole_sa_active_cycles", whole_sa_active_cycles},
      {"whole_sa_idle_cycles", whole_sa_idle_cycles},
      {"whole_sa_idle_episode_count", whole_sa_idle_episode_count},
      {"whole_sa_idle_episode_cycles", whole_sa_idle_episode_cycles},
      {"whole_sa_shortest_idle_episode_cycles",
       whole_sa_idle_episode_histogram.empty()
           ? 0
           : whole_sa_idle_episode_histogram.begin()->first},
      {"whole_sa_longest_idle_episode_cycles",
       whole_sa_idle_episode_histogram.empty()
           ? 0
           : whole_sa_idle_episode_histogram.rbegin()->first},
      {"whole_sa_average_idle_episode_cycles",
       whole_sa_idle_episode_count == 0
           ? 0.0
           : static_cast<double>(whole_sa_idle_episode_cycles) /
                 static_cast<double>(whole_sa_idle_episode_count)},
      {"whole_sa_idle_episode_length_histogram",
       histogram_json(whole_sa_idle_episode_histogram)},
      {"whole_sa_committed_cycles", whole_sa_committed_cycles},
      {"whole_sa_uncommitted_cycles", whole_sa_uncommitted_cycles},
      {"baseline_sa_active_cycles", _stat_tot_systolic_active_cycle},
      {"baseline_active_match", baseline_active_match},
      {"pe_state_opportunity",
       json{
           {"evidence_class", "derived"},
           {"model", "packed-footprint structural opportunity approximation"},
           {"completed_intervals", pe_state.completed_intervals},
           {"incomplete_intervals", pe_state.incomplete_intervals},
           {"invalid_dimension_intervals",
            pe_state.invalid_dimension_intervals},
           {"insufficient_duration_intervals",
            pe_state.insufficient_duration_intervals},
           {"union_active_cycles", pe_state.union_active_cycles},
           {"active_capacity_pe_cycles",
            pe_state.active_capacity_pe_cycles},
           {"on_compute_pe_cycles", pe_state.on_compute_pe_cycles},
           {"resident_weight_pe_cycles",
            pe_state.resident_weight_pe_cycles},
           {"w_on_candidate_pe_cycles",
            pe_state.w_on_candidate_pe_cycles},
           {"off_unmapped_candidate_pe_cycles",
            pe_state.off_unmapped_candidate_pe_cycles},
           {"temporal_idle_unclassified_pe_cycles",
            pe_state.temporal_idle_unclassified_pe_cycles},
           {"total_capacity_pe_cycles",
            pe_state.total_capacity_pe_cycles},
           {"on_within_resident_capacity",
            pe_state.on_within_resident_capacity},
           {"resident_within_active_capacity",
            pe_state.resident_within_active_capacity},
           {"accounting_exact", pe_state.accounting_exact},
           {"accounting_exact_semantics",
            "Arithmetic partition is exact under the stated origin-packed "
            "footprint and aggregate useful-MAC assumptions; it is not an "
            "exact functional ReGate state trace."},
           {"interval_classes", pe_state_interval_classes},
           {"equations",
            json{
                {"per_interval_on_compute",
                 "tile_m * tile_k * tile_n"},
                {"resident_weight_union",
                 "integral over merged [start,finish) intervals of the union "
                 "of origin-packed tile_m-by-tile_k footprints"},
                {"w_on_candidate",
                 "resident_weight_pe_cycles - on_compute_pe_cycles"},
                {"off_unmapped_candidate",
                 "array_area * whole_sa_active_cycles - "
                 "resident_weight_pe_cycles"},
                {"temporal_idle_unclassified",
                 "array_area * whole_sa_idle_cycles"}}},
           {"semantics",
            "The aggregate useful-work count assigns tile_n MACs per mapped "
            "PE. The remaining mapped capacity is only a W_on opportunity "
            "upper candidate, and unmapped capacity during merged SA-active "
            "time is only an OFF candidate; no physical per-cycle PE state "
            "trace is reconstructed."},
           {"dimension_mapping",
            "For ONNXim GEMM/GEMM_PRELOAD metadata, tile_n is the number "
            "of streamed input rows/waves (ReGate M), tile_k is the "
            "reduction dimension (ReGate K), and tile_m is the number of "
            "output channels/stationary-weight rows (ReGate N). The packed "
            "physical footprint follows ONNXim's existing tile_m-by-tile_k "
            "region convention."},
           {"classification_strength",
            "Derived structural candidate accounting, not an artifact-"
            "equivalent or measured ON/W_on/OFF state trace."},
           {"model_assumptions",
            json::array({
                "one useful MAC occupies one mapped PE for one cycle",
                "each mapped stationary weight is used by tile_n input waves",
                "tile footprints are origin-packed as in the existing "
                "ONNXim region profiler",
                "numerical value sparsity is not inspected",
                "overlapping instruction footprints are unioned before "
                "resident and unmapped capacity is calculated"})},
           {"interval_class_semantics",
            "Interval classes expose the per-instruction derivation grouped "
            "by equal dimensions and duration. They are not additive physical "
            "time when instruction windows overlap; the top-level unioned "
            "fields are the physical-time accounting."},
           {"pre_start_semantics",
            "The [issue_cycle,start_cycle) interval is not classified as "
            "OFF; it remains available only in the separately reported "
            "committed/reserved accounting."},
           {"temporal_idle_semantics",
            "Whole-SA idle PE-cycles are deliberately unclassified: this "
            "profile alone does not prove whether resident weights are live "
            "and require W_on/retention or are dead and permit OFF."},
           {"not_claimed",
            "No per-cycle PE_on trace, energy, voltage, entry/wake latency, "
            "transition cost, BET, scheduling effect, row/column zero "
            "detection, prefix forwarding, diagonal PE_on propagation, or "
            "functional ReGate state transition is modeled."}}},
      {"regate_analytical_wave_lifetime",
       json{
           {"evidence_class", "derived"},
           {"model",
            "analytical origin-packed WS wave plus compiler-visible exact "
            "weight-block lifetime"},
           {"schedule_equation", "local_cycle = row + column + wave"},
           {"useful_mac_equation",
            "U(t) is the coefficient of "
            "(1+x+...+x^(tile_m-1))*(1+x+...+x^(tile_k-1))*"
            "(1+x+...+x^(tile_n-1))"},
           {"analytical_on_pe_cycles",
            wave_lifetime.analytical_on_pe_cycles},
           {"simulator_active_capacity_pe_cycles",
            wave_lifetime.simulator_active_capacity_pe_cycles},
           {"simulator_active_non_wave_pe_cycles",
            wave_lifetime.simulator_active_non_wave_pe_cycles},
           {"resident_weight_pe_cycles",
            wave_lifetime.resident_weight_pe_cycles},
           {"pre_start_committed_pe_cycles",
            wave_lifetime.pre_start_committed_pe_cycles},
           {"analytical_w_on_pe_cycles",
            wave_lifetime.analytical_w_on_pe_cycles},
           {"structural_off_candidate_pe_cycles",
            wave_lifetime.analytical_off_pe_cycles},
           {"unclassified_pe_cycles",
            wave_lifetime.unclassified_pe_cycles},
           {"total_capacity_pe_cycles",
            wave_lifetime.total_capacity_pe_cycles},
           {"completed_intervals", wave_lifetime.completed_intervals},
           {"incomplete_intervals", wave_lifetime.incomplete_intervals},
           {"invalid_dimension_intervals",
            wave_lifetime.invalid_dimension_intervals},
           {"wave_exceeds_instruction_intervals",
            wave_lifetime.wave_exceeds_instruction_intervals},
           {"overlapping_wave_intervals",
            wave_lifetime.overlapping_wave_intervals},
           {"unknown_weight_block_intervals",
            wave_lifetime.unknown_weight_block_intervals},
           {"gemm_without_preload_intervals",
            wave_lifetime.gemm_without_preload_intervals},
           {"weight_block_mismatch_intervals",
            wave_lifetime.weight_block_mismatch_intervals},
           {"footprint_mismatch_intervals",
            wave_lifetime.footprint_mismatch_intervals},
           {"wave_overlap_free", wave_lifetime.wave_overlap_free},
           {"weight_lifetime_complete",
            wave_lifetime.weight_lifetime_complete},
           {"partition_exact_under_model",
            wave_lifetime.state_accounting_exact},
           {"weight_lifetimes", weight_lifetimes},
           {"atomic_preload_approximation",
            "The complete mapped weight footprint becomes valid at the "
            "ONNXim instruction start cycle. ONNXim does not expose "
            "wire-by-wire preload movement."},
           {"pre_start_semantics",
            "Mapped [issue_cycle,start_cycle) capacity is committed and "
            "unclassified; it is never silently called OFF or W_on."},
           {"off_semantics",
            "OFF is a structural candidate under an offline exact-liveness "
            "oracle: no useful MAC, no compiler-live weight footprint, and "
            "no unresolved preload commitment. It is not a generated or "
            "measured ReGate control signal."},
           {"duration_semantics",
            "The useful analytical wave uses mapped tile dimensions. Extra "
            "cycles in ONNXim's full-array instruction duration remain "
            "visible as simulator_active_non_wave_pe_cycles."},
           {"not_energy_result", true}}},
      {"granularities", granularities},
      {"validation_pass", validation_pass},
      {"footprint_semantics",
       "tile_m-by-tile_k stationary-operand footprint packed from the "
       "array origin; modeled spatial utilization uses "
       "[start_cycle,finish_cycle)"},
      {"active_interval_semantics",
       "[start_cycle,finish_cycle) reproduces ONNXim's existing whole-SA "
       "active-cycle count, so whole-array modeled idle equals the baseline"},
      {"conservative_interval_semantics",
       "[issue_cycle,finish_cycle) is reported separately as a policy "
       "reservation bound; ONNXim does not expose internal PE activity "
       "during the pre-start interval"},
      {"overlap_semantics",
       "per-region interval union; overlapping pipelined instructions are "
       "counted active once"},
      {"limitation",
       "ONNXim does not model individual PE wavefront activity or ReGate's "
       "row/column zero detection, prefix forwarding, and diagonal PE_on "
       "propagation; spatial candidates are not cycle-accurate multiplier "
       "or power-state traces"},
      {"regate_input_semantics",
       "modeled_useful_mac_count is the sum of tile_m*tile_k*tile_n over "
       "issued GEMM and GEMM_PRELOAD instructions; it is an architectural "
       "work count, not a measured PE activation trace"}};
  spdlog::info("SA_REGION_PROFILE {}", record.dump());
}

void SystolicWS::print_stats() {
  Core::print_stats();
  emit_regate_functional_profile();
  emit_regate_two_bank_profile();
  emit_sa_region_profile();
  spdlog::info("Core [{}] : Systolic Inst Issue Count : {}", _id,
               _stat_systolic_inst_issue_count);
  spdlog::info("Core [{}] : Systolic PRELOAD Issue Count : {}", _id,
               _stat_systolic_preload_issue_count);

  if (_retention_shadow_enabled) {
    close_retention_shadow_episode(true);
    auto percent = [&](uint64_t cycles) {
      return _core_cycle == 0
          ? 0.0
          : 100.0 * static_cast<double>(cycles) /
                static_cast<double>(_core_cycle);
    };
    json duration_histogram = json::object();
    for (const auto& [duration, count] :
         _retention_shadow_duration_histogram) {
      duration_histogram[std::to_string(duration)] = count;
    }
    json entry_bound_histogram = json::object();
    for (const auto& [bound, count] :
         _retention_shadow_entry_bound_histogram) {
      entry_bound_histogram[std::to_string(bound)] = count;
    }
    json entry_threshold_sweep = json::object();
    for (size_t index = 0;
         index < RETENTION_ENTRY_THRESHOLDS.size(); ++index) {
      entry_threshold_sweep[
          std::to_string(RETENTION_ENTRY_THRESHOLDS[index])] = {
          {"entry_qualified_episode_count",
           _retention_shadow_threshold_episode_count[index]},
          {"observed_retention_cycles",
           _retention_shadow_threshold_cycle_count[index]},
          {"episodes_shorter_than_threshold",
           _retention_shadow_threshold_early_exit_count[index]},
          {"right_censored_episodes",
           _retention_shadow_threshold_right_censored_count[index]}};
    }
    json record{
        {"version", 2},
        {"core_id", _id},
        {"core_cycles", _core_cycle},
        {"measurement_only", true},
        {"control_path_modified", false},
        {"zero_wake_latency_assumption", true},
        {"break_even_filter_applied", false},
        {"entry_pipeline_applied", false},
        {"ordinary_gemm_retention_allowed", true},
        {"two_resident_candidate_requirement", false},
        {"minimum_causal_bound_cycles", 1},
        {"sa_idle_cycles", _retention_shadow_sa_idle_cycles},
        {"retention_cycles", _retention_shadow_cycles},
        {"retention_percent", percent(_retention_shadow_cycles)},
        {"retention_percent_of_sa_idle",
         _retention_shadow_sa_idle_cycles == 0
             ? 0.0
             : 100.0 *
                   static_cast<double>(_retention_shadow_cycles) /
                   static_cast<double>(
                       _retention_shadow_sa_idle_cycles)},
        {"episode_count", _retention_shadow_episode_count},
        {"longest_episode", _retention_shadow_longest_episode},
        {"state_reuse_episode_count",
         _retention_shadow_reuse_episode_count},
        {"preload_only_episode_count",
         _retention_shadow_preload_only_episode_count},
        {"unknown_future_episode_count",
         _retention_shadow_unknown_future_episode_count},
        {"immediate_wake_events",
         _retention_shadow_immediate_wake_events},
        {"causal_deadline_violations",
         _retention_shadow_causal_deadline_violations},
        {"right_censored_episodes",
         _retention_shadow_right_censored_episodes},
        {"episode_duration_histogram", duration_histogram},
        {"entry_bound_histogram", entry_bound_histogram},
        {"entry_threshold_sweep", entry_threshold_sweep},
        {"threshold_sweep_semantics",
         "episodes selected causally by the bound available at entry; "
         "observed durations include early invalidation and right-censored "
         "episodes"},
        {"entry_semantics",
         "actual SA completion followed by a positive causal lower bound"},
        {"retained_state_semantics",
         "all PE state required by a subsequent ordinary GEMM"}};
    spdlog::info("SA_RETENTION_SHADOW {}", record.dump());
  }

  if (_sa_state_measure) {
    constexpr std::array<const char*,
                         static_cast<size_t>(SaArchitecturalState::COUNT)>
        state_names = {
            "ACTIVE_GEMM",
            "ACTIVE_GEMM_PRELOAD_OPCODE",
            "ACTIVE_UNKNOWN",
            "IDLE_COMMITTED_GEMM_START_DELAY",
            "IDLE_COMMITTED_GEMM_PRELOAD_START_DELAY",
            "IDLE_VECTOR_ACTIVE",
            "IDLE_INSTRUCTION_DISPATCH",
            "IDLE_MEMORY_STALL",
            "IDLE_DEPENDENCY_STALL",
            "IDLE_SCHEDULER_STALL",
            "IDLE_OTHER"};
    uint64_t measured_cycles = 0;
    json cycle_counts = json::object();
    json percentages = json::object();
    for (size_t index = 0; index < state_names.size(); ++index) {
      measured_cycles += _sa_state_cycles[index];
      cycle_counts[state_names[index]] = _sa_state_cycles[index];
      percentages[state_names[index]] =
          _core_cycle == 0
              ? 0.0
              : 100.0 * static_cast<double>(_sa_state_cycles[index]) /
                    static_cast<double>(_core_cycle);
    }
    json record{{"version", 1},
                {"core_id", _id},
                {"core_cycles", _core_cycle},
                {"measured_cycles", measured_cycles},
                {"accounting_exact", measured_cycles == _core_cycle},
                {"cycle_counts", cycle_counts},
                {"percent_of_core_cycles", percentages},
                {"active_preload_semantics",
                 "GEMM_PRELOAD instruction active interval; ONNXim does "
                 "not expose internal preload-only versus MAC subcycles"},
                {"committed_delay_semantics",
                 "instruction is in compute pipeline but its modeled "
                 "start_cycle is still in the future"}};
    spdlog::info("SA_STATE_BREAKDOWN {}", record.dump());
  }

  if (_retention_reason_measure) {
    close_response_drain_shadow_episode(_core_cycle);
    constexpr std::array<const char*,
                         static_cast<size_t>(RetentionSourceReason::COUNT)>
        reason_names = {
            "READY",
            "PENDING_MEMORY_TAIL_QUEUED",
            "PENDING_MEMORY_TAIL_DEQUEUED",
            "PENDING_COMPUTE",
            "INVALID_NO_PENDING",
            "UNALLOCATED",
            "TICKET_WINDOW_OVERFLOW",
            "CORE_QUEUE_TICKETS_INEXACT"};
    json cycle_counts = json::object();
    json source_counts = json::object();
    for (size_t index = 0; index < reason_names.size(); ++index) {
      cycle_counts[reason_names[index]] =
          _retention_reason_cycle_counts[index];
      source_counts[reason_names[index]] =
          _retention_reason_source_counts[index];
    }
    json record{{"version", 2},
                {"core_id", _id},
                {"core_cycles", _core_cycle},
                {"physical_precondition_cycles",
                 _retention_reason_context_cycles},
                {"rtl_sources_supported_cycles",
                 _retention_reason_supported_cycles},
                {"rtl_sources_unsupported_cycles",
                 _retention_reason_unsupported_cycles},
                {"more_than_two_sources_cycles",
                 _retention_reason_more_than_two_sources_cycles},
                {"more_than_two_sources_tile_observations",
                 _retention_reason_more_than_two_sources_tiles},
                {"unique_source_observations",
                 _retention_reason_unique_source_observations},
                {"cycle_reason_counts", cycle_counts},
                {"source_reason_observations", source_counts}};
    json response_threshold_cycles = json::object();
    for (size_t index = 0; index < RESPONSE_DRAIN_THRESHOLDS.size();
         ++index) {
      response_threshold_cycles[
          std::to_string(RESPONSE_DRAIN_THRESHOLDS[index])] =
          _response_drain_threshold_cycles[index];
    }
    record["response_drain_ports_per_core_cycle_bound"] =
        _response_drain_ports_per_core_cycle_bound;
    record["response_drain_evaluated_cycles"] =
        _response_drain_evaluated_cycles;
    record["response_drain_zero_bound_cycles"] =
        _response_drain_zero_bound_cycles;
    record["response_drain_max_bound"] =
        _response_drain_max_bound;
    record["response_drain_threshold_cycles"] =
        response_threshold_cycles;
    record["response_drain_shadow_entry_required_cycles"] =
        POWER_GATE_BREAK_EVEN_CYCLES + POWER_GATE_WAKE_GUARD_CYCLES;
    record["response_drain_shadow_episode_count"] =
        _response_drain_shadow_episode_count;
    record["response_drain_shadow_sleep_cycles"] =
        _response_drain_shadow_sleep_cycles;
    record["response_drain_shadow_longest_episode"] =
        _response_drain_shadow_longest_episode;
    record["response_drain_shadow_safety_violations"] =
        _response_drain_shadow_safety_violations;
    spdlog::info("SA_RETENTION_SOURCE_REASONS {}", record.dump());
  }

  if (_scheduler_opportunity_measure) {
    // An interval still open at simulation termination has no observed next
    // admission, so it is right-censored rather than a safety failure.
    close_scheduler_opportunity_context(_core_cycle, true);
    constexpr cycle_type ENTRY_PIPELINE_CYCLES = 3;
    constexpr cycle_type PIPELINED_ENTRY_REQUIRED =
        POWER_GATE_BREAK_EVEN_CYCLES +
        POWER_GATE_WAKE_GUARD_CYCLES + ENTRY_PIPELINE_CYCLES;
    auto percent = [&](uint64_t cycles) {
      return _core_cycle == 0
                 ? 0.0
                 : 100.0 * static_cast<double>(cycles) /
                       static_cast<double>(_core_cycle);
    };
    json record{
        {"version", 1},
        {"core_id", _id},
        {"core_cycles", _core_cycle},
        {"measurement_only", true},
        {"scheduler_semantics", "simple scheduler next-layer admission"},
        {"context_semantics",
         "SA and core fully drained, per-core scheduler queue empty, "
         "scheduler attempted but could not admit a tile"},
        {"bound_semantics",
         "causal lower bound from maximum remaining instruction count "
         "across resident and per-core queued tiles; one instruction issue "
         "per core per cycle"},
        {"break_even_cycles", POWER_GATE_BREAK_EVEN_CYCLES},
        {"wake_guard_cycles", POWER_GATE_WAKE_GUARD_CYCLES},
        {"entry_pipeline_cycles", ENTRY_PIPELINE_CYCLES},
        {"entry_required_bound", PIPELINED_ENTRY_REQUIRED},
        {"fully_drained_scheduler_stall_cycles",
         _scheduler_opportunity_context_cycles},
        {"fully_drained_scheduler_stall_percent",
         percent(_scheduler_opportunity_context_cycles)},
        {"causal_bound_available_cycles",
         _scheduler_opportunity_certified_cycles},
        {"causal_bound_available_percent",
         percent(_scheduler_opportunity_certified_cycles)},
        {"entry_qualified_cycles",
         _scheduler_opportunity_entry_qualified_cycles},
        {"entry_qualified_percent",
         percent(_scheduler_opportunity_entry_qualified_cycles)},
        {"maximum_causal_bound",
         _scheduler_opportunity_max_bound},
        {"scheduler_stall_interval_count",
         _scheduler_opportunity_interval_count},
        {"certified_interval_count",
         _scheduler_opportunity_certified_interval_count},
        {"oracle_intervals_at_least_entry_required",
         _scheduler_opportunity_oracle_long_interval_count},
        {"oracle_cycles_in_long_intervals",
         _scheduler_opportunity_oracle_long_interval_cycles},
        {"longest_scheduler_stall_interval",
         _scheduler_opportunity_longest_interval},
        {"causal_qualified_run_count",
         _scheduler_opportunity_qualified_run_count},
        {"causal_qualified_run_cycles",
         _scheduler_opportunity_qualified_run_cycles},
        {"longest_causal_qualified_run",
         _scheduler_opportunity_longest_qualified_run},
        {"certificate_violations",
         _scheduler_opportunity_certificate_violations},
        {"right_censored_intervals",
         _scheduler_opportunity_censored_intervals}};
    spdlog::info("SA_SCHEDULER_ADMISSION_OPPORTUNITY {}", record.dump());
  }

  if (_scheduler_unlock_measure) {
    auto shadow_json = [](const SchedulerUnlockShadow& shadow) {
      return json{{"entry_attempts", shadow.entry_attempts},
                  {"expired_before_entry", shadow.expired_before_entry},
                  {"episode_count", shadow.episode_count},
                  {"sleep_cycles", shadow.sleep_cycles},
                  {"longest_episode", shadow.longest_episode},
                  {"early_context_end_violations",
                   shadow.early_context_end_violations},
                  {"censored_episodes", shadow.censored_episodes}};
    };
    auto percent = [&](uint64_t cycles) {
      return _core_cycle == 0
                 ? 0.0
                 : 100.0 * static_cast<double>(cycles) /
                       static_cast<double>(_core_cycle);
    };
    json record{
        {"version", 1},
        {"core_id", _id},
        {"core_cycles", _core_cycle},
        {"measurement_only", true},
        {"control_path_modified", false},
        {"context_cycles", _scheduler_opportunity_context_cycles},
        {"context_percent", percent(_scheduler_opportunity_context_cycles)},
        {"issue_only_bound_available_cycles",
         _scheduler_opportunity_certified_cycles},
        {"enhanced_bound_available_cycles",
         _scheduler_unlock_certified_cycles},
        {"issue_only_entry_qualified_cycles",
         _scheduler_opportunity_entry_qualified_cycles},
        {"enhanced_entry_qualified_cycles",
         _scheduler_unlock_entry_qualified_cycles},
        {"enhanced_bound_improved_cycles",
         _scheduler_unlock_improved_cycles},
        {"compute_bound_binding_cycles",
         _scheduler_unlock_compute_binding_cycles},
        {"vector_bound_binding_cycles",
         _scheduler_unlock_vector_binding_cycles},
        {"memory_injection_bound_binding_cycles",
         _scheduler_unlock_memory_binding_cycles},
        {"maximum_issue_only_bound", _scheduler_opportunity_max_bound},
        {"maximum_enhanced_bound",
         _scheduler_unlock_max_enhanced_bound},
        {"maximum_committed_compute_bound",
         _scheduler_unlock_max_compute_bound},
        {"maximum_committed_vector_bound",
         _scheduler_unlock_max_vector_bound},
        {"maximum_future_movin_injection_bound",
         _scheduler_unlock_max_memory_bound},
        {"issue_only_certificate_violations",
         _scheduler_opportunity_certificate_violations},
        {"enhanced_certificate_violations",
         _scheduler_unlock_certificate_violations},
        {"issue_only_shadow", shadow_json(_scheduler_issue_shadow)},
        {"enhanced_shadow", shadow_json(_scheduler_enhanced_shadow)},
        {"bound_semantics",
         "max of one-instruction-per-cycle issue count, committed front "
         "compute completion, committed front vector completion, and "
         "unavoidable injection time for future MOVIN requests"},
        {"memory_bound_exclusions",
         "already-generated requests and MOVOUT traffic are excluded "
         "because they are not universally required to delay scheduler "
         "admission"},
        {"shadow_semantics",
         "three-cycle entry pipeline, frozen causal wake frontier, and "
         "one-cycle wake guard; no simulated state or timing is changed"}};
    spdlog::info("SA_SCHEDULER_UNLOCK_DIAGNOSTIC {}", record.dump());
  }
}
