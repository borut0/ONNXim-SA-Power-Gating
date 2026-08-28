#include "SystolicWS.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <limits>
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

  const char* reason_measure =
      std::getenv("ONNXSIM_SA_RETENTION_REASON_MEASURE");
  _retention_reason_measure =
      reason_measure != nullptr && reason_measure[0] != '\0' &&
      std::string(reason_measure) != "0";
  if (_retention_reason_measure && _config.core_freq > 0 &&
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

  if (_controller_trace.is_open()) {
    trace_controller_inputs(is_compute_busy);
  }
  if (_retention_reason_measure) {
    measure_retention_source_reasons(is_compute_busy);
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

  const Instruction* candidate = nullptr;
  uint64_t prefix = 0;
  for (const std::unique_ptr<Instruction>& inst : tile->instructions) {
    if (inst->opcode == Opcode::GEMM ||
        inst->opcode == Opcode::GEMM_PRELOAD) {
      candidate = inst.get();
      break;
    }
    prefix++;
  }
  if (candidate == nullptr) {
    return descriptor;
  }

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
  size_t prefix_length = 0;

  for (const std::unique_ptr<Instruction>& inst : tile.instructions) {
    if (inst->opcode != Opcode::GEMM &&
        inst->opcode != Opcode::GEMM_PRELOAD) {
      prefix_length++;
      continue;
    }

    candidate.found = true;
    candidate.opcode = inst->opcode;
    candidate.horizon = _core_cycle + prefix_length;
    if (prefix_length == 0) {
      bool fifo_certified = false;
      bool unknown = false;
      cycle_type fifo_horizon = predictor_fifo_horizon(
          *inst, tile.spad_id, tile.accum_spad_id,
          fifo_certified, unknown);
      candidate.horizon = MAX(candidate.horizon, fifo_horizon);
      candidate.fifo_certified = fifo_certified;
      candidate.unknown = unknown;
    }
    return candidate;
  }

  return candidate;
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
      } else {
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
    return _core_cycle + 1;
  }
  if (_tiles.size() != 2 || tile_candidates.size() != _tiles.size()) {
    unknown = true;
    return _core_cycle;
  }

  cycle_type cap = std::numeric_limits<cycle_type>::max();
  for (size_t i = 0; i < _tiles.size(); ++i) {
    size_t remaining = _tiles[i]->instructions.size();
    cycle_type retirement_lower_bound =
        _core_cycle + MAX(static_cast<size_t>(1), remaining);
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
      if (!candidate.found || candidate.opcode != Opcode::GEMM_PRELOAD) {
        snapshot.retention_safe_preload = false;
        break;
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
  if (sa_active) {
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

    if (inst->opcode == Opcode::GEMM ||
        inst->opcode == Opcode::GEMM_PRELOAD)
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

void SystolicWS::print_stats() {
  Core::print_stats();
  spdlog::info("Core [{}] : Systolic Inst Issue Count : {}", _id,
               _stat_systolic_inst_issue_count);
  spdlog::info("Core [{}] : Systolic PRELOAD Issue Count : {}", _id,
               _stat_systolic_preload_issue_count);

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
