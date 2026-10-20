#include "Core.h"
#include "SystolicWS.h"
#include "SystolicOS.h"
#include "MemoryProfiler.h"

#include <cstdlib>
#include <string>

#include "helper/HelperFunctions.h"

std::unique_ptr<Core> Core::create(uint32_t id, SimulationConfig config) {
  if (config.core_config[id].core_type == CoreType::SYSTOLIC_WS) {
    return std::make_unique<SystolicWS>(id, config);
  } else if (config.core_config[id].core_type == CoreType::SYSTOLIC_OS) {
    return std::make_unique<SystolicOS>(id, config);
  } else {
      spdlog::error("[Configuration] Invalid core type...!");
    exit(EXIT_FAILURE);
  }
}

Core::Core(uint32_t id, SimulationConfig config)
    : _id(id),
      _config(config),
      _core_cycle(0),
      _stat_idle_cycle(0),
      _stat_memory_idle_cycle(0),
      _stat_vec_compute_cycle(0),
      _stat_matmul_cycle(0),
      _spad(Sram(config, _core_cycle, false, id)),
      _acc_spad(Sram(config, _core_cycle, true, id)) {
  const char* component_measure =
      std::getenv("ONNXSIM_COMPONENT_CHARACTERIZATION");
  _component_characterization =
      component_measure != nullptr && component_measure[0] != '\0' &&
      std::string(component_measure) != "0";
  auto environment_enabled = [](const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' &&
           std::string(value) != "0";
  };
  _next_sa_metadata_enabled =
      environment_enabled("ONNXSIM_SA_CONTROLLER_TRACE_DIR") ||
      environment_enabled("ONNXSIM_SA_PREDICTOR_MEASURE") ||
      environment_enabled("ONNXSIM_SA_RETENTION_SHADOW") ||
      environment_enabled("ONNXSIM_SA_EVENT_TRACE_DIR") ||
      environment_enabled("ONNXSIM_SA_HYBRID_CONFIG") ||
      environment_enabled("ONNXSIM_SA_UPDATED_PREDICTOR") ||
      environment_enabled("ONNXSIM_REGATE_UPDATED_PROFILE") ||
      environment_enabled("ONNXSIM_REGATE_EVENT_DRIVEN_PROFILE");
  _waiting_write_reqs = 0;
  _running_layer = -1;
}

bool Core::can_issue(bool is_accum_tile) {
  return _tiles.size() < 2;  // double buffer
}

void Core::issue(std::unique_ptr<Tile> op) {
  op->measurement_tile_id = _next_measurement_tile_id++;
  op->stat = {.start_cycle = _core_cycle,
             .cycles = 0,
             .compute_cycles = 0,
             .memory_stall = 0,
             .sram_reads = 0,
             .sram_writes = 0};
  int spad_id = 0;
  int acc_spad_id = 0;

  if (_tiles.size() == 1) {
    spad_id = _tiles[0]->spad_id;
    acc_spad_id = _tiles[0]->accum_spad_id;
  }

  /* Double buffer */
  spad_id = (spad_id + 1) % 2;
  _spad.flush(spad_id);
  if (!op->accum || !(_current_layer_id == op->layer_id && _current_fused_op_id == op->fused_op_id)) {
    /* Accumeulate tile uses same acc spad buffer */
    acc_spad_id = (acc_spad_id + 1) % 2;
    _acc_spad.flush(acc_spad_id);
  }
  _current_layer_id = op->layer_id;
  _current_fused_op_id = op->fused_op_id;

  op->spad_id = spad_id;
  op->accum_spad_id = acc_spad_id;
  if (_next_sa_metadata_enabled) {
    if (!op->sa_metadata_finalized) {
      throw std::logic_error(
          "tile reached a core before SA metadata finalization");
    }
    op->reset_runtime_sa_metadata();
  }
  op->status = Tile::Status::RUNNING;
  if (op->skip) {
    op->status = Tile::Status::FINISH;
    _finished_tiles.push(std::move(op));
    return;
  }
  if (_running_layer != op->layer_id) {
    _running_layer = op->layer_id;
  }
  _tiles.push_back(std::move(op));
}

std::unique_ptr<Tile> Core::pop_finished_tile() {
  std::unique_ptr<Tile> result = std::make_unique<Tile>(Tile{});
  result->status = Tile::Status::EMPTY;
  if (_finished_tiles.size() > 0) {
    result = std::move(_finished_tiles.front());
    _finished_tiles.pop();
  }
  return result;
}

void Core::set_scheduler_issue_state(SchedulerIssueState state) {
  _scheduler_issue_state = state;
}

void Core::set_scheduler_core_tile_waiting(
    bool waiting, bool front_skip, bool front_contains_gemm,
    size_t front_pre_gemm_instruction_count) {
  _scheduler_core_tile_waiting = waiting;
  _scheduler_front_tile_skip = front_skip;
  _scheduler_front_tile_contains_gemm = front_contains_gemm;
  _scheduler_front_tile_pre_gemm_instruction_count =
      front_pre_gemm_instruction_count;
}

cycle_type Core::get_resident_instruction_count() const {
  cycle_type instruction_count = 0;
  for (const std::unique_ptr<Tile>& tile : _tiles) {
    instruction_count += tile->instructions.size();
  }
  return instruction_count;
}

Core::SchedulerDrainSnapshot Core::get_scheduler_drain_snapshot() const {
  SchedulerDrainSnapshot snapshot;

  // The front instruction must complete before the current layer can finish.
  // Using only the front (rather than assuming finish-cycle ordering for the
  // complete queue) keeps this a conservative lower bound.
  if (!_compute_pipeline.empty() &&
      _compute_pipeline.front()->finish_cycle > _core_cycle) {
    snapshot.committed_compute_cycles =
        _compute_pipeline.front()->finish_cycle - _core_cycle;
  }
  if (!_vector_pipeline.empty() &&
      _vector_pipeline.front()->finish_cycle > _core_cycle) {
    snapshot.committed_vector_cycles =
        _vector_pipeline.front()->finish_cycle - _core_cycle;
  }

  auto count_memory_requests = [](const Instruction& inst) -> uint64_t {
    // Only future MOVIN traffic is a scheduler-unlock prerequisite.  A final
    // MOVOUT marks its tile complete when the write requests are generated,
    // so counting writes here would not be a valid admission lower bound.
    return inst.opcode == Opcode::MOVIN ? inst.src_addrs.size() : 0;
  };
  for (const std::unique_ptr<Tile>& tile : _tiles) {
    for (const std::unique_ptr<Instruction>& inst : tile->instructions) {
      snapshot.resident_future_memory_requests +=
          count_memory_requests(*inst);
    }
  }
  if (!_ld_inst_queue.empty()) {
    snapshot.staged_memory_requests +=
        count_memory_requests(*_ld_inst_queue.front());
  }
  snapshot.core_request_queue_depth = _request_queue.size();
  snapshot.outstanding_memory_requests =
      _outstanding_memory_accesses.size();
  snapshot.waiting_write_requests = _waiting_write_reqs;
  return snapshot;
}

void Core::set_scheduler_global_issue_drain_bound(cycle_type bound) {
  _scheduler_global_issue_drain_bound = bound;
}

void Core::set_scheduler_global_unlock_bounds(
    cycle_type committed_compute_bound,
    cycle_type committed_vector_bound,
    cycle_type memory_injection_bound,
    cycle_type enhanced_bound) {
  _scheduler_global_committed_compute_bound = committed_compute_bound;
  _scheduler_global_committed_vector_bound = committed_vector_bound;
  _scheduler_global_memory_injection_bound = memory_injection_bound;
  _scheduler_global_enhanced_drain_bound = enhanced_bound;
}

void Core::cycle() {
  _last_core_cycle_issued_instruction = false;
  _last_core_cycle_had_pending_instruction = false;
  _last_core_cycle_memory_block = false;
  _last_core_cycle_dependency_block = false;
  _last_core_cycle_unknown_block = false;
  _last_core_cycle_compute_issue_block = false;
  _last_instruction_flow_events.clear();
  _last_retired_tile_ids.clear();

  _core_cycle++;
  const bool spad_idle = _spad.cycle();
  const bool accum_spad_idle = _acc_spad.cycle();
  if (spad_idle && accum_spad_idle) {
    _all_sram_modeled_idle_cycles++;
  }
  for (int tile_iter = 0; tile_iter < _tiles.size(); tile_iter++) {
    int i = (tile_iter + tile_rr) % _tiles.size();
    if(_tiles[i]->instructions.empty()) 
      continue;
    _last_core_cycle_had_pending_instruction = true;
    std::unique_ptr<Instruction>& inst = _tiles[i]->instructions.front();
    const Instruction* issued_instruction = inst.get();
    inst->measurement_tile_id = _tiles[i]->measurement_tile_id;
    if(_tiles[i]->instructions.size() == 1) {
      inst->last_inst = true;
      inst->my_tile = _tiles[i].get();
    }
    inst->spad_id = _tiles[i]->spad_id;
    inst->accum_spad_id = _tiles[i]->accum_spad_id;
    Sram *buffer;
    int buffer_id;
    if (inst->dest_addr >= ACCUM_SPAD_BASE) {
      buffer = &_acc_spad;
      buffer_id = _tiles[i]->accum_spad_id;
    } else {
      buffer = &_spad;
      buffer_id = _tiles[i]->spad_id;
    }
    bool issued = false;
    InstructionBlockReason block_reason = InstructionBlockReason::NONE;
    if (inst->opcode == Opcode::MOVIN) {
      /*LD inst queue */
      if (inst->size == 0) {
        spdlog::error("[Core {}] MVIN issue addr: {:x}, size: {:x}", _id, inst->dest_addr, inst->size);
      }
      if (!buffer->check_allocated(inst->dest_addr, buffer_id) &&
          buffer->check_remain(inst->size, buffer_id)) {
        _ld_inst_queue.push(std::move(inst));
        issued = true;
      } else {
        /*Invalid state */
        spdlog::error("Destination allocated: {} Size remain: {}", buffer->check_allocated(inst->dest_addr, buffer_id), buffer->check_remain(inst->size, buffer_id));
        spdlog::error("[Core {}] MVIN issue panic addr: {:x}, size: {} B", _id, inst->dest_addr, inst->size*_config.dram_req_size);
        buffer->print_all(buffer_id);
        exit(EXIT_FAILURE);
      }
    } else if (inst->opcode == Opcode::MOVOUT ||
               inst->opcode == Opcode::MOVOUT_POOL) {
      /* ST inst queue */
      if (buffer->check_hit(inst->dest_addr, buffer_id)) {
        _st_inst_queue.push(std::move(inst));
        issued = true;
      } else {
        record_instruction_issue_wait(get_store_wait_reason(*inst));
        OperandWaitReason wait_reason = get_store_wait_reason(*inst);
        block_reason = wait_reason == OperandWaitReason::MEMORY
            ? InstructionBlockReason::MEMORY
            : wait_reason == OperandWaitReason::DEPENDENCY
                ? InstructionBlockReason::DEPENDENCY
                : InstructionBlockReason::UNKNOWN;
      }
    } else {
      /* Ex inst queue */
      if (inst.get() == 0) {
        spdlog::error("null instruction!");
      }
      if(_ex_inst_queue.empty() && can_issue_compute(inst)) {
        _stat_ex_issue++;
        //end
        _ex_inst_queue.push(std::move(inst));
        issued = true;
      } else if (!_ex_inst_queue.empty()) {
        block_reason = InstructionBlockReason::EXECUTION_QUEUE_BUSY;
      } else if (_ex_inst_queue.empty()) {
        _last_core_cycle_compute_issue_block = true;
        OperandWaitReason wait_reason = get_operand_wait_reason(*inst);
        if (wait_reason == OperandWaitReason::NONE) {
          _last_core_cycle_unknown_block = true;
          block_reason = specialized_compute_block_reason();
        } else {
          record_instruction_issue_wait(wait_reason);
          block_reason = wait_reason == OperandWaitReason::MEMORY
              ? InstructionBlockReason::MEMORY
              : wait_reason == OperandWaitReason::DEPENDENCY
                  ? InstructionBlockReason::DEPENDENCY
                  : InstructionBlockReason::UNKNOWN;
        }
      }
    }
    InstructionFlowEvent flow_event{
        .cycle = _core_cycle,
        .tile_id = _tiles[i]->measurement_tile_id,
        .layer_id = _tiles[i]->layer_id,
        .fused_op_id = _tiles[i]->fused_op_id,
        .operation = _tiles[i]->optype,
        .instruction_ordinal = inst
            ? inst->tile_instruction_ordinal
            : issued_instruction->tile_instruction_ordinal,
        .opcode = issued_instruction->opcode,
        .issued = issued,
        .block_reason = issued ? InstructionBlockReason::NONE : block_reason,
        .previous_issued_tile_id = _last_issued_tile_id,
        .tile_switch = issued && _last_issued_tile_id != 0 &&
                       _last_issued_tile_id != _tiles[i]->measurement_tile_id,
        .next_sa_prefix_before = _tiles[i]->next_sa_prefix_count,
        .weight_block_id = issued_instruction->sa_weight_block_id,
        .weight_block_known = issued_instruction->sa_weight_block_known};
    _last_instruction_flow_events.push_back(std::move(flow_event));
    if (issued) {
      _last_core_cycle_issued_instruction = true;
      if (_next_sa_metadata_enabled) {
        _tiles[i]->advance_sa_metadata(issued_instruction);
      }
      _tiles[i]->instructions.pop_front();
      _last_issued_tile_id = _tiles[i]->measurement_tile_id;
      tile_rr = i;
      break;
    }
  }
  for (auto tile = _tiles.begin() ; tile < _tiles.end(); tile++) {
    if ((*tile)->instructions.empty() && (*tile)->inst_finished) {
      _last_retired_tile_ids.push_back((*tile)->measurement_tile_id);
      (*tile)->status = Tile::Status::FINISH;
      (*tile)->stat.cycles = _core_cycle - (*tile)->stat.start_cycle;
      (*tile)->stat.memory_stall =
          (*tile)->stat.cycles - (*tile)->stat.compute_cycles;
      _finished_tiles.push(std::move(*tile));
      _tiles.erase(tile);
      break;
    }
  }
  if(_config.core_print_interval && _core_cycle % _config.core_print_interval == 0) {
    print_current_stats();
  }
}

Core::InstructionBlockReason Core::specialized_compute_block_reason() const {
  return InstructionBlockReason::UNKNOWN;
}

bool Core::running() {
  bool running = false;
  running = running || _tiles.size() > 0;
  running = running || !_compute_pipeline.empty();
  running = running ||
            !_vector_pipeline.empty();  // Vector unit (Might need to modify)
  running = running || _waiting_write_reqs != 0;
  running = running || !_ld_inst_queue.empty();
  running = running || !_st_inst_queue.empty();
  running = running || !_ex_inst_queue.empty();
  return running;
}

bool Core::has_memory_request() { return _request_queue.size() > 0; }

void Core::pop_memory_request() {
  assert(has_memory_request());
  if (_request_queue.front()->core_queue_ticket !=
      _last_dequeued_core_queue_ticket + 1) {
    _core_queue_tickets_exact = false;
  }
  _last_dequeued_core_queue_ticket =
      _request_queue.front()->core_queue_ticket;
  _request_queue.pop();
}

void Core::push_memory_response(MemoryAccess *response) {
  assert(!response->request);  // can only push response

  transition_memory_stage(
      response,
      MemoryStage::DONE,
      _core_cycle);

  MemoryProfiler::instance().finalize(response);

  if (response->write) {
    _waiting_write_reqs--;
  } else if (response->spad_address >= ACCUM_SPAD_BASE) {
    _acc_spad.fill(response->spad_address, response->buffer_id,
                   Sram::PendingProducer::MEMORY);
  } else {
    assert(_spad.check_allocated(response->spad_address,
                                 response->buffer_id));
    _spad.fill(response->spad_address, response->buffer_id,
               Sram::PendingProducer::MEMORY);
  }

  _outstanding_memory_accesses.erase(response->id);
  delete response;
}

Core::OperandWaitReason Core::get_data_wait_reason(const Sram& buffer,
                                                     addr_type address,
                                                     int buffer_id) const {
  Sram::EntryState state = buffer.get_entry_state(address, buffer_id);
  if (state.allocated && state.valid) {
    return OperandWaitReason::NONE;
  }

  if (!state.allocated) {
    return OperandWaitReason::UNKNOWN;
  }

  if (state.pending_memory_count > 0 && state.pending_compute_count > 0) {
    return OperandWaitReason::UNKNOWN;
  }
  if (state.pending_memory_count > 0) {
    return OperandWaitReason::MEMORY;
  }
  if (state.pending_compute_count > 0) {
    return OperandWaitReason::DEPENDENCY;
  }

  return OperandWaitReason::UNKNOWN;
}

Core::OperandWaitReason Core::get_operand_wait_reason(const Instruction& inst) const {
  bool saw_memory = false;
  bool saw_dependency = false;
  bool saw_unknown = false;

  for (addr_type addr : inst.src_addrs) {
    const Sram* buffer;
    int buffer_id;
    if (inst.src_from_accum && addr >= ACCUM_SPAD_BASE) {
      buffer = &_acc_spad;
      buffer_id = inst.accum_spad_id;
    } else {
      buffer = &_spad;
      buffer_id = inst.spad_id;
    }

    OperandWaitReason reason = get_data_wait_reason(*buffer, addr, buffer_id);
    if (reason == OperandWaitReason::MEMORY) {
      saw_memory = true;
    } else if (reason == OperandWaitReason::DEPENDENCY) {
      saw_dependency = true;
    } else if (reason == OperandWaitReason::UNKNOWN) {
      saw_unknown = true;
    }
  }

  if (saw_unknown || (saw_memory && saw_dependency)) {
    return OperandWaitReason::UNKNOWN;
  }
  if (saw_memory) {
    return OperandWaitReason::MEMORY;
  }
  if (saw_dependency) {
    return OperandWaitReason::DEPENDENCY;
  }
  return OperandWaitReason::NONE;
}

Core::OperandWaitReason Core::get_store_wait_reason(const Instruction& inst) const {
  const Sram* buffer;
  int buffer_id;
  if (inst.dest_addr >= ACCUM_SPAD_BASE) {
    buffer = &_acc_spad;
    buffer_id = inst.accum_spad_id;
  } else {
    buffer = &_spad;
    buffer_id = inst.spad_id;
  }

  return get_data_wait_reason(*buffer, inst.dest_addr, buffer_id);
}

bool Core::can_issue_compute(std::unique_ptr<Instruction>& inst) {
  bool result = get_operand_wait_reason(*inst) == OperandWaitReason::NONE;

  if (!result) {
    for (addr_type addr : inst->src_addrs) {
      spdlog::trace("Core[{}] Dependency fail : {} , {}", _id, addr,
                    _spad.check_hit(addr, inst->spad_id));
    }
  }
  return result;
}

void Core::record_compute_operand_reads(const Instruction& inst) {
  for (addr_type address : inst.src_addrs) {
    if (inst.src_from_accum && address >= ACCUM_SPAD_BASE) {
      _acc_spad.record_modeled_read(address, inst.accum_spad_id);
    } else {
      _spad.record_modeled_read(address, inst.spad_id);
    }
  }
}

void Core::record_instruction_issue_wait(OperandWaitReason reason) {
  switch (reason) {
    case OperandWaitReason::NONE:
      break;
    case OperandWaitReason::MEMORY:
      _last_core_cycle_memory_block = true;
      break;
    case OperandWaitReason::DEPENDENCY:
      _last_core_cycle_dependency_block = true;
      break;
    case OperandWaitReason::UNKNOWN:
      _last_core_cycle_unknown_block = true;
      break;
  }
}

void Core::attribute_core_cycle(CoreAttributionCategory category) {
  switch (category) {
    case CoreAttributionCategory::COMPUTE:
      _attrib_compute_cycles++;
      break;
    case CoreAttributionCategory::MEMORY_STALL:
      _attrib_memory_stall_cycles++;
      break;
    case CoreAttributionCategory::DEPENDENCY_STALL:
      _attrib_dependency_stall_cycles++;
      break;
    case CoreAttributionCategory::SCHEDULER_STALL:
      _attrib_scheduler_stall_cycles++;
      break;
    case CoreAttributionCategory::OTHER:
      _attrib_other_cycles++;
      break;
  }
}

cycle_type Core::attribution_total_cycles() const {
  return _attrib_compute_cycles +
         _attrib_memory_stall_cycles +
         _attrib_dependency_stall_cycles +
         _attrib_scheduler_stall_cycles +
         _attrib_other_cycles;
}

void Core::print_execution_time_attribution() {
  cycle_type total = attribution_total_cycles();
  auto pct = [&](cycle_type value) {
    return total ? (100.0 * value / total) : 0.0;
  };

  spdlog::info("========== Exact Core Cycle Attribution ==========");
  spdlog::info("Core [{}] : Attribution Total Core Cycles : {}", _id, total);
  spdlog::info("Core [{}] : Compute          : {} ({:.2f}%)",
               _id, _attrib_compute_cycles, pct(_attrib_compute_cycles));
  spdlog::info("Core [{}] : Memory Stall     : {} ({:.2f}%)",
               _id, _attrib_memory_stall_cycles, pct(_attrib_memory_stall_cycles));
  spdlog::info("Core [{}] : Dependency Stall : {} ({:.2f}%)",
               _id, _attrib_dependency_stall_cycles, pct(_attrib_dependency_stall_cycles));
  spdlog::info("Core [{}] : Scheduler Stall  : {} ({:.2f}%)",
               _id, _attrib_scheduler_stall_cycles, pct(_attrib_scheduler_stall_cycles));
  spdlog::info("Core [{}] : Other            : {} ({:.2f}%)",
               _id, _attrib_other_cycles, pct(_attrib_other_cycles));

  if (total != _core_cycle) {
    spdlog::error("Core [{}] : Attribution mismatch. total={} core_cycles={}",
                  _id, total, _core_cycle);
    assert(total == _core_cycle);
  }
}

void Core::print_stats() {
  update_stats();
  spdlog::info(
      "Core [{}] : MatMul active cycle {} Vector active cycle {} ",
      _id, _stat_tot_matmul_cycle, _stat_tot_vec_compute_cycle);

  spdlog::info(
      "Core [{}] : Memory unit idle cycle {} Systolic bubble cycle {} "
      "Core idle cycle {} ",
      _id, _stat_tot_memory_idle_cycle, _stat_tot_systolic_bubble_cycle, _stat_tot_idle_cycle);

  spdlog::info("Core [{}] : Systolic Array Utilization(%) {:.2f} ({:.2f}% PE util), Vector Unit Utilization(%) {:.2f}, Total cycle: {}",
      _id, static_cast<float>(_stat_tot_systolic_active_cycle * 100) / _core_cycle,
      static_cast<float>(_stat_tot_matmul_cycle * 100) / _core_cycle,
      static_cast<float>(_stat_tot_vec_compute_cycle * 100) / _core_cycle, _core_cycle);

  // for analysis
  spdlog::info(
    "Core [{}] : EX instructions issued = {}",
    _id,
    _stat_ex_issue);
  //end

  if (_component_characterization) {
    print_execution_time_attribution();
    auto percent = [&](cycle_type value) {
      return _core_cycle == 0
                 ? 0.0
                 : 100.0 * static_cast<double>(value) /
                       static_cast<double>(_core_cycle);
    };
    const cycle_type sa_idle_cycles =
        _core_cycle - _stat_tot_systolic_active_cycle;
    const cycle_type vector_idle_cycles =
        _core_cycle - _stat_tot_vec_compute_cycle;
    const cycle_type neither_compute_active_cycles =
        _core_cycle - _stat_tot_compute_cycle;
    const cycle_type core_request_fifo_nonempty_cycles =
        _core_cycle - _stat_tot_memory_idle_cycle;
    auto sram_record = [&](const Sram& sram) {
      return json{{"modeled_cycles", sram.modeled_cycles()},
                  {"modeled_idle_cycles", sram.modeled_idle_cycles()},
                  {"modeled_active_cycles",
                   sram.modeled_cycles() - sram.modeled_idle_cycles()},
                  {"modeled_read_active_cycles",
                   sram.modeled_read_active_cycles()},
                  {"modeled_write_active_cycles",
                   sram.modeled_write_active_cycles()},
                  {"modeled_read_events", sram.modeled_read_events()},
                  {"modeled_write_events", sram.modeled_write_events()}};
    };
    json record{
        {"version", 2},
        {"core_id", _id},
        {"core_cycles", _core_cycle},
        {"sa_active_cycles", _stat_tot_systolic_active_cycle},
        {"sa_idle_cycles", sa_idle_cycles},
        {"sa_active_percent", percent(_stat_tot_systolic_active_cycle)},
        {"sa_idle_percent", percent(sa_idle_cycles)},
        {"vector_active_cycles", _stat_tot_vec_compute_cycle},
        {"vector_idle_cycles", vector_idle_cycles},
        {"vector_active_percent", percent(_stat_tot_vec_compute_cycle)},
        {"vector_idle_percent", percent(vector_idle_cycles)},
        {"sa_or_vector_active_cycles", _stat_tot_compute_cycle},
        {"sa_and_vector_active_cycles",
         _stat_tot_sa_vector_overlap_cycle},
        {"neither_sa_nor_vector_active_cycles",
         neither_compute_active_cycles},
        {"core_request_fifo_nonempty_cycles",
         core_request_fifo_nonempty_cycles},
        {"core_request_fifo_nonempty_percent",
         percent(core_request_fifo_nonempty_cycles)},
        {"core_request_fifo_empty_cycles", _stat_tot_memory_idle_cycle},
        {"spad_model",
         {{"main", sram_record(_spad)},
          {"accumulator", sram_record(_acc_spad)},
          {"all_devices_modeled_idle_cycles",
           _all_sram_modeled_idle_cycles},
          {"all_devices_modeled_active_cycles",
           _core_cycle - _all_sram_modeled_idle_cycles},
          {"cycle_count_evidence_class", "measured"},
          {"measurement_scope", "instruction-level simulator events"},
          {"idle_predicate",
           "main_read_or_write_events == 0 && accumulator_read_or_write_events == 0"},
          {"all_devices_model_semantics",
           "no explicit logical read or write event in either whole SRAM "
           "during the core cycle"},
          {"model_limit",
           "logical event-level model only; no bank, port, SRAM latency, "
           "or physical macro activity is inferred"}}},
        {"execution_attribution",
         {{"compute", _attrib_compute_cycles},
          {"memory_stall", _attrib_memory_stall_cycles},
          {"dependency_stall", _attrib_dependency_stall_cycles},
          {"scheduler_stall", _attrib_scheduler_stall_cycles},
          {"other", _attrib_other_cycles}}},
        {"accounting_exact", attribution_total_cycles() == _core_cycle},
        {"fifo_semantics",
         "core-to-NoC request FIFO occupancy sampled once per core cycle"}};
    spdlog::info("COMPONENT_CORE {}", record.dump());
  }
}

void Core::print_current_stats() {
  auto level = spdlog::level::info;
  if(_id != 0) 
    level = spdlog::level::debug;
    spdlog::log(level,
      "Core [{}] : MatMul active cycle {} Vector active cycle {} ",
      _id, _stat_matmul_cycle, _stat_vec_compute_cycle);

  spdlog::log(level,
      "Core [{}] : issued tile {} ", _id, _tiles.size());

  spdlog::log(level,
      "Core [{}] : Memory unit idle cycle {} Systolic bubble cycle {} "
      "Core idle cycle {} ",
      _id, _stat_memory_idle_cycle, _stat_systolic_bubble_cycle, _stat_idle_cycle);
  spdlog::log(level,"Core [{}] : Systolic Array Utilization(%) {:.2f} ({:.2f}% PE util), Vector Unit Utilization(%) {:.2f}, Total cycle: {}",
      _id, static_cast<float>(_stat_systolic_active_cycle * 100) / _config.core_print_interval,
      static_cast<float>(_stat_matmul_cycle * 100) / _config.core_print_interval,
      static_cast<float>(_stat_vec_compute_cycle * 100) / _config.core_print_interval, _core_cycle);
  update_stats();
}

void Core::update_stats() {
  _stat_tot_compute_cycle += _stat_compute_cycle;
  _stat_tot_sa_vector_overlap_cycle += _stat_sa_vector_overlap_cycle;
  _stat_tot_systolic_active_cycle += _stat_systolic_active_cycle;
  _stat_tot_systolic_bubble_cycle += _stat_systolic_bubble_cycle;
  _stat_tot_memory_idle_cycle += _stat_memory_idle_cycle;
  _stat_tot_idle_cycle += _stat_idle_cycle;
  _stat_tot_vec_compute_cycle += _stat_vec_compute_cycle;
  _stat_tot_matmul_cycle += _stat_matmul_cycle;
  _stat_compute_cycle = 0;
  _stat_sa_vector_overlap_cycle = 0;
  _stat_systolic_active_cycle = 0;
  _stat_systolic_bubble_cycle = 0;
  _stat_memory_idle_cycle = 0;
  _stat_idle_cycle = 0;
  _stat_vec_compute_cycle = 0;
  _stat_matmul_cycle = 0;
}

void Core::finish_compute_pipeline(){
  if (!_compute_pipeline.empty() &&
      _compute_pipeline.front()->finish_cycle <= _core_cycle) {
    std::unique_ptr<Instruction> inst = std::move(_compute_pipeline.front());
    if (inst->dest_addr >= ACCUM_SPAD_BASE)
      _acc_spad.fill(inst->dest_addr, inst->accum_spad_id,
                     Sram::PendingProducer::COMPUTE);
    else
      _spad.fill(inst->dest_addr, inst->spad_id,
                 Sram::PendingProducer::COMPUTE);
    if(inst->last_inst) {
      spdlog::trace("Finished last GEMM {}", inst->spad_id);
      inst->my_tile->inst_finished = true;
    }
    double compute_size = inst->tile_k * inst->tile_m * inst->tile_n
                            / (_config.core_config[_id].core_height * _config.core_config[_id].core_width);
    spdlog::trace("Compute size {} tile m {} tile k {} tile n {}", inst->compute_size, inst->tile_m, inst->tile_k, inst->tile_n);
    spdlog::trace("Compute size {} , compute time {}", compute_size, inst->finish_cycle - inst->start_cycle);
    _stat_matmul_cycle += compute_size;
    _compute_pipeline.pop();
  }
}

void Core::finish_vector_pipeline() {
  if (!_vector_pipeline.empty() &&
      _vector_pipeline.front()->finish_cycle <= _core_cycle) {
    std::unique_ptr<Instruction> inst = std::move(_vector_pipeline.front());
    if (inst->dest_addr >= ACCUM_SPAD_BASE) {
      if(!_acc_spad.check_allocated(inst->dest_addr, inst->accum_spad_id)) {
        spdlog::error("Vector pipeline -> accum");
        spdlog::error("Destination not allocated {}", inst->dest_addr);
      }
      _acc_spad.fill(inst->dest_addr, inst->accum_spad_id,
                     Sram::PendingProducer::COMPUTE);
    }
    else {
      if(!_spad.check_allocated(inst->dest_addr, inst->accum_spad_id)) {
        spdlog::error("Vector pipeline -> spad");
        spdlog::error("Destination not allocated {}", inst->dest_addr);
      }
      _spad.fill(inst->dest_addr, inst->spad_id,
                 Sram::PendingProducer::COMPUTE);
    }
      
    if(inst->last_inst)
      inst->my_tile->inst_finished = true;
    _vector_pipeline.pop();
  }
}

void Core::handle_ld_inst_queue() {
  if (!_ld_inst_queue.empty()) {
    std::unique_ptr<Instruction> front = std::move(_ld_inst_queue.front());
    if (front->opcode == Opcode::MOVIN) {
      Sram *buffer;
      int buffer_id;
      if (front->dest_addr >= ACCUM_SPAD_BASE) {
        buffer = &_acc_spad;
        buffer_id = front->accum_spad_id;
      } else {
        buffer = &_spad;
        buffer_id = front->spad_id;
      }
      if (front->size==0) {
        spdlog::error("Destination size is 0! opcode: {}, addr: 0x{:x}", (int)front->opcode, front->dest_addr);
      }
      uint64_t memory_tail_core_queue_ticket =
          front->src_addrs.empty()
              ? 0
              : _next_core_queue_ticket + front->src_addrs.size() - 1;
      int ret = buffer->prefetch(front->dest_addr, buffer_id, front->size,
                                 front->size,
                                 Sram::PendingProducer::MEMORY,
                                 memory_tail_core_queue_ticket);
      if (!ret) {
        spdlog::error("Destination allocated: {} Size remain: {}", buffer->check_allocated(front->dest_addr, buffer_id), buffer->check_remain(front->size, buffer_id));
        spdlog::error("instruction panic opcode: {:x}, addr: {:x}, size: {} B", (int)front->opcode, front->dest_addr, front->size*_config.dram_req_size);
        std::exit(EXIT_FAILURE);
      }
      for (addr_type addr : front->src_addrs) {
        assert(front->base_addr != GARBEGE_ADDR);
        MemoryAccess *access =
            new MemoryAccess({.id = generate_mem_access_id(),
                              .dram_address = addr + front->base_addr,
                              .spad_address = front->dest_addr,
                              .size = _config.dram_req_size,
                              .write = false,
                              .request = true,
                              .core_id = _id,
                              .start_cycle = _core_cycle,
                              .buffer_id = buffer_id,
                              .core_queue_ticket =
                                  _next_core_queue_ticket++});
        access->stage = MemoryStage::CORE_REQUEST_QUEUE;
        access->stage_enter_cycle = _core_cycle;
        _outstanding_memory_accesses.emplace(access->id, access);
        _request_queue.push(access);
      }
      _ld_inst_queue.pop();
    } else {
      assert(0);
    }
  }
}

void Core::handle_st_inst_queue() {
  if (!_st_inst_queue.empty()) {
    std::unique_ptr<Instruction> front = std::move(_st_inst_queue.front());
    if (front->opcode == Opcode::MOVOUT || front->opcode == Opcode::MOVOUT_POOL) {
      Sram *buffer;
      int buffer_id;
      if (front->dest_addr >= ACCUM_SPAD_BASE) {
        buffer = &_acc_spad;
        buffer_id = front->accum_spad_id;
      } else {
        buffer = &_spad;
        buffer_id = front->spad_id;
      }
      if(buffer->check_hit(front->dest_addr, buffer_id)) {
        if (!front->src_addrs.empty()) {
          buffer->record_modeled_read(front->dest_addr, buffer_id,
                                      front->src_addrs.size());
        }
        for (addr_type addr : front->src_addrs) {
          assert(front->base_addr != GARBEGE_ADDR);
          MemoryAccess *access =
              new MemoryAccess{.id = generate_mem_access_id(),
                              .dram_address = addr + front->base_addr,
                              .spad_address = front->dest_addr,
                              .size = _config.dram_req_size,
                              .write = true,
                              .request = true,
                              .core_id = _id,
                              .start_cycle = _core_cycle,
                              .buffer_id = buffer_id,
                              .core_queue_ticket =
                                  _next_core_queue_ticket++};
          access->stage = MemoryStage::CORE_REQUEST_QUEUE;
          access->stage_enter_cycle = _core_cycle;
          _outstanding_memory_accesses.emplace(access->id, access);
          _waiting_write_reqs++;
          _request_queue.push(access);
        }
        if(front->last_inst) {
          spdlog::trace("Finished last store {}", front->spad_id);
          front->my_tile->inst_finished = true;
        }
        _st_inst_queue.pop();
      }
    } else {
      assert(0);
    }
  }
}

cycle_type Core::calculate_add_tree_iterations(uint32_t vector_size) {
  uint32_t calculation_unit = _config.core_config[_id].vector_process_bit >> 3;
  if (vector_size <= calculation_unit) {
    return 1;
  }

  uint32_t ret = vector_size / calculation_unit;
  if (vector_size % calculation_unit != 0) {
    ret++;
  }
  return ret + calculate_add_tree_iterations(ret);
}

cycle_type Core::calculate_vector_op_iterations(uint32_t vector_size) {
  uint32_t calculation_unit = _config.core_config[_id].vector_process_bit >> 3;
  uint32_t ret = vector_size / calculation_unit;
  if (vector_size % calculation_unit != 0) {
    ret++;
  }
  return ret;
}
