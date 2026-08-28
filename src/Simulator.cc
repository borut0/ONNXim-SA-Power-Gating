#include "Simulator.h"

#include <algorithm>
#include <filesystem>
#include <cstdlib>
#include <string>

#include "SystolicOS.h"
#include "SystolicWS.h"

#include "MemoryProfiler.h"

namespace fs = std::filesystem;

Simulator::Simulator(SimulationConfig config, bool language_mode)
    : _config(config), _core_cycles(0), _language_mode(language_mode) {
  const char* component_measure =
      std::getenv("ONNXSIM_COMPONENT_CHARACTERIZATION");
  _component_characterization =
      component_measure != nullptr && component_measure[0] != '\0' &&
      std::string(component_measure) != "0";
  const char* scheduler_unlock_measure =
      std::getenv("ONNXSIM_SA_SCHEDULER_UNLOCK_MEASURE");
  _scheduler_unlock_measure =
      scheduler_unlock_measure != nullptr &&
      scheduler_unlock_measure[0] != '\0' &&
      std::string(scheduler_unlock_measure) != "0";
  if (_component_characterization) {
    // Broad paper runs need only final aggregate records.  Suppressing
    // periodic progress statistics changes logging, never simulated state.
    config.core_print_interval = 0;
    config.icnt_print_interval = 0;
    config.dram_print_interval = 0;
    _config.core_print_interval = 0;
    _config.icnt_print_interval = 0;
    _config.dram_print_interval = 0;
  }
  // Create dram object
  spdlog::info("Simulator Configuration:");
  for (int i=0; i<config.num_cores;i++)
    spdlog::info("[Core {}] Systolic Array Throughput: {} GFLOPS, Spad size: {} KB, Accumulator size: {} KB",
      i, config.max_systolic_flops(i), config.core_config[i].spad_size, config.core_config[i].accum_spad_size);
  spdlog::info("DRAM Bandwidth {} GB/s", config.max_dram_bandwidth());
  _core_period = 1000000 / (config.core_freq);
  _icnt_period = 1000000 / (config.icnt_freq);
  _dram_period = 1000000 / (config.dram_freq);
  _core_time = 0;
  _dram_time = 0;
  _icnt_time = 0;
  char* onnxim_path_env = std::getenv("ONNXIM_HOME");
  std::string onnxim_path = onnxim_path_env != NULL?
  std::string(onnxim_path_env) : std::string("./");
  if (config.dram_type == DramType::SIMPLE) {
    _dram = std::make_unique<SimpleDram>(config);
  } else if (config.dram_type == DramType::RAMULATOR1) {
    std::string ramulator_config = fs::path(onnxim_path)
                                       .append("configs")
                                       .append(config.dram_config_path)
                                       .string();
    spdlog::info("Ramulator config: {}", ramulator_config);
    config.dram_config_path = ramulator_config;
    _dram = std::make_unique<DramRamulator>(config);
  } 
  else if (config.dram_type == DramType::RAMULATOR2) 
  {
    std::string ramulator_config = fs::path(onnxim_path)
                                       .append("configs")
                                       .append(config.dram_config_path)
                                       .string();
    spdlog::info("Ramulator2 config: {}", ramulator_config);
    config.dram_config_path = ramulator_config;
    _dram = std::make_unique<DramRamulator2>(config);
  } 
  else {
    spdlog::error("[Configuration] Invalid DRAM type...!");
    exit(EXIT_FAILURE);
  }

  // Create interconnect object
  if (config.icnt_type == IcntType::SIMPLE) {
    _icnt = std::make_unique<SimpleInterconnect>(config);
  } else if (config.icnt_type == IcntType::BOOKSIM2) {
    _icnt = std::make_unique<Booksim2Interconnect>(config);
  } else {
    spdlog::error("[Configuration] {} Invalid interconnect type...!");
    exit(EXIT_FAILURE);
  }
  _icnt_interval = config.icnt_print_interval;

  // Create core objects
  _cores.resize(config.num_cores);
  _n_cores = config.num_cores;
  _n_memories = config.dram_channels;
  _noc_node_per_core = config.icnt_injection_ports_per_core;
  _memory_req_size = config.dram_req_size;
  for (int core_index = 0; core_index < _n_cores; core_index++) {
    _cores[core_index] = Core::create(core_index, config);
  }

  //Configure Hardware Scheduler
  _scheduler = Scheduler::create(_config, &_core_cycles, &_core_time, this);
  
  /* Create heap */
  std::make_heap(_models.begin(), _models.end(), CompareModel());
}

void Simulator::run_simulator() {
  spdlog::info("======Start Simulation=====");
  cycle();
}

void Simulator::handle_model() {
  if(_language_mode) {
    _lang_scheduler->cycle();
    if(_lang_scheduler->can_schedule_model()) {
      _models.push_back(_lang_scheduler->pop_model());
      std::push_heap(_models.begin(), _models.end(), CompareModel());
    }
  }
  while (!_models.empty() && _models.front()->get_request_time() <= _core_time) {
    std::unique_ptr<Model> launch_model = std::move(_models.front());
    std::pop_heap(_models.begin(), _models.end(), CompareModel());
    _models.pop_back();

    launch_model->initialize_model(_weight_table[launch_model->get_name()]);
    launch_model->set_request_time(_core_time);
    spdlog::info("Schedule model: {} at {} us", launch_model->get_name(), _core_time);
    _scheduler->schedule_model(std::move(launch_model), 1);
  }
}

void Simulator::cycle() {
  OpStat op_stat;
  ModelStat model_stat;
  uint32_t tile_count;
  bool is_accum_tile;
  while (running()) {
    int model_id = 0;

    set_cycle_mask();
    // Core Cycle
    if (_cycle_mask & CORE_MASK) {
      /* Handle requested model */
      handle_model();

      for (int core_id = 0; core_id < _n_cores; core_id++) {
        std::unique_ptr<Tile> finished_tile = _cores[core_id]->pop_finished_tile();
        if (finished_tile->status == Tile::Status::FINISH) {
          _scheduler->finish_tile(core_id, finished_tile->layer_id);
        }
        // Issue new tile to core
        Core::SchedulerIssueState scheduler_issue_state =
            Core::SchedulerIssueState::NO_SCHEDULED_REQUEST;
        if (!_scheduler->empty()) {
          is_accum_tile = _scheduler->is_accum_tile(core_id, 0);
          if (_cores[core_id]->can_issue(is_accum_tile)) {
            std::unique_ptr<Tile> tile = _scheduler->get_tile(core_id);
            if (tile->status == Tile::Status::INITIALIZED) {
              _cores[core_id]->issue(std::move(tile));
              _tile_timestamp.push_back(std::chrono::high_resolution_clock::now());
              scheduler_issue_state = Core::SchedulerIssueState::TILE_ISSUED;
            } else {
              scheduler_issue_state =
                  Core::SchedulerIssueState::NO_TILE_AVAILABLE;
            }
          } else {
            scheduler_issue_state = Core::SchedulerIssueState::CORE_NOT_READY;
          }
        }
        bool front_skip = false;
        bool front_contains_gemm = false;
        size_t front_pre_gemm_instruction_count = 0;
        bool core_tile_waiting = _scheduler->get_core_front_tile_info(
            core_id, front_skip, front_contains_gemm,
            front_pre_gemm_instruction_count);
        _cores[core_id]->set_scheduler_core_tile_waiting(
            core_tile_waiting, front_skip, front_contains_gemm,
            front_pre_gemm_instruction_count);
        cycle_type global_issue_drain_bound = 1;
        cycle_type global_committed_compute_bound = 0;
        cycle_type global_committed_vector_bound = 0;
        cycle_type global_memory_injection_bound = 0;
        cycle_type global_enhanced_drain_bound = 0;
        if (_config.scheduler_type == "simple" &&
            _scheduler->simple_global_tile_queue_empty()) {
          for (int measured_core = 0; measured_core < _n_cores;
               measured_core++) {
            cycle_type core_instruction_count =
                _cores[measured_core]->get_resident_instruction_count() +
                _scheduler->get_core_queued_instruction_count(measured_core);
            if (core_instruction_count > global_issue_drain_bound) {
              global_issue_drain_bound = core_instruction_count;
            }

            if (_scheduler_unlock_measure) {
              Core::SchedulerDrainSnapshot snapshot =
                  _cores[measured_core]->get_scheduler_drain_snapshot();
              global_committed_compute_bound =
                  std::max(global_committed_compute_bound,
                           snapshot.committed_compute_cycles);
              global_committed_vector_bound =
                  std::max(global_committed_vector_bound,
                           snapshot.committed_vector_cycles);

              uint64_t request_count =
                  snapshot.resident_future_memory_requests +
                  snapshot.staged_memory_requests +
                  _scheduler->get_core_queued_memory_request_count(
                      measured_core);
              uint64_t maximum_icnt_events_per_core_cycle =
                  _icnt_period == 0
                      ? 0
                      : (_core_period + _icnt_period - 1) /
                            _icnt_period;
              uint64_t request_capacity_per_core_cycle =
                  maximum_icnt_events_per_core_cycle *
                  _config.icnt_injection_ports_per_core;
              cycle_type memory_bound =
                  request_count == 0 || request_capacity_per_core_cycle == 0
                      ? 0
                      : (request_count - 1) /
                            request_capacity_per_core_cycle;
              global_memory_injection_bound =
                  std::max(global_memory_injection_bound, memory_bound);
            }
          }
          global_enhanced_drain_bound = std::max(
              {global_issue_drain_bound,
               global_committed_compute_bound,
               global_committed_vector_bound,
               global_memory_injection_bound});
        }
        _cores[core_id]->set_scheduler_global_issue_drain_bound(
            global_issue_drain_bound);
        _cores[core_id]->set_scheduler_global_unlock_bounds(
            global_committed_compute_bound,
            global_committed_vector_bound,
            global_memory_injection_bound,
            global_enhanced_drain_bound);
        _cores[core_id]->set_scheduler_issue_state(scheduler_issue_state);
        _cores[core_id]->cycle();
      }
      _core_cycles++;
    }

    // DRAM cycle
    if (_cycle_mask & DRAM_MASK) {
      _dram->cycle();
    }
    // Interconnect cycle
    if (_cycle_mask & ICNT_MASK) {
      _icnt_cycle++;

      for (int core_id = 0; core_id < _n_cores; core_id++) {
        for (int noc_id = 0; noc_id < _noc_node_per_core; noc_id++) {
          // PUHS core to ICNT. memory request
          int port_id = core_id * _noc_node_per_core + noc_id;
          if (_cores[core_id]->has_memory_request()) {
            MemoryAccess *front = _cores[core_id]->top_memory_request();
            front->core_id = core_id;
            if (!_icnt->is_full(port_id, front)) {
              transition_memory_stage(front,MemoryStage::ICNT_TO_MEMORY,_core_cycles);
              _icnt->push(port_id, get_dest_node(front), front);
              _cores[core_id]->pop_memory_request();
              _nr_from_core++;
              _total_nr_from_core++;
            }
          }
          // Push response from ICNT. to Core.
          if (!_icnt->is_empty(port_id)) {
            _cores[core_id]->push_memory_response(_icnt->top(port_id));
            _icnt->pop(port_id);
            _nr_to_core++;
            _total_nr_to_core++;
          }
        }
      }

      for (int mem_id = 0; mem_id < _n_memories; mem_id++) {
        // ICNT to memory
        int core_offset = _n_cores * _noc_node_per_core;
        if (!_icnt->is_empty(core_offset + mem_id) &&
            !_dram->is_full(mem_id, _icnt->top(core_offset + mem_id))) {

          MemoryAccess *front = _icnt->top(core_offset + mem_id);

          transition_memory_stage(
              front,
              MemoryStage::DRAM,
              _core_cycles);

          _dram->push(mem_id, front);
          _icnt->pop(core_offset + mem_id);
          _nr_to_mem++;
          _total_nr_to_mem++;
        }
        // Pop response to ICNT from dram
        if (!_dram->is_empty(mem_id) &&
            !_icnt->is_full(core_offset + mem_id, _dram->top(mem_id))) {

          MemoryAccess *front = _dram->top(mem_id);

          transition_memory_stage(
              front,
              MemoryStage::ICNT_TO_CORE,
              _core_cycles);


          _icnt->push(core_offset + mem_id,
                      get_dest_node(front),
                      front);

          _dram->pop(mem_id);
          _nr_from_mem++;
          _total_nr_from_mem++;
        }
      }
      if (_icnt_interval!=0 && _icnt_cycle % _icnt_interval == 0) {
        spdlog::info("[ICNT] Core->ICNT request {}GB/Sec", ((_memory_req_size*_nr_from_core*(1000/_icnt_period)/_icnt_interval)));
        spdlog::info("[ICNT] Core<-ICNT request {}GB/Sec", ((_memory_req_size*_nr_to_core*(1000/_icnt_period)/_icnt_interval)));
        spdlog::info("[ICNT] ICNT->MEM request {}GB/Sec", ((_memory_req_size*_nr_to_mem*(1000/_icnt_period)/_icnt_interval)));
        spdlog::info("[ICNT] ICNT<-MEM request {}GB/Sec", ((_memory_req_size*_nr_from_mem*(1000/_icnt_period)/_icnt_interval)));
        _nr_from_core=0;
        _nr_to_core=0;
        _nr_to_mem=0;
        _nr_from_mem=0;
      }
      _icnt->cycle();
    }
  }
  spdlog::info("Simulation Finished at {} cycle {} us", _core_cycles, _core_cycles / (_config.core_freq) );
  /* Print simulation stats */
  for (int core_id = 0; core_id < _n_cores; core_id++) {
    _cores[core_id]->print_stats();
  }
  if (_component_characterization) {
    json record{{"version", 1},
                {"core_cycles", _core_cycles},
                {"icnt_cycles", _icnt_cycle},
                {"num_cores", _n_cores},
                {"dram_channels", _n_memories},
                {"core_frequency_mhz", _config.core_freq},
                {"icnt_frequency_mhz", _config.icnt_freq},
                {"dram_frequency_mhz", _config.dram_freq},
                {"core_to_noc_packets", _total_nr_from_core},
                {"noc_to_core_packets", _total_nr_to_core},
                {"noc_to_dram_packets", _total_nr_to_mem},
                {"dram_to_noc_packets", _total_nr_from_mem}};
    spdlog::info("COMPONENT_SYSTEM {}", record.dump());
    _icnt->print_component_characterization();
    _dram->print_component_characterization();
  }
  _icnt->print_stats();
  _dram->print_stat();
  MemoryProfiler::instance().print_stats();
}

void Simulator::register_model(std::unique_ptr<Model> model) {
  if(_weight_table.find(model->get_name()) == _weight_table.end()) {
    model->initialize_weight(_weight_table[model->get_name()]);
  } 
  _models.push_back(std::move(model));
  std::push_heap(_models.begin(), _models.end(), CompareModel());
}

void Simulator::register_language_model(json info, std::unique_ptr<LanguageModel> model) {
  std::string name = info["name"];
  std::string trace_file = info["trace_file"];
  char* onnxim_path_env = std::getenv("ONNXIM_HOME");
  std::string onnxim_path = onnxim_path_env != NULL?
  std::string(onnxim_path_env) : std::string("./");
  trace_file = fs::path(onnxim_path).append("traces").append(trace_file).string();
  if(_weight_table.find(name) == _weight_table.end()) {
    model->initialize_weight(_weight_table[name]);
  }
  _lang_scheduler = LangScheduler::create(name, trace_file, std::move(model), _config, info);
}

void Simulator::finish_language_model(uint32_t model_id) {
  _lang_scheduler->finish_model(model_id);
}

bool Simulator::running() {
  bool running = false;
  running |= !_models.empty();
  for (auto &core : _cores) {
    running = running || core->running();
  }
  running = running || _icnt->running();
  running = running || _dram->running();
  running = running || !_scheduler->empty();
  if(_language_mode) {
    running = running || _lang_scheduler->busy();
  }
  return running;
}

void Simulator::set_cycle_mask() {
  _cycle_mask = 0x0;
  uint64_t minimum_time = MIN3(_core_time, _dram_time, _icnt_time);
  if (_core_time <= minimum_time) {
    _cycle_mask |= CORE_MASK;
    _core_time += _core_period;
  }
  if (_dram_time <= minimum_time) {
    _cycle_mask |= DRAM_MASK;
    _dram_time += _dram_period;
  }
  if (_icnt_time <= minimum_time) {
    _cycle_mask |= ICNT_MASK;
    _icnt_time += _icnt_period;
  }
}

uint32_t Simulator::get_dest_node(MemoryAccess *access) {
  if (access->request) {
    return _config.num_cores * _config.icnt_injection_ports_per_core + _dram->get_channel_id(access);
  } else {
    return access->core_id * _config.icnt_injection_ports_per_core + (_dram->get_channel_id(access) % _config.icnt_injection_ports_per_core);
  }
}

const double Simulator::get_tile_ops() {
  std::chrono::duration<double> duration = _tile_timestamp.back() - _tile_timestamp.front();
  if (_tile_timestamp.empty())
    return 0.0;
  else
    return _tile_timestamp.size() / duration.count();
}
