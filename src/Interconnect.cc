#include "Interconnect.h"
#include "booksim2/Interconnect.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

namespace {

bool component_characterization_enabled() {
  const char* value = std::getenv("ONNXSIM_COMPONENT_CHARACTERIZATION");
  return value != nullptr && value[0] != '\0' &&
         std::string(value) != "0";
}

}  // namespace

void Interconnect::record_packet_injected() {
  _packet_activity_since_last_cycle = true;
  _outstanding_packets++;
  _injected_packets++;
  _peak_outstanding_packets =
      std::max(_peak_outstanding_packets, _outstanding_packets);
}

void Interconnect::record_packet_delivered() {
  _packet_activity_since_last_cycle = true;
  assert(_outstanding_packets > 0);
  _outstanding_packets--;
  _delivered_packets++;
}

void Interconnect::record_quiescence_cycle(
    const ModelQuiescenceStatus& before,
    const ModelQuiescenceStatus& after) {
  _quiescence_sampled_cycles++;
  if (!_packet_activity_since_last_cycle &&
      before.outstanding_packets_zero &&
      after.outstanding_packets_zero) {
    _zero_outstanding_boundary_cycles++;
  }
  if (!before.observable || !after.observable) {
    _quiescence_available = false;
  } else if (!_packet_activity_since_last_cycle &&
             before.quiescent() && after.quiescent()) {
    _modeled_quiescent_cycles++;
  }
  _packet_activity_since_last_cycle = false;
}

void Interconnect::print_component_characterization() const {
  const bool available = _quiescence_available &&
      _quiescence_sampled_cycles == _cycles;
  json record{{"version", 2},
              {"icnt_cycles", _cycles},
              {"modeled_quiescence_available", available},
              {"modeled_quiescent_cycles", nullptr},
              {"modeled_active_cycles", nullptr},
              {"modeled_quiescent_percent", nullptr},
              {"injected_packets", _injected_packets},
              {"delivered_packets", _delivered_packets},
              {"peak_outstanding_packets", _peak_outstanding_packets},
              {"outstanding_packets_at_end", _outstanding_packets},
              {"zero_outstanding_boundary_cycles",
               _zero_outstanding_boundary_cycles},
              {"cycle_count_evidence_class", "measured"},
              {"percentage_evidence_class", "derived"},
              {"percentage_equation",
               "100 * modeled_quiescent_cycles / icnt_cycles"},
              {"measurement_scope", "simulator model state"},
              {"activity_semantics",
               "pre/post cycle model quiescence plus no injection or delivery "
               "event during the sampled cycle; unsupported internals make "
               "the result unavailable"}};
  if (available) {
    record["modeled_quiescent_cycles"] = _modeled_quiescent_cycles;
    record["modeled_active_cycles"] = _cycles - _modeled_quiescent_cycles;
    record["modeled_quiescent_percent"] =
        _cycles == 0 ? 0.0 :
        100.0 * static_cast<double>(_modeled_quiescent_cycles) /
            static_cast<double>(_cycles);
  }
  spdlog::info("COMPONENT_NOC {}", record.dump());
}

SimpleInterconnect::SimpleInterconnect(SimulationConfig config)
  :  _latency(config.icnt_latency) {
  spdlog::info("Initialize SimpleInterconnect");
  _cycles = 0;
  _component_characterization = component_characterization_enabled();
  _config = config;
  _n_nodes = config.num_cores * _config.icnt_injection_ports_per_core + config.dram_channels;
  _in_buffers.resize(_n_nodes);
  _out_buffers.resize(_n_nodes);
  _busy_node.resize(_n_nodes);
  _rr_next_src.resize(_n_nodes);
  for(int node = 0; node < _n_nodes; node++) {
    _busy_node[node] = false;
    _in_buffers.at(node).resize(_n_nodes);
    _rr_next_src[node] = 0;
  }
}


bool SimpleInterconnect::running() {
  return false;
}

void SimpleInterconnect::cycle() {
  ModelQuiescenceStatus before = get_model_quiescence_status();
  for(int dest = 0; dest < _n_nodes; dest++) {
    int src_start = _rr_next_src[dest];
    bool pushed = false;

    for(int i = 0; i < _n_nodes; i++) {
      int src = (src_start + i) % _n_nodes;

      if (!_in_buffers[src][dest].empty() &&
          _in_buffers[src][dest].front().finish_cycle <= _cycles) {

        _out_buffers[dest].push(_in_buffers[src][dest].front().access);
        _in_buffers[src][dest].pop();
        _rr_next_src[dest] = (src + 1) % _n_nodes;
        pushed = true;
        break;
      }
    }
  }
  _cycles++;
  ModelQuiescenceStatus after = get_model_quiescence_status();
  record_quiescence_cycle(before, after);
}

Interconnect::ModelQuiescenceStatus
SimpleInterconnect::get_model_quiescence_status() {
  ModelQuiescenceStatus status;
  status.observable = true;
  status.outstanding_packets_zero = _outstanding_packets == 0;
  status.modeled_buffers_empty = true;
  for (const auto& sources : _in_buffers) {
    for (const auto& queue : sources) {
      if (!queue.empty()) status.modeled_buffers_empty = false;
    }
  }
  for (const auto& queue : _out_buffers) {
    if (!queue.empty()) status.modeled_buffers_empty = false;
  }
  for (bool busy : _busy_node) {
    if (busy) status.modeled_buffers_empty = false;
  }
  // The simple model has no routers, VCs, links, or credits.
  status.backend_quiescent = true;
  return status;
}

void SimpleInterconnect::push(uint32_t src, uint32_t dest, MemoryAccess* request) {
  SimpleInterconnect::Entity entity;
  if(_in_buffers[src][dest].empty())
    entity.finish_cycle =  _cycles + _latency;
  else
    entity.finish_cycle =  _in_buffers[src][dest].back().finish_cycle + 1;
  entity.dest = dest;
  entity.access = request;
  _in_buffers[src][dest].push(entity);
  record_packet_injected();
}

bool SimpleInterconnect::is_full(uint32_t nid, MemoryAccess* request) {
  //TODO: limit buffersize
  return false;
}

bool SimpleInterconnect::is_empty(uint32_t nid) {
  return _out_buffers[nid].empty();
}

MemoryAccess* SimpleInterconnect::top(uint32_t nid) {
  assert(!is_empty(nid));
  return _out_buffers[nid].front();
}

void SimpleInterconnect::pop(uint32_t nid) {
  _out_buffers[nid].pop();
  record_packet_delivered();
  // spdlog::trace("PUSH {}", _cycles);
}


Booksim2Interconnect::Booksim2Interconnect(SimulationConfig config) {
  _cycles = 0;
  _component_characterization = component_characterization_enabled();
  _config = config;
  _n_nodes = config.num_cores * _config.icnt_injection_ports_per_core + config.dram_channels;
  spdlog::info("Initialize Booksim2"); 
  char* onnxim_path_env = std::getenv("ONNXIM_HOME");
  std::string onnxim_path = onnxim_path_env != NULL?
    std::string(onnxim_path_env) : std::string("./");

  _config_path = fs::path(onnxim_path).append("configs").append((std::string)config.icnt_config_path).string();
  spdlog::info("Config path : {}", _config_path);
  _booksim = std::make_unique<booksim2::Interconnect>(_config_path, _n_nodes);
  _ctrl_size = 8;
}

bool Booksim2Interconnect::running() {
  return false;
}

void Booksim2Interconnect::cycle() {
  ModelQuiescenceStatus before = get_model_quiescence_status();
  _booksim->run();
  _cycles++;
  ModelQuiescenceStatus after = get_model_quiescence_status();
  record_quiescence_cycle(before, after);
}

void Booksim2Interconnect::push(uint32_t src, uint32_t dest, MemoryAccess* request) {
  booksim2::Interconnect::Type type = get_booksim_type(request);
  uint32_t size = get_packet_size(request);
  _booksim->push(request, 0, 0, size, type, src, dest);
  record_packet_injected();
}

bool Booksim2Interconnect::is_full(uint32_t nid, MemoryAccess* request) {
  uint32_t size = get_packet_size(request);
  return _booksim->is_full(nid, 0, size);
}

bool Booksim2Interconnect::is_empty(uint32_t nid) {
  return _booksim->is_empty(nid, 0);
}

MemoryAccess* Booksim2Interconnect::top(uint32_t nid) {
  assert(!is_empty(nid));
  return (MemoryAccess*) _booksim->top(nid, 0);
}

void Booksim2Interconnect::pop(uint32_t nid) {
  assert(!is_empty(nid));
  _booksim->pop(nid, 0);
  record_packet_delivered();
}

void Booksim2Interconnect::print_stats() {
  _booksim->print_stats();
}

Interconnect::ModelQuiescenceStatus
Booksim2Interconnect::get_model_quiescence_status() {
  booksim2::Interconnect::QuiescenceStatus backend =
      _booksim->get_quiescence_status();
  ModelQuiescenceStatus status;
  status.observable = backend.observable;
  status.outstanding_packets_zero = _outstanding_packets == 0;
  status.modeled_buffers_empty =
      backend.injection_queues_empty &&
      backend.source_buffer_states_quiescent &&
      backend.ejection_buffers_empty &&
      backend.boundary_buffers_empty &&
      backend.retired_flit_queues_empty &&
      backend.retired_packet_heads_empty &&
      backend.in_flight_flits_empty;
  status.backend_quiescent = backend.routers_quiescent &&
      backend.flit_channels_empty && backend.credit_channels_empty;
  return status;
}

booksim2::Interconnect::Type Booksim2Interconnect::get_booksim_type(MemoryAccess* access) {
  booksim2::Interconnect::Type type;
  if(access->write && access->request) {
    /* Write request */
    type = booksim2::Interconnect::Type::WRITE;
  }
  else if(access->write && !access->request) {
    /* Write response */
    type = booksim2::Interconnect::Type::WRITE_REPLY;
  }
  else if(!access->write && access->request){
    /* Read request */
    type = booksim2::Interconnect::Type::READ;
  } 
  else if(!access->write && !access->request) {
    /* Read reply */
    type = booksim2::Interconnect::Type::READ_REPLY;
  }
  return type;
}

uint32_t Booksim2Interconnect::get_packet_size(MemoryAccess* access) {
  uint32_t size;
  if(access->write && access->request) {
    /* Write request */
    size = access->size;
  }
  else if(access->write && !access->request) {
    /* Write response */
    size = _ctrl_size;
  }
  else if(!access->write && access->request){
    /* Read request */
    size = _ctrl_size;
  } 
  else if(!access->write && !access->request) {
    /* Read reply */
    size = access->size;
  }
  return size;
}
