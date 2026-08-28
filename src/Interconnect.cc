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
  if (!_component_characterization) {
    return;
  }
  _outstanding_packets++;
  _injected_packets++;
  _peak_outstanding_packets =
      std::max(_peak_outstanding_packets, _outstanding_packets);
}

void Interconnect::record_packet_delivered() {
  if (!_component_characterization) {
    return;
  }
  assert(_outstanding_packets > 0);
  _outstanding_packets--;
  _delivered_packets++;
}

void Interconnect::record_activity_cycle() {
  if (_component_characterization && _outstanding_packets != 0) {
    _demand_active_cycles++;
  }
}

void Interconnect::print_component_characterization() const {
  const uint64_t idle_cycles = _cycles - _demand_active_cycles;
  const double active_percent =
      _cycles == 0
          ? 0.0
          : 100.0 * static_cast<double>(_demand_active_cycles) /
                static_cast<double>(_cycles);
  json record{{"version", 1},
              {"icnt_cycles", _cycles},
              {"demand_active_cycles", _demand_active_cycles},
              {"demand_idle_cycles", idle_cycles},
              {"demand_active_percent", active_percent},
              {"demand_idle_percent", 100.0 - active_percent},
              {"injected_packets", _injected_packets},
              {"delivered_packets", _delivered_packets},
              {"peak_outstanding_packets", _peak_outstanding_packets},
              {"outstanding_packets_at_end", _outstanding_packets},
              {"activity_semantics",
               "at least one ONNXim memory packet is in flight from "
               "injection through destination delivery"}};
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
  record_activity_cycle();
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
  record_activity_cycle();
  _booksim->run();
  _cycles++;
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
