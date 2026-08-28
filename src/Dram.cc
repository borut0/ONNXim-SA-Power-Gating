#include "Dram.h"

#include "helper/HelperFunctions.h"
#include "Hashing.h"
#include <cstdlib>
#include <string>

namespace {

bool component_characterization_enabled() {
  const char* value = std::getenv("ONNXSIM_COMPONENT_CHARACTERIZATION");
  return value != nullptr && value[0] != '\0' &&
         std::string(value) != "0";
}

}  // namespace

void Dram::initialize_component_counters() {
  _component_characterization = component_characterization_enabled();
  if (!_component_characterization) {
    return;
  }
  _outstanding_demand_requests.assign(_n_ch, 0);
  _channel_demand_active_cycles.assign(_n_ch, 0);
  _channel_demand_active_start.assign(_n_ch, 0);
}

void Dram::record_request_accepted(uint32_t channel,
                                   const MemoryAccess* request) {
  if (!_component_characterization) {
    return;
  }
  assert(channel < _outstanding_demand_requests.size());
  if (_outstanding_demand_requests[channel] == 0) {
    _channel_demand_active_start[channel] = _cycles;
  }
  if (_total_outstanding_requests == 0) {
    _any_channel_demand_active_start = _cycles;
  }
  _outstanding_demand_requests[channel]++;
  _total_outstanding_requests++;
  _accepted_requests++;
  if (request->write) {
    _accepted_write_requests++;
    _accepted_write_bytes += request->size;
  } else {
    _accepted_read_requests++;
    _accepted_read_bytes += request->size;
  }
  _peak_total_outstanding_requests =
      MAX(_peak_total_outstanding_requests,
          _total_outstanding_requests);
}

void Dram::record_response_ready(uint32_t channel,
                                 MemoryAccess* response) {
  if (!_component_characterization) {
    return;
  }
  if (response->dram_completion_counted) {
    return;
  }
  assert(channel < _outstanding_demand_requests.size());
  assert(_outstanding_demand_requests[channel] > 0);
  _outstanding_demand_requests[channel]--;
  _total_outstanding_requests--;
  if (_outstanding_demand_requests[channel] == 0) {
    _channel_demand_active_cycles[channel] +=
        _cycles - _channel_demand_active_start[channel];
  }
  if (_total_outstanding_requests == 0) {
    _any_channel_demand_active_cycles +=
        _cycles - _any_channel_demand_active_start;
  }
  _completed_requests++;
  response->dram_completion_counted = true;
}

void Dram::print_component_characterization() const {
  uint64_t active_channel_cycles = 0;
  std::vector<uint64_t> final_channel_active_cycles =
      _channel_demand_active_cycles;
  for (uint32_t channel = 0; channel < _n_ch; ++channel) {
    if (_outstanding_demand_requests[channel] != 0) {
      final_channel_active_cycles[channel] +=
          _cycles - _channel_demand_active_start[channel];
    }
    active_channel_cycles += final_channel_active_cycles[channel];
  }
  uint64_t final_any_channel_active_cycles =
      _any_channel_demand_active_cycles;
  if (_total_outstanding_requests != 0) {
    final_any_channel_active_cycles +=
        _cycles - _any_channel_demand_active_start;
  }
  const uint64_t channel_cycles = _cycles * _n_ch;
  const double mean_channel_active_percent =
      channel_cycles == 0
          ? 0.0
          : 100.0 * static_cast<double>(active_channel_cycles) /
                static_cast<double>(channel_cycles);
  const double any_channel_active_percent =
      _cycles == 0
          ? 0.0
          : 100.0 * static_cast<double>(
                        final_any_channel_active_cycles) /
                static_cast<double>(_cycles);
  const double transaction_bandwidth_utilization_percent =
      channel_cycles == 0
          ? 0.0
          : 100.0 * static_cast<double>(_completed_requests) *
                static_cast<double>(_config.dram_nbl) /
                static_cast<double>(channel_cycles);
  json record{
      {"version", 1},
      {"dram_cycles", _cycles},
      {"dram_channels", _n_ch},
      {"accepted_requests", _accepted_requests},
      {"completed_requests", _completed_requests},
      {"accepted_read_requests", _accepted_read_requests},
      {"accepted_write_requests", _accepted_write_requests},
      {"accepted_read_bytes", _accepted_read_bytes},
      {"accepted_write_bytes", _accepted_write_bytes},
      {"channel_demand_active_cycles", final_channel_active_cycles},
      {"mean_channel_demand_active_percent",
       mean_channel_active_percent},
      {"mean_channel_demand_idle_percent",
       100.0 - mean_channel_active_percent},
      {"any_channel_demand_active_cycles",
       final_any_channel_active_cycles},
      {"any_channel_demand_active_percent",
       any_channel_active_percent},
      {"all_channels_demand_idle_percent",
       100.0 - any_channel_active_percent},
      {"transaction_bandwidth_utilization_percent",
       transaction_bandwidth_utilization_percent},
      {"peak_total_outstanding_requests",
       _peak_total_outstanding_requests},
      {"outstanding_requests_at_end", _total_outstanding_requests},
      {"activity_semantics",
       "channel has at least one accepted demand request whose DRAM "
       "completion has not yet become visible to ONNXim"},
      {"bandwidth_semantics",
       "completed transactions * configured nBL / "
       "(DRAM cycles * channels); same normalization as the current "
       "Ramulator2 wrapper"}};
  spdlog::info("COMPONENT_DRAM {}", record.dump());
}

uint32_t Dram::get_channel_id(MemoryAccess* access) {
  return map_channel_id(access->dram_address, _config.dram_req_size, _n_ch);
}

uint32_t Dram::map_channel_id(addr_type address, uint32_t request_size,
                              uint32_t channel_count) {
  new_addr_type request_address =
      static_cast<new_addr_type>(address) / request_size;
  if (channel_count >= 16) {
    return ipoly_hash_function(request_address, 0, channel_count);
  }
  return ipoly_hash_function(request_address, 0, 16) % channel_count;
}

/* FIXME: Simple DRAM has bugs */
SimpleDram::SimpleDram(SimulationConfig config)
    : _latency(config.dram_latency) {
  _cycles = 0;
  _config = config;
  _n_ch = config.dram_channels;
  initialize_component_counters();
  _waiting_queue.resize(_n_ch);
  _response_queue.resize(_n_ch);
}

bool SimpleDram::running() { return false; }

void SimpleDram::cycle() {
  for (uint32_t ch = 0; ch < _n_ch; ch++) {
    if (!_waiting_queue[ch].empty() &&
        _waiting_queue[ch].front().first <= _cycles) {
      _response_queue[ch].push(_waiting_queue[ch].front().second);
      _waiting_queue[ch].pop();
    }
  }

  _cycles++;
}

bool SimpleDram::is_full(uint32_t cid, MemoryAccess* request) { return false; }

void SimpleDram::push(uint32_t cid, MemoryAccess* request) {
  record_request_accepted(cid, request);
  request->request = false;
  std::pair<uint64_t, MemoryAccess*> entity;
  entity.first = MAX(_cycles + _latency, _last_finish_cycle);
  _last_finish_cycle = entity.first;
  entity.second = request;
  _waiting_queue[cid].push(entity);
}

bool SimpleDram::is_empty(uint32_t cid) { return _response_queue[cid].empty(); }

MemoryAccess* SimpleDram::top(uint32_t cid) {
  assert(!is_empty(cid));
  MemoryAccess* response = _response_queue[cid].front();
  record_response_ready(cid, response);
  return response;
}

void SimpleDram::pop(uint32_t cid) {
  assert(!is_empty(cid));
  _response_queue[cid].pop();
}

DramRamulator::DramRamulator(SimulationConfig config)
    : _mem(std::make_unique<ram::Ramulator>(config.dram_config_path,
                                            config.num_cores, false)) {
  _n_ch = config.dram_channels;
  initialize_component_counters();
  _config = config;
  _cycles = 0;
  _total_processed_requests.resize(_n_ch);
  _processed_requests.resize(_n_ch);
  for (int ch = 0; ch < _n_ch; ch++) {
    _total_processed_requests[ch] = 0;
    _processed_requests[ch] = 0;
  }
}

bool DramRamulator::running() { return false; }

void DramRamulator::cycle() {
  _mem->tick();
  _cycles++;
  int interval = _config.dram_print_interval? _config.dram_print_interval: INT32_MAX;
  int average = 0;
  if (_cycles % interval == 0) {
    for (int ch = 0; ch < _n_ch; ch++) {
      float util = ((float)_processed_requests[ch]) / interval * 100;
      _total_processed_requests[ch] += _processed_requests[ch];
      average += _processed_requests[ch];
      _processed_requests[ch] = 0;
    }
    spdlog::info("Avg DRAM: BW Util {:.2f}%", (float)average / (interval * _n_ch) * 100);
  }
}

bool DramRamulator::is_full(uint32_t cid, MemoryAccess* request) {
  return !_mem->isAvailable(cid, request->dram_address, request->write);
}

void DramRamulator::push(uint32_t cid, MemoryAccess* request) {
  record_request_accepted(cid, request);
  const addr_type atomic_bytes = _mem->getAtomicBytes();
  const addr_type target_addr = request->dram_address;
  // align address
  const addr_type start_addr = target_addr - (target_addr % atomic_bytes);
  assert(start_addr == target_addr);
  assert(request->size == atomic_bytes);
  int count = 0;
  request->request = false;
  _mem->push(cid, target_addr, request->write, request->core_id, request);
}

bool DramRamulator::is_empty(uint32_t cid) { return _mem->isEmpty(cid); }

MemoryAccess* DramRamulator::top(uint32_t cid) {
  assert(!is_empty(cid));
  MemoryAccess* response = (MemoryAccess*)_mem->top(cid);
  record_response_ready(cid, response);
  return response;
}

void DramRamulator::pop(uint32_t cid) {
  assert(!is_empty(cid));
  _mem->pop(cid);
  _processed_requests[cid]++;
}

void DramRamulator::print_stat() {
  uint32_t total_reqs = 0;
  for (int ch = 0; ch < _n_ch; ch++) {
    _total_processed_requests[ch] += _processed_requests[ch];
    float util = ((float)_total_processed_requests[ch]) / _cycles * 100;
    spdlog::info("DRAM CH[{}]: AVG BW Util {:.2f}%", ch, util);
    total_reqs += _total_processed_requests[ch];
  }
  float util = ((float)total_reqs / _n_ch) / _cycles * 100;
  spdlog::info("DRAM: AVG BW Util {:.2f}%", util);
  _mem->print_stats();
}

DramRamulator2::DramRamulator2(SimulationConfig config) {
  _n_ch = config.dram_channels;
  _req_size = config.dram_req_size;
  _config = config;
  _cycles = 0;
  initialize_component_counters();
  _mem.resize(_n_ch);
  for (int ch = 0; ch < _n_ch; ch++) {
    _mem[ch] = std::make_unique<NDPSim::Ramulator2>(
      ch, _n_ch, config.dram_config_path, "Ramulator2", _config.dram_print_interval, 1);
  }
  _tx_log2 = log2(_req_size);
  _tx_ch_log2 = log2(_n_ch) + _tx_log2;
}

bool DramRamulator2::running() { 
  return false;
}

void DramRamulator2::cycle() {
  for (int ch = 0; ch < _n_ch; ch++) {
    _mem[ch]->cycle();
  }
  _cycles++;
}

bool DramRamulator2::is_full(uint32_t cid, MemoryAccess* request) {
  return _mem[cid]->full();
}

void DramRamulator2::push(uint32_t cid, MemoryAccess* request) {
  record_request_accepted(cid, request);
  addr_type atomic_bytes =_config.dram_req_size;
  addr_type target_addr = request->dram_address;
  // align address
  addr_type start_addr = target_addr - (target_addr % atomic_bytes);
  assert(start_addr == target_addr);
  assert(request->size == atomic_bytes);
  target_addr = (target_addr >> _tx_ch_log2) << _tx_log2;
  NDPSim::mem_fetch* mf = new NDPSim::mem_fetch();
  mf->addr = target_addr;
  mf->size = request->size;
  mf->write = request->write;
  mf->request = true;
  mf->origin_data = request;
  _mem[cid]->push(mf);
}

bool DramRamulator2::is_empty(uint32_t cid) { 
  return _mem[cid]->return_queue_top() == NULL;
}

MemoryAccess* DramRamulator2::top(uint32_t cid) {
  assert(!is_empty(cid));
  NDPSim::mem_fetch* mf = _mem[cid]->return_queue_top();
  ((MemoryAccess*)mf->origin_data)->request = false;
  MemoryAccess* response = (MemoryAccess*)mf->origin_data;
  record_response_ready(cid, response);
  return response;
}

void DramRamulator2::pop(uint32_t cid) {
  assert(!is_empty(cid));
  NDPSim::mem_fetch* mf = _mem[cid]->return_queue_pop();
  delete mf;
}

void DramRamulator2::print_stat() {
  for (int ch = 0; ch < _n_ch; ch++) {
    _mem[ch]->print(stdout);
  }
}
