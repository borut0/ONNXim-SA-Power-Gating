#include "Sram.h"
#define NUM_PORTS 3

Sram::Sram(SimulationConfig config, const cycle_type& core_cycle, bool accum, uint32_t core_id)
    : _core_cycle(core_cycle) {
  if (!accum) {
    _size = config.core_config[core_id].spad_size KB / 2;
  } else {
    _size = config.core_config[core_id].accum_spad_size KB / 2;
  }
  _data_width = config.dram_req_size;
  int precision = config.precision;
  _current_size[0] = 0; // multiple of _data_width
  _current_size[1] = 0;
  _accum = accum;
}

bool Sram::check_hit(addr_type address, int buffer_id) {
  if (_cache_table[buffer_id].find(address) == _cache_table[buffer_id].end())
    return false;
  _cache_table[buffer_id].at(address).timestamp = _core_cycle;
  return _cache_table[buffer_id].at(address).valid;
}

bool Sram::check_full(int buffer_id) {
  return _current_size[buffer_id] * _data_width < _size;
}

bool Sram::check_remain(size_t size, int buffer_id) {
  return (_current_size[buffer_id] + size) * _data_width <= _size;
}

bool Sram::check_allocated(addr_type address, int buffer_id) {
  return _cache_table[buffer_id].find(address) != _cache_table[buffer_id].end();
}

Sram::EntryState Sram::get_entry_state(addr_type address, int buffer_id) const {
  auto iter = _cache_table[buffer_id].find(address);
  if (iter == _cache_table[buffer_id].end()) {
    return EntryState{};
  }

  return EntryState{.allocated = true,
                    .valid = iter->second.valid,
                    .pending_memory_count = iter->second.pending_memory_count,
                    .pending_compute_count = iter->second.pending_compute_count,
                    .remain_req_count = iter->second.remain_req_count,
                    .memory_tail_core_queue_ticket =
                        iter->second.memory_tail_core_queue_ticket};
}

bool Sram::cycle() {
  const bool idle = _reads_this_cycle == 0 && _writes_this_cycle == 0;
  _modeled_cycles++;
  if (idle) _modeled_idle_cycles++;
  if (_reads_this_cycle != 0) _modeled_read_active_cycles++;
  if (_writes_this_cycle != 0) _modeled_write_active_cycles++;
  _modeled_read_events += _reads_this_cycle;
  _modeled_write_events += _writes_this_cycle;
  _reads_this_cycle = 0;
  _writes_this_cycle = 0;
  return idle;
}

void Sram::record_modeled_read(addr_type address, int buffer_id,
                               uint64_t access_count) {
  assert(buffer_id >= 0 && buffer_id < 2);
  assert(check_allocated(address, buffer_id));
  assert(_cache_table[buffer_id].at(address).valid);
  _reads_this_cycle += access_count;
}

void Sram::flush(int buffer_id) {
  _current_size[buffer_id] = 0;
  _cache_table[buffer_id].clear();
  spdlog::trace("{}SRAM[{}] Flush", _accum? "Acc-": "", buffer_id);
}

int Sram::prefetch(addr_type address, int buffer_id, size_t allocated_size,
                    size_t count, PendingProducer pending_producer,
                    uint64_t memory_tail_core_queue_ticket){
  if (_cache_table[buffer_id].find(address) == _cache_table[buffer_id].end()) {
    if (!check_remain(allocated_size, buffer_id)) {
      print_all(buffer_id);
      assert(0);
      return 0;
    }
    _current_size[buffer_id] += allocated_size;
  } else if (_cache_table[buffer_id].find(address) !=
                 _cache_table[buffer_id].end() &&
             _accum) {
    assert(_cache_table[buffer_id].at(address).size == allocated_size);
    return 0;
  } else {
    assert(0);
    return 0;
  }

  assert(pending_producer != PendingProducer::NONE);
  size_t pending_memory_count = pending_producer == PendingProducer::MEMORY ? count : 0;
  size_t pending_compute_count = pending_producer == PendingProducer::COMPUTE ? count : 0;

  _cache_table[buffer_id][address] = SramEntry{.valid = false,
                                               .address = address,
                                               .size = allocated_size,
                                               .remain_req_count = count,
                                               .pending_memory_count = pending_memory_count,
                                               .pending_compute_count = pending_compute_count,
                                               .memory_tail_core_queue_ticket =
                                                   memory_tail_core_queue_ticket,
                                               .timestamp = _core_cycle};
  return 1;
}

void Sram::fill(addr_type address, int buffer_id, PendingProducer completed_producer) {
  assert(check_allocated(address, buffer_id));
  _writes_this_cycle++;
  SramEntry& entry = _cache_table[buffer_id].at(address);
  assert(entry.remain_req_count > 0 && !entry.valid);

  switch (completed_producer) {
    case PendingProducer::MEMORY:
      assert(entry.pending_memory_count > 0);
      entry.pending_memory_count--;
      break;
    case PendingProducer::COMPUTE:
      assert(entry.pending_compute_count > 0);
      entry.pending_compute_count--;
      break;
    case PendingProducer::NONE:
      assert(0);
      break;
  }

  entry.remain_req_count--;
  if (entry.remain_req_count == 0) {
    assert(entry.pending_memory_count == 0);
    assert(entry.pending_compute_count == 0);
    entry.valid = true;
    spdlog::trace("MAKE valid {} {}F", buffer_id, address);
  }
}

void Sram::count_up(addr_type address, int buffer_id, PendingProducer pending_producer) {
  assert(check_allocated(address, buffer_id));
  assert(pending_producer != PendingProducer::NONE);
  SramEntry& entry = _cache_table[buffer_id].at(address);
  entry.remain_req_count++;
  if (pending_producer == PendingProducer::MEMORY) {
    entry.pending_memory_count++;
  } else {
    entry.pending_compute_count++;
  }
  if (entry.valid) {
    entry.valid = false;
    spdlog::trace("MAKE valid {} {}F", buffer_id, address);
  }
}

void Sram::print_all(int buffer_id) {
  for (auto& [key, val] : _cache_table[buffer_id]) {
    spdlog::info("0x{:x} : {} B", key, val.size * _data_width);
  }
  spdlog::info("Allocated size: {} B", _current_size[buffer_id]*_data_width);
  spdlog::info("Size: {} B", _size);
}