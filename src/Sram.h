#pragma once
#include "Common.h"

class Sram {
 public:
  enum class PendingProducer {
    NONE,
    MEMORY,
    COMPUTE
  };

  struct EntryState {
    bool allocated = false;
    bool valid = false;
    size_t pending_memory_count = 0;
    size_t pending_compute_count = 0;
    size_t remain_req_count = 0;
    uint64_t memory_tail_core_queue_ticket = 0;
  };

  Sram(SimulationConfig config, const cycle_type& core_cycle, bool accum, uint32_t core_id);

  bool check_hit(addr_type address, int buffer_id);
  bool check_full(int buffer_id);
  bool check_remain(size_t size, int buffer_id);
  bool check_allocated(addr_type address, int buffer_id);
  EntryState get_entry_state(addr_type address, int buffer_id) const;

  bool cycle();
  void record_modeled_read(addr_type address, int buffer_id,
                           uint64_t access_count = 1);
  uint64_t modeled_cycles() const { return _modeled_cycles; }
  uint64_t modeled_idle_cycles() const { return _modeled_idle_cycles; }
  uint64_t modeled_read_active_cycles() const { return _modeled_read_active_cycles; }
  uint64_t modeled_write_active_cycles() const { return _modeled_write_active_cycles; }
  uint64_t modeled_read_events() const { return _modeled_read_events; }
  uint64_t modeled_write_events() const { return _modeled_write_events; }
  void flush(int buffer_id);
  int prefetch(addr_type address, int buffer_id, size_t allocated_size, size_t count,
               PendingProducer pending_producer,
               uint64_t memory_tail_core_queue_ticket = 0);
  void count_up(addr_type, int buffer_id, PendingProducer pending_producer);
  void fill(addr_type address, int buffer_id, PendingProducer completed_producer);
  int get_size() { return _size; }
  int get_current_size(int buffer_id) { return _current_size[buffer_id]; }
  void print_all(int buffer_id);
 private:
  struct SramEntry {
    bool valid;
    addr_type address;
    size_t size;
    size_t remain_req_count;
    size_t pending_memory_count;
    size_t pending_compute_count;
    uint64_t memory_tail_core_queue_ticket;
    cycle_type timestamp;
  };

  int _size;
  int _data_width;
  int _current_size[2];
  bool _accum;
  uint64_t _reads_this_cycle = 0;
  uint64_t _writes_this_cycle = 0;
  uint64_t _modeled_cycles = 0;
  uint64_t _modeled_idle_cycles = 0;
  uint64_t _modeled_read_active_cycles = 0;
  uint64_t _modeled_write_active_cycles = 0;
  uint64_t _modeled_read_events = 0;
  uint64_t _modeled_write_events = 0;

  const cycle_type& _core_cycle;

  robin_hood::unordered_map<addr_type, SramEntry> _cache_table[2];
};
