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

  void cycle();
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

  const cycle_type& _core_cycle;

  robin_hood::unordered_map<addr_type, SramEntry> _cache_table[2];
};
