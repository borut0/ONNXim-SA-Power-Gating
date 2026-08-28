#ifndef DRAM_H
#define DRAM_H
#include <robin_hood.h>
#include <cstdint>
#include <queue>
#include <utility>

#include "Common.h"
#include "ramulator/Ramulator.hpp"
#include "ramulator2.hh"


class Dram {
 public:
  virtual ~Dram() = default;
  virtual bool running() = 0;
  virtual void cycle() = 0;
  virtual bool is_full(uint32_t cid, MemoryAccess* request) = 0;
  virtual void push(uint32_t cid, MemoryAccess* request) = 0;
  virtual bool is_empty(uint32_t cid) = 0;
  virtual MemoryAccess* top(uint32_t cid) = 0;
  virtual void pop(uint32_t cid) = 0;
  uint32_t get_channel_id(MemoryAccess* request);
  static uint32_t map_channel_id(addr_type address, uint32_t request_size,
                                 uint32_t channel_count);
  virtual void print_stat() {}
  void print_component_characterization() const;

 protected:
  SimulationConfig _config;
  uint32_t _n_ch;
  cycle_type _cycles;
  std::vector<uint64_t> _outstanding_demand_requests;
  std::vector<uint64_t> _channel_demand_active_cycles;
  std::vector<uint64_t> _channel_demand_active_start;
  uint64_t _any_channel_demand_active_cycles = 0;
  uint64_t _any_channel_demand_active_start = 0;
  uint64_t _total_outstanding_requests = 0;
  uint64_t _accepted_requests = 0;
  uint64_t _completed_requests = 0;
  uint64_t _accepted_read_requests = 0;
  uint64_t _accepted_write_requests = 0;
  uint64_t _accepted_read_bytes = 0;
  uint64_t _accepted_write_bytes = 0;
  uint64_t _peak_total_outstanding_requests = 0;
  bool _component_characterization = false;

  void initialize_component_counters();
  void record_request_accepted(uint32_t channel,
                               const MemoryAccess* request);
  void record_response_ready(uint32_t channel, MemoryAccess* response);
};

class SimpleDram : public Dram {
 public:
  SimpleDram(SimulationConfig config);
  virtual bool running() override;
  virtual void cycle() override;
  virtual bool is_full(uint32_t cid, MemoryAccess* request) override;
  virtual void push(uint32_t cid, MemoryAccess* request) override;
  virtual bool is_empty(uint32_t cid) override;
  virtual MemoryAccess* top(uint32_t cid) override;
  virtual void pop(uint32_t cid) override;

 private:
  uint32_t _latency;
  double _bandwidth;

  uint64_t _last_finish_cycle;
  std::vector<std::queue<std::pair<addr_type, MemoryAccess*>>> _waiting_queue;
  std::vector<std::queue<MemoryAccess*>> _response_queue;
};

class DramRamulator : public Dram {
 public:
  DramRamulator(SimulationConfig config);

  virtual bool running() override;
  virtual void cycle() override;
  virtual bool is_full(uint32_t cid, MemoryAccess* request) override;
  virtual void push(uint32_t cid, MemoryAccess* request) override;
  virtual bool is_empty(uint32_t cid) override;
  virtual MemoryAccess* top(uint32_t cid) override;
  virtual void pop(uint32_t cid) override;
  virtual void print_stat() override;

 private:
  std::unique_ptr<ram::Ramulator> _mem;
  robin_hood::unordered_flat_map<uint64_t, MemoryAccess*> _waiting_mem_access;
  std::queue<MemoryAccess*> _responses;

  std::vector<uint64_t> _total_processed_requests;
  std::vector<uint64_t> _processed_requests;
};

class DramRamulator2 : public Dram {
 public:
  DramRamulator2(SimulationConfig config);

  virtual bool running() override;
  virtual void cycle() override;
  virtual bool is_full(uint32_t cid, MemoryAccess* request) override;
  virtual void push(uint32_t cid, MemoryAccess* request) override;
  virtual bool is_empty(uint32_t cid) override;
  virtual MemoryAccess* top(uint32_t cid) override;
  virtual void pop(uint32_t cid) override;
  virtual void print_stat() override;

 private:
  std::vector<std::unique_ptr<NDPSim::Ramulator2>> _mem;
  int _tx_ch_log2;
  int _tx_log2;
  int _req_size;
};
#endif
