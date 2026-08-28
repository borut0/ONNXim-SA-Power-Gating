#pragma once
#include <robin_hood.h>

#include <memory>
#include <vector>

#include "Dram.h"
#include "SimulationConfig.h"
#include "Sram.h"
#include "Stat.h"

class Core {
 public:
  struct SchedulerDrainSnapshot {
    cycle_type committed_compute_cycles = 0;
    cycle_type committed_vector_cycles = 0;
    uint64_t resident_future_memory_requests = 0;
    uint64_t staged_memory_requests = 0;
    uint64_t core_request_queue_depth = 0;
    uint64_t outstanding_memory_requests = 0;
    uint64_t waiting_write_requests = 0;
  };

  static std::unique_ptr<Core> create(uint32_t id, SimulationConfig config);
  Core(uint32_t id, SimulationConfig config);
  virtual ~Core() = default;
  virtual bool running();
  virtual bool can_issue(bool is_accum_tile=false);
  virtual void issue(std::unique_ptr<Tile> tile);
  virtual std::unique_ptr<Tile> pop_finished_tile();

  virtual void cycle();

  virtual bool has_memory_request();
  virtual void pop_memory_request();
  virtual MemoryAccess* top_memory_request() { return _request_queue.front(); }
  virtual void push_memory_response(MemoryAccess* response);
  enum class SchedulerIssueState {
    NO_SCHEDULED_REQUEST,
    CORE_NOT_READY,
    TILE_ISSUED,
    NO_TILE_AVAILABLE
  };

  virtual void print_stats();
  virtual void print_current_stats();
  void set_scheduler_issue_state(SchedulerIssueState state);
  void set_scheduler_core_tile_waiting(
      bool waiting, bool front_skip, bool front_contains_gemm,
      size_t front_pre_gemm_instruction_count);
  cycle_type get_resident_instruction_count() const;
  SchedulerDrainSnapshot get_scheduler_drain_snapshot() const;
  void set_scheduler_global_issue_drain_bound(cycle_type bound);
  void set_scheduler_global_unlock_bounds(
      cycle_type committed_compute_bound,
      cycle_type committed_vector_bound,
      cycle_type memory_injection_bound,
      cycle_type enhanced_bound);

  virtual cycle_type get_compute_cycles() { return _stat_tot_compute_cycle; }

 protected:
  enum class OperandWaitReason {
    NONE,
    MEMORY,
    DEPENDENCY,
    UNKNOWN
  };

  enum class CoreAttributionCategory {
    COMPUTE,
    MEMORY_STALL,
    DEPENDENCY_STALL,
    SCHEDULER_STALL,
    OTHER
  };

  virtual bool can_issue_compute(std::unique_ptr<Instruction>& inst);
  virtual cycle_type get_inst_compute_cycles(std::unique_ptr<Instruction>& inst) = 0;
  virtual void update_stats();
  virtual void finish_compute_pipeline();
  virtual void finish_vector_pipeline();
  virtual void handle_ld_inst_queue();
  virtual void handle_st_inst_queue();
  OperandWaitReason get_operand_wait_reason(const Instruction& inst) const;
  OperandWaitReason get_data_wait_reason(const Sram& buffer,
                                         addr_type address,
                                         int buffer_id) const;
  OperandWaitReason get_store_wait_reason(const Instruction& inst) const;
  void record_instruction_issue_wait(OperandWaitReason reason);
  void attribute_core_cycle(CoreAttributionCategory category);
  virtual void print_execution_time_attribution();
  cycle_type attribution_total_cycles() const;
  virtual cycle_type calculate_add_tree_iterations(uint32_t vector_size);
  virtual cycle_type calculate_vector_op_iterations(uint32_t vector_size);

  const uint32_t _id;
  const SimulationConfig _config;

  cycle_type _core_cycle;
  
  cycle_type _stat_idle_cycle;
  cycle_type _stat_tot_idle_cycle = 0;

  cycle_type _stat_systolic_bubble_cycle = 0;
  cycle_type _stat_tot_systolic_bubble_cycle = 0;

  cycle_type _stat_memory_idle_cycle;
  cycle_type _stat_tot_memory_idle_cycle = 0;

  cycle_type _stat_compute_cycle = 0;
  cycle_type _stat_tot_compute_cycle = 0;
  cycle_type _stat_sa_vector_overlap_cycle = 0;
  cycle_type _stat_tot_sa_vector_overlap_cycle = 0;

  cycle_type _accum_request_rr_cycle;
  cycle_type _max_request_rr_cycle;
  cycle_type _min_request_rr_cycle;
  
  /* Vector Unit Params */
  cycle_type _stat_vec_compute_cycle;
  cycle_type _stat_tot_vec_compute_cycle = 0;

  cycle_type _stat_systolic_active_cycle = 0;
  cycle_type _stat_tot_systolic_active_cycle = 0;
  double _stat_matmul_cycle = 0;
  double _stat_tot_matmul_cycle = 0;

  int _running_layer;
  uint32_t tile_rr = 0;
  uint64_t _next_measurement_tile_id = 1;
  std::deque<std::unique_ptr<Tile>> _tiles;
  std::queue<std::unique_ptr<Tile>> _finished_tiles;

  std::queue<std::unique_ptr<Instruction>> _compute_pipeline;
  std::queue<std::unique_ptr<Instruction>> _vector_pipeline;

  std::queue<std::unique_ptr<Instruction>> _ld_inst_queue;
  std::queue<std::unique_ptr<Instruction>> _st_inst_queue;
  std::queue<std::unique_ptr<Instruction>> _ex_inst_queue;

  std::queue<MemoryAccess*> _request_queue;
  std::queue<MemoryAccess*> _response_queue;
  robin_hood::unordered_flat_map<uint32_t, MemoryAccess*>
      _outstanding_memory_accesses;
  uint64_t _next_core_queue_ticket = 1;
  uint64_t _last_dequeued_core_queue_ticket = 0;
  bool _core_queue_tickets_exact = true;
  uint32_t _waiting_write_reqs;

  uint32_t _current_layer_id;
  uint32_t _current_fused_op_id;
  Sram _spad;
  Sram _acc_spad;

  SchedulerIssueState _scheduler_issue_state =
      SchedulerIssueState::NO_SCHEDULED_REQUEST;
  bool _scheduler_core_tile_waiting = false;
  bool _scheduler_front_tile_skip = false;
  bool _scheduler_front_tile_contains_gemm = false;
  size_t _scheduler_front_tile_pre_gemm_instruction_count = 0;
  cycle_type _scheduler_global_issue_drain_bound = 1;
  cycle_type _scheduler_global_committed_compute_bound = 0;
  cycle_type _scheduler_global_committed_vector_bound = 0;
  cycle_type _scheduler_global_memory_injection_bound = 0;
  cycle_type _scheduler_global_enhanced_drain_bound = 0;

  bool _last_core_cycle_issued_instruction = false;
  bool _last_core_cycle_had_pending_instruction = false;
  bool _last_core_cycle_memory_block = false;
  bool _last_core_cycle_dependency_block = false;
  bool _last_core_cycle_unknown_block = false;
  bool _last_core_cycle_compute_issue_block = false;

  cycle_type _attrib_compute_cycles = 0;
  cycle_type _attrib_memory_stall_cycles = 0;
  cycle_type _attrib_dependency_stall_cycles = 0;
  cycle_type _attrib_scheduler_stall_cycles = 0;
  cycle_type _attrib_other_cycles = 0;

  // Compact aggregate-only paper characterization.  This flag never feeds
  // scheduling or timing decisions.
  bool _component_characterization = false;

  // for analysis
  uint64_t _stat_ex_issue = 0;
};
