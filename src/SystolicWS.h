#include <array>
#include <fstream>
#include <map>

#include "Core.h"

class SystolicWS : public Core {
 public:
  SystolicWS(uint32_t id, SimulationConfig config);
  virtual ~SystolicWS() override;
  virtual void issue(std::unique_ptr<Tile> tile) override;
  virtual void push_memory_response(MemoryAccess* response) override;
  virtual void cycle() override;
  virtual void print_stats() override;

 protected:
  virtual bool can_issue_compute(std::unique_ptr<Instruction>& inst) override;
  virtual cycle_type get_inst_compute_cycles(std::unique_ptr<Instruction>& inst) override;
  uint32_t _stat_systolic_inst_issue_count = 0;
  uint32_t _stat_systolic_preload_issue_count = 0;
  cycle_type get_vector_compute_cycles(std::unique_ptr<Instruction>& inst);

  // for analysis
  enum class ComputeBlockReason {
    NONE,
    OPERAND_WAIT,
    PIPELINE_FULL,
    VECTOR_BUSY
  };

  ComputeBlockReason _last_compute_block_reason = ComputeBlockReason::NONE;
  //end

  enum class PhaseWaitClass {
    READY,
    MEMORY_WAIT,
    COMPUTE_DEPENDENCY,
    UNKNOWN_OTHER
  };

  struct PhaseRequiredSource {
    bool accum = false;
    int buffer_id = 0;
    addr_type address = 0;
    size_t pending_memory = 0;
  };

  struct PhaseAdmission {
    uint64_t tile_id = 0;
    cycle_type admission_cycle = 0;
    uint32_t layer_id = 0;
    std::string operation;
    bool skip = false;
    bool contains_gemm = false;
    bool resident = false;
    bool first_gemm_set = false;
    cycle_type first_gemm_cycle = 0;
    bool wait_class_set = false;
    PhaseWaitClass wait_class = PhaseWaitClass::UNKNOWN_OTHER;
    bool response_drain_set = false;
    cycle_type response_drain_cycle = 0;
    size_t total_unique_required_pending_memory = 0;
    std::vector<PhaseRequiredSource> required_sources;
    bool request_cohort_frozen = false;
    bool request_cohort_exact = false;
    size_t request_cohort_matched = 0;
    size_t request_cohort_completed = 0;
    size_t cohort_core_queue = 0;
    size_t cohort_icnt_to_memory = 0;
    size_t cohort_dram = 0;
    size_t cohort_icnt_to_core = 0;
    size_t core_request_queue_depth = 0;
    bool fifo_position_exact = false;
    size_t tailmost_required_fifo_position_1based = 0;
    bool channel_counts_exact = false;
    std::vector<size_t> required_not_yet_dram_by_channel;
    robin_hood::unordered_flat_set<uint32_t> request_cohort_ids;
    cycle_type last_required_dram_enter_cycle = 0;
    cycle_type last_required_dram_exit_cycle = 0;
    cycle_type last_required_core_arrival_cycle = 0;
    bool retirement_set = false;
    cycle_type retirement_cycle = 0;
  };

  struct PredictorCandidate {
    bool found = false;
    cycle_type horizon = 0;
    Opcode opcode = Opcode::BAR;
    bool fifo_certified = false;
    bool unknown = false;
  };

  struct PredictorSnapshot {
    cycle_type safe_horizon = 0;
    cycle_type visible_horizon = 0;
    cycle_type future_tile_cap = 0;
    bool visible = false;
    bool fifo_limited = false;
    bool future_cap_limited = false;
    bool retention_safe_preload = false;
    bool unknown = false;
  };

  struct PowerGateShadowEpisode {
    cycle_type start_offset = 0;
    cycle_type duration = 0;
    cycle_type entry_fresh_remaining = 0;
    cycle_type minimum_fresh_remaining = 0;
  };

  struct SchedulerUnlockShadow {
    bool pipeline_active = false;
    uint32_t pipeline_cycles = 0;
    cycle_type pipeline_frontier = 0;
    bool sleeping = false;
    cycle_type sleep_start = 0;
    cycle_type wake_cycle = 0;
    uint64_t entry_attempts = 0;
    uint64_t expired_before_entry = 0;
    uint64_t episode_count = 0;
    uint64_t sleep_cycles = 0;
    cycle_type longest_episode = 0;
    uint64_t early_context_end_violations = 0;
    uint64_t censored_episodes = 0;
  };

  enum class RetentionSourceReason : size_t {
    READY,
    PENDING_MEMORY_TAIL_QUEUED,
    PENDING_MEMORY_TAIL_DEQUEUED,
    PENDING_COMPUTE,
    INVALID_NO_PENDING,
    UNALLOCATED,
    TICKET_WINDOW_OVERFLOW,
    CORE_QUEUE_TICKETS_INEXACT,
    COUNT
  };

  enum class SaArchitecturalState : size_t {
    ACTIVE_GEMM,
    ACTIVE_GEMM_PRELOAD,
    ACTIVE_UNKNOWN,
    IDLE_COMMITTED_GEMM_START_DELAY,
    IDLE_COMMITTED_GEMM_PRELOAD_START_DELAY,
    IDLE_VECTOR_ACTIVE,
    IDLE_INSTRUCTION_DISPATCH,
    IDLE_MEMORY_STALL,
    IDLE_DEPENDENCY_STALL,
    IDLE_SCHEDULER_STALL,
    IDLE_OTHER,
    COUNT
  };

  // Change-only workload trace for replaying the separately synthesized
  // controller. It is enabled only by ONNXSIM_SA_CONTROLLER_TRACE_DIR.
  static constexpr size_t CONTROLLER_DESCRIPTOR_FIELDS = 14;
  static constexpr size_t CONTROLLER_TRACE_INPUT_FIELDS = 42;
  using ControllerDescriptor =
      std::array<uint64_t, CONTROLLER_DESCRIPTOR_FIELDS>;
  using ControllerTraceInputs =
      std::array<uint64_t, CONTROLLER_TRACE_INPUT_FIELDS>;

  static constexpr cycle_type SA_PHASE_MIN_IDLE = 5000;
  static constexpr std::array<cycle_type, 6>
      PREDICTOR_CAUSAL_THRESHOLDS = {10, 50, 100, 250, 500, 1000};
  static constexpr std::array<cycle_type, 9>
      RESPONSE_DRAIN_THRESHOLDS = {1, 10, 50, 100, 167,
                                   250, 500, 1000, 2000};
  static constexpr cycle_type POWER_GATE_BREAK_EVEN_CYCLES = 166;
  static constexpr cycle_type POWER_GATE_WAKE_GUARD_CYCLES = 1;

  // Historical characterization experiments are opt-in.  Keeping these
  // disabled avoids their per-cycle scans during broad workload evaluation.
  bool _phase_measure_enabled = false;
  bool _predictor_measure_enabled = false;
  bool _phase_sa_idle = false;
  bool _phase_measure = false;
  uint64_t _phase_old_tile_id = 0;
  uint64_t _phase_next_tile_id = 0;
  bool _phase_pre_t0_committed = false;
  std::vector<PhaseAdmission> _phase_admissions;
  bool _phase_t1_set = false;
  bool _phase_t2_set = false;
  bool _phase_t3_set = false;
  bool _phase_t3_classified = false;
  cycle_type _phase_t0 = 0;
  cycle_type _phase_t1 = 0;
  cycle_type _phase_t2 = 0;
  cycle_type _phase_t3 = 0;
  size_t _phase_resident_count = 0;
  bool _phase_remaining_known = false;
  size_t _phase_remaining = 0;
  bool _phase_final_opcode_known = false;
  Opcode _phase_final_opcode = Opcode::BAR;
  bool _phase_vector_active = false;
  bool _phase_core_tile_waiting = false;
  PhaseWaitClass _phase_t3_wait = PhaseWaitClass::UNKNOWN_OTHER;

  bool _predictor_sa_idle = false;
  cycle_type _predictor_idle_start = 0;
  cycle_type _predictor_initial_horizon = 0;
  cycle_type _predictor_frontier = 0;
  cycle_type _predictor_last_horizon = 0;
  cycle_type _predictor_max_lookahead = 0;
  cycle_type _predictor_max_lookahead_cycle = 0;
  cycle_type _predictor_max_fifo_lookahead = 0;
  cycle_type _predictor_max_causal_remaining = 0;
  std::array<uint64_t, PREDICTOR_CAUSAL_THRESHOLDS.size()>
      _predictor_causal_qualified_cycles{};
  std::array<uint64_t, PREDICTOR_CAUSAL_THRESHOLDS.size()>
      _predictor_causal_current_runs{};
  std::array<uint64_t, PREDICTOR_CAUSAL_THRESHOLDS.size()>
      _predictor_causal_longest_runs{};
  uint64_t _predictor_now1_cap_binding_count = 0;
  uint64_t _predictor_now1_queue_empty_count = 0;
  uint64_t _predictor_now1_queue_front_skip_count = 0;
  uint64_t _predictor_now1_queue_front_no_gemm_count = 0;
  uint64_t _predictor_now1_queue_front_gemm_count = 0;
  std::map<size_t, uint64_t> _predictor_now1_front_prefix_histogram;
  cycle_type _predictor_work_drain_frontier = 0;
  cycle_type _predictor_work_drain_max_remaining = 0;
  cycle_type _predictor_work_drain_max_issue_bound = 0;
  std::array<uint64_t, PREDICTOR_CAUSAL_THRESHOLDS.size()>
      _predictor_work_drain_qualified_cycles{};
  std::array<uint64_t, PREDICTOR_CAUSAL_THRESHOLDS.size()>
      _predictor_work_drain_current_runs{};
  std::array<uint64_t, PREDICTOR_CAUSAL_THRESHOLDS.size()>
      _predictor_work_drain_longest_runs{};
  uint64_t _predictor_recomputations = 0;
  uint64_t _predictor_frontier_extensions = 0;
  uint64_t _predictor_fifo_limited_count = 0;
  uint64_t _predictor_future_cap_limited_count = 0;
  uint64_t _predictor_no_visible_count = 0;
  uint64_t _predictor_unknown_count = 0;
  bool _pg_shadow_sleeping = false;
  cycle_type _pg_shadow_episode_start = 0;
  cycle_type _pg_shadow_episode_entry_remaining = 0;
  cycle_type _pg_shadow_episode_min_remaining = 0;
  cycle_type _pg_shadow_max_fresh_remaining = 0;
  uint64_t _pg_shadow_retention_safe_cycles = 0;
  uint64_t _pg_shadow_stale_only_qualified_cycles = 0;
  bool _pg_shadow_late_preload_commit = false;
  std::vector<PowerGateShadowEpisode> _pg_shadow_episodes;

  std::ofstream _controller_trace;
  bool _controller_trace_have_inputs = false;
  std::array<ControllerDescriptor, 2> _controller_trace_descriptors{};
  ControllerTraceInputs _controller_trace_last_inputs{};

  // Optional aggregate-only diagnosis for the SRAM-source eligibility loss.
  // Enabled by ONNXSIM_SA_RETENTION_REASON_MEASURE; it never feeds control.
  bool _retention_reason_measure = false;
  uint64_t _retention_reason_context_cycles = 0;
  uint64_t _retention_reason_supported_cycles = 0;
  uint64_t _retention_reason_unsupported_cycles = 0;
  uint64_t _retention_reason_more_than_two_sources_cycles = 0;
  uint64_t _retention_reason_more_than_two_sources_tiles = 0;
  uint64_t _retention_reason_unique_source_observations = 0;
  std::array<uint64_t,
             static_cast<size_t>(RetentionSourceReason::COUNT)>
      _retention_reason_cycle_counts{};
  std::array<uint64_t,
             static_cast<size_t>(RetentionSourceReason::COUNT)>
      _retention_reason_source_counts{};
  uint64_t _response_drain_ports_per_core_cycle_bound = 0;
  uint64_t _response_drain_evaluated_cycles = 0;
  uint64_t _response_drain_zero_bound_cycles = 0;
  cycle_type _response_drain_max_bound = 0;
  std::array<uint64_t, RESPONSE_DRAIN_THRESHOLDS.size()>
      _response_drain_threshold_cycles{};
  bool _response_drain_shadow_sleeping = false;
  cycle_type _response_drain_shadow_start = 0;
  uint64_t _response_drain_shadow_episode_count = 0;
  uint64_t _response_drain_shadow_sleep_cycles = 0;
  cycle_type _response_drain_shadow_longest_episode = 0;
  uint64_t _response_drain_shadow_safety_violations = 0;

  // Measurement-only audit for the scheduler/tile-admission (CPU IF-like)
  // opportunity.  It never feeds scheduling, timing, or power-gating control.
  // A certificate is considered usable only while this core is completely
  // drained, its per-core scheduler queue is empty, and the simple scheduler
  // exposes a causal lower bound before a new layer can be admitted.
  bool _scheduler_opportunity_measure = false;
  bool _scheduler_unlock_measure = false;
  bool _scheduler_opportunity_context_active = false;
  cycle_type _scheduler_opportunity_context_start = 0;
  cycle_type _scheduler_opportunity_context_frontier = 0;
  cycle_type _scheduler_opportunity_context_max_bound = 0;
  bool _scheduler_opportunity_context_had_certificate = false;
  bool _scheduler_opportunity_qualified_run_active = false;
  cycle_type _scheduler_opportunity_qualified_run_start = 0;
  uint64_t _scheduler_opportunity_context_cycles = 0;
  uint64_t _scheduler_opportunity_certified_cycles = 0;
  uint64_t _scheduler_opportunity_entry_qualified_cycles = 0;
  cycle_type _scheduler_opportunity_max_bound = 0;
  uint64_t _scheduler_opportunity_interval_count = 0;
  uint64_t _scheduler_opportunity_certified_interval_count = 0;
  uint64_t _scheduler_opportunity_oracle_long_interval_count = 0;
  uint64_t _scheduler_opportunity_oracle_long_interval_cycles = 0;
  cycle_type _scheduler_opportunity_longest_interval = 0;
  uint64_t _scheduler_opportunity_qualified_run_count = 0;
  uint64_t _scheduler_opportunity_qualified_run_cycles = 0;
  cycle_type _scheduler_opportunity_longest_qualified_run = 0;
  uint64_t _scheduler_opportunity_certificate_violations = 0;
  uint64_t _scheduler_opportunity_censored_intervals = 0;
  cycle_type _scheduler_unlock_context_frontier = 0;
  uint64_t _scheduler_unlock_certified_cycles = 0;
  uint64_t _scheduler_unlock_entry_qualified_cycles = 0;
  uint64_t _scheduler_unlock_improved_cycles = 0;
  uint64_t _scheduler_unlock_compute_binding_cycles = 0;
  uint64_t _scheduler_unlock_vector_binding_cycles = 0;
  uint64_t _scheduler_unlock_memory_binding_cycles = 0;
  cycle_type _scheduler_unlock_max_enhanced_bound = 0;
  cycle_type _scheduler_unlock_max_compute_bound = 0;
  cycle_type _scheduler_unlock_max_vector_bound = 0;
  cycle_type _scheduler_unlock_max_memory_bound = 0;
  uint64_t _scheduler_unlock_certificate_violations = 0;
  SchedulerUnlockShadow _scheduler_issue_shadow;
  SchedulerUnlockShadow _scheduler_enhanced_shadow;

  // Optional aggregate-only SA architectural-state breakdown. Enabled by
  // ONNXSIM_SA_STATE_BREAKDOWN; it never feeds scheduling or control.
  bool _sa_state_measure = false;
  std::array<uint64_t,
             static_cast<size_t>(SaArchitecturalState::COUNT)>
      _sa_state_cycles{};

  bool has_resident_gemm_candidate() const;
  Tile* find_resident_tile(uint64_t tile_id) const;
  void begin_phase_measurement(bool vector_active);
  void observe_phase_events();
  void classify_phase_t3();
  void complete_phase_measurement();
  PhaseWaitClass classify_gemm_operands(const Instruction& inst,
                                        const Tile& tile) const;
  void capture_response_drain(const Instruction& inst, const Tile& tile,
                              PhaseAdmission& admission) const;
  void freeze_required_request_cohort(PhaseAdmission& admission) const;
  void observe_required_response(const MemoryAccess& response);
  ControllerDescriptor controller_descriptor(const Tile* tile) const;
  void trace_controller_inputs(bool sa_active);
  void write_controller_trace_record(
      cycle_type cycle, bool final,
      const ControllerTraceInputs& inputs);
  PredictorCandidate predictor_tile_candidate(const Tile& tile) const;
  cycle_type predictor_fifo_horizon(const Instruction& inst, int spad_id,
                                    int accum_spad_id,
                                    bool& certified,
                                    bool& unknown) const;
  cycle_type predictor_future_tile_cap(
      const std::vector<PredictorCandidate>& tile_candidates,
      bool& unknown) const;
  PredictorSnapshot compute_idle_predictor_snapshot() const;
  void measure_retention_source_reasons(bool sa_active);
  void close_response_drain_shadow_episode(cycle_type end_cycle);
  void measure_scheduler_opportunity(bool sa_active);
  void close_scheduler_opportunity_context(cycle_type end_cycle,
                                           bool censored);
  void close_scheduler_opportunity_qualified_run(cycle_type end_cycle);
  void advance_scheduler_unlock_shadow(SchedulerUnlockShadow& shadow,
                                       cycle_type bound);
  void close_scheduler_unlock_shadow(SchedulerUnlockShadow& shadow,
                                     cycle_type end_cycle,
                                     bool censored_context);
  void update_idle_predictor(bool sa_active);
  void complete_idle_predictor_interval();
  void close_power_gate_shadow_episode(cycle_type end_cycle);

};
