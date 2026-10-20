#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Parameter-free, measurement-only accounting for a weight-stationary SA.
// This model derives PE-cycle opportunities from ONNXim instruction metadata;
// it does not assign power, voltage, transition latency, or break-even time.
namespace sa_pe_state {

struct Interval {
  uint64_t start_cycle = 0;
  uint64_t finish_cycle = 0;
  uint32_t tile_m = 0;
  uint32_t tile_k = 0;
  uint32_t tile_n = 0;
};

struct IntervalClass {
  uint64_t duration_cycles = 0;
  uint32_t tile_m = 0;
  uint32_t tile_k = 0;
  uint32_t tile_n = 0;
  uint64_t occurrences = 0;
  uint64_t on_compute_pe_cycles_per_interval = 0;
  uint64_t resident_weight_pe_cycles_per_interval = 0;
  uint64_t w_on_candidate_pe_cycles_per_interval = 0;
  uint64_t off_unmapped_candidate_pe_cycles_per_interval = 0;
};

struct Result {
  uint64_t completed_intervals = 0;
  uint64_t incomplete_intervals = 0;
  uint64_t invalid_dimension_intervals = 0;
  uint64_t insufficient_duration_intervals = 0;

  // Physical-time accounting. Overlapping instruction windows are unioned.
  uint64_t union_active_cycles = 0;
  uint64_t active_capacity_pe_cycles = 0;
  uint64_t on_compute_pe_cycles = 0;
  uint64_t resident_weight_pe_cycles = 0;
  uint64_t w_on_candidate_pe_cycles = 0;
  uint64_t off_unmapped_candidate_pe_cycles = 0;
  uint64_t temporal_idle_unclassified_pe_cycles = 0;
  uint64_t total_capacity_pe_cycles = 0;

  bool on_within_resident_capacity = false;
  bool resident_within_active_capacity = false;
  bool accounting_exact = false;
  std::vector<IntervalClass> interval_classes;
};

// Analytical WS wavefront model.  This is deliberately separate from Result:
// Result preserves the older packed-footprint opportunity accounting over
// ONNXim's complete instruction intervals, whereas the structures below model
// the useful MAC wave and compiler-visible stationary-weight lifetime.
enum class WaveOpcode {
  PRELOAD,
  GEMM
};

struct WaveInterval {
  uint64_t issue_cycle = 0;
  uint64_t start_cycle = 0;
  uint64_t finish_cycle = 0;
  uint32_t tile_m = 0;
  uint32_t tile_k = 0;
  uint32_t tile_n = 0;
  WaveOpcode opcode = WaveOpcode::GEMM;
  uint64_t weight_block_id = 0;
  bool weight_block_known = false;
};

struct WeightLifetime {
  uint64_t start_cycle = 0;
  uint64_t finish_cycle = 0;
  uint32_t rows = 0;
  uint32_t columns = 0;
  uint64_t weight_block_id = 0;
  uint64_t consumer_instructions = 0;
};

struct WaveLifetimeResult {
  uint64_t completed_intervals = 0;
  uint64_t incomplete_intervals = 0;
  uint64_t invalid_dimension_intervals = 0;
  uint64_t wave_exceeds_instruction_intervals = 0;
  uint64_t overlapping_wave_intervals = 0;
  uint64_t unknown_weight_block_intervals = 0;
  uint64_t gemm_without_preload_intervals = 0;
  uint64_t weight_block_mismatch_intervals = 0;
  uint64_t footprint_mismatch_intervals = 0;

  uint64_t analytical_on_pe_cycles = 0;
  uint64_t simulator_active_capacity_pe_cycles = 0;
  uint64_t simulator_active_non_wave_pe_cycles = 0;
  uint64_t resident_weight_pe_cycles = 0;
  // Unioned mapped-footprint capacity in [issue_cycle,start_cycle). This is
  // reported as committed/unknown rather than silently classified as OFF.
  // It can overlap an older resident footprint, so it is not necessarily a
  // disjoint component of total_capacity_pe_cycles.
  uint64_t pre_start_committed_pe_cycles = 0;
  uint64_t analytical_w_on_pe_cycles = 0;
  uint64_t analytical_off_pe_cycles = 0;
  uint64_t unclassified_pe_cycles = 0;
  uint64_t total_capacity_pe_cycles = 0;

  bool wave_overlap_free = false;
  bool weight_lifetime_complete = false;
  bool state_accounting_exact = false;
  std::vector<WeightLifetime> weight_lifetimes;
};

struct RegionStateResult {
  uint32_t region_size_rows = 0;
  uint32_t region_size_columns = 0;
  uint32_t region_rows = 0;
  uint32_t region_columns = 0;
  uint64_t region_count = 0;
  uint64_t capacity_region_cycles = 0;
  uint64_t analytical_on_region_cycles = 0;
  uint64_t analytical_w_on_region_cycles = 0;
  uint64_t analytical_off_region_cycles = 0;
  uint64_t unclassified_region_cycles = 0;
  bool state_accounting_exact = false;
  std::vector<uint64_t> per_region_on_cycles;
  std::vector<uint64_t> per_region_w_on_cycles;
  std::vector<uint64_t> per_region_off_cycles;
};

// Functional ReGate approximation supported by ONNXim's instruction
// metadata. It models structural row/column prefix masks and the diagonal
// useful-input wave, but deliberately does not guess value sparsity, circuit
// power, transition energy, or physical entry/wake latency.
struct ReGateControlClass {
  uint32_t tile_m = 0;
  uint32_t tile_k = 0;
  uint32_t tile_n = 0;
  uint64_t occurrences = 0;
  uint32_t row_on_prefix_length = 0;
  uint32_t col_on_prefix_length = 0;
  uint64_t pe_on_wave_cycles = 0;
  uint64_t on_pe_cycles_per_instruction = 0;
  uint64_t pe_on_assert_events_per_instruction = 0;
  uint64_t pe_on_deassert_events_per_instruction = 0;
};

struct ReGateFunctionalResult {
  uint64_t completed_intervals = 0;
  uint64_t incomplete_intervals = 0;
  uint64_t invalid_dimension_intervals = 0;
  uint64_t wave_exceeds_instruction_intervals = 0;
  uint64_t overlapping_wave_intervals = 0;
  uint64_t unknown_weight_block_intervals = 0;
  uint64_t gemm_without_preload_intervals = 0;
  uint64_t weight_block_mismatch_intervals = 0;
  uint64_t footprint_mismatch_intervals = 0;
  uint64_t capacity_pe_cycles = 0;
  uint64_t on_pe_cycles = 0;
  uint64_t w_on_pe_cycles = 0;
  uint64_t off_pe_cycles = 0;
  uint64_t unclassified_pe_cycles = 0;
  uint64_t pe_on_assert_events = 0;
  uint64_t pe_on_deassert_events = 0;
  bool control_schedule_valid = false;
  bool control_event_counts_valid = false;
  bool weight_lifetime_complete = false;
  bool state_partition_exact_under_model = false;
  std::vector<ReGateControlClass> control_classes;
};

// Version-2 analytical shadow with two logical stationary-weight banks per
// PE. Bank residency is inferred from explicit PRELOAD instructions and is
// retained until that bank is overwritten. All state counts are withheld when
// the instruction stream cannot be represented by the two-bank model.
struct TwoBankTimingCase {
  // Peak distinct preload instances. The global counts treat any footprints
  // as sharing the same two banks; mapped-PE counts only include a preload at
  // PEs inside its checked origin-packed footprint.
  bool global_applicable = true;
  uint64_t global_peak_live_preloads = 0;
  uint64_t global_over_two_cycles = 0;
  uint64_t mapped_pe_peak_live_preloads = 0;
  uint64_t mapped_pe_over_two_pe_cycles = 0;
  uint64_t mapped_pes_ever_over_two = 0;
};

struct TwoBankTimingAudit {
  // Derived timing counterfactuals, never validated bank schedules. The
  // start and local-first-use load times assume zero-duration loading;
  // GEMM_PRELOAD is fused load+compute in ONNXim.
  bool available = false;
  uint64_t matched_gemm_consumers = 0;
  TwoBankTimingCase issue_to_whole_finish;
  TwoBankTimingCase start_to_whole_finish;
  TwoBankTimingCase issue_to_pe_last_mac;
  TwoBankTimingCase start_to_pe_last_mac;
  TwoBankTimingCase pe_first_to_pe_last_mac;
};

// Explicit architectural hypothesis, not timing recovered from ONNXim or
// ReGate. One row is written to all mapped columns simultaneously. All four
// fields must be supplied by the caller; duration and period must be > 0.
enum class RowWriteOrder { ASCENDING_PE_ROW, DESCENDING_PE_ROW };

struct RowWriteTiming {
  uint64_t first_row_lead_cycles_before_start;
  uint64_t row_start_period_cycles;
  uint64_t row_write_duration_cycles;
  RowWriteOrder row_order;
};

// Per-row readiness check for one explicit GEMM_PRELOAD under RowWriteTiming.
// first_use is the earliest modeled use in the row (column zero).  A valid
// event needs required_stall_cycles = max(0, bank_ready_cycle -
// first_use_cycle). A planned load before issue is separately diagnosed even
// when the causal load can be delayed without adding an execution cycle.
struct RowwisePreloadRowAudit {
  uint32_t pe_row = 0;
  uint32_t ordered_row_index = 0;
  bool planned_load_begin_available = false;
  uint64_t planned_load_begin_cycle = 0;
  uint64_t load_begin_cycle = 0;
  uint64_t bank_ready_cycle = 0;
  uint64_t first_use_cycle = 0;
  uint64_t required_stall_cycles = 0;
  uint64_t max_local_bank_wait_cycles = 0;
  bool load_before_issue = false;
  bool no_added_cycle = false;
};

struct RowwiseBankConflictAudit {
  uint64_t preload_index = 0;
  uint32_t pe_row = 0;
  uint32_t pe_column = 0;
  uint64_t planned_load_begin_cycle = 0;
  uint64_t earliest_bank_release_cycle = 0;
  uint64_t load_begin_cycle = 0;
  uint64_t bank_ready_cycle = 0;
  uint64_t first_use_cycle = 0;
  uint64_t required_stall_cycles = 0;
};

struct RowwisePreloadAudit {
  uint64_t preload_index = 0;
  uint64_t instruction_index = 0;
  uint64_t weight_block_id = 0;
  uint64_t issue_cycle = 0;
  uint64_t start_cycle = 0;
  uint32_t tile_m = 0;
  uint32_t tile_k = 0;
  uint32_t tile_n = 0;
  bool planned_load_begin_available = false;
  uint64_t planned_load_begin_cycle = 0;
  uint64_t load_begin_cycle = 0;
  uint64_t conflicting_rows = 0;
  uint64_t schedule_diagnostic_rows = 0;
  uint64_t sum_row_required_stall_cycles = 0;
  uint64_t required_stall_cycles = 0;
  bool no_added_cycle = false;
  std::vector<RowwisePreloadRowAudit> rows;
  std::vector<RowwiseBankConflictAudit> bank_conflicts;
};

struct RowwiseFeasibility {
  bool available = false;
  bool secondary_counts_complete = false;
  bool necessary_conditions_pass = false;
  uint64_t mapped_pe_weight_write_events = 0;
  uint64_t row_write_before_issue_events = 0;
  uint64_t row_ready_after_first_mac_events = 0;
  uint64_t max_row_ready_lateness_cycles = 0;
  uint64_t mapped_pe_peak_live_preloads = 0;
  uint64_t mapped_pe_over_two_pe_cycles = 0;
  uint64_t mapped_pes_ever_over_two = 0;
  // Local no-stall diagnostics with earlier reservations held fixed. Neither
  // is a solved global reschedule or an actual ONNXim stall measurement.
  uint64_t locally_blocked_pe_weight_write_events = 0;
  uint64_t max_local_bank_wait_cycles = 0;
  uint64_t max_local_first_mac_delay_cycles = 0;
  // Exact row-readiness audit. Counts here are per preload-row, unlike the
  // mapped-PE counters above, and never modify ONNXim execution.
  uint64_t audited_preloads = 0;
  uint64_t audited_preload_rows = 0;
  uint64_t no_added_cycle_preloads = 0;
  uint64_t conflicting_preloads = 0;
  uint64_t conflicting_preload_rows = 0;
  uint64_t schedule_diagnostic_preloads = 0;
  uint64_t schedule_diagnostic_rows = 0;
  uint64_t sum_row_required_stall_cycles = 0;
  uint64_t total_required_stall_cycles = 0;
  uint64_t max_required_stall_cycles = 0;
  std::vector<RowwisePreloadAudit> preload_audit;
};

// Version-1 disjoint PE-cycle partition under an explicitly supplied
// RowWriteTiming hypothesis. This is independent of the legacy atomic
// whole-footprint bank assignment. Preload writes are reported as a
// separate (potentially overlapping) exposure: a write does not by itself
// make an otherwise idle PE ON. The GEMM_PRELOAD activation wave is ON just
// like the activation wave of a matched later GEMM.
struct RowwiseStateAccounting {
  bool available = false;
  bool validation_pass = false;
  bool partition_check_pass = false;
  uint64_t capacity_pe_cycles = 0;
  uint64_t on_pe_cycles = 0;
  uint64_t on_preload_wave_pe_cycles = 0;
  uint64_t on_gemm_wave_pe_cycles = 0;
  uint64_t on_zero_bank_pe_cycles = 0;
  uint64_t on_one_bank_pe_cycles = 0;
  uint64_t on_two_bank_pe_cycles = 0;
  uint64_t w_on_one_bank_pe_cycles = 0;
  uint64_t w_on_two_bank_pe_cycles = 0;
  uint64_t off_pe_cycles = 0;
  uint64_t unclassified_pe_cycles = 0;
  // Unioned PE-cycles in which at least one assumed row write is active.
  // This exposure overlaps the disjoint state partition above.
  uint64_t preload_write_pe_cycles = 0;
  uint64_t preload_write_while_on_pe_cycles = 0;
};

// Whole-bank row-stream readiness shadow for every explicit GEMM_PRELOAD.
// This assumes rows can be delivered at the SA input at the supplied period
// and duration. It does not include source-memory/read latency or prove a
// physical row bus, global bank selection, or circuit timing.
struct PreloadFeasibilityEvent {
  uint64_t preload_index = 0;
  uint64_t instruction_index = 0;
  uint64_t weight_block_id = 0;
  uint64_t issue_cycle = 0;
  uint64_t start_cycle = 0;
  uint32_t rows = 0;
  uint32_t columns = 0;
  uint64_t fill_span_cycles = 0;
  bool planned_load_begin_available = false;
  uint64_t planned_load_begin_cycle = 0;
  uint64_t load_begin_cycle = 0;
  uint64_t bank_ready_cycle = 0;
  uint64_t first_use_cycle = 0;
  uint64_t required_stall_cycles = 0;
  bool no_added_cycle = false;
};

struct PreloadFeasibilityAudit {
  bool available = false;
  uint64_t audited_preloads = 0;
  uint64_t no_added_cycle_preloads = 0;
  uint64_t conflicting_preloads = 0;
  uint64_t total_required_stall_cycles = 0;
  uint64_t max_required_stall_cycles = 0;
  std::vector<PreloadFeasibilityEvent> events;
};

struct TwoBankReGateResult {
  uint64_t capacity_pe_cycles = 0;
  uint64_t on_pe_cycles = 0;
  uint64_t on_zero_bank_pe_cycles = 0;
  uint64_t on_one_bank_pe_cycles = 0;
  uint64_t on_two_bank_pe_cycles = 0;
  uint64_t w_on_one_bank_pe_cycles = 0;
  uint64_t w_on_two_bank_pe_cycles = 0;
  uint64_t off_pe_cycles = 0;
  uint64_t unclassified_pe_cycles = 0;
  uint64_t preload_bank_pe_cycles = 0;
  uint64_t preload_events = 0;
  uint64_t gemm_events = 0;
  uint64_t incomplete_intervals = 0;
  uint64_t invalid_intervals = 0;
  uint64_t unknown_weight_intervals = 0;
  uint64_t gemm_without_weight_intervals = 0;
  uint64_t footprint_mismatch_intervals = 0;
  uint64_t weight_not_ready_intervals = 0;
  uint64_t weight_evicted_before_use_intervals = 0;
  uint64_t wave_exceeds_instruction_intervals = 0;
  uint64_t compute_collisions = 0;
  uint64_t two_bank_conflicts = 0;
  bool validation_pass = false;
  TwoBankTimingAudit timing_audit;
  PreloadFeasibilityAudit preload_feasibility_audit;
  RowwiseFeasibility rowwise_feasibility;
  RowwiseStateAccounting rowwise_state_accounting;
};

// Identity-independent shadow used by the updated predictor experiment.
// `stream_id` is a tile instance.  Every PRELOAD creates a distinct physical
// copy and every following GEMM in that stream consumes that copy until the
// next PRELOAD.  Weight values/addresses are intentionally not inputs.
struct UpdatedShadowInterval {
  uint64_t stream_id = 0;
  uint64_t issue_cycle = 0;
  uint64_t simulator_start_cycle = 0;
  uint64_t simulator_finish_cycle = 0;
  uint32_t tile_m = 0;
  uint32_t tile_k = 0;
  uint32_t tile_n = 0;
  WaveOpcode opcode = WaveOpcode::GEMM;
  uint64_t preload_sequence = 0;
};

struct UpdatedShadowEvent {
  uint64_t input_index = 0;
  uint64_t stream_id = 0;
  uint64_t preload_instance = 0;
  WaveOpcode opcode = WaveOpcode::GEMM;
  int bank = -1;
  uint64_t issue_cycle = 0;
  uint64_t loading_start_cycle = 0;
  uint64_t loading_finish_cycle = 0;
  uint64_t compute_start_cycle = 0;
  uint64_t compute_finish_cycle = 0;
  uint64_t simulator_start_cycle = 0;
  uint64_t simulator_finish_cycle = 0;
  uint64_t start_lateness_cycles = 0;
  uint64_t finish_lateness_cycles = 0;
};

struct UpdatedTwoBankShadowResult {
  uint64_t analytical_horizon_cycles = 0;
  uint64_t capacity_pe_cycles = 0;
  uint64_t on_pe_cycles = 0;
  uint64_t on_zero_bank_pe_cycles = 0;
  uint64_t on_one_bank_pe_cycles = 0;
  uint64_t on_two_bank_pe_cycles = 0;
  uint64_t w_on_pe_cycles = 0;
  uint64_t w_on_one_bank_pe_cycles = 0;
  uint64_t w_on_two_bank_pe_cycles = 0;
  uint64_t loading_pe_cycles = 0;
  // Raw assumed write exposure overlaps the disjoint state partition.
  uint64_t preload_write_pe_cycles = 0;
  uint64_t preload_write_while_on_pe_cycles = 0;
  uint64_t off_pe_cycles = 0;
  uint64_t unclassified_pe_cycles = 0;
  uint64_t preload_events = 0;
  uint64_t gemm_events = 0;
  uint64_t invalid_intervals = 0;
  uint64_t sequence_errors = 0;
  uint64_t footprint_mismatch_intervals = 0;
  uint64_t bank_assignment_conflicts = 0;
  // Event-local disagreement is diagnostic: its sum double-counts delay
  // propagated from an earlier event and is not a trace-level penalty.
  uint64_t sum_event_start_lateness_cycles = 0;
  uint64_t max_event_start_lateness_cycles = 0;
  uint64_t sum_event_finish_lateness_cycles = 0;
  uint64_t max_event_finish_lateness_cycles = 0;
  uint64_t simulator_trace_makespan_cycles = 0;
  uint64_t analytical_trace_makespan_cycles = 0;
  uint64_t trace_makespan_extension_cycles = 0;
  bool preload_first_valid = false;
  bool bank_assignment_valid = false;
  bool state_partition_exact = false;
  bool validation_pass = false;
  std::vector<UpdatedShadowEvent> events;
};

// Exact coefficient of
// (1+x+...+x^(rows-1))(1+x+...+x^(columns-1))
// (1+x+...+x^(waves-1)).  For the assumed WS schedule, this is the number of
// useful MACs at zero-based local cycle t, where t = row + column + wave.
uint64_t useful_macs_at_cycle(uint32_t rows, uint32_t columns,
                              uint32_t waves, uint64_t t);

WaveLifetimeResult derive_wave_lifetime(
    const std::vector<WaveInterval>& intervals,
    uint32_t array_rows,
    uint32_t array_columns,
    uint64_t horizon_cycles,
    uint64_t expected_union_active_cycles);

RegionStateResult derive_region_states(
    const std::vector<WaveInterval>& intervals,
    const WaveLifetimeResult& wave_lifetime,
    uint32_t array_rows,
    uint32_t array_columns,
    uint32_t region_size_rows,
    uint32_t region_size_columns,
    uint64_t horizon_cycles);

// ReGate-only, zero-latency functional approximation. For a valid interval,
// row_on and col_on are structural prefix masks of lengths tile_m and tile_k.
// For mapped PE (r,c), PE_on is asserted over the half-open interval
// [start+r+c, start+r+c+tile_n); its deassertion drains diagonally. W_on/OFF
// are classified from the proven stationary-weight lifetimes used by
// derive_region_states. Unknown state remains unclassified.
ReGateFunctionalResult derive_regate_functional(
    const std::vector<WaveInterval>& intervals,
    uint32_t array_rows,
    uint32_t array_columns,
    uint64_t horizon_cycles);

TwoBankReGateResult derive_two_bank_regate(
    const std::vector<WaveInterval>& intervals,
    uint32_t array_rows,
    uint32_t array_columns,
    uint64_t horizon_cycles,
    const RowWriteTiming* row_write_timing = nullptr);

// Analytical schedule independent of ONNXim's GEMM latency.  The explicit
// architecture hypothesis is one stationary-weight row written per cycle,
// two logical banks per PE, and one shared MAC wave.  Loading may overlap the
// other bank's compute; compute waves are serialized.  ONNXim timestamps are
// retained only to anchor issue and report disagreement/possible penalty.
UpdatedTwoBankShadowResult derive_updated_two_bank_shadow(
    const std::vector<UpdatedShadowInterval>& intervals,
    uint32_t array_rows,
    uint32_t array_columns,
    uint64_t simulator_horizon_cycles);

Result derive(const std::vector<Interval>& intervals,
              uint32_t array_rows,
              uint32_t array_columns,
              uint64_t horizon_cycles,
              uint64_t expected_union_active_cycles);

}  // namespace sa_pe_state
