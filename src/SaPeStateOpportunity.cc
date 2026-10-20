#include "SaPeStateOpportunity.h"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <tuple>

namespace sa_pe_state {
namespace {

uint64_t checked_multiply(uint64_t left, uint64_t right,
                          const char* description) {
  if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
    throw std::overflow_error(std::string(description) + " overflows uint64_t");
  }
  return left * right;
}

uint64_t checked_add(uint64_t left, uint64_t right,
                     const char* description) {
  if (right > std::numeric_limits<uint64_t>::max() - left) {
    throw std::overflow_error(std::string(description) + " overflows uint64_t");
  }
  return left + right;
}

struct Event {
  uint64_t cycle = 0;
  bool entering = false;
  uint32_t rows = 0;
  uint32_t columns = 0;
};

using Span = std::pair<uint64_t, uint64_t>;

std::vector<Span> merge_spans(std::vector<Span> spans,
                              uint64_t horizon_cycles) {
  for (Span& span : spans) {
    span.first = std::min(span.first, horizon_cycles);
    span.second = std::min(span.second, horizon_cycles);
  }
  spans.erase(std::remove_if(spans.begin(), spans.end(),
                             [](const Span& span) {
                               return span.second <= span.first;
                             }),
              spans.end());
  std::sort(spans.begin(), spans.end());
  std::vector<Span> result;
  for (const Span& span : spans) {
    if (result.empty() || span.first > result.back().second) {
      result.push_back(span);
    } else {
      result.back().second = std::max(result.back().second, span.second);
    }
  }
  return result;
}

uint64_t span_cycles(const std::vector<Span>& spans) {
  uint64_t result = 0;
  for (const Span& span : spans) {
    result = checked_add(result, span.second - span.first,
                         "merged span cycles");
  }
  return result;
}

uint64_t overlap_cycles(const std::vector<Span>& left,
                        const std::vector<Span>& right) {
  uint64_t result = 0;
  size_t left_index = 0;
  size_t right_index = 0;
  while (left_index < left.size() && right_index < right.size()) {
    const uint64_t begin = std::max(left[left_index].first,
                                    right[right_index].first);
    const uint64_t end = std::min(left[left_index].second,
                                  right[right_index].second);
    if (begin < end) {
      result = checked_add(result, end - begin, "span overlap cycles");
    }
    if (left[left_index].second < right[right_index].second) {
      ++left_index;
    } else {
      ++right_index;
    }
  }
  return result;
}

std::vector<Span> intersect_merged_spans(const std::vector<Span>& left,
                                         const std::vector<Span>& right) {
  std::vector<Span> result;
  size_t left_index = 0;
  size_t right_index = 0;
  while (left_index < left.size() && right_index < right.size()) {
    const uint64_t begin = std::max(left[left_index].first,
                                    right[right_index].first);
    const uint64_t finish = std::min(left[left_index].second,
                                     right[right_index].second);
    if (begin < finish) {
      result.emplace_back(begin, finish);
    }
    if (left[left_index].second < right[right_index].second) {
      ++left_index;
    } else {
      ++right_index;
    }
  }
  return result;
}

struct OccupancySummary {
  uint64_t peak = 0;
  uint64_t over_two_cycles = 0;
};

OccupancySummary sweep_preload_occupancy(const std::vector<Span>& spans) {
  struct Boundary {
    uint64_t cycle;
    bool entering;
  };
  std::vector<Boundary> boundaries;
  boundaries.reserve(spans.size() * 2);
  for (const Span& span : spans) {
    if (span.second <= span.first) continue;
    boundaries.push_back(Boundary{span.first, true});
    boundaries.push_back(Boundary{span.second, false});
  }
  std::sort(boundaries.begin(), boundaries.end(),
            [](const Boundary& a, const Boundary& b) {
              if (a.cycle != b.cycle) return a.cycle < b.cycle;
              return a.entering < b.entering;
            });
  OccupancySummary summary;
  uint64_t live = 0;
  uint64_t previous = 0;
  size_t index = 0;
  while (index < boundaries.size()) {
    const uint64_t cycle = boundaries[index].cycle;
    if (live > 2) {
      summary.over_two_cycles = checked_add(
          summary.over_two_cycles, cycle - previous,
          "over-two preload occupancy");
    }
    while (index < boundaries.size() &&
           boundaries[index].cycle == cycle) {
      if (boundaries[index].entering) {
        ++live;
      } else {
        --live;
      }
      ++index;
    }
    summary.peak = std::max(summary.peak, live);
    previous = cycle;
  }
  return summary;
}

uint64_t tetrahedral_prefix(int64_t value) {
  if (value < 0) return 0;
  const uint64_t x = static_cast<uint64_t>(value);
  const uint64_t left = checked_add(x, 1, "wave prefix");
  const uint64_t right = checked_add(x, 2, "wave prefix");
  // Divide before multiplying to retain the exact integer and reduce the
  // overflow range.
  return left % 2 == 0
      ? checked_multiply(left / 2, right, "wave prefix")
      : checked_multiply(left, right / 2, "wave prefix");
}

bool intersects_origin_footprint(uint32_t row_start, uint32_t row_finish,
                                 uint32_t column_start,
                                 uint32_t column_finish,
                                 uint32_t used_rows,
                                 uint32_t used_columns) {
  return row_start < used_rows && column_start < used_columns &&
         row_finish != 0 && column_finish != 0;
}

}  // namespace

uint64_t useful_macs_at_cycle(uint32_t rows, uint32_t columns,
                              uint32_t waves, uint64_t t) {
  if (rows == 0 || columns == 0 || waves == 0) return 0;
  if (t > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    throw std::overflow_error("wave cycle exceeds int64_t");
  }
  const int64_t n = static_cast<int64_t>(t);
  const int64_t r = rows;
  const int64_t c = columns;
  const int64_t b = waves;
  // Inclusion-exclusion over the three bounded coordinates.  Use a signed
  // accumulator because intermediate subtractions may be negative although
  // the final coefficient is nonnegative.
  __int128 count = tetrahedral_prefix(n);
  count -= tetrahedral_prefix(n - r);
  count -= tetrahedral_prefix(n - c);
  count -= tetrahedral_prefix(n - b);
  count += tetrahedral_prefix(n - r - c);
  count += tetrahedral_prefix(n - r - b);
  count += tetrahedral_prefix(n - c - b);
  count -= tetrahedral_prefix(n - r - c - b);
  if (count < 0 || count > std::numeric_limits<uint64_t>::max()) {
    throw std::overflow_error("invalid bounded wave coefficient");
  }
  return static_cast<uint64_t>(count);
}

Result derive(const std::vector<Interval>& intervals,
              uint32_t array_rows,
              uint32_t array_columns,
              uint64_t horizon_cycles,
              uint64_t expected_union_active_cycles) {
  if (array_rows == 0 || array_columns == 0) {
    throw std::invalid_argument("SA dimensions must be nonzero");
  }
  if (expected_union_active_cycles > horizon_cycles) {
    throw std::invalid_argument("SA-active cycles exceed the profile horizon");
  }

  Result result;
  const uint64_t array_area =
      checked_multiply(array_rows, array_columns, "SA area");
  result.active_capacity_pe_cycles = checked_multiply(
      array_area, expected_union_active_cycles, "active PE-cycle capacity");
  result.temporal_idle_unclassified_pe_cycles = checked_multiply(
      array_area, horizon_cycles - expected_union_active_cycles,
      "temporal-idle PE-cycle capacity");
  result.total_capacity_pe_cycles =
      checked_multiply(array_area, horizon_cycles, "total PE-cycle capacity");

  std::vector<Event> events;
  events.reserve(intervals.size() * 2);
  using ClassKey = std::tuple<uint64_t, uint32_t, uint32_t, uint32_t>;
  std::map<ClassKey, IntervalClass> classes;

  for (const Interval& interval : intervals) {
    if (interval.finish_cycle > horizon_cycles) {
      result.incomplete_intervals++;
      continue;
    }
    result.completed_intervals++;
    if (interval.finish_cycle <= interval.start_cycle ||
        interval.tile_m == 0 || interval.tile_k == 0 ||
        interval.tile_n == 0 || interval.tile_m > array_rows ||
        interval.tile_k > array_columns) {
      result.invalid_dimension_intervals++;
      continue;
    }

    const uint64_t duration = interval.finish_cycle - interval.start_cycle;
    const uint64_t mapped_pes = checked_multiply(
        interval.tile_m, interval.tile_k, "mapped PE count");
    const uint64_t on_compute = checked_multiply(
        mapped_pes, interval.tile_n, "ON-compute PE cycles");
    const uint64_t resident = checked_multiply(
        mapped_pes, duration, "resident-weight PE cycles");
    if (on_compute > resident) {
      // The one-valid-MAC-per-mapped-PE-per-cycle approximation cannot
      // represent this interval. Keep the inconsistency visible and exclude
      // it from state accounting instead of silently clipping it.
      result.insufficient_duration_intervals++;
      continue;
    }

    result.on_compute_pe_cycles = checked_add(
        result.on_compute_pe_cycles, on_compute,
        "aggregate ON-compute PE cycles");
    events.push_back(Event{interval.start_cycle, true,
                           interval.tile_m, interval.tile_k});
    events.push_back(Event{interval.finish_cycle, false,
                           interval.tile_m, interval.tile_k});

    const uint64_t interval_capacity =
        checked_multiply(array_area, duration, "interval PE-cycle capacity");
    const ClassKey key{duration, interval.tile_m,
                       interval.tile_k, interval.tile_n};
    auto [iterator, inserted] = classes.try_emplace(key);
    IntervalClass& item = iterator->second;
    if (inserted) {
      item.duration_cycles = duration;
      item.tile_m = interval.tile_m;
      item.tile_k = interval.tile_k;
      item.tile_n = interval.tile_n;
      item.on_compute_pe_cycles_per_interval = on_compute;
      item.resident_weight_pe_cycles_per_interval = resident;
      item.w_on_candidate_pe_cycles_per_interval = resident - on_compute;
      item.off_unmapped_candidate_pe_cycles_per_interval =
          interval_capacity - resident;
    }
    item.occurrences++;
  }

  std::sort(events.begin(), events.end(), [](const Event& left,
                                              const Event& right) {
    if (left.cycle != right.cycle) return left.cycle < right.cycle;
    // Process exits before entries at a shared half-open boundary.
    return left.entering < right.entering;
  });

  // Every mapped footprint is a tile_m-by-tile_k rectangle anchored at the
  // array origin. For each physical row, the union width is therefore the
  // maximum active tile_k. Maintaining a multiset per row gives the exact
  // union of overlapping packed footprints without instantiating every PE.
  std::vector<std::multiset<uint32_t>> active_widths(array_rows);
  uint64_t previous_cycle = 0;
  uint64_t resident_area = 0;
  size_t event_index = 0;
  while (event_index < events.size()) {
    const uint64_t cycle = events[event_index].cycle;
    const uint64_t elapsed = cycle - previous_cycle;
    result.resident_weight_pe_cycles = checked_add(
        result.resident_weight_pe_cycles,
        checked_multiply(resident_area, elapsed,
                         "resident-weight PE-cycle union"),
        "resident-weight PE-cycle union");
    if (resident_area != 0) {
      result.union_active_cycles = checked_add(
          result.union_active_cycles, elapsed, "union SA-active cycles");
    }

    while (event_index < events.size() && events[event_index].cycle == cycle) {
      const Event& event = events[event_index];
      for (uint32_t row = 0; row < event.rows; ++row) {
        if (event.entering) {
          active_widths[row].insert(event.columns);
        } else {
          auto match = active_widths[row].find(event.columns);
          if (match == active_widths[row].end()) {
            throw std::logic_error("unmatched SA footprint exit event");
          }
          active_widths[row].erase(match);
        }
      }
      ++event_index;
    }
    resident_area = 0;
    for (const auto& widths : active_widths) {
      if (!widths.empty()) resident_area += *widths.rbegin();
    }
    previous_cycle = cycle;
  }

  result.on_within_resident_capacity =
      result.on_compute_pe_cycles <= result.resident_weight_pe_cycles;
  result.resident_within_active_capacity =
      result.resident_weight_pe_cycles <= result.active_capacity_pe_cycles;
  if (result.on_within_resident_capacity) {
    result.w_on_candidate_pe_cycles =
        result.resident_weight_pe_cycles - result.on_compute_pe_cycles;
  }
  if (result.resident_within_active_capacity) {
    result.off_unmapped_candidate_pe_cycles =
        result.active_capacity_pe_cycles - result.resident_weight_pe_cycles;
  }

  const uint64_t classified_active = checked_add(
      checked_add(result.on_compute_pe_cycles,
                  result.w_on_candidate_pe_cycles,
                  "classified active PE cycles"),
      result.off_unmapped_candidate_pe_cycles,
      "classified active PE cycles");
  const uint64_t all_capacity = checked_add(
      classified_active, result.temporal_idle_unclassified_pe_cycles,
      "classified total PE cycles");
  result.accounting_exact =
      result.incomplete_intervals == 0 &&
      result.invalid_dimension_intervals == 0 &&
      result.insufficient_duration_intervals == 0 &&
      result.union_active_cycles == expected_union_active_cycles &&
      result.on_within_resident_capacity &&
      result.resident_within_active_capacity &&
      classified_active == result.active_capacity_pe_cycles &&
      all_capacity == result.total_capacity_pe_cycles;

  result.interval_classes.reserve(classes.size());
  for (const auto& [key, item] : classes) {
    (void)key;
    result.interval_classes.push_back(item);
  }
  return result;
}

WaveLifetimeResult derive_wave_lifetime(
    const std::vector<WaveInterval>& input_intervals,
    uint32_t array_rows,
    uint32_t array_columns,
    uint64_t horizon_cycles,
    uint64_t expected_union_active_cycles) {
  if (array_rows == 0 || array_columns == 0) {
    throw std::invalid_argument("SA dimensions must be nonzero");
  }
  if (expected_union_active_cycles > horizon_cycles) {
    throw std::invalid_argument("SA-active cycles exceed the profile horizon");
  }

  WaveLifetimeResult result;
  const uint64_t array_area =
      checked_multiply(array_rows, array_columns, "SA area");
  result.total_capacity_pe_cycles =
      checked_multiply(array_area, horizon_cycles, "total PE-cycle capacity");
  result.simulator_active_capacity_pe_cycles = checked_multiply(
      array_area, expected_union_active_cycles,
      "simulator-active PE-cycle capacity");

  std::vector<WaveInterval> intervals = input_intervals;
  std::sort(intervals.begin(), intervals.end(),
            [](const WaveInterval& left, const WaveInterval& right) {
              if (left.start_cycle != right.start_cycle) {
                return left.start_cycle < right.start_cycle;
              }
              // A preload establishes the block used by an ordinary GEMM at
              // the same modeled start boundary.
              return left.opcode == WaveOpcode::PRELOAD &&
                     right.opcode != WaveOpcode::PRELOAD;
            });

  uint64_t top_left_busy_until = 0;
  bool have_top_left_wave = false;
  bool current_lifetime = false;
  WeightLifetime current;
  auto close_lifetime = [&]() {
    if (!current_lifetime) return;
    result.weight_lifetimes.push_back(current);
    current = WeightLifetime{};
    current_lifetime = false;
  };

  for (const WaveInterval& interval : intervals) {
    if (interval.finish_cycle > horizon_cycles) {
      result.incomplete_intervals++;
      continue;
    }
    result.completed_intervals++;
    if (interval.finish_cycle <= interval.start_cycle ||
        interval.tile_m == 0 || interval.tile_k == 0 ||
        interval.tile_n == 0 || interval.tile_m > array_rows ||
        interval.tile_k > array_columns) {
      result.invalid_dimension_intervals++;
      continue;
    }

    const uint64_t wave_duration = checked_add(
        checked_add(interval.tile_m, interval.tile_k, "wave duration"),
        interval.tile_n, "wave duration") - 2;
    const uint64_t wave_finish = checked_add(
        interval.start_cycle, wave_duration, "wave finish cycle");
    if (wave_finish > interval.finish_cycle) {
      result.wave_exceeds_instruction_intervals++;
      continue;
    }

    // Every footprint is origin-packed. Consequently, two useful waves can
    // overlap at a PE only if they overlap at PE(0,0), whose active interval
    // is [start,start+tile_n).
    if (have_top_left_wave && interval.start_cycle < top_left_busy_until) {
      result.overlapping_wave_intervals++;
    }
    top_left_busy_until = std::max(
        top_left_busy_until,
        checked_add(interval.start_cycle, interval.tile_n,
                    "top-left wave finish"));
    have_top_left_wave = true;

    const uint64_t footprint = checked_multiply(
        interval.tile_m, interval.tile_k, "wave footprint");
    result.analytical_on_pe_cycles = checked_add(
        result.analytical_on_pe_cycles,
        checked_multiply(footprint, interval.tile_n,
                         "analytical ON PE cycles"),
        "analytical ON PE cycles");

    if (!interval.weight_block_known || interval.weight_block_id == 0) {
      result.unknown_weight_block_intervals++;
    }

    if (interval.opcode == WaveOpcode::PRELOAD) {
      close_lifetime();
      if (interval.weight_block_known && interval.weight_block_id != 0) {
        current_lifetime = true;
        current.start_cycle = interval.start_cycle;
        current.finish_cycle = interval.finish_cycle;
        current.rows = interval.tile_m;
        current.columns = interval.tile_k;
        current.weight_block_id = interval.weight_block_id;
        current.consumer_instructions = 1;
      }
      continue;
    }

    if (!current_lifetime) {
      result.gemm_without_preload_intervals++;
      continue;
    }
    if (!interval.weight_block_known || interval.weight_block_id == 0 ||
        interval.weight_block_id != current.weight_block_id) {
      result.weight_block_mismatch_intervals++;
      continue;
    }
    if (interval.tile_m != current.rows ||
        interval.tile_k != current.columns) {
      result.footprint_mismatch_intervals++;
      continue;
    }
    current.finish_cycle = std::max(current.finish_cycle,
                                    interval.finish_cycle);
    current.consumer_instructions++;
  }
  close_lifetime();

  result.wave_overlap_free = result.overlapping_wave_intervals == 0;
  result.weight_lifetime_complete =
      result.incomplete_intervals == 0 &&
      result.invalid_dimension_intervals == 0 &&
      result.wave_exceeds_instruction_intervals == 0 &&
      result.unknown_weight_block_intervals == 0 &&
      result.gemm_without_preload_intervals == 0 &&
      result.weight_block_mismatch_intervals == 0 &&
      result.footprint_mismatch_intervals == 0;

  if (result.analytical_on_pe_cycles <=
      result.simulator_active_capacity_pe_cycles) {
    result.simulator_active_non_wave_pe_cycles =
        result.simulator_active_capacity_pe_cycles -
        result.analytical_on_pe_cycles;
  }

  // Compute the union of all known, compiler-live, origin-packed stationary
  // weight footprints. Lifetimes may overlap while one wave drains and the
  // next preload begins; the union prevents double-counting those PE-cycles.
  std::vector<Event> events;
  events.reserve(result.weight_lifetimes.size() * 2);
  for (const WeightLifetime& lifetime : result.weight_lifetimes) {
    events.push_back(Event{lifetime.start_cycle, true,
                           lifetime.rows, lifetime.columns});
    events.push_back(Event{lifetime.finish_cycle, false,
                           lifetime.rows, lifetime.columns});
  }
  std::sort(events.begin(), events.end(), [](const Event& left,
                                              const Event& right) {
    if (left.cycle != right.cycle) return left.cycle < right.cycle;
    return left.entering < right.entering;
  });
  std::vector<std::multiset<uint32_t>> active_widths(array_rows);
  uint64_t previous_cycle = 0;
  uint64_t resident_area = 0;
  size_t event_index = 0;
  while (event_index < events.size()) {
    const uint64_t cycle = events[event_index].cycle;
    result.resident_weight_pe_cycles = checked_add(
        result.resident_weight_pe_cycles,
        checked_multiply(resident_area, cycle - previous_cycle,
                         "weight-lifetime PE-cycle union"),
        "weight-lifetime PE-cycle union");
    while (event_index < events.size() && events[event_index].cycle == cycle) {
      const Event& event = events[event_index];
      for (uint32_t row = 0; row < event.rows; ++row) {
        if (event.entering) {
          active_widths[row].insert(event.columns);
        } else {
          auto match = active_widths[row].find(event.columns);
          if (match == active_widths[row].end()) {
            throw std::logic_error("unmatched weight-lifetime exit event");
          }
          active_widths[row].erase(match);
        }
      }
      ++event_index;
    }
    resident_area = 0;
    for (const auto& widths : active_widths) {
      if (!widths.empty()) resident_area += *widths.rbegin();
    }
    previous_cycle = cycle;
  }

  // Preserve ONNXim's issue-to-start scheduling gap as an explicitly unknown
  // committed footprint. The simulator exposes no per-PE preload movement in
  // this interval, so treating it as OFF or W_on would be an unsupported
  // inference. This aggregate can overlap an older live footprint.
  events.clear();
  for (const WaveInterval& interval : intervals) {
    if (interval.issue_cycle >= interval.start_cycle ||
        interval.start_cycle > horizon_cycles || interval.tile_m == 0 ||
        interval.tile_k == 0 || interval.tile_m > array_rows ||
        interval.tile_k > array_columns) {
      continue;
    }
    events.push_back(Event{interval.issue_cycle, true,
                           interval.tile_m, interval.tile_k});
    events.push_back(Event{interval.start_cycle, false,
                           interval.tile_m, interval.tile_k});
  }
  std::sort(events.begin(), events.end(), [](const Event& left,
                                              const Event& right) {
    if (left.cycle != right.cycle) return left.cycle < right.cycle;
    return left.entering < right.entering;
  });
  active_widths = std::vector<std::multiset<uint32_t>>(array_rows);
  previous_cycle = 0;
  uint64_t committed_area = 0;
  event_index = 0;
  while (event_index < events.size()) {
    const uint64_t cycle = events[event_index].cycle;
    result.pre_start_committed_pe_cycles = checked_add(
        result.pre_start_committed_pe_cycles,
        checked_multiply(committed_area, cycle - previous_cycle,
                         "pre-start committed PE-cycle union"),
        "pre-start committed PE-cycle union");
    while (event_index < events.size() && events[event_index].cycle == cycle) {
      const Event& event = events[event_index];
      for (uint32_t row = 0; row < event.rows; ++row) {
        if (event.entering) {
          active_widths[row].insert(event.columns);
        } else {
          auto match = active_widths[row].find(event.columns);
          if (match == active_widths[row].end()) {
            throw std::logic_error("unmatched committed-footprint exit event");
          }
          active_widths[row].erase(match);
        }
      }
      ++event_index;
    }
    committed_area = 0;
    for (const auto& widths : active_widths) {
      if (!widths.empty()) committed_area += *widths.rbegin();
    }
    previous_cycle = cycle;
  }

  const bool capacities_consistent =
      result.analytical_on_pe_cycles <=
          result.simulator_active_capacity_pe_cycles &&
      result.analytical_on_pe_cycles <= result.resident_weight_pe_cycles &&
      result.resident_weight_pe_cycles <= result.total_capacity_pe_cycles;
  result.state_accounting_exact = result.wave_overlap_free &&
      result.weight_lifetime_complete && capacities_consistent &&
      result.pre_start_committed_pe_cycles == 0;
  if (result.state_accounting_exact) {
    result.analytical_w_on_pe_cycles =
        result.resident_weight_pe_cycles - result.analytical_on_pe_cycles;
    result.analytical_off_pe_cycles =
        result.total_capacity_pe_cycles - result.resident_weight_pe_cycles;
  } else if (result.wave_overlap_free &&
             result.analytical_on_pe_cycles <=
                 result.total_capacity_pe_cycles) {
    // Useful MAC work remains classified even when the weight lifetime cannot
    // be proven. Do not guess whether any remaining cycle is W_on or OFF.
    result.unclassified_pe_cycles =
        result.total_capacity_pe_cycles - result.analytical_on_pe_cycles;
  } else {
    result.unclassified_pe_cycles = result.total_capacity_pe_cycles;
  }
  return result;
}

RegionStateResult derive_region_states(
    const std::vector<WaveInterval>& intervals,
    const WaveLifetimeResult& wave_lifetime,
    uint32_t array_rows,
    uint32_t array_columns,
    uint32_t region_size_rows,
    uint32_t region_size_columns,
    uint64_t horizon_cycles) {
  if (array_rows == 0 || array_columns == 0 || region_size_rows == 0 ||
      region_size_columns == 0) {
    throw std::invalid_argument("array and region dimensions must be nonzero");
  }
  RegionStateResult result;
  result.region_size_rows = region_size_rows;
  result.region_size_columns = region_size_columns;
  result.region_rows =
      (array_rows + region_size_rows - 1) / region_size_rows;
  result.region_columns =
      (array_columns + region_size_columns - 1) / region_size_columns;
  result.region_count = checked_multiply(
      result.region_rows, result.region_columns, "region count");
  result.capacity_region_cycles = checked_multiply(
      result.region_count, horizon_cycles, "region-cycle capacity");
  result.per_region_on_cycles.reserve(result.region_count);
  result.per_region_w_on_cycles.reserve(result.region_count);
  result.per_region_off_cycles.reserve(result.region_count);

  if (!wave_lifetime.weight_lifetime_complete ||
      !wave_lifetime.wave_overlap_free) {
    result.unclassified_region_cycles = result.capacity_region_cycles;
    return result;
  }

  bool exact = true;
  for (uint32_t region_row = 0; region_row < result.region_rows;
       ++region_row) {
    const uint32_t row_start = region_row * region_size_rows;
    const uint32_t row_finish =
        std::min(array_rows, row_start + region_size_rows);
    for (uint32_t region_column = 0;
         region_column < result.region_columns; ++region_column) {
      const uint32_t column_start = region_column * region_size_columns;
      const uint32_t column_finish =
          std::min(array_columns, column_start + region_size_columns);
      std::vector<Span> on_spans;
      std::vector<Span> resident_spans;
      std::vector<Span> committed_unknown_spans;

      for (const WaveInterval& interval : intervals) {
        if (interval.finish_cycle > horizon_cycles ||
            interval.finish_cycle <= interval.start_cycle ||
            interval.tile_m == 0 || interval.tile_k == 0 ||
            interval.tile_n == 0 || interval.tile_m > array_rows ||
            interval.tile_k > array_columns ||
            !intersects_origin_footprint(
                row_start, row_finish, column_start, column_finish,
                interval.tile_m, interval.tile_k)) {
          continue;
        }
        const uint32_t active_row_start = row_start;
        const uint32_t active_row_finish =
            std::min(row_finish, interval.tile_m);
        const uint32_t active_column_start = column_start;
        const uint32_t active_column_finish =
            std::min(column_finish, interval.tile_k);
        // The set of row+column+wave values over a rectangular region is a
        // contiguous integer interval, so the shared region is ON throughout
        // this half-open span whenever at least one member PE performs a MAC.
        const uint64_t first = checked_add(
            interval.start_cycle,
            static_cast<uint64_t>(active_row_start) + active_column_start,
            "region wave start");
        const uint64_t last_exclusive = checked_add(
            interval.start_cycle,
            static_cast<uint64_t>(active_row_finish) +
                active_column_finish + interval.tile_n - 2,
            "region wave finish");
        on_spans.emplace_back(first, std::min(last_exclusive,
                                              interval.finish_cycle));
        if (interval.issue_cycle < interval.start_cycle) {
          committed_unknown_spans.emplace_back(interval.issue_cycle,
                                               interval.start_cycle);
        }
      }
      for (const WeightLifetime& lifetime :
           wave_lifetime.weight_lifetimes) {
        if (intersects_origin_footprint(
                row_start, row_finish, column_start, column_finish,
                lifetime.rows, lifetime.columns)) {
          resident_spans.emplace_back(lifetime.start_cycle,
                                      lifetime.finish_cycle);
        }
      }

      const std::vector<Span> merged_on =
          merge_spans(std::move(on_spans), horizon_cycles);
      const std::vector<Span> merged_resident =
          merge_spans(std::move(resident_spans), horizon_cycles);
      const std::vector<Span> merged_unknown =
          merge_spans(std::move(committed_unknown_spans), horizon_cycles);
      const uint64_t on_cycles = span_cycles(merged_on);
      const uint64_t resident_cycles = span_cycles(merged_resident);
      // Shared-domain priority is ON > committed/unknown > W_on > OFF. The
      // priority prevents normal pipelining (a new preload commitment while
      // an old wave drains) from double-counting region cycles.
      std::vector<Span> on_or_unknown_spans = merged_on;
      on_or_unknown_spans.insert(on_or_unknown_spans.end(),
                                 merged_unknown.begin(),
                                 merged_unknown.end());
      const std::vector<Span> merged_on_or_unknown =
          merge_spans(std::move(on_or_unknown_spans), horizon_cycles);
      const uint64_t unknown_cycles =
          span_cycles(merged_unknown) -
          overlap_cycles(merged_unknown, merged_on);
      const uint64_t on_resident_cycles =
          overlap_cycles(merged_on, merged_resident);
      const uint64_t resident_covered_cycles =
          overlap_cycles(merged_resident, merged_on_or_unknown);
      if (on_resident_cycles != on_cycles ||
          resident_cycles > horizon_cycles ||
          resident_covered_cycles > resident_cycles) {
        exact = false;
      }
      const uint64_t w_on_cycles =
          resident_covered_cycles <= resident_cycles
              ? resident_cycles - resident_covered_cycles
              : 0;
      const uint64_t classified_non_off =
          on_cycles + unknown_cycles + w_on_cycles;
      const uint64_t off_cycles = classified_non_off <= horizon_cycles
          ? horizon_cycles - classified_non_off
          : 0;
      result.per_region_on_cycles.push_back(on_cycles);
      result.per_region_w_on_cycles.push_back(w_on_cycles);
      result.per_region_off_cycles.push_back(off_cycles);
      result.analytical_on_region_cycles = checked_add(
          result.analytical_on_region_cycles, on_cycles,
          "ON region cycles");
      result.analytical_w_on_region_cycles = checked_add(
          result.analytical_w_on_region_cycles, w_on_cycles,
          "W_on region cycles");
      result.analytical_off_region_cycles = checked_add(
          result.analytical_off_region_cycles, off_cycles,
          "OFF region cycles");
      result.unclassified_region_cycles = checked_add(
          result.unclassified_region_cycles, unknown_cycles,
          "unclassified region cycles");
    }
  }
  result.state_accounting_exact = exact &&
      result.analytical_on_region_cycles +
          result.analytical_w_on_region_cycles +
          result.analytical_off_region_cycles ==
          result.capacity_region_cycles - result.unclassified_region_cycles;
  if (!result.state_accounting_exact) {
    result.unclassified_region_cycles = result.capacity_region_cycles;
  }
  return result;
}

ReGateFunctionalResult derive_regate_functional(
    const std::vector<WaveInterval>& intervals,
    uint32_t array_rows,
    uint32_t array_columns,
    uint64_t horizon_cycles) {
  if (array_rows == 0 || array_columns == 0) {
    throw std::invalid_argument("SA dimensions must be nonzero");
  }

  std::vector<Span> active_spans;
  active_spans.reserve(intervals.size());
  for (const WaveInterval& interval : intervals) {
    active_spans.emplace_back(interval.start_cycle, interval.finish_cycle);
  }
  const uint64_t active_cycles =
      span_cycles(merge_spans(std::move(active_spans), horizon_cycles));
  const WaveLifetimeResult lifetime = derive_wave_lifetime(
      intervals, array_rows, array_columns, horizon_cycles, active_cycles);
  const RegionStateResult pe_states = derive_region_states(
      intervals, lifetime, array_rows, array_columns, 1, 1, horizon_cycles);

  ReGateFunctionalResult result;
  result.completed_intervals = lifetime.completed_intervals;
  result.incomplete_intervals = lifetime.incomplete_intervals;
  result.invalid_dimension_intervals =
      lifetime.invalid_dimension_intervals;
  result.wave_exceeds_instruction_intervals =
      lifetime.wave_exceeds_instruction_intervals;
  result.overlapping_wave_intervals =
      lifetime.overlapping_wave_intervals;
  result.unknown_weight_block_intervals =
      lifetime.unknown_weight_block_intervals;
  result.gemm_without_preload_intervals =
      lifetime.gemm_without_preload_intervals;
  result.weight_block_mismatch_intervals =
      lifetime.weight_block_mismatch_intervals;
  result.footprint_mismatch_intervals =
      lifetime.footprint_mismatch_intervals;
  result.capacity_pe_cycles = pe_states.capacity_region_cycles;
  result.control_schedule_valid =
      lifetime.incomplete_intervals == 0 &&
      lifetime.invalid_dimension_intervals == 0 &&
      lifetime.wave_exceeds_instruction_intervals == 0 &&
      lifetime.overlapping_wave_intervals == 0;
  result.control_event_counts_valid = result.control_schedule_valid;
  result.weight_lifetime_complete = lifetime.weight_lifetime_complete;
  result.state_partition_exact_under_model =
      pe_states.state_accounting_exact;
  if (result.state_partition_exact_under_model) {
    result.on_pe_cycles = pe_states.analytical_on_region_cycles;
    result.w_on_pe_cycles = pe_states.analytical_w_on_region_cycles;
    result.off_pe_cycles = pe_states.analytical_off_region_cycles;
    result.unclassified_pe_cycles = pe_states.unclassified_region_cycles;
  } else {
    result.unclassified_pe_cycles = result.capacity_pe_cycles;
  }

  using ClassKey = std::tuple<uint32_t, uint32_t, uint32_t>;
  std::map<ClassKey, ReGateControlClass> classes;
  for (const WaveInterval& interval : intervals) {
    if (interval.finish_cycle > horizon_cycles ||
        interval.finish_cycle <= interval.start_cycle ||
        interval.tile_m == 0 || interval.tile_k == 0 ||
        interval.tile_n == 0 || interval.tile_m > array_rows ||
        interval.tile_k > array_columns) {
      continue;
    }
    const uint64_t wave_cycles = checked_add(
        checked_add(interval.tile_m, interval.tile_k, "wave duration"),
        interval.tile_n, "wave duration") - 2;
    if (checked_add(interval.start_cycle, wave_cycles,
                    "wave finish cycle") > interval.finish_cycle) {
      continue;
    }
    const uint64_t footprint = checked_multiply(
        interval.tile_m, interval.tile_k, "ReGate mapped PE count");
    const ClassKey key{interval.tile_m, interval.tile_k, interval.tile_n};
    auto [iterator, inserted] = classes.try_emplace(key);
    ReGateControlClass& item = iterator->second;
    if (inserted) {
      item.tile_m = interval.tile_m;
      item.tile_k = interval.tile_k;
      item.tile_n = interval.tile_n;
      item.row_on_prefix_length = interval.tile_m;
      item.col_on_prefix_length = interval.tile_k;
      item.pe_on_wave_cycles = wave_cycles;
      item.on_pe_cycles_per_instruction = checked_multiply(
          footprint, interval.tile_n, "ReGate ON PE cycles");
      item.pe_on_assert_events_per_instruction = footprint;
      item.pe_on_deassert_events_per_instruction = footprint;
    }
    item.occurrences++;
    if (result.control_event_counts_valid) {
      result.pe_on_assert_events = checked_add(
          result.pe_on_assert_events, footprint, "PE_on assert events");
      result.pe_on_deassert_events = checked_add(
          result.pe_on_deassert_events, footprint, "PE_on deassert events");
    }
  }
  result.control_classes.reserve(classes.size());
  for (const auto& [key, item] : classes) {
    (void)key;
    result.control_classes.push_back(item);
  }
  return result;
}

TwoBankReGateResult derive_two_bank_regate(
    const std::vector<WaveInterval>& input_intervals,
    uint32_t array_rows,
    uint32_t array_columns,
    uint64_t horizon_cycles,
    const RowWriteTiming* row_write_timing) {
  if (array_rows == 0 || array_columns == 0) {
    throw std::invalid_argument("SA dimensions must be nonzero");
  }
  if (row_write_timing &&
      (row_write_timing->row_start_period_cycles == 0 ||
       row_write_timing->row_write_duration_cycles == 0)) {
    throw std::invalid_argument("row-write period and duration must be nonzero");
  }

  TwoBankReGateResult result;
  result.capacity_pe_cycles = checked_multiply(
      checked_multiply(array_rows, array_columns, "SA area"),
      horizon_cycles, "two-bank PE-cycle capacity");
  result.rowwise_state_accounting.capacity_pe_cycles =
      result.capacity_pe_cycles;
  result.rowwise_state_accounting.unclassified_pe_cycles =
      result.capacity_pe_cycles;
  auto fail_closed = [&]() {
    result.on_pe_cycles = 0;
    result.on_zero_bank_pe_cycles = 0;
    result.on_one_bank_pe_cycles = 0;
    result.on_two_bank_pe_cycles = 0;
    result.w_on_one_bank_pe_cycles = 0;
    result.w_on_two_bank_pe_cycles = 0;
    result.off_pe_cycles = 0;
    result.preload_bank_pe_cycles = 0;
    result.unclassified_pe_cycles = result.capacity_pe_cycles;
    result.validation_pass = false;
    return result;
  };

  std::vector<WaveInterval> intervals = input_intervals;
  // Equal-issue instructions retain their recorded order. In particular,
  // each explicit PRELOAD remains a distinct load even when its weight ID
  // matches an older preload.
  std::stable_sort(intervals.begin(), intervals.end(),
                   [](const WaveInterval& a, const WaveInterval& b) {
                     return a.issue_cycle < b.issue_cycle;
                   });

  std::vector<Span> top_left_waves;
  top_left_waves.reserve(intervals.size());
  for (const WaveInterval& interval : intervals) {
    if (interval.opcode == WaveOpcode::PRELOAD) {
      ++result.preload_events;
    } else {
      ++result.gemm_events;
    }
    if (interval.finish_cycle > horizon_cycles) {
      ++result.incomplete_intervals;
    }
    if (interval.issue_cycle > interval.start_cycle ||
        interval.start_cycle >= interval.finish_cycle ||
        interval.tile_m == 0 || interval.tile_k == 0 ||
        interval.tile_n == 0 || interval.tile_m > array_rows ||
        interval.tile_k > array_columns) {
      ++result.invalid_intervals;
      continue;
    }
    if (!interval.weight_block_known || interval.weight_block_id == 0) {
      ++result.unknown_weight_intervals;
    }
    const uint64_t wave_length = checked_add(
        checked_add(interval.tile_m, interval.tile_k, "wave length"),
        interval.tile_n, "wave length") - 2;
    const uint64_t wave_finish = checked_add(
        interval.start_cycle, wave_length, "wave finish");
    if (wave_finish > interval.finish_cycle) {
      ++result.wave_exceeds_instruction_intervals;
    }
    top_left_waves.emplace_back(
        interval.start_cycle,
        checked_add(interval.start_cycle, interval.tile_n,
                    "top-left wave finish"));
  }
  std::sort(top_left_waves.begin(), top_left_waves.end());
  uint64_t busy_until = 0;
  for (const Span& wave : top_left_waves) {
    if (wave.first < busy_until) {
      ++result.compute_collisions;
    }
    busy_until = std::max(busy_until, wave.second);
  }
  if (result.incomplete_intervals || result.invalid_intervals ||
      result.unknown_weight_intervals ||
      result.wave_exceeds_instruction_intervals ||
      result.compute_collisions) {
    return fail_closed();
  }

  struct Preload {
    size_t instruction_index = 0;
    uint64_t last_consumer_finish = 0;
    uint64_t residency_finish = 0;
    int bank = -1;
  };
  using WeightKey = std::tuple<uint64_t, uint32_t, uint32_t>;
  std::map<WeightKey, size_t> latest_preload;
  std::map<uint64_t, size_t> latest_identity;
  std::vector<Preload> preloads;
  preloads.reserve(result.preload_events);
  std::vector<size_t> owner(intervals.size(),
                            std::numeric_limits<size_t>::max());
  for (size_t i = 0; i < intervals.size(); ++i) {
    const WaveInterval& interval = intervals[i];
    const WeightKey key{interval.weight_block_id,
                        interval.tile_m, interval.tile_k};
    if (interval.opcode == WaveOpcode::PRELOAD) {
      const size_t index = preloads.size();
      preloads.push_back(Preload{i, interval.finish_cycle,
                                 horizon_cycles, -1});
      latest_preload[key] = index;
      latest_identity[interval.weight_block_id] = index;
      owner[i] = index;
      continue;
    }
    const auto match = latest_preload.find(key);
    if (match == latest_preload.end()) {
      if (latest_identity.count(interval.weight_block_id)) {
        ++result.footprint_mismatch_intervals;
      } else {
        ++result.gemm_without_weight_intervals;
      }
      continue;
    }
    const size_t index = match->second;
    owner[i] = index;
    if (interval.start_cycle <
        intervals[preloads[index].instruction_index].start_cycle) {
      ++result.weight_not_ready_intervals;
    }
    preloads[index].last_consumer_finish =
        std::max(preloads[index].last_consumer_finish,
                 interval.finish_cycle);
  }
  if (result.gemm_without_weight_intervals ||
      result.footprint_mismatch_intervals ||
      result.weight_not_ready_intervals) {
    return fail_closed();
  }

  // This audit uses the same exact latest-prior-preload association as the
  // validated model, but never assigns banks or classifies power states.
  // It remains available when the issue-time bank assignment below fails.
  TwoBankTimingAudit& audit = result.timing_audit;
  audit.available = true;
  std::vector<uint64_t> last_wave_finish(preloads.size(), 0);
  for (size_t i = 0; i < intervals.size(); ++i) {
    const WaveInterval& interval = intervals[i];
    if (interval.opcode == WaveOpcode::GEMM) {
      ++audit.matched_gemm_consumers;
    }
    const uint64_t wave_length = checked_add(
        checked_add(interval.tile_m, interval.tile_k,
                    "timing audit wave length"),
        interval.tile_n, "timing audit wave length") - 2;
    last_wave_finish[owner[i]] = std::max(
        last_wave_finish[owner[i]],
        checked_add(interval.start_cycle, wave_length,
                    "timing audit wave finish"));
  }
  auto global_occupancy = [&](TwoBankTimingCase& timing_case,
                              bool load_at_start,
                              bool release_at_last_mac) {
    std::vector<Span> spans;
    spans.reserve(preloads.size());
    for (size_t p = 0; p < preloads.size(); ++p) {
      const WaveInterval& preload = intervals[preloads[p].instruction_index];
      spans.emplace_back(
          load_at_start ? preload.start_cycle : preload.issue_cycle,
          release_at_last_mac ? last_wave_finish[p]
                              : preloads[p].last_consumer_finish);
    }
    const OccupancySummary summary = sweep_preload_occupancy(spans);
    timing_case.global_peak_live_preloads = summary.peak;
    timing_case.global_over_two_cycles = summary.over_two_cycles;
  };
  global_occupancy(audit.issue_to_whole_finish, false, false);
  global_occupancy(audit.start_to_whole_finish, true, false);
  global_occupancy(audit.issue_to_pe_last_mac, false, true);
  global_occupancy(audit.start_to_pe_last_mac, true, true);
  audit.pe_first_to_pe_last_mac.global_applicable = false;

  TwoBankTimingCase* timing_cases[] = {
      &audit.issue_to_whole_finish, &audit.start_to_whole_finish,
      &audit.issue_to_pe_last_mac, &audit.start_to_pe_last_mac,
      &audit.pe_first_to_pe_last_mac};
  if (row_write_timing) {
    PreloadFeasibilityAudit& feasibility =
        result.preload_feasibility_audit;
    feasibility.available = !preloads.empty() || intervals.empty();
    feasibility.events.reserve(preloads.size());
    for (size_t p = 0; p < preloads.size(); ++p) {
      const WaveInterval& preload =
          intervals[preloads[p].instruction_index];
      PreloadFeasibilityEvent event;
      event.preload_index = p;
      event.instruction_index = preloads[p].instruction_index;
      event.weight_block_id = preload.weight_block_id;
      event.issue_cycle = preload.issue_cycle;
      event.start_cycle = preload.start_cycle;
      event.rows = preload.tile_m;
      event.columns = preload.tile_k;
      event.fill_span_cycles = checked_add(
          checked_multiply(
              preload.tile_m - 1,
              row_write_timing->row_start_period_cycles,
              "preload row-stream fill span"),
          row_write_timing->row_write_duration_cycles,
          "preload row-stream fill span");
      if (event.fill_span_cycles <= preload.start_cycle) {
        event.planned_load_begin_available = true;
        event.planned_load_begin_cycle =
            preload.start_cycle - event.fill_span_cycles;
        event.load_begin_cycle = std::max(
            preload.issue_cycle, event.planned_load_begin_cycle);
      } else {
        event.load_begin_cycle = preload.issue_cycle;
      }
      event.bank_ready_cycle = checked_add(
          event.load_begin_cycle, event.fill_span_cycles,
          "preload whole-bank ready");
      event.first_use_cycle = preload.start_cycle;
      if (event.bank_ready_cycle > event.first_use_cycle) {
        event.required_stall_cycles =
            event.bank_ready_cycle - event.first_use_cycle;
      }
      event.no_added_cycle = event.required_stall_cycles == 0;
      ++feasibility.audited_preloads;
      if (event.no_added_cycle) {
        ++feasibility.no_added_cycle_preloads;
      } else {
        ++feasibility.conflicting_preloads;
      }
      feasibility.total_required_stall_cycles = checked_add(
          feasibility.total_required_stall_cycles,
          event.required_stall_cycles,
          "total independent preload required stalls");
      feasibility.max_required_stall_cycles = std::max(
          feasibility.max_required_stall_cycles,
          event.required_stall_cycles);
      feasibility.events.push_back(event);
    }

    RowwiseFeasibility& rowwise = result.rowwise_feasibility;
    rowwise.preload_audit.reserve(preloads.size());
    for (size_t p = 0; p < preloads.size(); ++p) {
      const WaveInterval& preload =
          intervals[preloads[p].instruction_index];
      RowwisePreloadAudit preload_audit;
      preload_audit.preload_index = p;
      preload_audit.instruction_index = preloads[p].instruction_index;
      preload_audit.weight_block_id = preload.weight_block_id;
      preload_audit.issue_cycle = preload.issue_cycle;
      preload_audit.start_cycle = preload.start_cycle;
      preload_audit.tile_m = preload.tile_m;
      preload_audit.tile_k = preload.tile_k;
      preload_audit.tile_n = preload.tile_n;
      const uint64_t lead =
          row_write_timing->first_row_lead_cycles_before_start;
      if (lead <= preload.start_cycle) {
        preload_audit.planned_load_begin_available = true;
        preload_audit.planned_load_begin_cycle =
            preload.start_cycle - lead;
      }
      preload_audit.rows.reserve(preload.tile_m);
      for (uint32_t row = 0; row < preload.tile_m; ++row) {
        RowwisePreloadRowAudit row_audit;
        row_audit.pe_row = row;
        row_audit.ordered_row_index =
            row_write_timing->row_order ==
                    RowWriteOrder::ASCENDING_PE_ROW
                ? row
                : preload.tile_m - 1 - row;
        row_audit.first_use_cycle = checked_add(
            preload.start_cycle, row,
            "rowwise preload earliest row use");
        const uint64_t start_plus_stagger = checked_add(
            preload.start_cycle,
            checked_multiply(
                row_audit.ordered_row_index,
                row_write_timing->row_start_period_cycles,
                "rowwise preload row stagger"),
            "rowwise preload row load begin before lead");
        if (lead <= start_plus_stagger) {
          row_audit.planned_load_begin_available = true;
          row_audit.planned_load_begin_cycle =
              start_plus_stagger - lead;
          row_audit.load_before_issue =
              row_audit.planned_load_begin_cycle < preload.issue_cycle;
          row_audit.load_begin_cycle = std::max(
              row_audit.planned_load_begin_cycle, preload.issue_cycle);
        } else {
          // The assumed schedule requests a negative cycle. The earliest
          // causal shadow load is issue; preserve the invalid plan as an
          // explicit conflict instead of unsigned wraparound.
          row_audit.load_before_issue = true;
          row_audit.load_begin_cycle = preload.issue_cycle;
        }
        row_audit.bank_ready_cycle = checked_add(
            row_audit.load_begin_cycle,
            row_write_timing->row_write_duration_cycles,
            "rowwise preload causal row bank ready");
        if (row_audit.bank_ready_cycle > row_audit.first_use_cycle) {
          row_audit.required_stall_cycles =
              row_audit.bank_ready_cycle - row_audit.first_use_cycle;
        }
        preload_audit.rows.push_back(row_audit);
      }
      rowwise.preload_audit.push_back(std::move(preload_audit));
    }
  }
  RowwiseStateAccounting rowwise_state_candidate;
  rowwise_state_candidate.capacity_pe_cycles = result.capacity_pe_cycles;
  for (uint32_t row = 0; row < array_rows; ++row) {
    for (uint32_t column = 0; column < array_columns; ++column) {
      std::vector<uint64_t> pe_last(preloads.size(), 0);
      for (size_t i = 0; i < intervals.size(); ++i) {
        const WaveInterval& interval = intervals[i];
        if (row >= interval.tile_m || column >= interval.tile_k) continue;
        const uint64_t first = checked_add(
            interval.start_cycle,
            static_cast<uint64_t>(row) + column,
            "timing audit PE first MAC");
        pe_last[owner[i]] = std::max(
            pe_last[owner[i]],
            checked_add(first, interval.tile_n,
                        "timing audit PE last MAC"));
      }
      std::vector<Span> candidate_spans[5];
      for (size_t p = 0; p < preloads.size(); ++p) {
        const WaveInterval& preload =
            intervals[preloads[p].instruction_index];
        if (row >= preload.tile_m || column >= preload.tile_k) continue;
        const uint64_t first = checked_add(
            preload.start_cycle,
            static_cast<uint64_t>(row) + column,
            "timing audit preload PE first MAC");
        const uint64_t whole = preloads[p].last_consumer_finish;
        const uint64_t local = pe_last[p];
        candidate_spans[0].emplace_back(preload.issue_cycle, whole);
        candidate_spans[1].emplace_back(preload.start_cycle, whole);
        candidate_spans[2].emplace_back(preload.issue_cycle, local);
        candidate_spans[3].emplace_back(preload.start_cycle, local);
        candidate_spans[4].emplace_back(first, local);
      }
      for (size_t s = 0; s < 5; ++s) {
        const OccupancySummary summary =
            sweep_preload_occupancy(candidate_spans[s]);
        TwoBankTimingCase& timing_case = *timing_cases[s];
        timing_case.mapped_pe_peak_live_preloads = std::max(
            timing_case.mapped_pe_peak_live_preloads, summary.peak);
        timing_case.mapped_pe_over_two_pe_cycles = checked_add(
            timing_case.mapped_pe_over_two_pe_cycles,
            summary.over_two_cycles,
            "timing audit mapped-PE over-two PE-cycles");
        if (summary.over_two_cycles != 0) {
          ++timing_case.mapped_pes_ever_over_two;
        }
      }
      if (row_write_timing) {
        RowwiseFeasibility& rowwise = result.rowwise_feasibility;
        rowwise.available = true;
        struct RowEvent {
          size_t preload_index;
          uint64_t write_start;
          uint64_t write_ready;
          uint64_t last_mac_end;
          uint64_t first_mac;
        };
        std::vector<RowEvent> events;
        std::vector<RowEvent> causal_events;
        std::vector<Span> reservations;
        bool row_events_complete = true;
        for (size_t p = 0; p < preloads.size(); ++p) {
          const WaveInterval& preload =
              intervals[preloads[p].instruction_index];
          if (row >= preload.tile_m || column >= preload.tile_k) continue;
          ++rowwise.mapped_pe_weight_write_events;
          const RowwisePreloadRowAudit& row_audit =
              rowwise.preload_audit[p].rows[row];
          const uint64_t first_mac = checked_add(
              row_audit.first_use_cycle, column,
              "row write first MAC");
          causal_events.push_back(RowEvent{
              p, row_audit.load_begin_cycle, row_audit.bank_ready_cycle,
              std::max(pe_last[p], row_audit.bank_ready_cycle), first_mac});
          if (!row_audit.planned_load_begin_available) {
            ++rowwise.row_write_before_issue_events;
            row_events_complete = false;
            continue;
          }
          const uint64_t write_start =
              row_audit.planned_load_begin_cycle;
          if (row_audit.load_before_issue) {
            ++rowwise.row_write_before_issue_events;
            row_events_complete = false;
            continue;
          }
          const uint64_t ready = row_audit.bank_ready_cycle;
          const uint64_t reservation_end = std::max(pe_last[p], ready);
          events.push_back(
              RowEvent{p, write_start, ready, reservation_end, first_mac});
          if (ready > first_mac) {
            ++rowwise.row_ready_after_first_mac_events;
            const uint64_t late = ready - first_mac;
            rowwise.max_row_ready_lateness_cycles = std::max(
                rowwise.max_row_ready_lateness_cycles, late);
            rowwise.max_local_first_mac_delay_cycles = std::max(
                rowwise.max_local_first_mac_delay_cycles, late);
          }
          reservations.emplace_back(write_start, reservation_end);
        }
        const OccupancySummary summary =
            sweep_preload_occupancy(reservations);
        rowwise.mapped_pe_peak_live_preloads = std::max(
            rowwise.mapped_pe_peak_live_preloads, summary.peak);
        rowwise.mapped_pe_over_two_pe_cycles = checked_add(
            rowwise.mapped_pe_over_two_pe_cycles,
            summary.over_two_cycles,
            "rowwise mapped-PE over-two PE-cycles");
        if (summary.over_two_cycles) ++rowwise.mapped_pes_ever_over_two;

        // Causally delay this PE-local stream when both logical banks are
        // occupied. This does not propagate through a shared row bus, solve
        // global bank indices, or modify the ONNXim execution trace.
        std::stable_sort(causal_events.begin(), causal_events.end(),
                         [](const RowEvent& a, const RowEvent& b) {
                           return a.write_start < b.write_start;
                         });
        std::vector<uint64_t> earlier_releases;
        for (const RowEvent& event : causal_events) {
          earlier_releases.erase(
              std::remove_if(earlier_releases.begin(),
                             earlier_releases.end(),
                             [&](uint64_t release) {
                               return release <= event.write_start;
                             }),
              earlier_releases.end());
          uint64_t causal_release = event.last_mac_end;
          if (earlier_releases.size() >= 2) {
            const auto release_to_replace =
                std::min_element(earlier_releases.begin(),
                                 earlier_releases.end());
            const uint64_t earliest_release = *release_to_replace;
            earlier_releases.erase(release_to_replace);
            ++rowwise.locally_blocked_pe_weight_write_events;
            rowwise.max_local_bank_wait_cycles = std::max(
                rowwise.max_local_bank_wait_cycles,
                earliest_release - event.write_start);
            const uint64_t delayed_ready = checked_add(
                earliest_release,
                row_write_timing->row_write_duration_cycles,
                "locally delayed row ready");
            causal_release = std::max(causal_release, delayed_ready);
            RowwisePreloadAudit& preload_audit =
                rowwise.preload_audit[event.preload_index];
            RowwisePreloadRowAudit& row_audit =
                preload_audit.rows[row];
            const uint64_t bank_wait =
                earliest_release - event.write_start;
            row_audit.max_local_bank_wait_cycles = std::max(
                row_audit.max_local_bank_wait_cycles, bank_wait);
            row_audit.load_begin_cycle = std::max(
                row_audit.load_begin_cycle, earliest_release);
            row_audit.bank_ready_cycle = checked_add(
                row_audit.load_begin_cycle,
                row_write_timing->row_write_duration_cycles,
                "causal two-bank row ready");
            row_audit.required_stall_cycles =
                row_audit.bank_ready_cycle > row_audit.first_use_cycle
                    ? row_audit.bank_ready_cycle -
                          row_audit.first_use_cycle
                    : 0;
            preload_audit.bank_conflicts.push_back(
                RowwiseBankConflictAudit{
                    event.preload_index, row, column,
                    row_audit.planned_load_begin_available
                        ? row_audit.planned_load_begin_cycle
                        : 0,
                    earliest_release, earliest_release, delayed_ready,
                    event.first_mac,
                    delayed_ready > event.first_mac
                        ? delayed_ready - event.first_mac
                        : 0});
            if (delayed_ready > event.first_mac) {
              rowwise.max_local_first_mac_delay_cycles = std::max(
                  rowwise.max_local_first_mac_delay_cycles,
                  delayed_ready - event.first_mac);
            }
          }
          if (event.write_start < causal_release) {
            earlier_releases.push_back(causal_release);
          }
        }

        // Candidate counts are published only after all row-wise timing and
        // partition checks pass. A write is separate exposure; without a MAC
        // wave its live bank remains in W_on rather than making the PE ON.
        if (!row_events_complete || summary.over_two_cycles != 0) continue;
        struct LocalBank {
          bool occupied = false;
          uint64_t residence_start = 0;
          uint64_t safe_replace_cycle = 0;
        };
        LocalBank local_banks[2];
        std::vector<Span> resident_spans[2];
        bool bank_assignment_pass = true;
        for (const RowEvent& event : events) {
          int selected = -1;
          for (int bank = 0; bank < 2; ++bank) {
            if (!local_banks[bank].occupied) {
              selected = bank;
              break;
            }
          }
          if (selected < 0) {
            for (int bank = 0; bank < 2; ++bank) {
              if (local_banks[bank].safe_replace_cycle <=
                  event.write_start) {
                selected = bank;
                break;
              }
            }
          }
          if (selected < 0) {
            bank_assignment_pass = false;
            break;
          }
          if (local_banks[selected].occupied) {
            resident_spans[selected].emplace_back(
                local_banks[selected].residence_start, event.write_start);
          }
          local_banks[selected] =
              LocalBank{true, event.write_start, event.last_mac_end};
        }
        if (!bank_assignment_pass) continue;
        for (int bank = 0; bank < 2; ++bank) {
          if (local_banks[bank].occupied) {
            // A last MAC makes a bank safe to replace; it does not erase the
            // weight. Residency ends only when that logical bank is actually
            // overwritten, otherwise it extends to the profile horizon.
            resident_spans[bank].emplace_back(
                local_banks[bank].residence_start, horizon_cycles);
          }
        }
        std::vector<Span> wave_spans;
        std::vector<Span> preload_wave_spans;
        std::vector<Span> gemm_wave_spans;
        for (const WaveInterval& interval : intervals) {
          if (row >= interval.tile_m || column >= interval.tile_k) continue;
          const uint64_t begin = checked_add(
              interval.start_cycle,
              static_cast<uint64_t>(row) + column,
              "rowwise PE wave begin");
          const Span wave{
              begin, checked_add(begin, interval.tile_n,
                                 "rowwise PE wave finish")};
          wave_spans.push_back(wave);
          (interval.opcode == WaveOpcode::PRELOAD
               ? preload_wave_spans
               : gemm_wave_spans).push_back(wave);
        }
        std::vector<Span> write_spans;
        write_spans.reserve(events.size());
        for (const RowEvent& event : events) {
          write_spans.emplace_back(event.write_start, event.write_ready);
        }
        const std::vector<Span> bank_zero =
            merge_spans(std::move(resident_spans[0]), horizon_cycles);
        const std::vector<Span> bank_one =
            merge_spans(std::move(resident_spans[1]), horizon_cycles);
        std::vector<Span> two_bank_spans;
        size_t bank_zero_index = 0;
        size_t bank_one_index = 0;
        while (bank_zero_index < bank_zero.size() &&
               bank_one_index < bank_one.size()) {
          const uint64_t begin = std::max(
              bank_zero[bank_zero_index].first,
              bank_one[bank_one_index].first);
          const uint64_t finish = std::min(
              bank_zero[bank_zero_index].second,
              bank_one[bank_one_index].second);
          if (begin < finish) two_bank_spans.emplace_back(begin, finish);
          if (bank_zero[bank_zero_index].second <
              bank_one[bank_one_index].second) {
            ++bank_zero_index;
          } else {
            ++bank_one_index;
          }
        }
        const std::vector<Span> waves =
            merge_spans(std::move(wave_spans), horizon_cycles);
        const std::vector<Span> preload_waves =
            merge_spans(std::move(preload_wave_spans), horizon_cycles);
        const std::vector<Span> gemm_waves =
            merge_spans(std::move(gemm_wave_spans), horizon_cycles);
        std::vector<Span> live_input = bank_zero;
        live_input.insert(live_input.end(), bank_one.begin(), bank_one.end());
        const std::vector<Span> live =
            merge_spans(std::move(live_input), horizon_cycles);
        const std::vector<Span> live_two =
            merge_spans(std::move(two_bank_spans), horizon_cycles);
        const std::vector<Span> writes =
            merge_spans(std::move(write_spans), horizon_cycles);
        const uint64_t on = span_cycles(waves);
        const uint64_t any_bank = span_cycles(live);
        const uint64_t two_banks = span_cycles(live_two);
        const uint64_t on_any = overlap_cycles(waves, live);
        const uint64_t on_two = overlap_cycles(waves, live_two);
        if (two_banks > any_bank || on_two > on_any) continue;
        const uint64_t on_one = on_any - on_two;
        const uint64_t on_zero = on - on_any;
        const uint64_t one_bank = any_bank - two_banks;
        if (on_one > one_bank) continue;
        const uint64_t w_two = two_banks - on_two;
        const uint64_t w_one = one_bank - on_one;
        const uint64_t classified = on + w_one + w_two;
        if (classified > horizon_cycles) continue;
        rowwise_state_candidate.on_pe_cycles = checked_add(
            rowwise_state_candidate.on_pe_cycles, on,
            "rowwise ON PE-cycles");
        rowwise_state_candidate.on_preload_wave_pe_cycles = checked_add(
            rowwise_state_candidate.on_preload_wave_pe_cycles,
            span_cycles(preload_waves),
            "rowwise preload-wave ON PE-cycles");
        rowwise_state_candidate.on_gemm_wave_pe_cycles = checked_add(
            rowwise_state_candidate.on_gemm_wave_pe_cycles,
            span_cycles(gemm_waves),
            "rowwise GEMM-wave ON PE-cycles");
        rowwise_state_candidate.on_zero_bank_pe_cycles = checked_add(
            rowwise_state_candidate.on_zero_bank_pe_cycles, on_zero,
            "rowwise ON zero-bank PE-cycles");
        rowwise_state_candidate.on_one_bank_pe_cycles = checked_add(
            rowwise_state_candidate.on_one_bank_pe_cycles, on_one,
            "rowwise ON one-bank PE-cycles");
        rowwise_state_candidate.on_two_bank_pe_cycles = checked_add(
            rowwise_state_candidate.on_two_bank_pe_cycles, on_two,
            "rowwise ON two-bank PE-cycles");
        rowwise_state_candidate.w_on_one_bank_pe_cycles = checked_add(
            rowwise_state_candidate.w_on_one_bank_pe_cycles, w_one,
            "rowwise W_on one-bank PE-cycles");
        rowwise_state_candidate.w_on_two_bank_pe_cycles = checked_add(
            rowwise_state_candidate.w_on_two_bank_pe_cycles, w_two,
            "rowwise W_on two-bank PE-cycles");
        rowwise_state_candidate.off_pe_cycles = checked_add(
            rowwise_state_candidate.off_pe_cycles,
            horizon_cycles - classified, "rowwise OFF PE-cycles");
        rowwise_state_candidate.preload_write_pe_cycles = checked_add(
            rowwise_state_candidate.preload_write_pe_cycles,
            span_cycles(writes), "rowwise preload-write PE-cycles");
        rowwise_state_candidate.preload_write_while_on_pe_cycles =
            checked_add(
                rowwise_state_candidate.preload_write_while_on_pe_cycles,
                overlap_cycles(writes, waves),
                "rowwise preload-write while-ON PE-cycles");
      }
    }
  }

  if (row_write_timing) {
    RowwiseFeasibility& rowwise = result.rowwise_feasibility;
    for (RowwisePreloadAudit& preload_audit : rowwise.preload_audit) {
      bool first_audited_row = true;
      for (RowwisePreloadRowAudit& row_audit : preload_audit.rows) {
        if (first_audited_row) {
          preload_audit.load_begin_cycle = row_audit.load_begin_cycle;
          first_audited_row = false;
        } else {
          preload_audit.load_begin_cycle = std::min(
              preload_audit.load_begin_cycle,
              row_audit.load_begin_cycle);
        }
        row_audit.no_added_cycle =
            row_audit.required_stall_cycles == 0;
        if (!row_audit.no_added_cycle) {
          ++preload_audit.conflicting_rows;
        }
        if (row_audit.load_before_issue ||
            row_audit.max_local_bank_wait_cycles != 0) {
          ++preload_audit.schedule_diagnostic_rows;
        }
        preload_audit.sum_row_required_stall_cycles = checked_add(
            preload_audit.sum_row_required_stall_cycles,
            row_audit.required_stall_cycles,
            "preload sum of row required stalls");
        preload_audit.required_stall_cycles = std::max(
            preload_audit.required_stall_cycles,
            row_audit.required_stall_cycles);
      }
      preload_audit.no_added_cycle =
          preload_audit.required_stall_cycles == 0;
      ++rowwise.audited_preloads;
      rowwise.audited_preload_rows = checked_add(
          rowwise.audited_preload_rows, preload_audit.rows.size(),
          "audited preload rows");
      if (preload_audit.no_added_cycle) {
        ++rowwise.no_added_cycle_preloads;
      } else {
        ++rowwise.conflicting_preloads;
      }
      rowwise.conflicting_preload_rows = checked_add(
          rowwise.conflicting_preload_rows,
          preload_audit.conflicting_rows,
          "conflicting preload rows");
      if (preload_audit.schedule_diagnostic_rows != 0) {
        ++rowwise.schedule_diagnostic_preloads;
      }
      rowwise.schedule_diagnostic_rows = checked_add(
          rowwise.schedule_diagnostic_rows,
          preload_audit.schedule_diagnostic_rows,
          "preload rows with schedule diagnostics");
      rowwise.sum_row_required_stall_cycles = checked_add(
          rowwise.sum_row_required_stall_cycles,
          preload_audit.sum_row_required_stall_cycles,
          "sum of all preload-row required stalls");
      // Rows of one preload would delay that instruction together. The
      // preload delay is therefore the maximum row delay, not their sum.
      rowwise.total_required_stall_cycles = checked_add(
          rowwise.total_required_stall_cycles,
          preload_audit.required_stall_cycles,
          "total of per-preload required stalls");
      rowwise.max_required_stall_cycles = std::max(
          rowwise.max_required_stall_cycles,
          preload_audit.required_stall_cycles);
    }
    // With an explicit timing hypothesis, an empty SA trace is a vacuous,
    // exact all-OFF partition. A nonempty GEMM-only trace returns earlier on
    // the missing-preload identity check and therefore remains unavailable.
    rowwise.available = !preloads.empty() || intervals.empty();
    rowwise.secondary_counts_complete = rowwise.available &&
        rowwise.row_write_before_issue_events == 0;
    rowwise.necessary_conditions_pass =
        rowwise.secondary_counts_complete &&
        rowwise.row_ready_after_first_mac_events == 0 &&
        rowwise.mapped_pe_over_two_pe_cycles == 0;

    RowwiseStateAccounting& state = result.rowwise_state_accounting;
    state.available = rowwise.available;
    const bool partition_pass =
        rowwise_state_candidate.on_pe_cycles ==
            rowwise_state_candidate.on_preload_wave_pe_cycles +
                rowwise_state_candidate.on_gemm_wave_pe_cycles &&
        rowwise_state_candidate.on_pe_cycles ==
            rowwise_state_candidate.on_zero_bank_pe_cycles +
                rowwise_state_candidate.on_one_bank_pe_cycles +
                rowwise_state_candidate.on_two_bank_pe_cycles &&
        rowwise_state_candidate.on_zero_bank_pe_cycles == 0 &&
        rowwise_state_candidate.on_pe_cycles +
                rowwise_state_candidate.w_on_one_bank_pe_cycles +
                rowwise_state_candidate.w_on_two_bank_pe_cycles +
                rowwise_state_candidate.off_pe_cycles ==
            rowwise_state_candidate.capacity_pe_cycles &&
        rowwise_state_candidate.preload_write_while_on_pe_cycles <=
            rowwise_state_candidate.preload_write_pe_cycles;
    state.partition_check_pass =
        rowwise.necessary_conditions_pass && partition_pass;
    state.validation_pass = state.available &&
        rowwise.necessary_conditions_pass && state.partition_check_pass;
    if (state.validation_pass) {
      state = rowwise_state_candidate;
      state.available = true;
      state.validation_pass = true;
      state.partition_check_pass = true;
      state.unclassified_pe_cycles = 0;
    }
  }

  struct BankSlot {
    bool occupied = false;
    size_t owner = 0;
    uint64_t release_cycle = 0;
  };
  BankSlot banks[2];
  for (size_t p = 0; p < preloads.size(); ++p) {
    Preload& preload = preloads[p];
    const WaveInterval& interval =
        intervals[preload.instruction_index];
    int selected = -1;
    for (int bank = 0; bank < 2; ++bank) {
      if (!banks[bank].occupied) {
        selected = bank;
        break;
      }
    }
    if (selected < 0) {
      for (int bank = 0; bank < 2; ++bank) {
        if (banks[bank].release_cycle <= interval.issue_cycle) {
          selected = bank;
          break;
        }
      }
    }
    if (selected < 0) {
      ++result.two_bank_conflicts;
      continue;
    }
    if (banks[selected].occupied) {
      preloads[banks[selected].owner].residency_finish =
          interval.issue_cycle;
    }
    preload.bank = selected;
    banks[selected] = BankSlot{true, p,
                              preload.last_consumer_finish};
    result.preload_bank_pe_cycles = checked_add(
        result.preload_bank_pe_cycles,
        checked_multiply(
            checked_multiply(interval.tile_m, interval.tile_k,
                             "preload footprint"),
            interval.start_cycle - interval.issue_cycle,
            "preload bank PE-cycles"),
        "preload bank PE-cycles");
  }
  if (result.two_bank_conflicts) {
    return fail_closed();
  }
  for (size_t i = 0; i < intervals.size(); ++i) {
    const Preload& preload = preloads[owner[i]];
    if (preload.bank < 0 ||
        intervals[i].finish_cycle > preload.residency_finish) {
      ++result.weight_evicted_before_use_intervals;
    }
  }
  if (result.weight_evicted_before_use_intervals) {
    return fail_closed();
  }

  // Sweep half-open spans per PE. This scales with instruction count and
  // mapped PE count, not with the number of simulated clock cycles.
  auto intersect_spans = [](const std::vector<Span>& a,
                            const std::vector<Span>& b) {
    std::vector<Span> common;
    size_t i = 0;
    size_t j = 0;
    while (i < a.size() && j < b.size()) {
      const uint64_t begin = std::max(a[i].first, b[j].first);
      const uint64_t finish = std::min(a[i].second, b[j].second);
      if (begin < finish) common.emplace_back(begin, finish);
      if (a[i].second < b[j].second) {
        ++i;
      } else {
        ++j;
      }
    }
    return common;
  };
  for (uint32_t row = 0; row < array_rows; ++row) {
    for (uint32_t column = 0; column < array_columns; ++column) {
      std::vector<Span> wave_spans;
      std::vector<Span> bank_spans[2];
      for (const Preload& preload : preloads) {
        const WaveInterval& interval =
            intervals[preload.instruction_index];
        if (row < interval.tile_m && column < interval.tile_k &&
            preload.residency_finish > interval.issue_cycle) {
          bank_spans[preload.bank].emplace_back(
              interval.issue_cycle, preload.residency_finish);
        }
      }
      for (const WaveInterval& interval : intervals) {
        if (row >= interval.tile_m || column >= interval.tile_k) {
          continue;
        }
        const uint64_t begin = checked_add(
            interval.start_cycle,
            static_cast<uint64_t>(row) + column, "PE wave begin");
        wave_spans.emplace_back(
            begin, checked_add(begin, interval.tile_n,
                               "PE wave finish"));
      }
      const std::vector<Span> waves =
          merge_spans(std::move(wave_spans), horizon_cycles);
      const std::vector<Span> b0 =
          merge_spans(std::move(bank_spans[0]), horizon_cycles);
      const std::vector<Span> b1 =
          merge_spans(std::move(bank_spans[1]), horizon_cycles);
      std::vector<Span> either_input = b0;
      either_input.insert(either_input.end(), b1.begin(), b1.end());
      const std::vector<Span> either =
          merge_spans(std::move(either_input), horizon_cycles);
      const std::vector<Span> both = intersect_spans(b0, b1);
      const uint64_t on = span_cycles(waves);
      const uint64_t any_bank = span_cycles(either);
      const uint64_t two_banks = span_cycles(both);
      const uint64_t on_any = overlap_cycles(waves, either);
      const uint64_t on_two = overlap_cycles(waves, both);
      const uint64_t on_one = on_any - on_two;
      const uint64_t on_zero = on - on_any;
      const uint64_t one_bank = any_bank - two_banks;
      const uint64_t w_two = two_banks - on_two;
      const uint64_t w_one = one_bank - on_one;
      const uint64_t classified = on + w_one + w_two;
      if (classified > horizon_cycles) {
        return fail_closed();
      }
      result.on_pe_cycles =
          checked_add(result.on_pe_cycles, on, "ON PE-cycles");
      result.on_zero_bank_pe_cycles = checked_add(
          result.on_zero_bank_pe_cycles, on_zero, "ON no-bank PE-cycles");
      result.on_one_bank_pe_cycles = checked_add(
          result.on_one_bank_pe_cycles, on_one, "ON one-bank PE-cycles");
      result.on_two_bank_pe_cycles = checked_add(
          result.on_two_bank_pe_cycles, on_two, "ON two-bank PE-cycles");
      result.w_on_one_bank_pe_cycles = checked_add(
          result.w_on_one_bank_pe_cycles, w_one, "W_on one-bank PE-cycles");
      result.w_on_two_bank_pe_cycles = checked_add(
          result.w_on_two_bank_pe_cycles, w_two, "W_on two-bank PE-cycles");
      result.off_pe_cycles = checked_add(
          result.off_pe_cycles, horizon_cycles - classified,
          "OFF PE-cycles");
    }
  }

  result.validation_pass =
      result.on_zero_bank_pe_cycles == 0 &&
      result.on_pe_cycles + result.w_on_one_bank_pe_cycles +
          result.w_on_two_bank_pe_cycles + result.off_pe_cycles ==
          result.capacity_pe_cycles;
  if (!result.validation_pass) {
    return fail_closed();
  }
  return result;
}

UpdatedTwoBankShadowResult derive_updated_two_bank_shadow(
    const std::vector<UpdatedShadowInterval>& input_intervals,
    uint32_t array_rows,
    uint32_t array_columns,
    uint64_t simulator_horizon_cycles) {
  if (array_rows == 0 || array_columns == 0) {
    throw std::invalid_argument("SA dimensions must be nonzero");
  }

  UpdatedTwoBankShadowResult result;
  std::vector<std::pair<size_t, UpdatedShadowInterval>> ordered;
  ordered.reserve(input_intervals.size());
  for (size_t i = 0; i < input_intervals.size(); ++i) {
    ordered.emplace_back(i, input_intervals[i]);
  }
  std::stable_sort(
      ordered.begin(), ordered.end(),
      [](const auto& left, const auto& right) {
        return left.second.issue_cycle < right.second.issue_cycle;
      });

  struct PreloadInstance {
    size_t event_index = 0;
    size_t last_event_index = 0;
    uint64_t instance_id = 0;
    int bank = -1;
    uint64_t load_start = 0;
    uint64_t release = 0;
    uint32_t tile_m = 0;
    uint32_t tile_k = 0;
  };
  std::vector<PreloadInstance> preloads;
  std::vector<size_t> owner(ordered.size(),
                            std::numeric_limits<size_t>::max());
  std::map<uint64_t, size_t> current_owner;
  std::map<uint64_t, uint64_t> current_sequence;
  for (size_t event_index = 0; event_index < ordered.size(); ++event_index) {
    const UpdatedShadowInterval& interval = ordered[event_index].second;
    if (interval.issue_cycle > interval.simulator_start_cycle ||
        interval.simulator_start_cycle > interval.simulator_finish_cycle ||
        interval.simulator_finish_cycle > simulator_horizon_cycles ||
        interval.tile_m == 0 || interval.tile_k == 0 ||
        interval.tile_n == 0 || interval.tile_m > array_rows ||
        interval.tile_k > array_columns) {
      ++result.invalid_intervals;
    }
    if (interval.opcode == WaveOpcode::PRELOAD) {
      ++result.preload_events;
      const uint64_t expected = current_sequence[interval.stream_id] + 1;
      if (interval.preload_sequence != expected) {
        ++result.sequence_errors;
      }
      current_sequence[interval.stream_id] = interval.preload_sequence;
      const size_t preload_index = preloads.size();
      preloads.push_back(PreloadInstance{
          event_index, event_index, preload_index + 1, -1, 0, 0,
          interval.tile_m, interval.tile_k});
      current_owner[interval.stream_id] = preload_index;
      owner[event_index] = preload_index;
    } else {
      ++result.gemm_events;
      const auto found = current_owner.find(interval.stream_id);
      if (found == current_owner.end() || interval.preload_sequence == 0 ||
          current_sequence[interval.stream_id] !=
              interval.preload_sequence) {
        ++result.sequence_errors;
        continue;
      }
      owner[event_index] = found->second;
      if (interval.tile_m != preloads[found->second].tile_m ||
          interval.tile_k != preloads[found->second].tile_k) {
        ++result.footprint_mismatch_intervals;
      }
      preloads[found->second].last_event_index = event_index;
    }
  }
  result.preload_first_valid = result.sequence_errors == 0;

  // Check logical two-bank colorability before assigning a schedule. A copy
  // remains protected through its last same-stream consumer.
  std::array<std::optional<size_t>, 2> live{};
  for (size_t p = 0; p < preloads.size(); ++p) {
    for (auto& bank_owner : live) {
      if (bank_owner &&
          preloads[*bank_owner].last_event_index <
              preloads[p].event_index) {
        bank_owner.reset();
      }
    }
    int selected = -1;
    for (int bank = 0; bank < 2; ++bank) {
      if (!live[bank]) {
        selected = bank;
        break;
      }
    }
    if (selected < 0) {
      ++result.bank_assignment_conflicts;
      continue;
    }
    live[selected] = p;
  }
  result.bank_assignment_valid = result.bank_assignment_conflicts == 0;

  const uint64_t array_area = checked_multiply(
      array_rows, array_columns, "updated shadow SA area");
  auto fail_closed = [&]() {
    result.analytical_horizon_cycles = simulator_horizon_cycles;
    result.capacity_pe_cycles = checked_multiply(
        array_area, result.analytical_horizon_cycles,
        "updated shadow capacity");
    result.on_pe_cycles = 0;
    result.on_zero_bank_pe_cycles = 0;
    result.on_one_bank_pe_cycles = 0;
    result.on_two_bank_pe_cycles = 0;
    result.w_on_pe_cycles = 0;
    result.w_on_one_bank_pe_cycles = 0;
    result.w_on_two_bank_pe_cycles = 0;
    result.loading_pe_cycles = 0;
    result.preload_write_pe_cycles = 0;
    result.preload_write_while_on_pe_cycles = 0;
    result.off_pe_cycles = 0;
    result.unclassified_pe_cycles = result.capacity_pe_cycles;
    result.state_partition_exact = false;
    result.validation_pass = false;
    return result;
  };
  if (result.invalid_intervals || result.sequence_errors ||
      result.footprint_mismatch_intervals ||
      result.bank_assignment_conflicts) {
    return fail_closed();
  }

  uint64_t compute_lane_ready = 0;
  uint64_t load_lane_ready = 0;
  std::array<uint64_t, 2> bank_replace_ready{};
  live = {};
  result.events.reserve(ordered.size());
  for (size_t event_index = 0; event_index < ordered.size(); ++event_index) {
    const auto& [input_index, interval] = ordered[event_index];
    const size_t preload_index = owner[event_index];
    if (preload_index == std::numeric_limits<size_t>::max()) {
      return fail_closed();
    }
    PreloadInstance& preload = preloads[preload_index];
    UpdatedShadowEvent event;
    event.input_index = input_index;
    event.stream_id = interval.stream_id;
    event.preload_instance = preload.instance_id;
    event.opcode = interval.opcode;
    event.issue_cycle = interval.issue_cycle;
    event.simulator_start_cycle = interval.simulator_start_cycle;
    event.simulator_finish_cycle = interval.simulator_finish_cycle;

    if (interval.opcode == WaveOpcode::PRELOAD) {
      for (auto& bank_owner : live) {
        if (bank_owner &&
            preloads[*bank_owner].last_event_index < event_index) {
          bank_owner.reset();
        }
      }
      int selected = -1;
      uint64_t earliest_load_start =
          std::numeric_limits<uint64_t>::max();
      for (int bank = 0; bank < 2; ++bank) {
        if (live[bank]) continue;
        const uint64_t candidate = std::max(
            {interval.issue_cycle, load_lane_ready,
             bank_replace_ready[bank]});
        if (candidate < earliest_load_start) {
          earliest_load_start = candidate;
          selected = bank;
        }
      }
      if (selected < 0) {
        ++result.bank_assignment_conflicts;
        result.bank_assignment_valid = false;
        return fail_closed();
      }
      preload.bank = selected;
      live[selected] = preload_index;
    }
    event.bank = preload.bank;
    uint64_t weight_ready = checked_add(
        preload.load_start, interval.tile_m,
        "updated shadow resident weight ready");
    if (interval.opcode == WaveOpcode::PRELOAD) {
      event.loading_start_cycle = std::max(
          {interval.issue_cycle, load_lane_ready,
           bank_replace_ready[event.bank]});
      event.loading_finish_cycle = checked_add(
          event.loading_start_cycle, interval.tile_m,
          "updated shadow row preload finish");
      preload.load_start = event.loading_start_cycle;
      weight_ready = event.loading_finish_cycle;
      load_lane_ready = event.loading_finish_cycle;
    }
    event.compute_start_cycle = std::max(
        {interval.issue_cycle, compute_lane_ready, weight_ready});
    const uint64_t wave_cycles = checked_add(
        checked_add(interval.tile_m, interval.tile_k,
                    "updated shadow wave duration"),
        interval.tile_n, "updated shadow wave duration") - 2;
    event.compute_finish_cycle = checked_add(
        event.compute_start_cycle, wave_cycles,
        "updated shadow compute finish");
    compute_lane_ready = event.compute_finish_cycle;
    if (event.compute_start_cycle > interval.simulator_start_cycle) {
      event.start_lateness_cycles =
          event.compute_start_cycle - interval.simulator_start_cycle;
    }
    if (event.compute_finish_cycle > interval.simulator_finish_cycle) {
      event.finish_lateness_cycles =
          event.compute_finish_cycle - interval.simulator_finish_cycle;
    }
    result.sum_event_start_lateness_cycles = checked_add(
        result.sum_event_start_lateness_cycles, event.start_lateness_cycles,
        "updated shadow sum of event start lateness");
    result.max_event_start_lateness_cycles = std::max(
        result.max_event_start_lateness_cycles, event.start_lateness_cycles);
    result.sum_event_finish_lateness_cycles = checked_add(
        result.sum_event_finish_lateness_cycles, event.finish_lateness_cycles,
        "updated shadow sum of event finish lateness");
    result.max_event_finish_lateness_cycles = std::max(
        result.max_event_finish_lateness_cycles, event.finish_lateness_cycles);
    result.simulator_trace_makespan_cycles = std::max(
        result.simulator_trace_makespan_cycles,
        interval.simulator_finish_cycle);
    result.analytical_trace_makespan_cycles = std::max(
        result.analytical_trace_makespan_cycles,
        std::max(event.compute_finish_cycle, event.loading_finish_cycle));
    if (event_index == preload.last_event_index) {
      preload.release = event.compute_finish_cycle;
      bank_replace_ready[event.bank] = event.compute_finish_cycle;
    }
    result.events.push_back(event);
  }

  if (result.analytical_trace_makespan_cycles >
      result.simulator_trace_makespan_cycles) {
    result.trace_makespan_extension_cycles =
        result.analytical_trace_makespan_cycles -
        result.simulator_trace_makespan_cycles;
  }

  result.analytical_horizon_cycles = std::max(
      simulator_horizon_cycles,
      std::max(compute_lane_ready, load_lane_ready));
  result.capacity_pe_cycles = checked_multiply(
      array_area, result.analytical_horizon_cycles,
      "updated shadow capacity");

  for (uint32_t row = 0; row < array_rows; ++row) {
    for (uint32_t column = 0; column < array_columns; ++column) {
      std::vector<Span> on_input;
      std::vector<Span> loading_input;
      std::vector<Span> resident_input[2];
      for (size_t event_index = 0; event_index < ordered.size();
           ++event_index) {
        const UpdatedShadowInterval& interval =
            ordered[event_index].second;
        if (row >= interval.tile_m || column >= interval.tile_k) {
          continue;
        }
        const UpdatedShadowEvent& event = result.events[event_index];
        const uint64_t on_begin = checked_add(
            event.compute_start_cycle,
            static_cast<uint64_t>(row) + column,
            "updated shadow PE wave begin");
        on_input.emplace_back(
            on_begin,
            checked_add(on_begin, interval.tile_n,
                        "updated shadow PE wave finish"));
      }
      for (const PreloadInstance& preload : preloads) {
        const UpdatedShadowInterval& interval =
            ordered[preload.event_index].second;
        if (row >= interval.tile_m || column >= interval.tile_k) {
          continue;
        }
        const uint64_t load_begin = checked_add(
            preload.load_start, row,
            "updated shadow PE row load begin");
        const uint64_t load_finish = checked_add(
            load_begin, 1, "updated shadow PE row load finish");
        loading_input.emplace_back(load_begin, load_finish);
        if (preload.release > load_finish) {
          resident_input[preload.bank].emplace_back(
              load_finish, preload.release);
        }
      }
      const std::vector<Span> on = merge_spans(
          std::move(on_input), result.analytical_horizon_cycles);
      const std::vector<Span> loading = merge_spans(
          std::move(loading_input), result.analytical_horizon_cycles);
      const std::vector<Span> bank_zero = merge_spans(
          std::move(resident_input[0]), result.analytical_horizon_cycles);
      const std::vector<Span> bank_one = merge_spans(
          std::move(resident_input[1]), result.analytical_horizon_cycles);
      std::vector<Span> resident_input_union = bank_zero;
      resident_input_union.insert(resident_input_union.end(),
                                  bank_one.begin(), bank_one.end());
      const std::vector<Span> resident = merge_spans(
          std::move(resident_input_union),
          result.analytical_horizon_cycles);
      const std::vector<Span> both_banks =
          intersect_merged_spans(bank_zero, bank_one);
      std::vector<Span> higher_priority_input = on;
      higher_priority_input.insert(higher_priority_input.end(),
                                   loading.begin(), loading.end());
      const std::vector<Span> higher_priority = merge_spans(
          std::move(higher_priority_input),
          result.analytical_horizon_cycles);
      const uint64_t on_cycles = span_cycles(on);
      const uint64_t raw_loading_cycles = span_cycles(loading);
      const uint64_t loading_while_on = overlap_cycles(loading, on);
      const uint64_t loading_cycles =
          raw_loading_cycles - loading_while_on;
      const uint64_t resident_cycles = span_cycles(resident);
      const uint64_t two_bank_cycles = span_cycles(both_banks);
      const uint64_t w_on_cycles = resident_cycles -
          overlap_cycles(resident, higher_priority);
      const uint64_t w_on_two_bank_cycles = two_bank_cycles -
          overlap_cycles(both_banks, higher_priority);
      const uint64_t w_on_one_bank_cycles =
          w_on_cycles - w_on_two_bank_cycles;
      const uint64_t on_any_bank = overlap_cycles(on, resident);
      const uint64_t on_two_banks = overlap_cycles(on, both_banks);
      const uint64_t on_one_bank = on_any_bank - on_two_banks;
      const uint64_t on_zero_banks = on_cycles - on_any_bank;
      const uint64_t classified = checked_add(
          checked_add(on_cycles, loading_cycles,
                      "updated shadow classified PE cycles"),
          w_on_cycles, "updated shadow classified PE cycles");
      if (classified > result.analytical_horizon_cycles) {
        return fail_closed();
      }
      result.on_pe_cycles = checked_add(
          result.on_pe_cycles, on_cycles,
          "updated shadow ON PE cycles");
      result.on_zero_bank_pe_cycles = checked_add(
          result.on_zero_bank_pe_cycles, on_zero_banks,
          "updated shadow ON zero-bank PE cycles");
      result.on_one_bank_pe_cycles = checked_add(
          result.on_one_bank_pe_cycles, on_one_bank,
          "updated shadow ON one-bank PE cycles");
      result.on_two_bank_pe_cycles = checked_add(
          result.on_two_bank_pe_cycles, on_two_banks,
          "updated shadow ON two-bank PE cycles");
      result.loading_pe_cycles = checked_add(
          result.loading_pe_cycles, loading_cycles,
          "updated shadow LOADING PE cycles");
      result.w_on_pe_cycles = checked_add(
          result.w_on_pe_cycles, w_on_cycles,
          "updated shadow W_on PE cycles");
      result.w_on_one_bank_pe_cycles = checked_add(
          result.w_on_one_bank_pe_cycles, w_on_one_bank_cycles,
          "updated shadow W_on one-bank PE cycles");
      result.w_on_two_bank_pe_cycles = checked_add(
          result.w_on_two_bank_pe_cycles, w_on_two_bank_cycles,
          "updated shadow W_on two-bank PE cycles");
      result.preload_write_pe_cycles = checked_add(
          result.preload_write_pe_cycles, raw_loading_cycles,
          "updated shadow preload-write PE cycles");
      result.preload_write_while_on_pe_cycles = checked_add(
          result.preload_write_while_on_pe_cycles, loading_while_on,
          "updated shadow preload-write while-ON PE cycles");
      result.off_pe_cycles = checked_add(
          result.off_pe_cycles,
          result.analytical_horizon_cycles - classified,
          "updated shadow OFF PE cycles");
    }
  }
  result.state_partition_exact =
      result.on_pe_cycles + result.w_on_pe_cycles +
          result.loading_pe_cycles + result.off_pe_cycles ==
          result.capacity_pe_cycles &&
      result.on_zero_bank_pe_cycles + result.on_one_bank_pe_cycles +
              result.on_two_bank_pe_cycles ==
          result.on_pe_cycles &&
      result.w_on_one_bank_pe_cycles +
              result.w_on_two_bank_pe_cycles ==
          result.w_on_pe_cycles &&
      result.preload_write_while_on_pe_cycles <=
          result.preload_write_pe_cycles;
  result.validation_pass = result.preload_first_valid &&
      result.bank_assignment_valid && result.state_partition_exact;
  if (!result.validation_pass) {
    return fail_closed();
  }
  return result;
}
}  // namespace sa_pe_state
