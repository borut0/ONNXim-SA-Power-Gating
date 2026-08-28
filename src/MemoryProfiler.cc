#include "MemoryProfiler.h"
#include "Common.h"

#include <spdlog/spdlog.h>
#include <cstdlib>
#include <string>

namespace {

bool environment_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' &&
           std::string(value) != "0";
}

}  // namespace

MemoryProfiler::MemoryProfiler() {
    _detailed = environment_enabled(
        "ONNXSIM_MEMORY_STAGE_PROFILE_DETAILED");
    _enabled = _detailed || environment_enabled(
        "ONNXSIM_COMPONENT_CHARACTERIZATION");
}

MemoryProfiler& MemoryProfiler::instance() {
    static MemoryProfiler profiler;
    return profiler;
}

void MemoryProfiler::finalize(MemoryAccess* access)
{
    if (!_enabled)
        return;

    _num_requests++;

    _core_queue_cycles += access->core_queue_cycles;
    _icnt_to_mem_cycles += access->icnt_to_mem_cycles;
    _dram_cycles += access->dram_cycles;
    _icnt_to_core_cycles += access->icnt_to_core_cycles;

    const std::array<cycle_type, 4> samples = {
        access->core_queue_cycles,
        access->icnt_to_mem_cycles,
        access->dram_cycles,
        access->icnt_to_core_cycles};
    for (size_t index = 0; index < samples.size(); ++index) {
        if (!_have_sample[index]) {
            _minimum_cycles[index] = samples[index];
            _maximum_cycles[index] = samples[index];
            _have_sample[index] = true;
        } else {
            _minimum_cycles[index] =
                std::min(_minimum_cycles[index], samples[index]);
            _maximum_cycles[index] =
                std::max(_maximum_cycles[index], samples[index]);
        }
    }

    if (!_detailed)
        return;

    auto update_histogram =
        [](std::vector<uint64_t>& hist, cycle_type latency)
        {
            if (latency >= hist.size())
                hist.resize(latency + 1, 0);

            hist[latency]++;
        };

    update_histogram(_core_queue_histogram,
                     access->core_queue_cycles);

    update_histogram(_icnt_to_mem_histogram,
                     access->icnt_to_mem_cycles);

    update_histogram(_dram_histogram,
                     access->dram_cycles);

    update_histogram(_icnt_to_core_histogram,
                     access->icnt_to_core_cycles);
}

struct StageStats {
    uint64_t min;
    uint64_t max;
    double avg;
    uint64_t p50;
    uint64_t p95;
    uint64_t p99;
};

static StageStats compute_stats(
    const std::vector<uint64_t>& histogram,
    uint64_t total_cycles)
{
    StageStats stats{};

    uint64_t total_samples = 0;
    bool first = true;

    for (size_t latency = 0; latency < histogram.size(); latency++)
    {
        uint64_t count = histogram[latency];

        if (count == 0)
            continue;

        total_samples += count;

        if (first)
        {
            stats.min = latency;
            first = false;
        }

        stats.max = latency;
    }

    if (total_samples == 0)
        return stats;

    stats.avg =
        static_cast<double>(total_cycles) /
        static_cast<double>(total_samples);

    // Percentile targets (same behavior as your previous code)
    uint64_t p50_target =
        std::max<uint64_t>(1, static_cast<uint64_t>(
            std::ceil(0.50 * total_samples)));

    uint64_t p95_target =
        std::max<uint64_t>(1, static_cast<uint64_t>(
            std::ceil(0.95 * total_samples)));

    uint64_t p99_target =
        std::max<uint64_t>(1, static_cast<uint64_t>(
            std::ceil(0.99 * total_samples)));

    uint64_t cumulative = 0;

    bool found50 = false;
    bool found95 = false;
    bool found99 = false;

    for (size_t latency = 0; latency < histogram.size(); latency++)
    {
        cumulative += histogram[latency];

        if (!found50 && cumulative >= p50_target)
        {
            stats.p50 = latency;
            found50 = true;
        }

        if (!found95 && cumulative >= p95_target)
        {
            stats.p95 = latency;
            found95 = true;
        }

        if (!found99 && cumulative >= p99_target)
        {
            stats.p99 = latency;
            found99 = true;
        }

        if (found50 && found95 && found99)
            break;
    }

    return stats;
}

void MemoryProfiler::print_stats()
{
    if (!_enabled || _num_requests == 0)
        return;

    const std::array<const char*, 4> names = {
        "core_queue", "icnt_to_memory", "dram", "icnt_to_core"};
    const std::array<uint64_t, 4> totals = {
        _core_queue_cycles, _icnt_to_mem_cycles,
        _dram_cycles, _icnt_to_core_cycles};
    json stages = json::object();
    for (size_t index = 0; index < names.size(); ++index) {
        stages[names[index]] =
            {{"minimum_cycles", _minimum_cycles[index]},
             {"maximum_cycles", _maximum_cycles[index]},
             {"average_cycles",
              static_cast<double>(totals[index]) /
                  static_cast<double>(_num_requests)}};
    }

    if (_detailed) {
        const std::array<const std::vector<uint64_t>*, 4> histograms = {
            &_core_queue_histogram, &_icnt_to_mem_histogram,
            &_dram_histogram, &_icnt_to_core_histogram};
        for (size_t index = 0; index < names.size(); ++index) {
            StageStats stats = compute_stats(*histograms[index],
                                             totals[index]);
            stages[names[index]]["p50_cycles"] = stats.p50;
            stages[names[index]]["p95_cycles"] = stats.p95;
            stages[names[index]]["p99_cycles"] = stats.p99;
        }
    }

    json record{{"version", 1},
                {"requests", _num_requests},
                {"detailed_percentiles_enabled", _detailed},
                {"stages", stages},
                {"cycle_semantics",
                 "elapsed simulator core-cycle timestamps at each "
                 "MemoryAccess stage transition"}};
    spdlog::info("COMPONENT_MEMORY_STAGES {}", record.dump());

    if (!_detailed)
        return;

    spdlog::info("========== Memory Request Profile ==========");

    spdlog::info("Requests : {}", _num_requests);

    StageStats cq = compute_stats(_core_queue_histogram,
                  _core_queue_cycles);

    StageStats im = compute_stats(_icnt_to_mem_histogram,
                  _icnt_to_mem_cycles);

StageStats dr = compute_stats(_dram_histogram,
                  _dram_cycles);

StageStats ic = compute_stats(_icnt_to_core_histogram,
                  _icnt_to_core_cycles);

    auto print_stage = [](const char* name,const StageStats& s)
    {
        spdlog::info(
            "{}",
            name);

        spdlog::info(
            "    Min : {}",
            s.min);

        spdlog::info(
            "    Max : {}",
            s.max);

        spdlog::info(
            "    Avg : {:.2f}",
            s.avg);

        spdlog::info(
            "    P50 : {}",
            s.p50);

        spdlog::info(
            "    P95 : {}",
            s.p95);

        spdlog::info(
            "    P99 : {}",
            s.p99);
    };
    print_stage("Core Queue", cq);
    print_stage("ICNT -> Memory", im);
    print_stage("DRAM", dr);
    print_stage("ICNT -> Core", ic);
}
