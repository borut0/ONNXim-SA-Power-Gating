#pragma once

#include <cstdint>
#include "Common.h"
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <array>

class MemoryProfiler {
public:
    static MemoryProfiler& instance();

    void finalize(MemoryAccess* access);

    void print_stats();

private:
    MemoryProfiler();

    bool _enabled = false;
    bool _detailed = false;

    uint64_t _num_requests = 0;

    uint64_t _core_queue_cycles = 0;
    uint64_t _icnt_to_mem_cycles = 0;
    uint64_t _dram_cycles = 0;
    uint64_t _icnt_to_core_cycles = 0;

    std::array<uint64_t, 4> _minimum_cycles{};
    std::array<uint64_t, 4> _maximum_cycles{};
    std::array<bool, 4> _have_sample{};

    std::vector<uint64_t> _core_queue_histogram;
    std::vector<uint64_t> _icnt_to_mem_histogram;
    std::vector<uint64_t> _dram_histogram;
    std::vector<uint64_t> _icnt_to_core_histogram;
};
