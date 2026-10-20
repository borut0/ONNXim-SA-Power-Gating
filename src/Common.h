#pragma once

#include <robin_hood.h>
#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <cassert>
#include <cstdint>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "SimulationConfig.h"
#include "Stat.h"
#include "helper/HelperFunctions.h"
#include "nlohmann/json.hpp"
#include "onnx/defs/schema.h"
#include "onnx/onnx-operators_pb.h"
#include "onnx/onnx_pb.h"

#define SPAD_BASE 0x10000000
#define ACCUM_SPAD_BASE 0x20000000
#define GARBEGE_ADDR 0xFFFFFFFFFFFFFFF
#define KB *1024

#define PAGE_SIZE 4096

using json = nlohmann::json;

typedef uint64_t addr_type;
typedef uint64_t cycle_type;

enum class MemoryStage {
  CORE_REQUEST_QUEUE,
  ICNT_TO_MEMORY,
  DRAM,
  ICNT_TO_CORE,
  DONE
};

typedef struct {
  uint32_t id;
  addr_type dram_address;
  addr_type spad_address;
  uint64_t size;
  bool write;
  bool request;
  uint32_t core_id;
  cycle_type start_cycle;
  cycle_type dram_enter_cycle;
  cycle_type dram_finish_cycle;
  int buffer_id;
  uint64_t core_queue_ticket = 0;

  // -------- Profiler --------
  MemoryStage stage = MemoryStage::CORE_REQUEST_QUEUE;
  cycle_type stage_enter_cycle = 0;

  cycle_type core_queue_cycles = 0;
  cycle_type icnt_to_mem_cycles = 0;
  cycle_type dram_cycles = 0;
  cycle_type icnt_to_core_cycles = 0;
  bool dram_completion_counted = false;
} MemoryAccess;

inline void transition_memory_stage(
    MemoryAccess* access,
    MemoryStage new_stage,
    cycle_type current_cycle)
{
    cycle_type elapsed = current_cycle - access->stage_enter_cycle;

    switch (access->stage) {
    case MemoryStage::CORE_REQUEST_QUEUE:
        access->core_queue_cycles += elapsed;
        break;

    case MemoryStage::ICNT_TO_MEMORY:
        access->icnt_to_mem_cycles += elapsed;
        break;

    case MemoryStage::DRAM:
        access->dram_cycles += elapsed;
        break;

    case MemoryStage::ICNT_TO_CORE:
        access->icnt_to_core_cycles += elapsed;
        break;

    case MemoryStage::DONE:
        break;
    }

    access->stage = new_stage;
    access->stage_enter_cycle = current_cycle;
}

enum class Opcode {
  MOVIN,
  MOVOUT,
  MOVOUT_POOL,
  GEMM_PRELOAD,
  GEMM,
  GEMM_WRITE,
  COMP,
  IM2COL,
  SOFTMAX,
  LAYERNORM,
  ADD,
  MUL,
  MAC,
  DIV,
  ADDTREE,
  EXP,
  GELU,
  SWISH,
  BAR
};
struct Tile;
typedef struct {
  Opcode opcode;
  cycle_type start_cycle;
  cycle_type finish_cycle;
  std::string id;
  std::vector<std::string> dependent_ids;
  std::string dest_id;
  addr_type dest_addr;
  uint32_t size;          // Used for sram allocation. Multiple of _config.dram_req_size
  uint32_t compute_size;
  std::vector<addr_type> src_addrs;
  int spad_id;
  int accum_spad_id;
  uint32_t operand_id  = 0;
  addr_type base_addr;

  uint32_t tile_m;
  uint32_t tile_k;
  uint32_t tile_n;

  bool src_from_accum = false;
  bool zero_init = false;
  bool last_inst = false;
  uint64_t measurement_tile_id = 0;
  Tile* my_tile;

  // Immutable, tile-generation metadata for systolic-array analysis. A
  // nonzero id denotes an exact interned weight-address/mapping signature,
  // not a scratchpad address (which can alias across tiles).
  uint64_t tile_instruction_ordinal = 0;
  uint64_t sa_weight_block_id = 0;
  bool sa_weight_block_known = false;
  // Boundary metadata for the assumed analytical two-bank model.  This is a
  // tile-local preload generation, not a physical bank selection.
  uint64_t sa_preload_sequence = 0;
  // Analytical preload boundary for an existing native GEMM.  This never
  // changes the opcode or ONNXim timing; it is consumed only by the updated
  // two-bank shadow and its causal safety predictor.
  bool sa_analytical_preload_boundary = false;
  std::string to_string();
} Instruction;

struct SaInstructionDescriptor {
  struct PlannedMemorySource {
    addr_type address = 0;
    size_t producer_instruction_ordinal = 0;
    uint64_t request_count = 0;
    uint64_t cumulative_request_count_through_producer = 0;
  };
  Instruction* instruction = nullptr;
  Opcode opcode = Opcode::BAR;
  size_t instruction_ordinal = 0;
  size_t prefix_from_previous_sa = 0;
  uint64_t weight_block_id = 0;
  bool weight_block_known = false;
  uint64_t preload_sequence = 0;
  bool analytical_preload_boundary = false;
  std::vector<PlannedMemorySource> planned_memory_sources;
  bool memory_plan_complete = false;
};

struct Tile {
  enum class Status {
    INITIALIZED,
    RUNNING,
    FINISH,
    BAR,
    EMPTY,
  };
  Status status = Status::EMPTY;
  std::string optype;
  uint32_t layer_id;
  uint32_t fused_op_id; /* For fused operation */
  uint32_t batch;
  uint32_t Q;
  uint32_t P;
  uint32_t M;
  uint32_t C;
  uint32_t S;
  uint32_t R;

  TileStat stat;
  std::deque<std::unique_ptr<Instruction>> instructions;
  bool accum;
  bool skip;
  int spad_id;
  int accum_spad_id;
  int core_id = -1;
  bool inst_finished = false;
  uint64_t measurement_tile_id = 0;

  // Immutable SA descriptors are finalized after tile generation. Runtime
  // state advances by descriptor index and never scans the instruction deque.
  std::vector<SaInstructionDescriptor> sa_descriptors;
  size_t next_sa_descriptor_index = 0;
  bool sa_metadata_finalized = false;

  // Cached current descriptor for the existing controller/predictor interface.
  // The raw pointer observes an Instruction owned by `instructions`; the
  // pointed-to object remains stable until that instruction is issued.
  bool next_sa_candidate_found = false;
  Opcode next_sa_opcode = Opcode::BAR;
  size_t next_sa_prefix_count = 0;
  Instruction* next_sa_instruction = nullptr;
  uint64_t next_sa_weight_block_id = 0;
  bool next_sa_weight_block_known = false;
  uint64_t next_sa_preload_sequence = 0;
  bool next_sa_analytical_preload_boundary = false;
  uint64_t issued_planned_memory_requests = 0;
  bool issued_planned_memory_requests_valid = true;

  void finalize_sa_metadata();
  void reset_runtime_sa_metadata();
  void advance_sa_metadata(const Instruction* issued_instruction);

  //for analysis
  cycle_type last_gemm_cycle = 0;
  bool last_gemm_seen = false;
  //end
} ;

// Returns a stable process-local identity for an exact global weight-address
// set and PE mapping. Equal keys receive equal ids; zero means unknown.
uint64_t intern_sa_weight_block(
    const std::vector<addr_type>& global_weight_addresses,
    uint32_t tile_m, uint32_t tile_k);

uint32_t generate_id();
uint32_t generate_mem_access_id();
addr_type allocate_address(uint32_t size);
SimulationConfig initialize_config(json config);
template <typename... Args>
std::string name_gen(Args... args) {
    std::vector<std::string> strs = {args...};
    assert(!strs.empty());
    std::string ret = "";
    for (auto &str : strs) {
        ret += str + ".";
    }
    ret.resize(ret.size() - 1);
    return ret;
}
uint32_t ceil_div(uint32_t src, uint32_t div);

std::vector<uint32_t> parse_dims(const std::string &str);

std::string dims_to_string(const std::vector<uint32_t> &dims);
