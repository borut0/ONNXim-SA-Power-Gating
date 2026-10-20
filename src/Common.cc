#include "Common.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <tuple>

uint32_t generate_id() {
  static uint32_t id_counter{0};
  return id_counter++;
}
uint32_t generate_mem_access_id() {
  static uint32_t id_counter{0};
  return id_counter++;
}

uint64_t intern_sa_weight_block(
    const std::vector<addr_type>& global_weight_addresses,
    uint32_t tile_m, uint32_t tile_k) {
  if (global_weight_addresses.empty() || tile_m == 0 || tile_k == 0) {
    return 0;
  }

  using WeightKey =
      std::tuple<uint32_t, uint32_t, std::vector<addr_type>>;
  static std::map<WeightKey, uint64_t> ids;
  static uint64_t next_id = 1;
  WeightKey key{tile_m, tile_k, global_weight_addresses};
  auto [iter, inserted] = ids.emplace(std::move(key), next_id);
  if (inserted) {
    next_id++;
  }
  return iter->second;
}

void Tile::finalize_sa_metadata() {
  if (sa_metadata_finalized) {
    return;
  }

  sa_descriptors.clear();
  size_t prefix = 0;
  size_t ordinal = 0;
  bool sa_phase_open = false;
  uint64_t preload_sequence = 0;
  uint64_t cumulative_memory_requests = 0;
  std::map<addr_type, SaInstructionDescriptor::PlannedMemorySource>
      planned_memory_sources;
  for (const std::unique_ptr<Instruction>& instruction : instructions) {
    instruction->tile_instruction_ordinal = ordinal;
    if (instruction->opcode == Opcode::MOVIN) {
      if (instruction->src_addrs.size() >
          std::numeric_limits<uint64_t>::max() -
              cumulative_memory_requests) {
        throw std::overflow_error(
            "tile planned memory-request count overflow");
      }
      cumulative_memory_requests += instruction->src_addrs.size();
      planned_memory_sources[instruction->dest_addr] =
          SaInstructionDescriptor::PlannedMemorySource{
              instruction->dest_addr, ordinal,
              instruction->src_addrs.size(), cumulative_memory_requests};
    }
    if (instruction->opcode == Opcode::GEMM ||
        instruction->opcode == Opcode::GEMM_PRELOAD) {
      const bool preload_boundary =
          instruction->opcode == Opcode::GEMM_PRELOAD ||
          instruction->sa_analytical_preload_boundary;
      if (!sa_phase_open && !preload_boundary) {
        throw std::logic_error(
            "each contiguous tile SA phase must begin with a preload boundary");
      }
      if (preload_boundary) {
        preload_sequence++;
      }
      sa_phase_open = true;
      instruction->sa_preload_sequence = preload_sequence;
      std::vector<SaInstructionDescriptor::PlannedMemorySource>
          required_memory_sources;
      bool memory_plan_complete = true;
      for (addr_type address : instruction->src_addrs) {
        if (address >= ACCUM_SPAD_BASE) {
          continue;
        }
        const auto producer = planned_memory_sources.find(address);
        if (producer == planned_memory_sources.end()) {
          memory_plan_complete = false;
          continue;
        }
        const auto duplicate = std::find_if(
            required_memory_sources.begin(), required_memory_sources.end(),
            [address](const auto& source) {
              return source.address == address;
            });
        if (duplicate == required_memory_sources.end()) {
          required_memory_sources.push_back(producer->second);
        }
      }
      sa_descriptors.push_back(SaInstructionDescriptor{
          .instruction = instruction.get(),
          .opcode = instruction->opcode,
          .instruction_ordinal = ordinal,
          .prefix_from_previous_sa = prefix,
          .weight_block_id = instruction->sa_weight_block_id,
          .weight_block_known = instruction->sa_weight_block_known,
          .preload_sequence = preload_sequence,
          .analytical_preload_boundary =
              instruction->sa_analytical_preload_boundary,
          .planned_memory_sources = std::move(required_memory_sources),
          .memory_plan_complete = memory_plan_complete});
      prefix = 0;
    } else {
      sa_phase_open = false;
      prefix++;
    }
    ordinal++;
  }
  sa_metadata_finalized = true;
  reset_runtime_sa_metadata();
}

void Tile::reset_runtime_sa_metadata() {
  if (!sa_metadata_finalized) {
    throw std::logic_error("SA metadata used before tile finalization");
  }

  next_sa_descriptor_index = 0;
  if (sa_descriptors.empty()) {
    next_sa_candidate_found = false;
    next_sa_opcode = Opcode::BAR;
    next_sa_prefix_count = 0;
    next_sa_instruction = nullptr;
    next_sa_weight_block_id = 0;
    next_sa_weight_block_known = false;
    next_sa_preload_sequence = 0;
    next_sa_analytical_preload_boundary = false;
    issued_planned_memory_requests = 0;
    issued_planned_memory_requests_valid = true;
    return;
  }

  const SaInstructionDescriptor& descriptor = sa_descriptors.front();
  next_sa_candidate_found = true;
  next_sa_opcode = descriptor.opcode;
  next_sa_prefix_count = descriptor.prefix_from_previous_sa;
  next_sa_instruction = descriptor.instruction;
  next_sa_weight_block_id = descriptor.weight_block_id;
  next_sa_weight_block_known = descriptor.weight_block_known;
  next_sa_preload_sequence = descriptor.preload_sequence;
  next_sa_analytical_preload_boundary =
      descriptor.analytical_preload_boundary;
  issued_planned_memory_requests = 0;
  issued_planned_memory_requests_valid = true;
}

void Tile::advance_sa_metadata(const Instruction* issued_instruction) {
  if (!sa_metadata_finalized) {
    throw std::logic_error("SA metadata advanced before tile finalization");
  }
  if (issued_instruction != nullptr &&
      issued_instruction->opcode == Opcode::MOVIN) {
    const uint64_t count = issued_instruction->src_addrs.size();
    if (count > std::numeric_limits<uint64_t>::max() -
                    issued_planned_memory_requests) {
      issued_planned_memory_requests_valid = false;
    } else {
      issued_planned_memory_requests += count;
    }
  }
  if (!next_sa_candidate_found) {
    return;
  }

  if (next_sa_instruction != issued_instruction) {
    if (next_sa_prefix_count == 0) {
      throw std::logic_error("issued instruction does not match SA descriptor");
    }
    next_sa_prefix_count--;
    return;
  }

  next_sa_descriptor_index++;
  if (next_sa_descriptor_index >= sa_descriptors.size()) {
    next_sa_candidate_found = false;
    next_sa_opcode = Opcode::BAR;
    next_sa_prefix_count = 0;
    next_sa_instruction = nullptr;
    next_sa_weight_block_id = 0;
    next_sa_weight_block_known = false;
    next_sa_preload_sequence = 0;
    next_sa_analytical_preload_boundary = false;
    return;
  }

  const SaInstructionDescriptor& descriptor =
      sa_descriptors[next_sa_descriptor_index];
  next_sa_candidate_found = true;
  next_sa_opcode = descriptor.opcode;
  next_sa_prefix_count = descriptor.prefix_from_previous_sa;
  next_sa_instruction = descriptor.instruction;
  next_sa_weight_block_id = descriptor.weight_block_id;
  next_sa_weight_block_known = descriptor.weight_block_known;
  next_sa_preload_sequence = descriptor.preload_sequence;
  next_sa_analytical_preload_boundary =
      descriptor.analytical_preload_boundary;
}

addr_type allocate_address(uint32_t size) {
  static addr_type base_addr{0};
  addr_type result = base_addr;
  int offset = 0;
  if (result % 256 != 0) {
    offset = 256 - (result % 256);
  }
  result += offset;
  assert(result % 256 == 0);
  base_addr += (size + offset);
  base_addr += (256 - base_addr % 256);
  return result;
}

template <typename T>
T get_config_value(json config, std::string key) {
  if (config.contains(key)) {
    return config[key];
  } else {
    throw std::runtime_error(fmt::format("Config key {} not found", key));
  }
}

const static std::map<std::string, CoreType> core_type_map = {
  {"systolic_os", CoreType::SYSTOLIC_OS},
  {"systolic_ws", CoreType::SYSTOLIC_WS}
};

const static std::map<std::string, DramType> dram_type_map = {
  {"simple", DramType::SIMPLE},
  {"ramulator", DramType::RAMULATOR1},
  {"ramulator2", DramType::RAMULATOR2}
};

const static std::map<std::string, IcntType> icnt_type_map = {
  {"simple", IcntType::SIMPLE},
  {"booksim2", IcntType::BOOKSIM2}
};

SimulationConfig initialize_config(json config) {
  SimulationConfig parsed_config;

  /* Core configs */
  parsed_config.num_cores = get_config_value<uint32_t>(config, "num_cores");
  parsed_config.core_config = new struct CoreConfig[parsed_config.num_cores];
  parsed_config.core_freq = get_config_value<uint32_t>(config, "core_freq");
  parsed_config.core_print_interval = get_config_value<uint32_t>(config, "core_print_interval");

  for (int i=0; i<parsed_config.num_cores; i++) {
    std::string core_id = "core_" + std::to_string(i);
    auto core_config = config["core_config"][core_id];
    std::string core_type = core_config["core_type"];
    if (core_type_map.contains(core_type)) {
      parsed_config.core_config[i].core_type = core_type_map.at(core_type);
    } else {
      throw std::runtime_error(fmt::format("Not implemented core type {} ", core_type));
    }
    parsed_config.core_config[i].core_width = core_config["core_width"];
    parsed_config.core_config[i].core_height = core_config["core_height"];

    /* Vector configs */
    parsed_config.core_config[i].vector_process_bit = core_config["vector_process_bit"];
    parsed_config.core_config[i].add_latency = core_config["add_latency"];
    parsed_config.core_config[i].mul_latency = core_config["mul_latency"];
    parsed_config.core_config[i].exp_latency = core_config["exp_latency"];
    parsed_config.core_config[i].gelu_latency = core_config["gelu_latency"];
    parsed_config.core_config[i].add_tree_latency = core_config["add_tree_latency"];
    parsed_config.core_config[i].scalar_sqrt_latency = core_config["scalar_sqrt_latency"];
    parsed_config.core_config[i].scalar_add_latency = core_config["scalar_add_latency"];
    parsed_config.core_config[i].scalar_mul_latency = core_config["scalar_mul_latency"];
    parsed_config.core_config[i].mac_latency = core_config["mac_latency"];
    parsed_config.core_config[i].div_latency = core_config["div_latency"];

    /* SRAM configs */
    parsed_config.core_config[i].sram_width = core_config["sram_width"];
    parsed_config.core_config[i].spad_size = core_config["spad_size"];
    parsed_config.core_config[i].accum_spad_size = core_config["accum_spad_size"];
  }

  /* DRAM config */
  std::string dram_type = get_config_value<std::string>(config, "dram_type");
  if (dram_type_map.contains(dram_type)) {
    parsed_config.dram_type = dram_type_map.at(dram_type);
  } else {
    throw std::runtime_error(fmt::format("Not implemented dram type {} ", dram_type));
  }

  parsed_config.dram_freq = get_config_value<uint32_t>(config, "dram_freq");
  if (config.contains("dram_latency"))
    parsed_config.dram_latency = config["dram_latency"];
  if (config.contains("dram_config_path"))
    parsed_config.dram_config_path = config["dram_config_path"];
  parsed_config.dram_channels = config["dram_channels"];
  if (config.contains("dram_req_size"))
    parsed_config.dram_req_size = config["dram_req_size"];
  if (config.contains("dram_print_interval"))
    parsed_config.dram_print_interval = config["dram_print_interval"];
  if(config.contains("dram_nbl"))
    parsed_config.dram_nbl = config["dram_nbl"];
  if (config.contains("dram_size"))
    parsed_config.dram_size = config["dram_size"];
  else
    parsed_config.dram_size = 0;

  /* Icnt config */
  std::string icnt_type = get_config_value<std::string>(config, "icnt_type");
  if (icnt_type_map.contains(icnt_type)) {
    parsed_config.icnt_type = icnt_type_map.at(icnt_type);
  } else {
    throw std::runtime_error(fmt::format("Not implemented icnt type {} ", icnt_type));
  }

  parsed_config.icnt_freq = get_config_value<uint32_t>(config, "icnt_freq");
  if (config.contains("icnt_latency"))
    parsed_config.icnt_latency = config["icnt_latency"];
  if (config.contains("icnt_config_path"))
    parsed_config.icnt_config_path = config["icnt_config_path"];
  if (config.contains("icnt_print_interval"))
    parsed_config.icnt_print_interval = config["icnt_print_interval"];
  if (config.contains("icnt_injection_ports_per_core"))
    parsed_config.icnt_injection_ports_per_core = config["icnt_injection_ports_per_core"];

  parsed_config.scheduler_type = get_config_value<std::string>(config, "scheduler");
  parsed_config.precision = get_config_value<uint32_t>(config, "precision");
  parsed_config.layout = get_config_value<std::string>(config, "layout");

  if (config.contains("partition")) {
    for (int i=0; i<parsed_config.num_cores; i++) {
      std::string core_partition = "core_" + std::to_string(i);
      uint32_t partition_id = uint32_t(config["partition"][core_partition]);
      parsed_config.partiton_map[partition_id].push_back(i);
      spdlog::info("CPU {}: Partition {}", i, partition_id);
    }
  } else {
    /* Default: all partition 0 */
    for (int i=0; i<parsed_config.num_cores; i++) {
      parsed_config.partiton_map[0].push_back(i);
      spdlog::info("CPU {}: Partition {}", i, 0);
    }
  }
  return parsed_config;
}

uint32_t ceil_div(uint32_t src, uint32_t div) { 
  return (src+div-1)/div; 
}

std::vector<uint32_t> parse_dims(const std::string &str) {
  std::vector<uint32_t> dims;
  std::string token;
  std::istringstream tokenStream(str);
  while (std::getline(tokenStream, token, ',')) {
      dims.push_back(std::stoi(token));
  }
  return dims;
}

std::string dims_to_string(const std::vector<uint32_t> &dims){
  std::string str;
  for (int i=0; i<dims.size(); i++) {
    str += std::to_string(dims[i]);
    if (i != dims.size()-1) {
      str += ",";
    }
  }
  return str;
}
