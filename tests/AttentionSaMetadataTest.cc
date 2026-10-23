#include <gtest/gtest.h>

#include <set>
#include <vector>

#include "Model.h"
#include "Tensor.h"
#include "operations/Attention.h"

namespace {

TEST(AttentionSaMetadataTest,
     UsesExactGlobalKvBlocksForPreloadAndConsumerInstructions) {
  CoreConfig core_config{};
  core_config.core_type = CoreType::SYSTOLIC_WS;
  core_config.core_height = 2;
  core_config.core_width = 2;

  SimulationConfig config{};
  config.num_cores = 1;
  config.core_config = &core_config;
  config.precision = 1;
  config.dram_req_size = 4;
  config.layout = "NHWC";

  Model model(json::object(), config, "attention_metadata_test");
  std::map<std::string, std::string> attributes = {
      {"num_tokens", "2"},
      {"num_heads", "4"},
      {"num_kv_heads", "2"},
      {"hidden_size", "16"}};
  Attention attention(config, &model, "attention", attributes, 0);

  std::vector<uint32_t> query_shape{4, 2, 4};
  std::vector<uint32_t> key_shape{2, 8, 4};
  std::vector<uint32_t> value_shape{2, 8, 4};
  std::vector<uint32_t> output_shape{2, 16};

  auto query = std::make_unique<Tensor>(0, "query", query_shape, 1, true);
  auto key = std::make_unique<Tensor>(0, "key", key_shape, 1, true);
  auto value = std::make_unique<Tensor>(0, "value", value_shape, 1, true);
  auto output =
      std::make_unique<Tensor>(0, "output", output_shape, 1, false);
  const uint32_t query_id = query->get_id();
  const uint32_t key_id = key->get_id();
  const uint32_t value_id = value->get_id();
  const uint32_t output_id = output->get_id();
  const addr_type key_base = key->get_address();
  const addr_type value_base = value->get_address();
  model.add_tensor(std::move(query));
  model.add_tensor(std::move(key));
  model.add_tensor(std::move(value));
  model.add_tensor(std::move(output));
  attention.add_input(query_id);
  attention.add_input(key_id);
  attention.add_input(value_id);
  attention.add_output(output_id);
  attention._seq = 8;
  attention._query_shape = query_shape;
  attention._key_shape = key_shape;
  attention._value_shape = value_shape;

  Mapping mapping{};
  mapping.tile_in_loop.N = 4;
  mapping.tile_in_loop.M = 4;
  mapping.tile_out_loop.N = 2;
  mapping.tile_out_loop.M = 2;

  Tile tile{};
  tile.batch = 0;
  tile.M = 1;
  attention.initialize_instructions(&tile, mapping, 2, 2);

  std::vector<const Instruction*> query_key_instructions;
  std::vector<const Instruction*> logits_value_instructions;
  for (const auto& instruction : tile.instructions) {
    if (instruction->opcode != Opcode::GEMM_PRELOAD &&
        instruction->opcode != Opcode::GEMM) {
      continue;
    }
    EXPECT_TRUE(instruction->sa_weight_block_known);
    EXPECT_NE(instruction->sa_weight_block_id, 0U);
    (instruction->src_from_accum ? logits_value_instructions
                                 : query_key_instructions)
        .push_back(instruction.get());
  }

  ASSERT_EQ(query_key_instructions.size(), 8U);
  ASSERT_EQ(logits_value_instructions.size(), 8U);

  std::set<uint64_t> query_key_ids;
  std::set<uint64_t> logits_value_ids;
  for (size_t i = 0; i < query_key_instructions.size(); i += 2) {
    EXPECT_EQ(query_key_instructions[i]->opcode, Opcode::GEMM_PRELOAD);
    EXPECT_EQ(query_key_instructions[i + 1]->opcode, Opcode::GEMM);
    EXPECT_EQ(query_key_instructions[i]->sa_weight_block_id,
              query_key_instructions[i + 1]->sa_weight_block_id);
    query_key_ids.insert(query_key_instructions[i]->sa_weight_block_id);
  }
  for (size_t i = 0; i < logits_value_instructions.size(); i += 2) {
    EXPECT_EQ(logits_value_instructions[i]->opcode, Opcode::GEMM_PRELOAD);
    EXPECT_EQ(logits_value_instructions[i + 1]->opcode, Opcode::GEMM);
    EXPECT_EQ(logits_value_instructions[i]->sa_weight_block_id,
              logits_value_instructions[i + 1]->sa_weight_block_id);
    logits_value_ids.insert(logits_value_instructions[i]->sa_weight_block_id);
  }
  EXPECT_EQ(query_key_ids.size(), 4U);
  EXPECT_EQ(logits_value_ids.size(), 4U);
  EXPECT_NE(query_key_instructions[0]->sa_weight_block_id,
            query_key_instructions[2]->sa_weight_block_id);
  EXPECT_NE(query_key_instructions[0]->sa_weight_block_id,
            query_key_instructions[4]->sa_weight_block_id);
  for (uint64_t query_key_id : query_key_ids) {
    EXPECT_EQ(logits_value_ids.count(query_key_id), 0U);
  }

  // head_idx 2 belongs to KV head 1, and tile M=1 starts at sequence 4.
  // The first 2x2 K/V block therefore contains these exact global addresses.
  const std::vector<addr_type> expected_key_addresses = {
      key_base + 48, key_base + 49, key_base + 52, key_base + 53};
  const std::vector<addr_type> expected_second_reduction_addresses = {
      key_base + 50, key_base + 51, key_base + 54, key_base + 55};
  const std::vector<addr_type> expected_value_addresses = {
      value_base + 48, value_base + 49, value_base + 52, value_base + 53};
  const uint64_t expected_query_key_id =
      intern_sa_weight_block(expected_key_addresses, 2, 2);
  const uint64_t expected_second_reduction_id =
      intern_sa_weight_block(expected_second_reduction_addresses, 2, 2);
  const uint64_t expected_logits_value_id =
      intern_sa_weight_block(expected_value_addresses, 2, 2);
  EXPECT_EQ(query_key_instructions.front()->sa_weight_block_id,
            expected_query_key_id);
  EXPECT_EQ(query_key_instructions[2]->sa_weight_block_id,
            expected_second_reduction_id);
  EXPECT_EQ(config.align_address(expected_key_addresses[0]),
            config.align_address(expected_second_reduction_addresses[0]));
  EXPECT_EQ(config.align_address(expected_key_addresses[2]),
            config.align_address(expected_second_reduction_addresses[2]));
  EXPECT_NE(expected_query_key_id, expected_second_reduction_id);
  EXPECT_EQ(logits_value_instructions.front()->sa_weight_block_id,
            expected_logits_value_id);
  EXPECT_NE(expected_query_key_id, expected_logits_value_id);

  Tile first_sequence_tile{};
  first_sequence_tile.batch = 0;
  first_sequence_tile.M = 0;
  attention.initialize_instructions(&first_sequence_tile, mapping, 2, 2);
  const Instruction* first_sequence_query_key = nullptr;
  for (const auto& instruction : first_sequence_tile.instructions) {
    if ((instruction->opcode == Opcode::GEMM_PRELOAD ||
         instruction->opcode == Opcode::GEMM) &&
        !instruction->src_from_accum) {
      first_sequence_query_key = instruction.get();
      break;
    }
  }
  ASSERT_NE(first_sequence_query_key, nullptr);
  EXPECT_NE(first_sequence_query_key->sa_weight_block_id,
            query_key_instructions.front()->sa_weight_block_id);

  // A padded stationary block has no exact global operand address. It must
  // remain unknown instead of receiving an inferred or fabricated identity.
  attention._seq = 5;
  attention._key_shape = {2, 5, 4};
  attention._value_shape = {2, 5, 4};
  Tile partial_sequence_tile{};
  partial_sequence_tile.batch = 0;
  partial_sequence_tile.M = 1;
  attention.initialize_instructions(&partial_sequence_tile, mapping, 2, 2);

  std::vector<const Instruction*> partial_query_key;
  std::vector<const Instruction*> partial_logits_value;
  for (const auto& instruction : partial_sequence_tile.instructions) {
    if (instruction->opcode != Opcode::GEMM_PRELOAD &&
        instruction->opcode != Opcode::GEMM) {
      continue;
    }
    (instruction->src_from_accum ? partial_logits_value : partial_query_key)
        .push_back(instruction.get());
  }
  ASSERT_EQ(partial_query_key.size(), 8U);
  ASSERT_EQ(partial_logits_value.size(), 8U);
  EXPECT_FALSE(partial_query_key[0]->sa_weight_block_known);
  EXPECT_EQ(partial_query_key[0]->sa_weight_block_id, 0U);
  EXPECT_FALSE(partial_query_key[1]->sa_weight_block_known);
  EXPECT_EQ(partial_query_key[1]->sa_weight_block_id, 0U);
  EXPECT_FALSE(partial_logits_value[0]->sa_weight_block_known);
  EXPECT_EQ(partial_logits_value[0]->sa_weight_block_id, 0U);
  EXPECT_FALSE(partial_logits_value[1]->sa_weight_block_known);
  EXPECT_EQ(partial_logits_value[1]->sa_weight_block_id, 0U);
  EXPECT_FALSE(partial_query_key[4]->sa_weight_block_known);
  EXPECT_EQ(partial_query_key[4]->sa_weight_block_id, 0U);
  EXPECT_FALSE(partial_query_key[5]->sa_weight_block_known);
  EXPECT_EQ(partial_query_key[5]->sa_weight_block_id, 0U);
  EXPECT_FALSE(partial_logits_value[2]->sa_weight_block_known);
  EXPECT_EQ(partial_logits_value[2]->sa_weight_block_id, 0U);
  EXPECT_FALSE(partial_logits_value[3]->sa_weight_block_known);
  EXPECT_EQ(partial_logits_value[3]->sa_weight_block_id, 0U);
}

TEST(AttentionSaMetadataTest,
     FusedOnnxPathUsesPhysicalSaSizedPreloadGenerations) {
  CoreConfig core_config{};
  core_config.core_type = CoreType::SYSTOLIC_WS;
  core_config.core_height = 4;
  core_config.core_width = 2;
  core_config.spad_size = 64;
  core_config.accum_spad_size = 16;
  SimulationConfig config{};
  config.num_cores = 1;
  config.core_config = &core_config;
  config.precision = 1;
  config.full_precision = 4;
  config.dram_req_size = 4;
  config.layout = "NHWC";
  Model model(json::object(), config, "attention_preload_first_test");

  std::vector<uint32_t> input_dims{1, 5, 4};
  std::vector<uint32_t> weight_dims{4, 12};
  std::vector<uint32_t> bias_dims{12};
  model.add_tensor(std::make_unique<Tensor>(
      0, "input", input_dims, config.precision, true));
  model.add_tensor(std::make_unique<Tensor>(
      0, "weight", weight_dims, config.precision, true));
  model.add_tensor(std::make_unique<Tensor>(
      0, "bias", bias_dims, config.precision, true));

  onnx::NodeProto node;
  node.set_name("fused_attention");
  node.set_op_type("Attention");
  node.add_input("input");
  node.add_input("weight");
  node.add_input("bias");
  node.add_output("output");
  auto* num_heads = node.add_attribute();
  num_heads->set_name("num_heads");
  num_heads->set_i(2);

  Attention attention(config, &model, node, 0);
  MappingTable mapping_table(config);
  attention.initialize_onnx_tiles(mapping_table);

  auto& tiles = attention.get_tiles();
  ASSERT_FALSE(tiles.empty());
  size_t sa_instruction_count = 0;
  std::set<uint64_t> qk_generations;
  std::set<uint64_t> pv_generations;
  for (const auto& tile : tiles) {
    ASSERT_NE(tile, nullptr);
    ASSERT_TRUE(tile->sa_metadata_finalized);
    ASSERT_FALSE(tile->sa_descriptors.empty());
    EXPECT_EQ(tile->sa_descriptors.front().opcode, Opcode::GEMM_PRELOAD);
    for (const auto& descriptor : tile->sa_descriptors) {
      ++sa_instruction_count;
      ASSERT_NE(descriptor.instruction, nullptr);
      EXPECT_EQ(descriptor.opcode, Opcode::GEMM_PRELOAD);
      EXPECT_FALSE(descriptor.analytical_preload_boundary);
      EXPECT_GT(descriptor.preload_sequence, 0U);
      EXPECT_GT(descriptor.instruction->tile_m, 0U);
      EXPECT_LE(descriptor.instruction->tile_m, core_config.core_height);
      EXPECT_GT(descriptor.instruction->tile_k, 0U);
      EXPECT_LE(descriptor.instruction->tile_k, core_config.core_width);
      EXPECT_GT(descriptor.instruction->tile_n, 0U);
      EXPECT_TRUE(descriptor.weight_block_known);
      EXPECT_NE(descriptor.weight_block_id, 0U);
      (descriptor.instruction->src_from_accum ? pv_generations
                                               : qk_generations)
          .insert(descriptor.weight_block_id);
    }
  }
  EXPECT_GT(sa_instruction_count, 0U);
  EXPECT_GT(qk_generations.size(), 1U);
  EXPECT_GT(pv_generations.size(), 1U);
  for (uint64_t qk_generation : qk_generations) {
    EXPECT_EQ(pv_generations.count(qk_generation), 0U);
  }

  delete attention._projection_node;
  attention._projection_node = nullptr;
}

TEST(AttentionSaMetadataTest,
     MappedPathMovoutUsesQueryTileOffsetAndCoversEveryQuery) {
  CoreConfig core_config{};
  core_config.core_type = CoreType::SYSTOLIC_WS;
  core_config.core_height = 4;
  core_config.core_width = 2;
  SimulationConfig config{};
  config.num_cores = 1;
  config.core_config = &core_config;
  config.precision = 1;
  config.full_precision = 4;
  config.dram_req_size = 1;
  config.layout = "NHWC";

  Model model(json::object(), config, "attention_movout_offset_test");
  std::map<std::string, std::string> attributes = {
      {"num_tokens", "4"}, {"num_heads", "4"},
      {"num_kv_heads", "2"}, {"hidden_size", "16"}};
  Attention attention(config, &model, "attention", attributes, 0);

  std::vector<uint32_t> query_shape{4, 4, 4};
  std::vector<uint32_t> key_shape{2, 4, 4};
  std::vector<uint32_t> value_shape{2, 4, 4};
  std::vector<uint32_t> output_shape{4, 16};
  auto query = std::make_unique<Tensor>(
      0, "query", query_shape, config.precision, true);
  auto key = std::make_unique<Tensor>(
      0, "key", key_shape, config.precision, true);
  auto value = std::make_unique<Tensor>(
      0, "value", value_shape, config.precision, true);
  auto output = std::make_unique<Tensor>(
      0, "output", output_shape, config.precision, false);
  const uint32_t query_id = query->get_id();
  const uint32_t key_id = key->get_id();
  const uint32_t value_id = value->get_id();
  const uint32_t output_id = output->get_id();
  model.add_tensor(std::move(query));
  model.add_tensor(std::move(key));
  model.add_tensor(std::move(value));
  model.add_tensor(std::move(output));
  attention.add_input(query_id);
  attention.add_input(key_id);
  attention.add_input(value_id);
  attention.add_output(output_id);
  attention._seq = 4;
  attention._query_shape = query_shape;
  attention._key_shape = key_shape;
  attention._value_shape = value_shape;

  Mapping mapping{};
  mapping.tile_in_loop.N = 4;   // two queries for each of two heads
  mapping.tile_in_loop.M = 4;
  mapping.tile_out_loop.N = 4;  // two query tiles for each of two KV heads
  mapping.tile_out_loop.M = 1;

  std::set<addr_type> all_output_addresses;
  std::set<addr_type> prior_tile_addresses;
  for (uint32_t query_tile = 0; query_tile < 2; ++query_tile) {
    Tile tile{};
    tile.batch = query_tile;
    tile.M = 0;
    attention.initialize_instructions(&tile, mapping, 0, 2);

    std::set<addr_type> tile_output_addresses;
    for (const auto& instruction : tile.instructions) {
      if (instruction->opcode != Opcode::MOVOUT) continue;
      tile_output_addresses.insert(instruction->src_addrs.begin(),
                                   instruction->src_addrs.end());
    }
    ASSERT_EQ(tile_output_addresses.size(), 16U);
    for (addr_type address : tile_output_addresses) {
      EXPECT_EQ(prior_tile_addresses.count(address), 0U);
      all_output_addresses.insert(address);
    }
    prior_tile_addresses.insert(tile_output_addresses.begin(),
                                tile_output_addresses.end());
  }
  EXPECT_EQ(all_output_addresses.size(), 32U);
}

TEST(AttentionSaMetadataTest, NonFusedSpliceTransfersLiveTilesOnly) {
  CoreConfig core_config{};
  core_config.core_type = CoreType::SYSTOLIC_WS;
  core_config.core_height = 8;
  core_config.core_width = 8;
  core_config.spad_size = 64;
  core_config.accum_spad_size = 16;
  SimulationConfig config{};
  config.num_cores = 1;
  config.core_config = &core_config;
  config.precision = 1;
  config.dram_req_size = 32;
  config.layout = "NHWC";
  Model model(json::object(), config, "attention_non_fused_splice_test");
  std::map<std::string, std::string> attributes = {
      {"num_tokens", "8"}, {"num_heads", "1"},
      {"num_kv_heads", "1"}, {"hidden_size", "8"}};
  Attention attention(config, &model, "attention", attributes, 0);
  attention._seq = 8;
  MappingTable mapping_table(config);

  attention.initialize_non_fused_tiles(mapping_table);
  auto& tiles = attention.get_tiles();
  ASSERT_FALSE(tiles.empty());
  size_t gemm_tiles = 0;
  size_t softmax_tiles = 0;
  for (const auto& tile : tiles) {
    ASSERT_NE(tile, nullptr);
    EXPECT_TRUE(tile->sa_metadata_finalized);
    if (tile->optype == "Gemm") ++gemm_tiles;
    for (const auto& instruction : tile->instructions) {
      if (instruction->opcode == Opcode::SOFTMAX) ++softmax_tiles;
    }
  }
  EXPECT_GT(gemm_tiles, 0U);
  EXPECT_GT(softmax_tiles, 0U);
  EXPECT_EQ(&attention.get_tiles(), &tiles);
}

}  // namespace
