#include <gtest/gtest.h>

#include "Common.h"

namespace {

std::unique_ptr<Instruction> make_instruction(Opcode opcode) {
  return std::make_unique<Instruction>(Instruction{.opcode = opcode});
}

TEST(SaMetadataTest, FinalizesOnceAndAdvancesWithoutRescanning) {
  Tile tile{};
  tile.instructions.push_back(make_instruction(Opcode::MOVIN));
  tile.instructions.push_back(make_instruction(Opcode::ADD));
  tile.instructions.push_back(make_instruction(Opcode::GEMM_PRELOAD));
  tile.instructions.push_back(make_instruction(Opcode::MOVOUT));
  tile.instructions.push_back(make_instruction(Opcode::GEMM_PRELOAD));

  tile.finalize_sa_metadata();
  ASSERT_EQ(tile.sa_descriptors.size(), 2U);
  EXPECT_EQ(tile.sa_descriptors[0].instruction_ordinal, 2U);
  EXPECT_EQ(tile.sa_descriptors[0].prefix_from_previous_sa, 2U);
  EXPECT_EQ(tile.sa_descriptors[1].instruction_ordinal, 4U);
  EXPECT_EQ(tile.sa_descriptors[1].prefix_from_previous_sa, 1U);
  EXPECT_EQ(tile.sa_descriptors[0].preload_sequence, 1U);
  EXPECT_EQ(tile.sa_descriptors[1].preload_sequence, 2U);

  tile.advance_sa_metadata(tile.instructions[0].get());
  EXPECT_EQ(tile.next_sa_prefix_count, 1U);
  tile.instructions.pop_front();
  tile.advance_sa_metadata(tile.instructions[0].get());
  EXPECT_EQ(tile.next_sa_prefix_count, 0U);
  tile.instructions.pop_front();

  Instruction* first_sa = tile.instructions[0].get();
  tile.advance_sa_metadata(first_sa);
  EXPECT_EQ(tile.next_sa_opcode, Opcode::GEMM_PRELOAD);
  EXPECT_EQ(tile.next_sa_prefix_count, 1U);
  tile.instructions.pop_front();
  tile.advance_sa_metadata(tile.instructions[0].get());
  EXPECT_EQ(tile.next_sa_prefix_count, 0U);
  tile.instructions.pop_front();
  tile.advance_sa_metadata(tile.instructions[0].get());
  EXPECT_FALSE(tile.next_sa_candidate_found);
}

TEST(SaMetadataTest, HandlesTileWithoutSaInstruction) {
  Tile tile{};
  tile.instructions.push_back(make_instruction(Opcode::MOVIN));
  tile.instructions.push_back(make_instruction(Opcode::MOVOUT));
  tile.finalize_sa_metadata();
  EXPECT_TRUE(tile.sa_metadata_finalized);
  EXPECT_TRUE(tile.sa_descriptors.empty());
  EXPECT_FALSE(tile.next_sa_candidate_found);
  tile.finalize_sa_metadata();
  EXPECT_TRUE(tile.sa_descriptors.empty());
}

TEST(SaMetadataTest, RejectsPlainGemmAtSaPhaseEntry) {
  Tile tile{};
  tile.instructions.push_back(make_instruction(Opcode::MOVIN));
  tile.instructions.push_back(make_instruction(Opcode::GEMM));
  EXPECT_THROW(tile.finalize_sa_metadata(), std::logic_error);
}

TEST(SaMetadataTest, RecordsOnlyBoundaryMemoryPlanAndPreloadSequence) {
  Tile tile{};
  auto movin = make_instruction(Opcode::MOVIN);
  movin->dest_addr = SPAD_BASE;
  movin->src_addrs = {10, 20, 30};
  tile.instructions.push_back(std::move(movin));
  auto preload = make_instruction(Opcode::GEMM_PRELOAD);
  preload->src_addrs = {SPAD_BASE};
  tile.instructions.push_back(std::move(preload));
  auto gemm = make_instruction(Opcode::GEMM);
  gemm->src_addrs = {SPAD_BASE};
  tile.instructions.push_back(std::move(gemm));

  tile.finalize_sa_metadata();
  ASSERT_EQ(tile.sa_descriptors.size(), 2U);
  ASSERT_EQ(tile.sa_descriptors[0].planned_memory_sources.size(), 1U);
  EXPECT_EQ(tile.sa_descriptors[0].planned_memory_sources[0].address,
            SPAD_BASE);
  EXPECT_EQ(tile.sa_descriptors[0].planned_memory_sources[0].request_count,
            3U);
  EXPECT_EQ(
      tile.sa_descriptors[0].planned_memory_sources[0]
          .producer_instruction_ordinal,
      0U);
  EXPECT_TRUE(tile.sa_descriptors[0].memory_plan_complete);
  EXPECT_EQ(tile.sa_descriptors[0].preload_sequence, 1U);
  EXPECT_EQ(tile.sa_descriptors[1].preload_sequence, 1U);
}

TEST(SaMetadataTest, InternsExactWeightIdentity) {
  const uint64_t first = intern_sa_weight_block({100, 200}, 8, 16);
  const uint64_t same = intern_sa_weight_block({100, 200}, 8, 16);
  const uint64_t different_address =
      intern_sa_weight_block({100, 201}, 8, 16);
  const uint64_t different_mapping =
      intern_sa_weight_block({100, 200}, 16, 8);
  EXPECT_NE(first, 0U);
  EXPECT_EQ(first, same);
  EXPECT_NE(first, different_address);
  EXPECT_NE(first, different_mapping);
}

}  // namespace
