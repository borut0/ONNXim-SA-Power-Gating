#include "SaPeStateOpportunity.h"

#include "gtest/gtest.h"

using sa_pe_state::Interval;

TEST(SaPeStateOpportunityTest, SplitsSingleWeightStationaryInterval) {
  const auto result = sa_pe_state::derive(
      {Interval{.start_cycle = 2, .finish_cycle = 9,
                .tile_m = 2, .tile_k = 3, .tile_n = 4}},
      4, 4, 10, 7);

  EXPECT_EQ(result.on_compute_pe_cycles, 24u);
  EXPECT_EQ(result.resident_weight_pe_cycles, 42u);
  EXPECT_EQ(result.w_on_candidate_pe_cycles, 18u);
  EXPECT_EQ(result.off_unmapped_candidate_pe_cycles, 70u);
  EXPECT_EQ(result.temporal_idle_unclassified_pe_cycles, 48u);
  EXPECT_EQ(result.total_capacity_pe_cycles, 160u);
  EXPECT_TRUE(result.accounting_exact);
  ASSERT_EQ(result.interval_classes.size(), 1u);
  EXPECT_EQ(result.interval_classes[0].occurrences, 1u);
}

TEST(SaPeStateOpportunityTest, UnionsOverlappingPackedFootprints) {
  // [0,4): 2x4 footprint (8 PEs)
  // [2,6): 4x2 footprint (8 PEs)
  // During [2,4), their origin-anchored union has 12 PEs, not 16.
  const auto result = sa_pe_state::derive(
      {Interval{.start_cycle = 0, .finish_cycle = 4,
                .tile_m = 2, .tile_k = 4, .tile_n = 1},
       Interval{.start_cycle = 2, .finish_cycle = 6,
                .tile_m = 4, .tile_k = 2, .tile_n = 1}},
      4, 4, 8, 6);

  EXPECT_EQ(result.union_active_cycles, 6u);
  EXPECT_EQ(result.resident_weight_pe_cycles, 56u);  // 16 + 24 + 16
  EXPECT_EQ(result.on_compute_pe_cycles, 16u);
  EXPECT_EQ(result.w_on_candidate_pe_cycles, 40u);
  EXPECT_EQ(result.off_unmapped_candidate_pe_cycles, 40u);
  EXPECT_EQ(result.temporal_idle_unclassified_pe_cycles, 32u);
  EXPECT_TRUE(result.accounting_exact);
}

TEST(SaPeStateOpportunityTest, RejectsUnrepresentableDurationWithoutClipping) {
  const auto result = sa_pe_state::derive(
      {Interval{.start_cycle = 0, .finish_cycle = 2,
                .tile_m = 2, .tile_k = 2, .tile_n = 3}},
      4, 4, 2, 2);

  EXPECT_EQ(result.insufficient_duration_intervals, 1u);
  EXPECT_FALSE(result.accounting_exact);
  EXPECT_EQ(result.on_compute_pe_cycles, 0u);
}

TEST(SaPeStateOpportunityTest, LeavesWholeSaIdleTimeUnclassified) {
  const auto result = sa_pe_state::derive({}, 8, 8, 5, 0);
  EXPECT_EQ(result.temporal_idle_unclassified_pe_cycles, 320u);
  EXPECT_EQ(result.w_on_candidate_pe_cycles, 0u);
  EXPECT_EQ(result.off_unmapped_candidate_pe_cycles, 0u);
  EXPECT_TRUE(result.accounting_exact);
}

TEST(SaPeStateOpportunityTest, RectangularWaveMatchesEnumeration) {
  constexpr uint32_t rows = 2;
  constexpr uint32_t columns = 3;
  constexpr uint32_t waves = 4;
  const std::vector<uint64_t> expected{1, 3, 5, 6, 5, 3, 1};
  uint64_t sum = 0;
  for (uint64_t cycle = 0; cycle < expected.size(); ++cycle) {
    const uint64_t actual = sa_pe_state::useful_macs_at_cycle(
        rows, columns, waves, cycle);
    EXPECT_EQ(actual, expected[cycle]);
    sum += actual;
  }
  EXPECT_EQ(sum, static_cast<uint64_t>(rows) * columns * waves);
  EXPECT_EQ(sa_pe_state::useful_macs_at_cycle(
                rows, columns, waves, expected.size()),
            0u);
}

TEST(SaPeStateOpportunityTest, TracksKnownPreloadAndWeightReuse) {
  using sa_pe_state::WaveInterval;
  using sa_pe_state::WaveOpcode;
  const auto result = sa_pe_state::derive_wave_lifetime(
      {WaveInterval{.issue_cycle = 0, .start_cycle = 0, .finish_cycle = 4,
                    .tile_m = 2, .tile_k = 2, .tile_n = 2,
                    .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 7, .weight_block_known = true},
       WaveInterval{.issue_cycle = 1, .start_cycle = 2, .finish_cycle = 6,
                    .tile_m = 2, .tile_k = 2, .tile_n = 2,
                    .opcode = WaveOpcode::GEMM,
                    .weight_block_id = 7, .weight_block_known = true}},
      2, 2, 6, 6);

  EXPECT_EQ(result.analytical_on_pe_cycles, 16u);
  EXPECT_EQ(result.resident_weight_pe_cycles, 24u);
  EXPECT_EQ(result.pre_start_committed_pe_cycles, 4u);
  EXPECT_TRUE(result.wave_overlap_free);
  EXPECT_TRUE(result.weight_lifetime_complete);
  // The aggregate PE model deliberately refuses to subtract a committed
  // pre-start footprint from an older diagonal wave without reconstructing
  // their intersection.
  EXPECT_FALSE(result.state_accounting_exact);
  EXPECT_EQ(result.unclassified_pe_cycles, 8u);
  ASSERT_EQ(result.weight_lifetimes.size(), 1u);
  EXPECT_EQ(result.weight_lifetimes[0].consumer_instructions, 2u);
  EXPECT_EQ(result.weight_lifetimes[0].finish_cycle, 6u);
}

TEST(SaPeStateOpportunityTest, RegionPartitionKeepsPrestartUnclassified) {
  using sa_pe_state::WaveInterval;
  using sa_pe_state::WaveOpcode;
  const std::vector<WaveInterval> intervals{
      WaveInterval{.issue_cycle = 0, .start_cycle = 2, .finish_cycle = 9,
                   .tile_m = 2, .tile_k = 3, .tile_n = 4,
                   .opcode = WaveOpcode::PRELOAD,
                   .weight_block_id = 9, .weight_block_known = true}};
  const auto wave = sa_pe_state::derive_wave_lifetime(
      intervals, 4, 4, 12, 7);
  const auto regions = sa_pe_state::derive_region_states(
      intervals, wave, 4, 4, 2, 2, 12);

  EXPECT_TRUE(wave.weight_lifetime_complete);
  EXPECT_TRUE(regions.state_accounting_exact);
  EXPECT_EQ(regions.region_count, 4u);
  EXPECT_EQ(regions.analytical_on_region_cycles, 11u);
  EXPECT_EQ(regions.unclassified_region_cycles, 4u);
  EXPECT_EQ(regions.analytical_w_on_region_cycles, 3u);
  EXPECT_EQ(regions.analytical_off_region_cycles, 30u);
  EXPECT_EQ(regions.analytical_on_region_cycles +
                regions.unclassified_region_cycles +
                regions.analytical_w_on_region_cycles +
                regions.analytical_off_region_cycles,
            regions.capacity_region_cycles);
}

TEST(SaPeStateOpportunityTest, UnknownWeightIdentityPreventsStateGuess) {
  using sa_pe_state::WaveInterval;
  using sa_pe_state::WaveOpcode;
  const std::vector<WaveInterval> intervals{
      WaveInterval{.issue_cycle = 0, .start_cycle = 0, .finish_cycle = 4,
                   .tile_m = 2, .tile_k = 2, .tile_n = 2,
                   .opcode = WaveOpcode::PRELOAD,
                   .weight_block_id = 0, .weight_block_known = false}};
  const auto wave = sa_pe_state::derive_wave_lifetime(
      intervals, 2, 2, 4, 4);
  const auto regions = sa_pe_state::derive_region_states(
      intervals, wave, 2, 2, 1, 1, 4);

  EXPECT_FALSE(wave.weight_lifetime_complete);
  EXPECT_EQ(wave.unknown_weight_block_intervals, 1u);
  EXPECT_EQ(wave.analytical_on_pe_cycles, 8u);
  EXPECT_EQ(wave.unclassified_pe_cycles, 8u);
  EXPECT_FALSE(regions.state_accounting_exact);
  EXPECT_EQ(regions.unclassified_region_cycles,
            regions.capacity_region_cycles);
}

TEST(SaPeStateOpportunityTest,
     ReGateFunctionalModelsPrefixAndDiagonalDrain) {
  using sa_pe_state::WaveInterval;
  using sa_pe_state::WaveOpcode;
  const auto result = sa_pe_state::derive_regate_functional(
      {WaveInterval{.issue_cycle = 0,
                    .start_cycle = 0,
                    .finish_cycle = 7,
                    .tile_m = 2,
                    .tile_k = 3,
                    .tile_n = 4,
                    .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 7,
                    .weight_block_known = true}},
      4, 4, 7);

  EXPECT_TRUE(result.control_schedule_valid);
  EXPECT_TRUE(result.control_event_counts_valid);
  EXPECT_TRUE(result.state_partition_exact_under_model);
  EXPECT_EQ(result.capacity_pe_cycles, 112u);
  EXPECT_EQ(result.on_pe_cycles, 24u);
  EXPECT_EQ(result.w_on_pe_cycles, 18u);
  EXPECT_EQ(result.off_pe_cycles, 70u);
  EXPECT_EQ(result.unclassified_pe_cycles, 0u);
  EXPECT_EQ(result.pe_on_assert_events, 6u);
  EXPECT_EQ(result.pe_on_deassert_events, 6u);
  ASSERT_EQ(result.control_classes.size(), 1u);
  EXPECT_EQ(result.control_classes[0].row_on_prefix_length, 2u);
  EXPECT_EQ(result.control_classes[0].col_on_prefix_length, 3u);
  EXPECT_EQ(result.control_classes[0].pe_on_wave_cycles, 7u);
}

TEST(SaPeStateOpportunityTest,
     ReGateFunctionalFailsClosedWithoutWeightIdentity) {
  using sa_pe_state::WaveInterval;
  using sa_pe_state::WaveOpcode;
  const auto result = sa_pe_state::derive_regate_functional(
      {WaveInterval{.issue_cycle = 0,
                    .start_cycle = 0,
                    .finish_cycle = 4,
                    .tile_m = 2,
                    .tile_k = 2,
                    .tile_n = 2,
                    .opcode = WaveOpcode::PRELOAD,
                    .weight_block_known = false}},
      2, 2, 4);

  EXPECT_TRUE(result.control_schedule_valid);
  EXPECT_TRUE(result.control_event_counts_valid);
  EXPECT_FALSE(result.state_partition_exact_under_model);
  EXPECT_EQ(result.unclassified_pe_cycles,
            result.capacity_pe_cycles);
  EXPECT_EQ(result.on_pe_cycles, 0u);
  EXPECT_EQ(result.w_on_pe_cycles, 0u);
  EXPECT_EQ(result.off_pe_cycles, 0u);
}

TEST(SaPeStateOpportunityTest,
     ReGateFunctionalKeepsIssueToStartUnclassified) {
  using sa_pe_state::WaveInterval;
  using sa_pe_state::WaveOpcode;
  const auto result = sa_pe_state::derive_regate_functional(
      {WaveInterval{.issue_cycle = 0,
                    .start_cycle = 2,
                    .finish_cycle = 6,
                    .tile_m = 2,
                    .tile_k = 2,
                    .tile_n = 2,
                    .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 4,
                    .weight_block_known = true}},
      2, 2, 6);

  EXPECT_TRUE(result.control_schedule_valid);
  EXPECT_TRUE(result.state_partition_exact_under_model);
  EXPECT_EQ(result.capacity_pe_cycles, 24u);
  EXPECT_EQ(result.on_pe_cycles, 8u);
  EXPECT_EQ(result.w_on_pe_cycles, 8u);
  EXPECT_EQ(result.off_pe_cycles, 0u);
  EXPECT_EQ(result.unclassified_pe_cycles, 8u);
}

TEST(SaPeStateOpportunityTest,
     ReGateFunctionalInvalidatesOverlappingControlEvents) {
  using sa_pe_state::WaveInterval;
  using sa_pe_state::WaveOpcode;
  const auto result = sa_pe_state::derive_regate_functional(
      {WaveInterval{.issue_cycle = 0,
                    .start_cycle = 0,
                    .finish_cycle = 4,
                    .tile_m = 1,
                    .tile_k = 1,
                    .tile_n = 4,
                    .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 3,
                    .weight_block_known = true},
       WaveInterval{.issue_cycle = 1,
                    .start_cycle = 2,
                    .finish_cycle = 6,
                    .tile_m = 1,
                    .tile_k = 1,
                    .tile_n = 4,
                    .opcode = WaveOpcode::GEMM,
                    .weight_block_id = 3,
                    .weight_block_known = true}},
      1, 1, 6);

  EXPECT_FALSE(result.control_schedule_valid);
  EXPECT_FALSE(result.control_event_counts_valid);
  EXPECT_EQ(result.overlapping_wave_intervals, 1u);
  EXPECT_EQ(result.pe_on_assert_events, 0u);
  EXPECT_EQ(result.pe_on_deassert_events, 0u);
  EXPECT_FALSE(result.state_partition_exact_under_model);
  EXPECT_EQ(result.unclassified_pe_cycles,
            result.capacity_pe_cycles);
}


TEST(SaPeStateOpportunityTest, TwoBankRetainsFootprintAfterWave) {
  using sa_pe_state::WaveInterval;
  using sa_pe_state::WaveOpcode;
  const auto result = sa_pe_state::derive_two_bank_regate(
      {WaveInterval{.issue_cycle = 0, .start_cycle = 2,
                    .finish_cycle = 6, .tile_m = 2, .tile_k = 2,
                    .tile_n = 2, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 7, .weight_block_known = true}},
      3, 3, 8);
  EXPECT_TRUE(result.validation_pass);
  EXPECT_EQ(result.capacity_pe_cycles, 72u);
  EXPECT_EQ(result.preload_bank_pe_cycles, 8u);
  EXPECT_EQ(result.on_pe_cycles, 8u);
  EXPECT_EQ(result.w_on_one_bank_pe_cycles, 24u);
  EXPECT_EQ(result.w_on_two_bank_pe_cycles, 0u);
  EXPECT_EQ(result.off_pe_cycles, 40u);
  EXPECT_EQ(result.unclassified_pe_cycles, 0u);
}

TEST(SaPeStateOpportunityTest,
     TwoBankCountsOverlapAndRepeatedExplicitPreload) {
  using sa_pe_state::WaveInterval;
  using sa_pe_state::WaveOpcode;
  const auto result = sa_pe_state::derive_two_bank_regate(
      {WaveInterval{.issue_cycle = 0, .start_cycle = 0,
                    .finish_cycle = 2, .tile_m = 1, .tile_k = 1,
                    .tile_n = 2, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 9, .weight_block_known = true},
       WaveInterval{.issue_cycle = 1, .start_cycle = 2,
                    .finish_cycle = 4, .tile_m = 1, .tile_k = 1,
                    .tile_n = 2, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 9, .weight_block_known = true}},
      1, 1, 6);
  EXPECT_TRUE(result.validation_pass);
  EXPECT_EQ(result.preload_events, 2u);
  EXPECT_EQ(result.preload_bank_pe_cycles, 1u);
  EXPECT_EQ(result.on_pe_cycles, 4u);
  EXPECT_EQ(result.on_one_bank_pe_cycles, 1u);
  EXPECT_EQ(result.on_two_bank_pe_cycles, 3u);
  EXPECT_EQ(result.w_on_two_bank_pe_cycles, 2u);
  EXPECT_EQ(result.unclassified_pe_cycles, 0u);
}

TEST(SaPeStateOpportunityTest, TwoBankFailsClosedOnThirdLiveWeight) {
  using sa_pe_state::WaveInterval;
  using sa_pe_state::WaveOpcode;
  const auto result = sa_pe_state::derive_two_bank_regate(
      {WaveInterval{.issue_cycle = 0, .start_cycle = 0,
                    .finish_cycle = 4, .tile_m = 1, .tile_k = 1,
                    .tile_n = 4, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 1, .weight_block_known = true},
       WaveInterval{.issue_cycle = 1, .start_cycle = 4,
                    .finish_cycle = 8, .tile_m = 1, .tile_k = 1,
                    .tile_n = 4, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 2, .weight_block_known = true},
       WaveInterval{.issue_cycle = 2, .start_cycle = 8,
                    .finish_cycle = 12, .tile_m = 1, .tile_k = 1,
                    .tile_n = 4, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 3, .weight_block_known = true}},
      1, 1, 12);
  EXPECT_FALSE(result.validation_pass);
  EXPECT_EQ(result.two_bank_conflicts, 1u);
  EXPECT_EQ(result.unclassified_pe_cycles, result.capacity_pe_cycles);
  EXPECT_EQ(result.on_pe_cycles, 0u);
}

TEST(SaPeStateOpportunityTest, TwoBankRequiresMatchingPreload) {
  using sa_pe_state::WaveInterval;
  using sa_pe_state::WaveOpcode;
  const auto result = sa_pe_state::derive_two_bank_regate(
      {WaveInterval{.issue_cycle = 0, .start_cycle = 0,
                    .finish_cycle = 2, .tile_m = 1, .tile_k = 1,
                    .tile_n = 2, .opcode = WaveOpcode::GEMM,
                    .weight_block_id = 5, .weight_block_known = true}},
      1, 1, 3);
  EXPECT_FALSE(result.validation_pass);
  EXPECT_EQ(result.gemm_without_weight_intervals, 1u);
  EXPECT_FALSE(result.timing_audit.available);
  EXPECT_EQ(result.unclassified_pe_cycles, 3u);
}


TEST(SaPeStateOpportunityTest, TwoBankMatchesOrdinaryGemmToResidentWeight) {
  using sa_pe_state::WaveInterval;
  using sa_pe_state::WaveOpcode;
  const auto result = sa_pe_state::derive_two_bank_regate(
      {WaveInterval{.issue_cycle = 0, .start_cycle = 0,
                    .finish_cycle = 2, .tile_m = 1, .tile_k = 1,
                    .tile_n = 2, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 5, .weight_block_known = true},
       WaveInterval{.issue_cycle = 1, .start_cycle = 2,
                    .finish_cycle = 4, .tile_m = 1, .tile_k = 1,
                    .tile_n = 2, .opcode = WaveOpcode::GEMM,
                    .weight_block_id = 5, .weight_block_known = true}},
      1, 1, 6);
  EXPECT_TRUE(result.validation_pass);
  EXPECT_EQ(result.preload_events, 1u);
  EXPECT_EQ(result.gemm_events, 1u);
  EXPECT_EQ(result.on_pe_cycles, 4u);
  EXPECT_EQ(result.w_on_one_bank_pe_cycles, 2u);
  EXPECT_EQ(result.unclassified_pe_cycles, 0u);
}

TEST(SaPeStateOpportunityTest, TwoBankFutureGemmProtectsOldBank) {
  using sa_pe_state::WaveInterval;
  using sa_pe_state::WaveOpcode;
  const auto result = sa_pe_state::derive_two_bank_regate(
      {WaveInterval{.issue_cycle = 0, .start_cycle = 0,
                    .finish_cycle = 2, .tile_m = 1, .tile_k = 1,
                    .tile_n = 2, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 1, .weight_block_known = true},
       WaveInterval{.issue_cycle = 1, .start_cycle = 2,
                    .finish_cycle = 4, .tile_m = 1, .tile_k = 1,
                    .tile_n = 2, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 2, .weight_block_known = true},
       WaveInterval{.issue_cycle = 2, .start_cycle = 4,
                    .finish_cycle = 6, .tile_m = 1, .tile_k = 1,
                    .tile_n = 2, .opcode = WaveOpcode::GEMM,
                    .weight_block_id = 1, .weight_block_known = true},
       WaveInterval{.issue_cycle = 3, .start_cycle = 6,
                    .finish_cycle = 8, .tile_m = 1, .tile_k = 1,
                    .tile_n = 2, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 3, .weight_block_known = true}},
      1, 1, 8);
  EXPECT_FALSE(result.validation_pass);
  EXPECT_EQ(result.two_bank_conflicts, 1u);
  EXPECT_EQ(result.unclassified_pe_cycles, result.capacity_pe_cycles);
  ASSERT_TRUE(result.timing_audit.available);
  EXPECT_EQ(result.timing_audit.matched_gemm_consumers, 1u);
  EXPECT_EQ(result.timing_audit.issue_to_whole_finish.global_over_two_cycles,
            1u);
  EXPECT_EQ(result.timing_audit.issue_to_pe_last_mac.global_over_two_cycles, 1u);
}

TEST(SaPeStateOpportunityTest, TwoBankRejectsComputeCollision) {
  using sa_pe_state::WaveInterval;
  using sa_pe_state::WaveOpcode;
  const auto result = sa_pe_state::derive_two_bank_regate(
      {WaveInterval{.issue_cycle = 0, .start_cycle = 0,
                    .finish_cycle = 3, .tile_m = 1, .tile_k = 1,
                    .tile_n = 3, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 1, .weight_block_known = true},
       WaveInterval{.issue_cycle = 1, .start_cycle = 2,
                    .finish_cycle = 5, .tile_m = 1, .tile_k = 1,
                    .tile_n = 3, .opcode = WaveOpcode::GEMM,
                    .weight_block_id = 1, .weight_block_known = true}},
      1, 1, 5);
  EXPECT_FALSE(result.validation_pass);
  EXPECT_EQ(result.compute_collisions, 1u);
  EXPECT_EQ(result.unclassified_pe_cycles, result.capacity_pe_cycles);
}

TEST(SaPeStateOpportunityTest, TwoBankFailsClosedOnIncompleteInterval) {
  using sa_pe_state::WaveInterval;
  using sa_pe_state::WaveOpcode;
  const auto result = sa_pe_state::derive_two_bank_regate(
      {WaveInterval{.issue_cycle = 0, .start_cycle = 0,
                    .finish_cycle = 4, .tile_m = 1, .tile_k = 1,
                    .tile_n = 4, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 1, .weight_block_known = true}},
      1, 1, 3);
  EXPECT_FALSE(result.validation_pass);
  EXPECT_EQ(result.incomplete_intervals, 1u);
  EXPECT_EQ(result.unclassified_pe_cycles, result.capacity_pe_cycles);
}

TEST(SaPeStateOpportunityTest, TwoBankTimingAuditSeparatesLoadReleaseAndFootprint) {
  using sa_pe_state::WaveInterval;
  using sa_pe_state::WaveOpcode;
  const auto result = sa_pe_state::derive_two_bank_regate(
      {WaveInterval{.issue_cycle = 0, .start_cycle = 5,
                    .finish_cycle = 13, .tile_m = 3, .tile_k = 3,
                    .tile_n = 4, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 1, .weight_block_known = true},
       WaveInterval{.issue_cycle = 4, .start_cycle = 9,
                    .finish_cycle = 17, .tile_m = 3, .tile_k = 3,
                    .tile_n = 4, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 2, .weight_block_known = true},
       WaveInterval{.issue_cycle = 11, .start_cycle = 13,
                    .finish_cycle = 21, .tile_m = 1, .tile_k = 1,
                    .tile_n = 4, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 3, .weight_block_known = true}},
      3, 3, 21);
  EXPECT_FALSE(result.validation_pass);
  EXPECT_EQ(result.two_bank_conflicts, 1u);
  EXPECT_EQ(result.unclassified_pe_cycles, result.capacity_pe_cycles);
  const auto& audit = result.timing_audit;
  ASSERT_TRUE(audit.available);
  EXPECT_EQ(audit.issue_to_whole_finish.global_peak_live_preloads, 3u);
  EXPECT_EQ(audit.issue_to_whole_finish.global_over_two_cycles, 2u);
  EXPECT_EQ(audit.issue_to_whole_finish.mapped_pe_over_two_pe_cycles, 2u);
  EXPECT_EQ(audit.issue_to_whole_finish.mapped_pes_ever_over_two, 1u);
  EXPECT_EQ(audit.start_to_whole_finish.global_over_two_cycles, 0u);
  EXPECT_EQ(audit.issue_to_pe_last_mac.global_over_two_cycles, 2u);
  EXPECT_EQ(audit.issue_to_pe_last_mac.mapped_pe_over_two_pe_cycles, 0u);
  EXPECT_EQ(audit.start_to_pe_last_mac.global_over_two_cycles, 0u);
  EXPECT_EQ(audit.pe_first_to_pe_last_mac.mapped_pe_over_two_pe_cycles, 0u);
  EXPECT_FALSE(audit.pe_first_to_pe_last_mac.global_applicable);
}

TEST(SaPeStateOpportunityTest, TwoBankTimingAuditUsesPeFirstMacAsIdealBound) {
  using sa_pe_state::WaveInterval;
  using sa_pe_state::WaveOpcode;
  const auto result = sa_pe_state::derive_two_bank_regate(
      {WaveInterval{.issue_cycle = 0, .start_cycle = 0,
                    .finish_cycle = 3, .tile_m = 2, .tile_k = 2,
                    .tile_n = 1, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 1, .weight_block_known = true},
       WaveInterval{.issue_cycle = 1, .start_cycle = 1,
                    .finish_cycle = 4, .tile_m = 2, .tile_k = 2,
                    .tile_n = 1, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 2, .weight_block_known = true},
       WaveInterval{.issue_cycle = 2, .start_cycle = 2,
                    .finish_cycle = 5, .tile_m = 2, .tile_k = 2,
                    .tile_n = 1, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 3, .weight_block_known = true}},
      2, 2, 5);
  EXPECT_FALSE(result.validation_pass);
  ASSERT_TRUE(result.timing_audit.available);
  EXPECT_EQ(
      result.timing_audit.start_to_pe_last_mac.mapped_pe_over_two_pe_cycles,
      1u);
  EXPECT_EQ(
      result.timing_audit.start_to_pe_last_mac.mapped_pes_ever_over_two,
      1u);
  EXPECT_EQ(
      result.timing_audit.pe_first_to_pe_last_mac
          .mapped_pe_over_two_pe_cycles,
      0u);
}

TEST(SaPeStateOpportunityTest,
     TwoBankTimingAuditExtendsLocalReleaseForMatchedFutureGemm) {
  using sa_pe_state::WaveInterval;
  using sa_pe_state::WaveOpcode;
  const auto result = sa_pe_state::derive_two_bank_regate(
      {WaveInterval{.issue_cycle = 0, .start_cycle = 0,
                    .finish_cycle = 2, .tile_m = 1, .tile_k = 1,
                    .tile_n = 2, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 1, .weight_block_known = true},
       WaveInterval{.issue_cycle = 1, .start_cycle = 2,
                    .finish_cycle = 8, .tile_m = 1, .tile_k = 1,
                    .tile_n = 2, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 2, .weight_block_known = true},
       WaveInterval{.issue_cycle = 2, .start_cycle = 4,
                    .finish_cycle = 6, .tile_m = 1, .tile_k = 1,
                    .tile_n = 2, .opcode = WaveOpcode::GEMM,
                    .weight_block_id = 1, .weight_block_known = true},
       WaveInterval{.issue_cycle = 3, .start_cycle = 6,
                    .finish_cycle = 8, .tile_m = 1, .tile_k = 1,
                    .tile_n = 2, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 3, .weight_block_known = true}},
      1, 1, 8);
  EXPECT_FALSE(result.validation_pass);
  ASSERT_TRUE(result.timing_audit.available);
  EXPECT_EQ(result.timing_audit.matched_gemm_consumers, 1u);
  // A's preload MAC ends at 2, but its matched GEMM extends the
  // PE-local release to 6. C's issue-time reservation overlaps at 3.
  EXPECT_EQ(
      result.timing_audit.issue_to_pe_last_mac.mapped_pe_over_two_pe_cycles,
      1u);
  // C's start-time reservation begins at the exact half-open release
  // boundary of A's matched GEMM, so that pair does not overlap.
  EXPECT_EQ(
      result.timing_audit.start_to_pe_last_mac.mapped_pe_over_two_pe_cycles,
      0u);
}

TEST(SaPeStateOpportunityTest,
     TwoBankTimingAuditTracksLaterGemmAcrossDifferentPeAndWholeEnds) {
  using sa_pe_state::WaveInterval;
  using sa_pe_state::WaveOpcode;
  std::vector<WaveInterval> with_gemm{
      WaveInterval{.issue_cycle = 0, .start_cycle = 5,
                   .finish_cycle = 13, .tile_m = 3, .tile_k = 3,
                   .tile_n = 4, .opcode = WaveOpcode::PRELOAD,
                   .weight_block_id = 1, .weight_block_known = true},
      WaveInterval{.issue_cycle = 4, .start_cycle = 9,
                   .finish_cycle = 21, .tile_m = 3, .tile_k = 3,
                   .tile_n = 8, .opcode = WaveOpcode::GEMM,
                   .weight_block_id = 1, .weight_block_known = true},
      WaveInterval{.issue_cycle = 8, .start_cycle = 17,
                   .finish_cycle = 25, .tile_m = 3, .tile_k = 3,
                   .tile_n = 4, .opcode = WaveOpcode::PRELOAD,
                   .weight_block_id = 2, .weight_block_known = true},
      WaveInterval{.issue_cycle = 13, .start_cycle = 21,
                   .finish_cycle = 29, .tile_m = 1, .tile_k = 1,
                   .tile_n = 4, .opcode = WaveOpcode::PRELOAD,
                   .weight_block_id = 3, .weight_block_known = true}};
  auto without_gemm = with_gemm;
  without_gemm.erase(without_gemm.begin() + 1);
  const auto extended =
      sa_pe_state::derive_two_bank_regate(with_gemm, 3, 3, 29);
  const auto baseline =
      sa_pe_state::derive_two_bank_regate(without_gemm, 3, 3, 29);
  ASSERT_TRUE(extended.timing_audit.available);
  ASSERT_TRUE(baseline.timing_audit.available);
  EXPECT_EQ(extended.timing_audit.matched_gemm_consumers, 1u);
  // A's top-left preload MAC ends at 9; the matching GEMM moves its
  // top-left last MAC to 17, while its whole instruction ends at 21.
  EXPECT_EQ(
      baseline.timing_audit.issue_to_pe_last_mac
          .mapped_pe_peak_live_preloads,
      2u);
  EXPECT_EQ(
      extended.timing_audit.issue_to_pe_last_mac
          .mapped_pe_peak_live_preloads,
      3u);
  EXPECT_EQ(
      extended.timing_audit.issue_to_whole_finish
          .mapped_pe_peak_live_preloads,
      3u);
  EXPECT_EQ(
      extended.timing_audit.issue_to_pe_last_mac
          .mapped_pe_over_two_pe_cycles,
      4u);
  auto at_local_release = with_gemm;
  at_local_release.back().issue_cycle = 17;
  const auto boundary = sa_pe_state::derive_two_bank_regate(
      at_local_release, 3, 3, 29);
  ASSERT_TRUE(boundary.timing_audit.available);
  EXPECT_EQ(
      boundary.timing_audit.issue_to_pe_last_mac
          .mapped_pe_peak_live_preloads,
      2u);
  EXPECT_EQ(
      boundary.timing_audit.issue_to_whole_finish
          .mapped_pe_peak_live_preloads,
      3u);
  EXPECT_EQ(
      boundary.timing_audit.issue_to_pe_last_mac
          .mapped_pe_over_two_pe_cycles,
      0u);
  EXPECT_EQ(boundary.timing_audit.issue_to_whole_finish
                .mapped_pe_over_two_pe_cycles, 4u);
}

TEST(SaPeStateOpportunityTest,
     RowwiseTimingCanAvoidAtomicIssueConflictButDoesNotProveHardware) {
  using sa_pe_state::RowWriteOrder;
  using sa_pe_state::RowWriteTiming;
  using sa_pe_state::WaveInterval;
  using sa_pe_state::WaveOpcode;
  const RowWriteTiming timing{1, 1, 1,
                              RowWriteOrder::ASCENDING_PE_ROW};
  const auto result = sa_pe_state::derive_two_bank_regate(
      {WaveInterval{.issue_cycle = 0, .start_cycle = 5,
                    .finish_cycle = 13, .tile_m = 3, .tile_k = 3,
                    .tile_n = 4, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 1, .weight_block_known = true},
       WaveInterval{.issue_cycle = 1, .start_cycle = 9,
                    .finish_cycle = 17, .tile_m = 3, .tile_k = 3,
                    .tile_n = 4, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 2, .weight_block_known = true},
       WaveInterval{.issue_cycle = 2, .start_cycle = 13,
                    .finish_cycle = 21, .tile_m = 3, .tile_k = 3,
                    .tile_n = 4, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 3, .weight_block_known = true}},
      3, 3, 21, &timing);
  EXPECT_FALSE(result.validation_pass);  // Existing atomic model unchanged.
  EXPECT_EQ(result.two_bank_conflicts, 1u);
  ASSERT_TRUE(result.rowwise_feasibility.available);
  EXPECT_TRUE(result.rowwise_feasibility.necessary_conditions_pass);
  EXPECT_EQ(result.rowwise_feasibility.mapped_pe_weight_write_events, 27u);
  EXPECT_EQ(result.rowwise_feasibility.mapped_pe_over_two_pe_cycles, 0u);
  EXPECT_EQ(result.rowwise_feasibility.row_ready_after_first_mac_events, 0u);
}

TEST(SaPeStateOpportunityTest, RowwiseTimingDiagnosesLateArrival) {
  using sa_pe_state::RowWriteOrder;
  using sa_pe_state::RowWriteTiming;
  using sa_pe_state::WaveInterval;
  using sa_pe_state::WaveOpcode;
  const RowWriteTiming timing{0, 1, 2,
                              RowWriteOrder::ASCENDING_PE_ROW};
  const auto result = sa_pe_state::derive_two_bank_regate(
      {WaveInterval{.issue_cycle = 0, .start_cycle = 1,
                    .finish_cycle = 3, .tile_m = 1, .tile_k = 1,
                    .tile_n = 2, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 1, .weight_block_known = true}},
      1, 1, 3, &timing);
  EXPECT_TRUE(result.validation_pass);
  EXPECT_FALSE(result.rowwise_feasibility.necessary_conditions_pass);
  EXPECT_EQ(result.rowwise_feasibility.row_ready_after_first_mac_events, 1u);
  EXPECT_EQ(result.rowwise_feasibility.max_row_ready_lateness_cycles, 2u);
  EXPECT_EQ(result.rowwise_feasibility.max_local_first_mac_delay_cycles, 2u);
  EXPECT_TRUE(result.rowwise_state_accounting.available);
  EXPECT_FALSE(result.rowwise_state_accounting.validation_pass);
  EXPECT_EQ(result.rowwise_state_accounting.unclassified_pe_cycles,
            result.rowwise_state_accounting.capacity_pe_cycles);
  EXPECT_EQ(result.rowwise_state_accounting.on_pe_cycles, 0u);
}

TEST(SaPeStateOpportunityTest,
     RowwisePreloadAuditReportsEveryPreloadAndExactRowStalls) {
  using sa_pe_state::RowWriteOrder;
  using sa_pe_state::RowWriteTiming;
  using sa_pe_state::WaveInterval;
  using sa_pe_state::WaveOpcode;
  const RowWriteTiming timing{1, 1, 3,
                              RowWriteOrder::ASCENDING_PE_ROW};
  const auto result = sa_pe_state::derive_two_bank_regate(
      {WaveInterval{.issue_cycle = 0, .start_cycle = 3,
                    .finish_cycle = 7, .tile_m = 2, .tile_k = 2,
                    .tile_n = 2, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 11, .weight_block_known = true},
       WaveInterval{.issue_cycle = 1, .start_cycle = 7,
                    .finish_cycle = 9, .tile_m = 1, .tile_k = 1,
                    .tile_n = 2, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 12, .weight_block_known = true}},
      2, 2, 9, &timing);
  const auto& audit = result.rowwise_feasibility;
  ASSERT_EQ(audit.preload_audit.size(), 2u);
  EXPECT_EQ(audit.audited_preloads, 2u);
  EXPECT_EQ(audit.audited_preload_rows, 3u);
  EXPECT_EQ(audit.no_added_cycle_preloads, 0u);
  EXPECT_EQ(audit.conflicting_preloads, 2u);
  EXPECT_EQ(audit.conflicting_preload_rows, 3u);
  EXPECT_EQ(audit.sum_row_required_stall_cycles, 6u);
  EXPECT_EQ(audit.total_required_stall_cycles, 4u);
  EXPECT_EQ(audit.max_required_stall_cycles, 2u);

  const auto& first = audit.preload_audit[0];
  EXPECT_EQ(first.preload_index, 0u);
  EXPECT_EQ(first.weight_block_id, 11u);
  ASSERT_TRUE(first.planned_load_begin_available);
  EXPECT_EQ(first.planned_load_begin_cycle, 2u);
  ASSERT_EQ(first.rows.size(), 2u);
  EXPECT_EQ(first.rows[0].load_begin_cycle, 2u);
  EXPECT_EQ(first.rows[0].bank_ready_cycle, 5u);
  EXPECT_EQ(first.rows[0].first_use_cycle, 3u);
  EXPECT_EQ(first.rows[0].required_stall_cycles, 2u);
  EXPECT_FALSE(first.rows[0].no_added_cycle);
  EXPECT_EQ(first.rows[1].load_begin_cycle, 3u);
  EXPECT_EQ(first.rows[1].bank_ready_cycle, 6u);
  EXPECT_EQ(first.rows[1].first_use_cycle, 4u);
  EXPECT_EQ(first.rows[1].required_stall_cycles, 2u);

  const auto& second = audit.preload_audit[1];
  ASSERT_EQ(second.rows.size(), 1u);
  EXPECT_EQ(second.rows[0].load_begin_cycle, 6u);
  EXPECT_EQ(second.rows[0].bank_ready_cycle, 9u);
  EXPECT_EQ(second.rows[0].first_use_cycle, 7u);
  EXPECT_EQ(second.rows[0].required_stall_cycles, 2u);
}

TEST(SaPeStateOpportunityTest,
     WholeBankPreloadFeasibilityUsesCausalRowStreamFill) {
  using sa_pe_state::RowWriteOrder;
  using sa_pe_state::RowWriteTiming;
  using sa_pe_state::WaveInterval;
  using sa_pe_state::WaveOpcode;
  const RowWriteTiming timing{1, 1, 1,
                              RowWriteOrder::ASCENDING_PE_ROW};
  const auto result = sa_pe_state::derive_two_bank_regate(
      {WaveInterval{.issue_cycle = 0, .start_cycle = 5,
                    .finish_cycle = 10, .tile_m = 3, .tile_k = 3,
                    .tile_n = 1, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 31, .weight_block_known = true},
       WaveInterval{.issue_cycle = 9, .start_cycle = 10,
                    .finish_cycle = 15, .tile_m = 3, .tile_k = 3,
                    .tile_n = 1, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 32, .weight_block_known = true}},
      3, 3, 15, &timing);
  const auto& audit = result.preload_feasibility_audit;
  ASSERT_TRUE(audit.available);
  ASSERT_EQ(audit.events.size(), 2u);
  EXPECT_EQ(audit.audited_preloads, 2u);
  EXPECT_EQ(audit.no_added_cycle_preloads, 1u);
  EXPECT_EQ(audit.conflicting_preloads, 1u);
  EXPECT_EQ(audit.total_required_stall_cycles, 2u);
  EXPECT_EQ(audit.max_required_stall_cycles, 2u);

  const auto& first = audit.events[0];
  EXPECT_EQ(first.rows, 3u);
  EXPECT_EQ(first.fill_span_cycles, 3u);
  ASSERT_TRUE(first.planned_load_begin_available);
  EXPECT_EQ(first.planned_load_begin_cycle, 2u);
  EXPECT_EQ(first.load_begin_cycle, 2u);
  EXPECT_EQ(first.bank_ready_cycle, 5u);
  EXPECT_EQ(first.first_use_cycle, 5u);
  EXPECT_EQ(first.required_stall_cycles, 0u);
  EXPECT_TRUE(first.no_added_cycle);

  const auto& second = audit.events[1];
  EXPECT_EQ(second.planned_load_begin_cycle, 7u);
  EXPECT_EQ(second.load_begin_cycle, 9u);
  EXPECT_EQ(second.bank_ready_cycle, 12u);
  EXPECT_EQ(second.first_use_cycle, 10u);
  EXPECT_EQ(second.required_stall_cycles, 2u);
  EXPECT_FALSE(second.no_added_cycle);
}

TEST(SaPeStateOpportunityTest,
     RowwisePreloadAuditMarksNoAddedCycleAndPreIssueConflict) {
  using sa_pe_state::RowWriteOrder;
  using sa_pe_state::RowWriteTiming;
  using sa_pe_state::WaveInterval;
  using sa_pe_state::WaveOpcode;
  const WaveInterval preload{
      .issue_cycle = 2, .start_cycle = 4, .finish_cycle = 6,
      .tile_m = 1, .tile_k = 1, .tile_n = 2,
      .opcode = WaveOpcode::PRELOAD,
      .weight_block_id = 21, .weight_block_known = true};

  const RowWriteTiming feasible{1, 1, 1,
                                RowWriteOrder::ASCENDING_PE_ROW};
  const auto feasible_result = sa_pe_state::derive_two_bank_regate(
      {preload}, 1, 1, 6, &feasible);
  const auto& feasible_audit =
      feasible_result.rowwise_feasibility.preload_audit[0];
  EXPECT_TRUE(feasible_audit.no_added_cycle);
  EXPECT_EQ(feasible_audit.rows[0].required_stall_cycles, 0u);
  EXPECT_EQ(feasible_result.rowwise_feasibility.no_added_cycle_preloads, 1u);

  const RowWriteTiming preissue{3, 1, 1,
                                RowWriteOrder::ASCENDING_PE_ROW};
  const auto preissue_result = sa_pe_state::derive_two_bank_regate(
      {preload}, 1, 1, 6, &preissue);
  const auto& preissue_audit =
      preissue_result.rowwise_feasibility.preload_audit[0];
  EXPECT_TRUE(preissue_audit.no_added_cycle);
  ASSERT_TRUE(preissue_audit.rows[0].planned_load_begin_available);
  EXPECT_TRUE(preissue_audit.rows[0].load_before_issue);
  EXPECT_EQ(preissue_audit.rows[0].load_begin_cycle, 2u);
  EXPECT_EQ(preissue_audit.rows[0].bank_ready_cycle, 3u);
  EXPECT_EQ(preissue_audit.rows[0].required_stall_cycles, 0u);
  EXPECT_EQ(preissue_result.rowwise_feasibility.conflicting_preload_rows, 0u);
  EXPECT_EQ(preissue_result.rowwise_feasibility.schedule_diagnostic_rows, 1u);
  EXPECT_EQ(preissue_result.rowwise_feasibility.schedule_diagnostic_preloads,
            1u);

  auto fused_at_issue = preload;
  fused_at_issue.start_cycle = 2;
  fused_at_issue.finish_cycle = 4;
  const RowWriteTiming one_cycle_early{1, 1, 1,
                                       RowWriteOrder::ASCENDING_PE_ROW};
  const auto causal_result = sa_pe_state::derive_two_bank_regate(
      {fused_at_issue}, 1, 1, 4, &one_cycle_early);
  const auto& causal_row =
      causal_result.rowwise_feasibility.preload_audit[0].rows[0];
  EXPECT_EQ(causal_row.planned_load_begin_cycle, 1u);
  EXPECT_EQ(causal_row.load_begin_cycle, 2u);  // Cannot precede issue.
  EXPECT_EQ(causal_row.bank_ready_cycle, 3u);
  EXPECT_EQ(causal_row.first_use_cycle, 2u);
  EXPECT_EQ(causal_row.required_stall_cycles, 1u);
}

TEST(SaPeStateOpportunityTest,
     RowwiseTimingProtectsMatchedFutureGemmAndReportsLocalWait) {
  using sa_pe_state::RowWriteOrder;
  using sa_pe_state::RowWriteTiming;
  using sa_pe_state::WaveInterval;
  using sa_pe_state::WaveOpcode;
  const RowWriteTiming timing{1, 1, 1,
                              RowWriteOrder::ASCENDING_PE_ROW};
  const auto result = sa_pe_state::derive_two_bank_regate(
      {WaveInterval{.issue_cycle = 0, .start_cycle = 2,
                    .finish_cycle = 4, .tile_m = 1, .tile_k = 1,
                    .tile_n = 2, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 1, .weight_block_known = true},
       WaveInterval{.issue_cycle = 1, .start_cycle = 4,
                    .finish_cycle = 6, .tile_m = 1, .tile_k = 1,
                    .tile_n = 2, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 2, .weight_block_known = true},
       WaveInterval{.issue_cycle = 2, .start_cycle = 6,
                    .finish_cycle = 8, .tile_m = 1, .tile_k = 1,
                    .tile_n = 2, .opcode = WaveOpcode::GEMM,
                    .weight_block_id = 1, .weight_block_known = true},
       WaveInterval{.issue_cycle = 3, .start_cycle = 8,
                    .finish_cycle = 10, .tile_m = 1, .tile_k = 1,
                    .tile_n = 2, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 3, .weight_block_known = true},
       WaveInterval{.issue_cycle = 4, .start_cycle = 10,
                    .finish_cycle = 12, .tile_m = 1, .tile_k = 1,
                    .tile_n = 2, .opcode = WaveOpcode::GEMM,
                    .weight_block_id = 2, .weight_block_known = true}},
      1, 1, 12, &timing);
  ASSERT_TRUE(result.rowwise_feasibility.available);
  EXPECT_FALSE(result.rowwise_feasibility.necessary_conditions_pass);
  EXPECT_EQ(result.rowwise_feasibility.mapped_pe_over_two_pe_cycles, 1u);
  EXPECT_EQ(result.rowwise_feasibility.locally_blocked_pe_weight_write_events, 1u);
  EXPECT_EQ(result.rowwise_feasibility.max_local_bank_wait_cycles, 1u);
  EXPECT_EQ(result.rowwise_feasibility.max_local_first_mac_delay_cycles, 1u);
  ASSERT_EQ(result.rowwise_feasibility.preload_audit.size(), 3u);
  const auto& third = result.rowwise_feasibility.preload_audit[2];
  EXPECT_FALSE(third.no_added_cycle);
  EXPECT_EQ(third.required_stall_cycles, 1u);
  ASSERT_EQ(third.bank_conflicts.size(), 1u);
  EXPECT_EQ(third.bank_conflicts[0].pe_row, 0u);
  EXPECT_EQ(third.bank_conflicts[0].pe_column, 0u);
  EXPECT_EQ(third.bank_conflicts[0].planned_load_begin_cycle, 7u);
  EXPECT_EQ(third.bank_conflicts[0].earliest_bank_release_cycle, 8u);
  EXPECT_EQ(third.bank_conflicts[0].bank_ready_cycle, 9u);
  EXPECT_EQ(third.bank_conflicts[0].first_use_cycle, 8u);
  EXPECT_EQ(third.bank_conflicts[0].required_stall_cycles, 1u);
  EXPECT_TRUE(result.rowwise_state_accounting.available);
  EXPECT_FALSE(result.rowwise_state_accounting.validation_pass);
  EXPECT_EQ(result.rowwise_state_accounting.unclassified_pe_cycles,
            result.rowwise_state_accounting.capacity_pe_cycles);
}

TEST(SaPeStateOpportunityTest, RowwiseTimingRejectsPreIssueWrite) {
  using sa_pe_state::RowWriteOrder;
  using sa_pe_state::RowWriteTiming;
  using sa_pe_state::WaveInterval;
  using sa_pe_state::WaveOpcode;
  const RowWriteTiming timing{2, 1, 1,
                              RowWriteOrder::ASCENDING_PE_ROW};
  const auto result = sa_pe_state::derive_two_bank_regate(
      {WaveInterval{.issue_cycle = 2, .start_cycle = 3,
                    .finish_cycle = 5, .tile_m = 1, .tile_k = 1,
                    .tile_n = 2, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 1, .weight_block_known = true}},
      1, 1, 5, &timing);
  EXPECT_TRUE(result.validation_pass);
  EXPECT_FALSE(result.rowwise_feasibility.necessary_conditions_pass);
  EXPECT_EQ(result.rowwise_feasibility.row_write_before_issue_events, 1u);
  EXPECT_FALSE(result.rowwise_feasibility.secondary_counts_complete);
  EXPECT_TRUE(result.rowwise_state_accounting.available);
  EXPECT_FALSE(result.rowwise_state_accounting.validation_pass);
  EXPECT_EQ(result.rowwise_state_accounting.unclassified_pe_cycles,
            result.rowwise_state_accounting.capacity_pe_cycles);
}

TEST(SaPeStateOpportunityTest, RowwiseTimingHasNoImplicitDefaults) {
  using sa_pe_state::RowWriteOrder;
  using sa_pe_state::RowWriteTiming;
  using sa_pe_state::WaveInterval;
  using sa_pe_state::WaveOpcode;
  const std::vector<WaveInterval> intervals{
      WaveInterval{.issue_cycle = 0, .start_cycle = 1,
                   .finish_cycle = 3, .tile_m = 1, .tile_k = 1,
                   .tile_n = 2, .opcode = WaveOpcode::PRELOAD,
                   .weight_block_id = 1, .weight_block_known = true}};
  const auto without_timing =
      sa_pe_state::derive_two_bank_regate(intervals, 1, 1, 3);
  EXPECT_TRUE(without_timing.validation_pass);
  EXPECT_FALSE(without_timing.rowwise_feasibility.available);
  const RowWriteTiming invalid{0, 0, 1,
                               RowWriteOrder::ASCENDING_PE_ROW};
  EXPECT_THROW(sa_pe_state::derive_two_bank_regate(
                   intervals, 1, 1, 3, &invalid), std::invalid_argument);
}

TEST(SaPeStateOpportunityTest,
     RowwiseTimingCountsPreIssueAfterEachRowStagger) {
  using sa_pe_state::RowWriteOrder;
  using sa_pe_state::RowWriteTiming;
  using sa_pe_state::WaveInterval;
  using sa_pe_state::WaveOpcode;
  const RowWriteTiming timing{2, 2, 1,
                              RowWriteOrder::ASCENDING_PE_ROW};
  const auto result = sa_pe_state::derive_two_bank_regate(
      {WaveInterval{.issue_cycle = 0, .start_cycle = 1,
                    .finish_cycle = 4, .tile_m = 3, .tile_k = 1,
                    .tile_n = 1, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 1, .weight_block_known = true}},
      3, 1, 4, &timing);
  ASSERT_TRUE(result.rowwise_feasibility.available);
  EXPECT_EQ(result.rowwise_feasibility.mapped_pe_weight_write_events, 3u);
  EXPECT_EQ(result.rowwise_feasibility.row_write_before_issue_events, 1u);
  EXPECT_FALSE(result.rowwise_feasibility.secondary_counts_complete);
  EXPECT_FALSE(result.rowwise_feasibility.necessary_conditions_pass);
}

TEST(SaPeStateOpportunityTest,
     RowwiseStateSeparatesWriteWaveResidencyAndUnmappedOff) {
  using sa_pe_state::RowWriteOrder;
  using sa_pe_state::RowWriteTiming;
  using sa_pe_state::WaveInterval;
  using sa_pe_state::WaveOpcode;
  const RowWriteTiming timing{1, 1, 1,
                              RowWriteOrder::ASCENDING_PE_ROW};
  const auto result = sa_pe_state::derive_two_bank_regate(
      {WaveInterval{.issue_cycle = 0, .start_cycle = 1,
                    .finish_cycle = 3, .tile_m = 1, .tile_k = 1,
                    .tile_n = 2, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 1, .weight_block_known = true}},
      2, 2, 4, &timing);
  const auto& state = result.rowwise_state_accounting;
  ASSERT_TRUE(state.available);
  ASSERT_TRUE(state.validation_pass);
  EXPECT_EQ(state.capacity_pe_cycles, 16u);
  EXPECT_EQ(state.on_pe_cycles, 2u);
  EXPECT_EQ(state.on_preload_wave_pe_cycles, 2u);
  EXPECT_EQ(state.on_gemm_wave_pe_cycles, 0u);
  EXPECT_EQ(state.on_one_bank_pe_cycles, 2u);
  EXPECT_EQ(state.w_on_one_bank_pe_cycles, 2u);
  EXPECT_EQ(state.w_on_two_bank_pe_cycles, 0u);
  EXPECT_EQ(state.off_pe_cycles, 12u);
  EXPECT_EQ(state.unclassified_pe_cycles, 0u);
  EXPECT_EQ(state.preload_write_pe_cycles, 1u);
  EXPECT_EQ(state.preload_write_while_on_pe_cycles, 0u);
}

TEST(SaPeStateOpportunityTest,
     RowwiseStateSplitsOneAndTwoLiveBanksDuringCompute) {
  using sa_pe_state::RowWriteOrder;
  using sa_pe_state::RowWriteTiming;
  using sa_pe_state::WaveInterval;
  using sa_pe_state::WaveOpcode;
  const RowWriteTiming timing{1, 1, 1,
                              RowWriteOrder::ASCENDING_PE_ROW};
  const auto result = sa_pe_state::derive_two_bank_regate(
      {WaveInterval{.issue_cycle = 0, .start_cycle = 1,
                    .finish_cycle = 3, .tile_m = 1, .tile_k = 1,
                    .tile_n = 2, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 1, .weight_block_known = true},
       WaveInterval{.issue_cycle = 2, .start_cycle = 3,
                    .finish_cycle = 5, .tile_m = 1, .tile_k = 1,
                    .tile_n = 2, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 2, .weight_block_known = true}},
      1, 1, 6, &timing);
  const auto& state = result.rowwise_state_accounting;
  ASSERT_TRUE(state.validation_pass);
  EXPECT_EQ(state.on_one_bank_pe_cycles, 1u);
  EXPECT_EQ(state.on_two_bank_pe_cycles, 3u);
  EXPECT_EQ(state.w_on_one_bank_pe_cycles, 1u);
  EXPECT_EQ(state.w_on_two_bank_pe_cycles, 1u);
  EXPECT_EQ(state.off_pe_cycles, 0u);
  EXPECT_EQ(state.preload_write_pe_cycles, 2u);
  EXPECT_EQ(state.preload_write_while_on_pe_cycles, 1u);
}

TEST(SaPeStateOpportunityTest,
     RowwiseStateAllowsHalfOpenReplacementAtLocalRelease) {
  using sa_pe_state::RowWriteOrder;
  using sa_pe_state::RowWriteTiming;
  using sa_pe_state::WaveInterval;
  using sa_pe_state::WaveOpcode;
  const RowWriteTiming timing{1, 1, 1,
                              RowWriteOrder::ASCENDING_PE_ROW};
  const auto result = sa_pe_state::derive_two_bank_regate(
      {WaveInterval{.issue_cycle = 0, .start_cycle = 1,
                    .finish_cycle = 2, .tile_m = 1, .tile_k = 1,
                    .tile_n = 1, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 1, .weight_block_known = true},
       WaveInterval{.issue_cycle = 1, .start_cycle = 2,
                    .finish_cycle = 3, .tile_m = 1, .tile_k = 1,
                    .tile_n = 1, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 2, .weight_block_known = true},
       WaveInterval{.issue_cycle = 2, .start_cycle = 3,
                    .finish_cycle = 4, .tile_m = 1, .tile_k = 1,
                    .tile_n = 1, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 3, .weight_block_known = true}},
      1, 1, 5, &timing);
  const auto& state = result.rowwise_state_accounting;
  ASSERT_TRUE(state.validation_pass);
  EXPECT_EQ(result.rowwise_feasibility.mapped_pe_over_two_pe_cycles, 0u);
  EXPECT_EQ(state.on_two_bank_pe_cycles, 3u);
  EXPECT_EQ(state.on_one_bank_pe_cycles, 0u);
  EXPECT_EQ(state.w_on_one_bank_pe_cycles, 1u);
  EXPECT_EQ(state.w_on_two_bank_pe_cycles, 1u);
}

TEST(SaPeStateOpportunityTest,
     RowwiseStateMatchedFutureGemmExtendsSafeReplacement) {
  using sa_pe_state::RowWriteOrder;
  using sa_pe_state::RowWriteTiming;
  using sa_pe_state::WaveInterval;
  using sa_pe_state::WaveOpcode;
  const RowWriteTiming timing{1, 1, 1,
                              RowWriteOrder::ASCENDING_PE_ROW};
  const auto result = sa_pe_state::derive_two_bank_regate(
      {WaveInterval{.issue_cycle = 0, .start_cycle = 1,
                    .finish_cycle = 2, .tile_m = 1, .tile_k = 1,
                    .tile_n = 1, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 1, .weight_block_known = true},
       WaveInterval{.issue_cycle = 1, .start_cycle = 2,
                    .finish_cycle = 3, .tile_m = 1, .tile_k = 1,
                    .tile_n = 1, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 2, .weight_block_known = true},
       WaveInterval{.issue_cycle = 2, .start_cycle = 3,
                    .finish_cycle = 4, .tile_m = 1, .tile_k = 1,
                    .tile_n = 1, .opcode = WaveOpcode::GEMM,
                    .weight_block_id = 1, .weight_block_known = true},
       WaveInterval{.issue_cycle = 3, .start_cycle = 4,
                    .finish_cycle = 5, .tile_m = 1, .tile_k = 1,
                    .tile_n = 1, .opcode = WaveOpcode::PRELOAD,
                    .weight_block_id = 3, .weight_block_known = true}},
      1, 1, 6, &timing);
  const auto& state = result.rowwise_state_accounting;
  ASSERT_TRUE(state.validation_pass);
  EXPECT_EQ(state.on_preload_wave_pe_cycles, 3u);
  EXPECT_EQ(state.on_gemm_wave_pe_cycles, 1u);
  EXPECT_EQ(state.unclassified_pe_cycles, 0u);
}

TEST(SaPeStateOpportunityTest, RowwiseStateEmptyTraceIsExactlyAllOff) {
  using sa_pe_state::RowWriteOrder;
  using sa_pe_state::RowWriteTiming;
  const RowWriteTiming timing{0, 1, 1,
                              RowWriteOrder::ASCENDING_PE_ROW};
  const auto result = sa_pe_state::derive_two_bank_regate(
      {}, 2, 3, 4, &timing);
  const auto& state = result.rowwise_state_accounting;
  ASSERT_TRUE(state.available);
  ASSERT_TRUE(state.validation_pass);
  EXPECT_EQ(state.capacity_pe_cycles, 24u);
  EXPECT_EQ(state.off_pe_cycles, 24u);
  EXPECT_EQ(state.unclassified_pe_cycles, 0u);
}

TEST(SaPeStateOpportunityTest,
     UpdatedShadowUsesOwnTimelineAndExactFourStatePartition) {
  const std::vector<sa_pe_state::UpdatedShadowInterval> intervals = {
      {.stream_id = 7,
       .issue_cycle = 0,
       .simulator_start_cycle = 0,
       .simulator_finish_cycle = 2,
       .tile_m = 2,
       .tile_k = 2,
       .tile_n = 1,
       .opcode = sa_pe_state::WaveOpcode::PRELOAD,
       .preload_sequence = 1},
      {.stream_id = 7,
       .issue_cycle = 1,
       .simulator_start_cycle = 2,
       .simulator_finish_cycle = 4,
       .tile_m = 2,
       .tile_k = 2,
       .tile_n = 1,
       .opcode = sa_pe_state::WaveOpcode::GEMM,
       .preload_sequence = 1}};

  const auto result = sa_pe_state::derive_updated_two_bank_shadow(
      intervals, 2, 2, 12);
  ASSERT_TRUE(result.validation_pass);
  EXPECT_TRUE(result.preload_first_valid);
  EXPECT_TRUE(result.bank_assignment_valid);
  EXPECT_TRUE(result.state_partition_exact);
  EXPECT_EQ(result.analytical_horizon_cycles, 12U);
  EXPECT_EQ(result.capacity_pe_cycles, 48U);
  EXPECT_EQ(result.loading_pe_cycles, 4U);
  EXPECT_EQ(result.on_pe_cycles, 8U);
  EXPECT_EQ(result.w_on_pe_cycles, 18U);
  EXPECT_EQ(result.off_pe_cycles, 18U);
  EXPECT_EQ(result.unclassified_pe_cycles, 0U);
  ASSERT_EQ(result.events.size(), 2U);
  EXPECT_EQ(result.events[0].loading_start_cycle, 0U);
  EXPECT_EQ(result.events[0].loading_finish_cycle, 2U);
  EXPECT_EQ(result.events[0].compute_start_cycle, 2U);
  EXPECT_EQ(result.events[0].compute_finish_cycle, 5U);
  EXPECT_EQ(result.events[1].compute_start_cycle, 5U);
  EXPECT_EQ(result.events[1].compute_finish_cycle, 8U);
  EXPECT_EQ(result.simulator_trace_makespan_cycles, 4U);
  EXPECT_EQ(result.analytical_trace_makespan_cycles, 8U);
  EXPECT_EQ(result.trace_makespan_extension_cycles, 4U);
}

TEST(SaPeStateOpportunityTest,
     UpdatedShadowUsesFreeBankForConsecutiveFusedPreloads) {
  const std::vector<sa_pe_state::UpdatedShadowInterval> intervals = {
      {.stream_id = 1, .issue_cycle = 0, .simulator_start_cycle = 0,
       .simulator_finish_cycle = 1, .tile_m = 1, .tile_k = 1,
       .tile_n = 1, .opcode = sa_pe_state::WaveOpcode::PRELOAD,
       .preload_sequence = 1},
      {.stream_id = 2, .issue_cycle = 1, .simulator_start_cycle = 1,
       .simulator_finish_cycle = 2, .tile_m = 1, .tile_k = 1,
       .tile_n = 1, .opcode = sa_pe_state::WaveOpcode::PRELOAD,
       .preload_sequence = 1}};
  const auto result = sa_pe_state::derive_updated_two_bank_shadow(
      intervals, 1, 1, 4);
  ASSERT_TRUE(result.validation_pass);
  ASSERT_EQ(result.events.size(), 2U);
  EXPECT_EQ(result.events[0].bank, 0);
  EXPECT_EQ(result.events[0].loading_start_cycle, 0U);
  EXPECT_EQ(result.events[0].compute_start_cycle, 1U);
  EXPECT_EQ(result.events[0].compute_finish_cycle, 2U);
  EXPECT_EQ(result.events[1].bank, 1);
  EXPECT_EQ(result.events[1].loading_start_cycle, 1U);
  EXPECT_EQ(result.events[1].loading_finish_cycle, 2U);
  EXPECT_EQ(result.events[1].compute_start_cycle, 2U);
  EXPECT_EQ(result.events[1].compute_finish_cycle, 3U);
  EXPECT_EQ(result.preload_write_while_on_pe_cycles, 1U);
  EXPECT_EQ(result.on_pe_cycles + result.w_on_pe_cycles +
                result.loading_pe_cycles + result.off_pe_cycles,
            result.capacity_pe_cycles);
  EXPECT_EQ(result.sum_event_finish_lateness_cycles, 2U);
  EXPECT_EQ(result.trace_makespan_extension_cycles, 1U);
}

TEST(SaPeStateOpportunityTest, UpdatedShadowRejectsMalformedSimulatorTimes) {
  const sa_pe_state::UpdatedShadowInterval valid = {
      .stream_id = 1, .issue_cycle = 1, .simulator_start_cycle = 2,
      .simulator_finish_cycle = 3, .tile_m = 1, .tile_k = 1,
      .tile_n = 1, .opcode = sa_pe_state::WaveOpcode::PRELOAD,
      .preload_sequence = 1};
  for (int malformed_case = 0; malformed_case < 3; ++malformed_case) {
    auto malformed = valid;
    if (malformed_case == 0) malformed.issue_cycle = 3;
    if (malformed_case == 1) malformed.simulator_start_cycle = 4;
    if (malformed_case == 2) malformed.simulator_finish_cycle = 5;
    const auto result = sa_pe_state::derive_updated_two_bank_shadow(
        {malformed}, 1, 1, 4);
    EXPECT_FALSE(result.validation_pass);
    EXPECT_EQ(result.invalid_intervals, 1U);
    EXPECT_EQ(result.unclassified_pe_cycles, 4U);
    EXPECT_EQ(result.off_pe_cycles, 0U);
  }
}

TEST(SaPeStateOpportunityTest,
     UpdatedShadowAssignsTwoOverlappingPreloadCopiesWithoutWeightIds) {
  const std::vector<sa_pe_state::UpdatedShadowInterval> intervals = {
      {.stream_id = 1, .issue_cycle = 0, .simulator_finish_cycle = 1,
       .tile_m = 1, .tile_k = 1,
       .tile_n = 1, .opcode = sa_pe_state::WaveOpcode::PRELOAD,
       .preload_sequence = 1},
      {.stream_id = 2, .issue_cycle = 1, .simulator_start_cycle = 1,
       .simulator_finish_cycle = 2, .tile_m = 1, .tile_k = 1,
       .tile_n = 1, .opcode = sa_pe_state::WaveOpcode::PRELOAD,
       .preload_sequence = 1},
      {.stream_id = 1, .issue_cycle = 2, .simulator_start_cycle = 2,
       .simulator_finish_cycle = 3, .tile_m = 1, .tile_k = 1,
       .tile_n = 1, .opcode = sa_pe_state::WaveOpcode::GEMM,
       .preload_sequence = 1},
      {.stream_id = 2, .issue_cycle = 3, .simulator_start_cycle = 3,
       .simulator_finish_cycle = 4, .tile_m = 1, .tile_k = 1,
       .tile_n = 1, .opcode = sa_pe_state::WaveOpcode::GEMM,
       .preload_sequence = 1}};
  const auto result = sa_pe_state::derive_updated_two_bank_shadow(
      intervals, 1, 1, 8);
  ASSERT_TRUE(result.validation_pass);
  ASSERT_EQ(result.events.size(), 4U);
  EXPECT_EQ(result.events[0].bank, 0);
  EXPECT_EQ(result.events[1].bank, 1);
  EXPECT_EQ(result.events[2].bank, 0);
  EXPECT_EQ(result.events[3].bank, 1);
}

TEST(SaPeStateOpportunityTest, UpdatedShadowFailsClosedWithoutPreload) {
  const std::vector<sa_pe_state::UpdatedShadowInterval> intervals = {
      {.stream_id = 1, .issue_cycle = 0, .simulator_finish_cycle = 1,
       .tile_m = 1, .tile_k = 1,
       .tile_n = 1, .opcode = sa_pe_state::WaveOpcode::GEMM,
       .preload_sequence = 0}};
  const auto result = sa_pe_state::derive_updated_two_bank_shadow(
      intervals, 1, 1, 5);
  EXPECT_FALSE(result.validation_pass);
  EXPECT_EQ(result.sequence_errors, 1U);
  EXPECT_EQ(result.unclassified_pe_cycles, 5U);
  EXPECT_EQ(result.off_pe_cycles, 0U);
}

TEST(SaPeStateOpportunityTest,
     UpdatedShadowRejectsConsumerFootprintDifferentFromItsPreload) {
  const std::vector<sa_pe_state::UpdatedShadowInterval> intervals = {
      {.stream_id = 1, .issue_cycle = 0, .simulator_finish_cycle = 1,
       .tile_m = 2, .tile_k = 2,
       .tile_n = 1, .opcode = sa_pe_state::WaveOpcode::PRELOAD,
       .preload_sequence = 1},
      {.stream_id = 1, .issue_cycle = 1, .simulator_start_cycle = 1,
       .simulator_finish_cycle = 2, .tile_m = 1, .tile_k = 2,
       .tile_n = 1, .opcode = sa_pe_state::WaveOpcode::GEMM,
       .preload_sequence = 1}};
  const auto result = sa_pe_state::derive_updated_two_bank_shadow(
      intervals, 2, 2, 5);
  EXPECT_FALSE(result.validation_pass);
  EXPECT_EQ(result.footprint_mismatch_intervals, 1U);
  EXPECT_EQ(result.unclassified_pe_cycles, 20U);
}

TEST(SaPeStateOpportunityTest, UpdatedShadowReportsThirdLiveCopyConflict) {
  const std::vector<sa_pe_state::UpdatedShadowInterval> intervals = {
      {.stream_id = 1, .issue_cycle = 0, .simulator_finish_cycle = 1,
       .tile_m = 1, .tile_k = 1,
       .tile_n = 1, .opcode = sa_pe_state::WaveOpcode::PRELOAD,
       .preload_sequence = 1},
      {.stream_id = 2, .issue_cycle = 1, .simulator_start_cycle = 1,
       .simulator_finish_cycle = 2, .tile_m = 1, .tile_k = 1,
       .tile_n = 1, .opcode = sa_pe_state::WaveOpcode::PRELOAD,
       .preload_sequence = 1},
      {.stream_id = 3, .issue_cycle = 2, .simulator_start_cycle = 2,
       .simulator_finish_cycle = 3, .tile_m = 1, .tile_k = 1,
       .tile_n = 1, .opcode = sa_pe_state::WaveOpcode::PRELOAD,
       .preload_sequence = 1},
      {.stream_id = 1, .issue_cycle = 3, .simulator_start_cycle = 3,
       .simulator_finish_cycle = 4, .tile_m = 1, .tile_k = 1,
       .tile_n = 1, .opcode = sa_pe_state::WaveOpcode::GEMM,
       .preload_sequence = 1},
      {.stream_id = 2, .issue_cycle = 4, .simulator_start_cycle = 4,
       .simulator_finish_cycle = 5, .tile_m = 1, .tile_k = 1,
       .tile_n = 1, .opcode = sa_pe_state::WaveOpcode::GEMM,
       .preload_sequence = 1},
      {.stream_id = 3, .issue_cycle = 5, .simulator_start_cycle = 5,
       .simulator_finish_cycle = 6, .tile_m = 1, .tile_k = 1,
       .tile_n = 1, .opcode = sa_pe_state::WaveOpcode::GEMM,
       .preload_sequence = 1}};
  const auto result = sa_pe_state::derive_updated_two_bank_shadow(
      intervals, 1, 1, 8);
  EXPECT_FALSE(result.validation_pass);
  EXPECT_EQ(result.bank_assignment_conflicts, 1U);
  EXPECT_EQ(result.unclassified_pe_cycles, 8U);
}
