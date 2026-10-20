#include <gtest/gtest.h>

#include "SimulationConfig.h"
#include "Sram.h"

TEST(SramIdlenessTest, CountsOnlyExplicitLogicalAccessEvents) {
  CoreConfig core_config{};
  core_config.spad_size = 1;
  core_config.accum_spad_size = 1;

  SimulationConfig config{};
  config.core_config = &core_config;
  config.dram_req_size = 64;
  config.precision = 1;

  cycle_type core_cycle = 0;
  Sram sram(config, core_cycle, false, 0);
  constexpr addr_type address = SPAD_BASE;

  EXPECT_EQ(sram.prefetch(address, 0, 1, 1,
                          Sram::PendingProducer::MEMORY), 1);
  EXPECT_TRUE(sram.cycle());

  // Allocation/readiness checks are deliberately not modeled accesses.
  EXPECT_TRUE(sram.check_allocated(address, 0));
  EXPECT_FALSE(sram.check_hit(address, 0));
  EXPECT_TRUE(sram.cycle());

  sram.fill(address, 0, Sram::PendingProducer::MEMORY);
  EXPECT_FALSE(sram.cycle());
  EXPECT_EQ(sram.modeled_write_events(), 1);

  sram.record_modeled_read(address, 0);
  EXPECT_FALSE(sram.cycle());
  EXPECT_EQ(sram.modeled_read_events(), 1);

  EXPECT_TRUE(sram.cycle());
  EXPECT_EQ(sram.modeled_cycles(), 5);
  EXPECT_EQ(sram.modeled_idle_cycles(), 3);
}
