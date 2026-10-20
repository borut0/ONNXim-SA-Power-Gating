#include <gtest/gtest.h>

#include "Dram.h"
#include "Interconnect.h"
#include "SimulationConfig.h"

TEST(ComponentQuiescenceTest, SimpleNoCChecksBoundaryBuffersAndOutstanding) {
  SimulationConfig config{};
  config.num_cores = 1;
  config.dram_channels = 1;
  config.icnt_injection_ports_per_core = 1;
  config.icnt_latency = 1;

  SimpleInterconnect interconnect(config);
  EXPECT_TRUE(interconnect.get_model_quiescence_status().quiescent());

  MemoryAccess request{};
  request.request = true;
  request.size = 64;
  interconnect.push(0, 1, &request);
  EXPECT_FALSE(interconnect.get_model_quiescence_status().quiescent());

  interconnect.cycle();
  interconnect.cycle();
  ASSERT_FALSE(interconnect.is_empty(1));
  EXPECT_FALSE(interconnect.get_model_quiescence_status().quiescent());

  interconnect.pop(1);
  EXPECT_TRUE(interconnect.get_model_quiescence_status().quiescent());
}

TEST(ComponentQuiescenceTest, UnsupportedDramBackendFailsClosed) {
  SimulationConfig config{};
  config.dram_channels = 1;
  config.dram_latency = 1;

  SimpleDram dram(config);
  EXPECT_FALSE(dram.get_model_quiescence_status().observable);
  EXPECT_FALSE(dram.get_model_quiescence_status().quiescent());
}
