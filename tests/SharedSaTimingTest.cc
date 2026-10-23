#include "Common.h"
#include "SimulationConfig.h"
#include "SystolicWS.h"
#include "gtest/gtest.h"

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

class ScopedSharedSaPolicy {
 public:
  explicit ScopedSharedSaPolicy(const char* policy) {
    const char* previous = std::getenv("ONNXSIM_SHARED_SA_POLICY");
    if (previous != nullptr) {
      previous_ = previous;
      had_previous_ = true;
    }
    setenv("ONNXSIM_SHARED_SA_POLICY", policy, 1);
  }
  ~ScopedSharedSaPolicy() {
    if (had_previous_) {
      setenv("ONNXSIM_SHARED_SA_POLICY", previous_.c_str(), 1);
    } else {
      unsetenv("ONNXSIM_SHARED_SA_POLICY");
    }
  }

 private:
  bool had_previous_ = false;
  std::string previous_;
};

SimulationConfig small_config() {
  SimulationConfig config;
  config.num_cores = 1;
  config.core_config = new CoreConfig;
  config.core_config[0].core_type = CoreType::SYSTOLIC_WS;
  config.core_config[0].core_height = 8;
  config.core_config[0].core_width = 8;
  config.core_config[0].spad_size = 192;
  config.core_config[0].accum_spad_size = 192;
  config.precision = 4;
  config.dram_req_size = 32;
  return config;
}

std::unique_ptr<Instruction> sa_instruction(Opcode opcode,
                                            bool analytical_boundary = false) {
  return std::make_unique<Instruction>(Instruction{
      .opcode = opcode,
      .dest_addr = ACCUM_SPAD_BASE,
      .compute_size = 8,
      .tile_m = 5,
      .tile_k = 3,
      .tile_n = 8,
      .sa_analytical_preload_boundary = analytical_boundary});
}

std::unique_ptr<Tile> one_preload_tile() {
  auto tile = std::make_unique<Tile>();
  tile->status = Tile::Status::INITIALIZED;
  tile->instructions.push_back(sa_instruction(Opcode::GEMM_PRELOAD));
  tile->finalize_sa_metadata();
  return tile;
}

class SharedSaTimingProbe : public SystolicWS {
 public:
  SharedSaTimingProbe(uint32_t id, SimulationConfig config)
      : SystolicWS(id, config) {}

  std::pair<cycle_type, cycle_type> schedule_separated_pair() {
    const SharedGeneration generation{17, 1};
    _shared_remaining_consumers[generation] = 2;

    Instruction preload = *sa_instruction(Opcode::GEMM_PRELOAD);
    preload.measurement_tile_id = generation.first;
    preload.sa_preload_sequence = generation.second;
    schedule_shared_sa_instruction(preload);

    _core_cycle = 11;
    Instruction consumer = *sa_instruction(Opcode::GEMM);
    consumer.measurement_tile_id = generation.first;
    consumer.sa_preload_sequence = generation.second;
    schedule_shared_sa_instruction(consumer);
    return {preload.start_cycle, consumer.start_cycle};
  }

  uint64_t lockout_cycles() const {
    return _shared_min_off_lockout_cycles;
  }
  uint64_t lockout_events() const {
    return _shared_min_off_lockout_events;
  }
};

cycle_type run_one_preload(const char* policy) {
  ScopedSharedSaPolicy selected(policy);
  auto config = small_config();
  SystolicWS core(0, config);
  core.issue(one_preload_tile());
  cycle_type cycles = 0;
  while (core.running() && cycles < 100) {
    core.cycle();
    ++cycles;
  }
  EXPECT_FALSE(core.running());
  return cycles;
}

}  // namespace

TEST(SharedSaTimingTest, FusedPreloadUsesMappedWaveAndExplicitFirstPeWake) {
  // The first row writes during cycle 1, so PE(0,0) first uses it in cycle 2.
  // The explicit wake moves PE(0,0) MAC use to cycle 3 for gated policies.
  // A 5x3 rectangle and 8 vectors finish in cycle 17 with that wake.
  EXPECT_EQ(run_one_preload("always_on"), 17);
  EXPECT_EQ(run_one_preload("regate"), 18);
  EXPECT_EQ(run_one_preload("ours"), 18);
}

TEST(SharedSaTimingTest, OursEnforcesOneWaveLevelBetLockout) {
  ScopedSharedSaPolicy selected("ours");
  auto config = small_config();
  SharedSaTimingProbe core(0, config);
  const auto starts = core.schedule_separated_pair();
  // First PE is live through cycle 10. The second wave's original gap is one
  // cycle: one wake plus 46 lockout cycles brings it to the 48-cycle bound.
  EXPECT_EQ(starts.first, 2);
  EXPECT_EQ(starts.second, 58);
  EXPECT_EQ(core.lockout_cycles(), 46u);
  EXPECT_EQ(core.lockout_events(), 1u);
}

TEST(SharedSaTimingTest, ReGateWakesImmediatelyWithoutBetLockout) {
  ScopedSharedSaPolicy selected("regate");
  auto config = small_config();
  SharedSaTimingProbe core(0, config);
  const auto starts = core.schedule_separated_pair();
  EXPECT_EQ(starts.first, 2);
  EXPECT_EQ(starts.second, 12);
  EXPECT_EQ(core.lockout_cycles(), 0u);
  EXPECT_EQ(core.lockout_events(), 0u);
}

TEST(SharedSaTimingTest, AnalyticalOnlyPreloadCannotRunFunctionally) {
  ScopedSharedSaPolicy selected("always_on");
  auto config = small_config();
  SystolicWS core(0, config);
  auto tile = std::make_unique<Tile>();
  tile->status = Tile::Status::INITIALIZED;
  tile->instructions.push_back(sa_instruction(Opcode::GEMM, true));
  tile->finalize_sa_metadata();
  EXPECT_THROW(core.issue(std::move(tile)), std::logic_error);
}

TEST(SharedSaTimingTest, TwoTilesKeepTheirOwnPreloadGenerations) {
  ScopedSharedSaPolicy selected("always_on");
  auto config = small_config();
  SystolicWS core(0, config);
  for (uint64_t block : {uint64_t{1}, uint64_t{2}}) {
    auto tile = std::make_unique<Tile>();
    tile->status = Tile::Status::INITIALIZED;
    for (Opcode opcode : {Opcode::GEMM_PRELOAD, Opcode::GEMM}) {
      auto instruction = sa_instruction(opcode);
      instruction->sa_weight_block_id = block;
      instruction->sa_weight_block_known = true;
      tile->instructions.push_back(std::move(instruction));
    }
    tile->finalize_sa_metadata();
    core.issue(std::move(tile));
  }
  cycle_type cycles = 0;
  while (core.running() && cycles < 100) {
    core.cycle();
    ++cycles;
  }
  EXPECT_FALSE(core.running());
  unsigned finished = 0;
  while (true) {
    auto tile = core.pop_finished_tile();
    if (tile->status == Tile::Status::EMPTY) {
      break;
    }
    ++finished;
  }
  EXPECT_EQ(finished, 2u);
}

TEST(SharedSaTimingTest, PlainGemmCannotSilentlySwitchWeights) {
  ScopedSharedSaPolicy selected("always_on");
  auto config = small_config();
  SystolicWS core(0, config);
  auto tile = std::make_unique<Tile>();
  tile->status = Tile::Status::INITIALIZED;
  auto preload = sa_instruction(Opcode::GEMM_PRELOAD);
  preload->sa_weight_block_id = 1;
  preload->sa_weight_block_known = true;
  auto consumer = sa_instruction(Opcode::GEMM);
  consumer->sa_weight_block_id = 2;
  consumer->sa_weight_block_known = true;
  tile->instructions.push_back(std::move(preload));
  tile->instructions.push_back(std::move(consumer));
  tile->finalize_sa_metadata();
  core.issue(std::move(tile));
  EXPECT_THROW({
    for (unsigned cycle = 0; cycle < 10; ++cycle) {
      core.cycle();
    }
  }, std::logic_error);
}
