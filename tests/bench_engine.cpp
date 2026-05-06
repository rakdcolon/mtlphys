// Benchmarks. Live under TEST_SUITE("bench") so they can be filtered:
//
//   ./mtlphys_tests                       # everything (unit + bench, ~10s)
//   ./mtlphys_tests -ts=bench             # only benchmarks
//   ./mtlphys_tests -tse=bench            # only unit tests (~1s)
//
// Each benchmark warms up for a few frames, then times N frames and reports
// p50 / p95 / p99 / mean in milliseconds (pure GPU time, completion-handler).
#include "doctest.h"
#include "TestUtils.hpp"

#include "mtlphys/Engine.hpp"

#include <Metal/Metal.hpp>

#include <vector>

using namespace mtlphys;
using namespace mtlphys::test;

namespace {

constexpr int kWarmup     = 8;
constexpr int kIterations = 64;

// Time `iters` invocations of `encode`. Each invocation gets its own command
// buffer so we measure pure GPU time per dispatch sequence (not pipelining).
template <typename Encode>
Stats benchEncoded(Encode&& encode) {
    std::vector<double> times;
    times.reserve(kIterations);
    for (int i = 0; i < kWarmup; ++i) {
        auto* cmd = queue()->commandBuffer();
        encode(cmd);
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    for (int i = 0; i < kIterations; ++i) {
        auto* cmd = queue()->commandBuffer();
        encode(cmd);
        cmd->commit();
        cmd->waitUntilCompleted();
        times.push_back(cmd->GPUEndTime() - cmd->GPUStartTime());
    }
    return computeStats(times);
}

} // namespace

TEST_SUITE("bench") {

TEST_CASE("integrator throughput") {
    printBenchHeader();

    for (uint32_t n : { 10'000u, 100'000u, 1'000'000u }) {
        Engine e(device());
        e.reset(n);
        auto s = benchEncoded([&](MTL::CommandBuffer* cmd) { e.step(cmd, 1.0f / 480.0f); });
        printBenchRow("integrator", n, s);
    }
}

TEST_CASE("spatial hash throughput (full 7-stage pass)") {
    for (uint32_t n : { 10'000u, 100'000u, 1'000'000u }) {
        Engine e(device());
        e.reset(n);
        // First-frame settle: positions are random, neighbor density is wild.
        // Run a few sim steps so particles fall into a more realistic layout.
        for (int i = 0; i < 10; ++i) {
            auto* cmd = queue()->commandBuffer();
            e.step(cmd, 1.0f / 240.0f);
            cmd->commit();
            cmd->waitUntilCompleted();
        }
        auto s = benchEncoded([&](MTL::CommandBuffer* cmd) { e.buildSpatialHash(cmd); });
        printBenchRow("spatial_hash", n, s);
    }
}

TEST_CASE("frame time (sim + hash)") {
    for (uint32_t n : { 100'000u, 1'000'000u }) {
        Engine e(device());
        e.reset(n);
        for (int i = 0; i < 10; ++i) {
            auto* cmd = queue()->commandBuffer();
            e.step(cmd, 1.0f / 240.0f);
            cmd->commit();
            cmd->waitUntilCompleted();
        }
        auto s = benchEncoded([&](MTL::CommandBuffer* cmd) {
            e.step(cmd, 1.0f / 240.0f);
            e.buildSpatialHash(cmd);
        });
        printBenchRow("sim+hash", n, s);
    }
}

TEST_CASE("standalone exclusive scan") {
    for (uint32_t n : { 1'024u, 65'536u, 635'626u }) {
        Engine e(device());
        const size_t bytes = n * sizeof(uint32_t);
        const uint32_t cc  = (n + Engine::kScanChunkSize - 1) / Engine::kScanChunkSize;

        auto* in  = device()->newBuffer(bytes,            MTL::ResourceStorageModePrivate);
        auto* out = device()->newBuffer(bytes,            MTL::ResourceStorageModePrivate);
        auto* sums= device()->newBuffer(cc * sizeof(uint32_t), MTL::ResourceStorageModePrivate);

        auto s = benchEncoded([&](MTL::CommandBuffer* cmd) {
            e.runExclusiveScan(cmd, in, out, sums, n);
        });
        printBenchRow("exclusive_scan", n, s);

        in->release(); out->release(); sums->release();
    }
}

} // TEST_SUITE("bench")
