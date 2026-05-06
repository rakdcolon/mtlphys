// TestUtils — small helpers shared by all tests/benchmarks.
//
// All tests rely on a single Metal device created lazily on first access. We
// also vend a command queue and a couple of helpers for getting data out of
// Private-storage buffers (which the engine uses for its hot path).
#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <simd/simd.h>

namespace mtlphys::test {

// Singleton-ish device + queue. Constructing them per-test would burn ms.
MTL::Device*       device();
MTL::CommandQueue* queue();

// Synchronous blit a Private buffer (or any buffer) into a host-readable
// vector<T>. Convenience wrapper — internally allocates a Shared scratch
// buffer of the right size, runs a blit, waits, copies, releases scratch.
template <typename T>
std::vector<T> readBuffer(MTL::Buffer* src, size_t count, size_t srcByteOffset = 0);

// Run a single command-buffer worth of work and wait for completion. Returns
// pure GPU time in seconds via the completion handler timestamps.
double runAndTime(void (*encode)(MTL::CommandBuffer*, void*), void* userdata);

// Same, but with a lambda. Convenience.
template <typename Fn>
double runAndTime(Fn&& encode) {
    auto* cmd = queue()->commandBuffer();
    encode(cmd);
    cmd->commit();
    cmd->waitUntilCompleted();
    return cmd->GPUEndTime() - cmd->GPUStartTime();
}

// CPU reference: same hash function as cellHashOf in Spatial.metal.
uint32_t cpuCellHash(simd::float3 pos,
                     simd::float3 origin,
                     float        cellSize,
                     simd::uint3  gridDim);

// CPU reference: exclusive prefix scan.
template <typename T>
std::vector<T> cpuExclusiveScan(const std::vector<T>& in) {
    std::vector<T> out(in.size());
    T running{};
    for (size_t i = 0; i < in.size(); ++i) {
        out[i] = running;
        running = static_cast<T>(running + in[i]);
    }
    return out;
}

// Tiny p50/p95 reporter.
struct Stats {
    double p50_ms = 0;
    double p95_ms = 0;
    double p99_ms = 0;
    double mean_ms = 0;
    double min_ms = 0;
    double max_ms = 0;
};
Stats computeStats(std::vector<double>& seconds);  // sorts in place
void  printBenchHeader();
void  printBenchRow(const std::string& name, uint32_t n, const Stats& s);

} // namespace mtlphys::test

// ---- Template impls --------------------------------------------------------

namespace mtlphys::test {

template <typename T>
std::vector<T> readBuffer(MTL::Buffer* src, size_t count, size_t srcByteOffset) {
    const size_t bytes = count * sizeof(T);
    auto* scratch = device()->newBuffer(bytes, MTL::ResourceStorageModeShared);
    auto* cmd     = queue()->commandBuffer();
    auto* blit    = cmd->blitCommandEncoder();
    blit->copyFromBuffer(src, srcByteOffset, scratch, 0, bytes);
    blit->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();

    std::vector<T> result(count);
    std::memcpy(result.data(), scratch->contents(), bytes);
    scratch->release();
    return result;
}

} // namespace mtlphys::test
