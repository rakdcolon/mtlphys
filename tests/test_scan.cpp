// Prefix-scan correctness — verify Engine::runExclusiveScan against a CPU
// reference for several sizes that exercise different code paths:
//   • single chunk, fully populated
//   • single chunk, partially populated (trailing padding)
//   • many chunks, exact multiple
//   • many chunks, ragged tail
#include "doctest.h"
#include "TestUtils.hpp"

#include "mtlphys/Engine.hpp"

#include <Metal/Metal.hpp>

#include <random>
#include <vector>

using namespace mtlphys;
using namespace mtlphys::test;

namespace {

std::vector<uint32_t> randomVec(uint32_t n, uint32_t seed, uint32_t maxVal = 7) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> d(0, maxVal);
    std::vector<uint32_t> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

// Run the GPU exclusive scan on the given input and return the result.
std::vector<uint32_t> gpuScan(const std::vector<uint32_t>& in) {
    Engine engine(device());

    const uint32_t n = uint32_t(in.size());
    const size_t   bytes = n * sizeof(uint32_t);
    const uint32_t chunkCount = (n + Engine::kScanChunkSize - 1) / Engine::kScanChunkSize;

    auto* inBuf  = device()->newBuffer(bytes, MTL::ResourceStorageModeShared);
    auto* outBuf = device()->newBuffer(bytes, MTL::ResourceStorageModeShared);
    auto* sums   = device()->newBuffer(chunkCount * sizeof(uint32_t),
                                        MTL::ResourceStorageModePrivate);

    std::memcpy(inBuf->contents(), in.data(), bytes);

    auto* cmd = queue()->commandBuffer();
    engine.runExclusiveScan(cmd, inBuf, outBuf, sums, n);
    cmd->commit();
    cmd->waitUntilCompleted();

    std::vector<uint32_t> result(n);
    std::memcpy(result.data(), outBuf->contents(), bytes);

    inBuf->release(); outBuf->release(); sums->release();
    return result;
}

} // namespace

TEST_SUITE("scan") {

TEST_CASE("all zeros stays zero") {
    auto in  = std::vector<uint32_t>(100, 0);
    auto got = gpuScan(in);
    for (auto v : got) CHECK(v == 0u);
}

TEST_CASE("all ones gives identity scan") {
    constexpr uint32_t N = 500;
    auto in  = std::vector<uint32_t>(N, 1);
    auto got = gpuScan(in);
    for (uint32_t i = 0; i < N; ++i) CHECK(got[i] == i);
}

TEST_CASE("single full chunk random") {
    auto in  = randomVec(1024, 0xA);
    auto got = gpuScan(in);
    auto ref = cpuExclusiveScan(in);
    REQUIRE(got.size() == ref.size());
    for (size_t i = 0; i < got.size(); ++i) CHECK(got[i] == ref[i]);
}

TEST_CASE("single partial chunk (700 of 1024)") {
    auto in  = randomVec(700, 0xB);
    auto got = gpuScan(in);
    auto ref = cpuExclusiveScan(in);
    for (size_t i = 0; i < got.size(); ++i) CHECK(got[i] == ref[i]);
}

TEST_CASE("two full chunks") {
    auto in  = randomVec(2048, 0xC);
    auto got = gpuScan(in);
    auto ref = cpuExclusiveScan(in);
    for (size_t i = 0; i < got.size(); ++i) CHECK(got[i] == ref[i]);
}

TEST_CASE("many chunks, ragged tail (635626 — real spatial-hash size)") {
    constexpr uint32_t N = 635626;
    auto in  = randomVec(N, 0xD, 3);
    auto got = gpuScan(in);
    auto ref = cpuExclusiveScan(in);
    bool ok = true;
    size_t firstBad = 0;
    for (size_t i = 0; i < N; ++i) {
        if (got[i] != ref[i]) { ok = false; firstBad = i; break; }
    }
    CHECK_MESSAGE(ok, "first mismatch at i=", firstBad,
                  " got=", got[firstBad], " ref=", ref[firstBad]);
}

} // TEST_SUITE("scan")
