// Spatial hash correctness — drive the engine with hand-placed particles,
// then verify cellHashes / cellStart / sortedIndex / neighborCounts agree
// with hand-computed expectations.
#include "doctest.h"
#include "TestUtils.hpp"

#include "mtlphys/Engine.hpp"

#include <Metal/Metal.hpp>
#include <simd/simd.h>

#include <cmath>
#include <map>
#include <unordered_set>

using namespace mtlphys;
using namespace mtlphys::test;

// Engine bakes these in as compile-time constants. Mirrored here for the
// hash-recomputation reference; cellSize is read from the engine to keep them
// in sync.
namespace {
constexpr simd::float3 kBoundsMin{ -3.0f, -3.0f, -3.0f };
constexpr simd::float3 kBoundsMax{  3.0f,  6.0f,  3.0f };

simd::uint3 gridDim(float cellSize) {
    auto extent = kBoundsMax - kBoundsMin;
    return simd::uint3{
        uint32_t(std::ceil(extent.x / cellSize)),
        uint32_t(std::ceil(extent.y / cellSize)),
        uint32_t(std::ceil(extent.z / cellSize)),
    };
}

// Configure the engine with `count` particles at user-supplied positions and
// run one full spatial-hash pass.
void seedAndHash(Engine& e, const std::vector<simd::float4>& positions) {
    e.reset(uint32_t(positions.size()));
    auto* pos = static_cast<simd::float4*>(e.positionsBuffer()->contents());
    for (size_t i = 0; i < positions.size(); ++i) pos[i] = positions[i];

    auto* cmd = queue()->commandBuffer();
    e.buildSpatialHash(cmd);
    cmd->commit();
    cmd->waitUntilCompleted();
}

} // namespace

TEST_SUITE("spatial") {

TEST_CASE("cell hash matches CPU reference") {
    Engine e(device());
    std::vector<simd::float4> ps{
        simd_make_float4( 0.0f,  0.0f,  0.0f, 1),
        simd_make_float4( 2.5f,  4.0f, -1.5f, 1),
        simd_make_float4(-2.9f,  5.5f,  2.5f, 1),
        simd_make_float4( 1.0f, -2.0f, -2.5f, 1),
    };
    seedAndHash(e, ps);
    auto hashes = readBuffer<uint32_t>(e.cellHashesBuffer(), ps.size());

    const float cs = e.cellSize();
    auto gd = gridDim(cs);
    for (size_t i = 0; i < ps.size(); ++i) {
        simd::float3 p{ ps[i].x, ps[i].y, ps[i].z };
        uint32_t expected = cpuCellHash(p, kBoundsMin, cs, gd);
        CHECK(hashes[i] == expected);
    }
}

TEST_CASE("sortedIndex groups particles by cell") {
    Engine e(device());
    std::vector<simd::float4> ps{
        simd_make_float4(0.00f, 0.00f, 0.00f, 1),   // cell A
        simd_make_float4(0.01f, 0.02f, 0.01f, 1),   // cell A
        simd_make_float4(0.03f, 0.01f, 0.02f, 1),   // cell A
        simd_make_float4(2.00f, 2.00f, 2.00f, 1),   // cell B
        simd_make_float4(2.01f, 2.02f, 2.03f, 1),   // cell B
        simd_make_float4(-2.50f,-2.50f,-2.50f, 1),  // cell C
    };
    seedAndHash(e, ps);
    auto hashes      = readBuffer<uint32_t>(e.cellHashesBuffer(),  ps.size());
    auto sortedIndex = readBuffer<uint32_t>(e.sortedIndexBuffer(), ps.size());
    auto cellStart   = readBuffer<uint32_t>(e.cellStartBuffer(),   e.totalCells() + 1);

    std::map<uint32_t, std::vector<uint32_t>> byHashCpu;
    for (uint32_t i = 0; i < ps.size(); ++i) byHashCpu[hashes[i]].push_back(i);

    for (auto& [hash, expectedSet] : byHashCpu) {
        const uint32_t s  = cellStart[hash];
        const uint32_t en = cellStart[hash + 1];
        REQUIRE(en - s == expectedSet.size());
        std::unordered_set<uint32_t> got;
        for (uint32_t i = s; i < en; ++i) got.insert(sortedIndex[i]);
        for (uint32_t exp : expectedSet) CHECK(got.count(exp) == 1);
    }

    CHECK(cellStart.back() == ps.size());
}

TEST_CASE("isolated particles have zero neighbors") {
    Engine e(device());
    std::vector<simd::float4> ps{
        simd_make_float4( 0.0f, 0.0f, 0.0f, 1),
        simd_make_float4( 1.5f, 1.5f, 1.5f, 1),
        simd_make_float4(-1.5f,-1.5f,-1.5f, 1),
    };
    seedAndHash(e, ps);
    auto neighbors = readBuffer<uint32_t>(e.neighborCountsBuffer(), ps.size());
    for (auto n : neighbors) CHECK(n == 0u);
}

TEST_CASE("close pair counts each other as neighbor") {
    Engine e(device());
    const float cs = e.cellSize();
    std::vector<simd::float4> ps{
        simd_make_float4(0.00f,        0.0f, 0.0f, 1),
        simd_make_float4(cs * 0.5f,    0.0f, 0.0f, 1),  // half a cell apart
    };
    seedAndHash(e, ps);
    auto neighbors = readBuffer<uint32_t>(e.neighborCountsBuffer(), ps.size());
    CHECK(neighbors[0] == 1u);
    CHECK(neighbors[1] == 1u);
}

TEST_CASE("3x3x3 lattice — center sees only face neighbors") {
    Engine e(device());
    // Spacing just under cellSize. Distances from center:
    //   face²   = s²   < cellSize² → INSIDE radius
    //   edge²   = 2s²  > cellSize² → outside
    //   corner² = 3s²  > cellSize² → outside
    const float s = e.cellSize() * 0.99f;
    std::vector<simd::float4> ps;
    for (int z = -1; z <= 1; ++z)
        for (int y = -1; y <= 1; ++y)
            for (int x = -1; x <= 1; ++x)
                ps.push_back(simd_make_float4(x * s, y * s, z * s, 1));

    seedAndHash(e, ps);
    auto neighbors = readBuffer<uint32_t>(e.neighborCountsBuffer(), ps.size());
    CHECK(neighbors[13] == 6u);
    CHECK(neighbors[0]  < neighbors[13]);
    CHECK(neighbors[26] < neighbors[13]);
}

} // TEST_SUITE("spatial")
