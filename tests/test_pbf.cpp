// PBF correctness — compare GPU density+lambda computation against a CPU
// reference for a small hand-placed particle layout.
#include "doctest.h"
#include "TestUtils.hpp"

#include "mtlphys/Engine.hpp"

#include <Metal/Metal.hpp>
#include <simd/simd.h>

#include <cmath>

using namespace mtlphys;
using namespace mtlphys::test;

// Mirror Engine.cpp constants. Drift here would be caught by the cellSize
// test in test_spatial.cpp, so we keep these local for clarity.
namespace {
constexpr float kPi          = 3.14159265358979323846f;
constexpr float kSmoothingH  = 0.10f;
constexpr float kRestDensity = 8000.0f;
constexpr float kEpsilonCFM  = 1500.0f;

float poly6_cpu(float r2, float h2, float poly6Norm) {
    if (r2 >= h2) return 0.0f;
    const float diff = h2 - r2;
    return poly6Norm * diff * diff * diff;
}

simd::float3 spikyGrad_cpu(simd::float3 r, float r_len, float h, float spikyGradNorm) {
    if (r_len <= 1e-6f || r_len >= h) return simd::float3{0, 0, 0};
    const float diff = h - r_len;
    return -spikyGradNorm * diff * diff * (r / r_len);
}

void runOnePBFStep(Engine& e, float dt) {
    auto* cmd = queue()->commandBuffer();
    e.step(cmd, dt);
    cmd->commit();
    cmd->waitUntilCompleted();
}

} // namespace

TEST_SUITE("pbf") {

TEST_CASE("single isolated particle stays put") {
    Engine e(device());
    e.reset(1);
    auto* pos = static_cast<simd::float4*>(e.positionsBuffer()->contents());
    auto* vel = static_cast<simd::float4*>(e.velocitiesBuffer()->contents());
    pos[0] = simd_make_float4(0, 1, 0, 1);
    vel[0] = simd_make_float4(0, 0, 0, 0);

    // One step of dt=0.01s. With nothing nearby, no constraint correction
    // applies — particle should fall under gravity by ~½·g·dt² ≈ 0.0005m.
    runOnePBFStep(e, 0.01f);

    const float py = pos[0].y;
    const float vy = vel[0].y;
    // Gravity-only fall over one frame is roughly v0=0 then v_new = g·dt;
    // x_new = x + (v0 + g·dt)·dt = 1 + (0 + (-9.81)·0.01)·0.01 ≈ 0.99902
    CHECK(py == doctest::Approx(0.99902f).epsilon(1e-3));
    CHECK(vy < 0.0f);
}

TEST_CASE("lambda sign matches density vs rest density") {
    // 3x3x3 cube of particles tightly packed — over rest density → λ < 0
    // (push apart). One far-away particle alone — density is just self-poly6,
    // well under rest density → λ > 0 (push together, but no neighbors so it
    // sits there). We verify both: signs and the analytical isolated value.
    Engine e(device());

    const float h  = kSmoothingH;
    const float h2 = h * h;
    const float h6 = h2 * h2 * h2;
    const float h9 = h6 * h2 * h;
    const float poly6Norm = 315.0f / (64.0f * kPi * h9);

    std::vector<simd::float4> ps;
    const float s = h * 0.4f;
    for (int z = -1; z <= 1; ++z)
        for (int y = -1; y <= 1; ++y)
            for (int x = -1; x <= 1; ++x)
                ps.push_back(simd_make_float4(x * s, y * s, z * s, 1));
    // Far-away isolated particle — distance >> h so it has no neighbors.
    ps.push_back(simd_make_float4(2.0f, 2.0f, 2.0f, 1));

    e.reset(uint32_t(ps.size()));
    auto* pos = static_cast<simd::float4*>(e.positionsBuffer()->contents());
    auto* vel = static_cast<simd::float4*>(e.velocitiesBuffer()->contents());
    for (size_t i = 0; i < ps.size(); ++i) { pos[i] = ps[i]; vel[i] = simd_make_float4(0, 0, 0, 0); }

    runOnePBFStep(e, 1e-6f);  // negligible-time step, lambdas reflect current layout

    auto lambdas = readBuffer<float>(e.lambdasBuffer(), ps.size());

    // Center of the 3x3x3 cube = index 13. It has 26 neighbors (well above
    // rest packing of ~26 at h/2 spacing), so density > rest density, λ < 0.
    CHECK(lambdas[13] < 0.0f);

    // Isolated particle: density = only self contribution.
    const float selfDensity         = poly6Norm * h6;
    const float aloneConstraint     = selfDensity / kRestDensity - 1.0f;
    const float expectedAloneLambda = -aloneConstraint / kEpsilonCFM;
    CHECK(lambdas[ps.size() - 1] == doctest::Approx(expectedAloneLambda).epsilon(1e-3));
}

TEST_CASE("step is stable: 50 frames of free fall doesn't NaN out") {
    Engine e(device());
    e.reset(1024);   // small enough to inspect all particles cheaply
    // Engine seeds particles with random positions in (-2..2, 1..5, -2..2).

    for (int i = 0; i < 50; ++i) {
        auto* cmd = queue()->commandBuffer();
        e.step(cmd, 1.0f / 240.0f);
        cmd->commit();
        cmd->waitUntilCompleted();
    }

    auto* pos = static_cast<simd::float4*>(e.positionsBuffer()->contents());
    int nans = 0, oob = 0;
    for (uint32_t i = 0; i < e.particleCount(); ++i) {
        const float x = pos[i].x, y = pos[i].y, z = pos[i].z;
        if (std::isnan(x) || std::isnan(y) || std::isnan(z)) ++nans;
        if (x < -3.01f || x > 3.01f || y < -3.01f || y > 6.01f || z < -3.01f || z > 3.01f) ++oob;
    }
    CHECK(nans == 0);
    CHECK(oob == 0);
}

} // TEST_SUITE("pbf")
