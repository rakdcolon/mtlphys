// Integrator correctness — compare GPU semi-implicit Euler against the same
// algorithm executed on the CPU. They must agree to single-precision tolerance.
#include "doctest.h"
#include "TestUtils.hpp"

#include "mtlphys/Engine.hpp"

#include <Metal/Metal.hpp>
#include <simd/simd.h>

using namespace mtlphys;
using namespace mtlphys::test;

namespace {

// Engine bakes these in. Mirror them so the CPU reference uses identical
// constants. If Engine.cpp tunes them, this test will fail loudly — which is
// what we want.
constexpr float        kGravityY  = -9.81f;
constexpr float        kDamping   = 0.999f;

// CPU reference for the integrator kernel: semi-implicit Euler + box bounds.
struct CpuParticle {
    simd::float3 p;
    simd::float3 v;
};
void cpuStep(CpuParticle& q, float dt) {
    q.v.y = (q.v.y + kGravityY * dt) * kDamping;
    q.v.x *= kDamping;
    q.v.z *= kDamping;
    q.p   += q.v * dt;
    // Match shader bounds + restitution.
    constexpr float r = 0.4f;
    constexpr simd::float3 lo{ -3, -3, -3 }, hi{ 3, 6, 3 };
    if (q.p.x < lo.x) { q.p.x = lo.x; q.v.x = -q.v.x * r; }
    if (q.p.y < lo.y) { q.p.y = lo.y; q.v.y = -q.v.y * r; }
    if (q.p.z < lo.z) { q.p.z = lo.z; q.v.z = -q.v.z * r; }
    if (q.p.x > hi.x) { q.p.x = hi.x; q.v.x = -q.v.x * r; }
    if (q.p.y > hi.y) { q.p.y = hi.y; q.v.y = -q.v.y * r; }
    if (q.p.z > hi.z) { q.p.z = hi.z; q.v.z = -q.v.z * r; }
}

} // namespace

TEST_SUITE("integrator") {

TEST_CASE("freefall matches CPU reference for 100 steps") {
    Engine e(device());
    e.reset(1);

    auto* pos = static_cast<simd::float4*>(e.positionsBuffer()->contents());
    auto* vel = static_cast<simd::float4*>(e.velocitiesBuffer()->contents());
    pos[0] = simd_make_float4(0, 5, 0, 1);
    vel[0] = simd_make_float4(0, 0, 0, 0);

    CpuParticle q{ {0, 5, 0}, {0, 0, 0} };

    constexpr float dt = 0.01f;
    constexpr int   N  = 100;

    for (int i = 0; i < N; ++i) {
        auto* cmd = queue()->commandBuffer();
        e.integrate(cmd, dt);
        cmd->commit();
        cmd->waitUntilCompleted();
        cpuStep(q, dt);
    }

    const float gpu_px = pos[0].x, gpu_py = pos[0].y, gpu_pz = pos[0].z;
    const float gpu_vy = vel[0].y;
    const float ref_px = q.p.x,    ref_py = q.p.y,    ref_pz = q.p.z;
    const float ref_vy = q.v.y;
    CHECK(gpu_px == doctest::Approx(ref_px).epsilon(1e-4));
    CHECK(gpu_py == doctest::Approx(ref_py).epsilon(1e-4));
    CHECK(gpu_pz == doctest::Approx(ref_pz).epsilon(1e-4));
    CHECK(gpu_vy == doctest::Approx(ref_vy).epsilon(1e-4));
}

TEST_CASE("particle bounces off the floor with restitution") {
    Engine e(device());
    e.reset(1);

    auto* pos = static_cast<simd::float4*>(e.positionsBuffer()->contents());
    auto* vel = static_cast<simd::float4*>(e.velocitiesBuffer()->contents());
    // Start near the floor, moving down hard. After one big step it should be
    // clamped to floor and have positive (upward) velocity.
    pos[0] = simd_make_float4(0, -2.95f, 0, 1);
    vel[0] = simd_make_float4(0, -10.0f, 0, 0);

    auto* cmd = queue()->commandBuffer();
    e.integrate(cmd, 0.05f);
    cmd->commit();
    cmd->waitUntilCompleted();

    const float py = pos[0].y;
    const float vy = vel[0].y;
    CHECK(py == doctest::Approx(-3.0f));   // clamped to floor
    CHECK(vy > 0.0f);                      // bounced upward
}

} // TEST_SUITE("integrator")
