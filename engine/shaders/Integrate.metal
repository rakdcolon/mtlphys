// Integrate.metal
// Semi-implicit (symplectic) Euler integrator with axis-aligned box bounds.
// One thread per particle. This is the simplest possible solver — Week 2
// will replace the boundary handling with constraint projection (XPBD).

#include <metal_stdlib>
#include "Shared.h"
using namespace metal;
using namespace mtlphys;

kernel void integrate(device float4*           positions  [[ buffer(0) ]],
                      device float4*           velocities [[ buffer(1) ]],
                      constant SimParams&      params     [[ buffer(2) ]],
                      uint                     gid        [[ thread_position_in_grid ]])
{
    if (gid >= params.particleCount) return;

    float3 p = positions[gid].xyz;
    float3 v = velocities[gid].xyz;

    // Symplectic Euler: integrate velocity first, then position.
    // (Reverse order — explicit Euler — drifts energy; this conserves it much better.)
    v += params.gravity * params.dt;
    v *= params.damping;
    p += v * params.dt;

    // Cheap box bounds: clamp + reflect with restitution. Placeholder until
    // Week 2's spatial-hash collision handling lands.
    constexpr float restitution = 0.4f;
    float3 lo = params.boundsMin;
    float3 hi = params.boundsMax;

    if (p.x < lo.x) { p.x = lo.x; v.x = -v.x * restitution; }
    if (p.y < lo.y) { p.y = lo.y; v.y = -v.y * restitution; }
    if (p.z < lo.z) { p.z = lo.z; v.z = -v.z * restitution; }
    if (p.x > hi.x) { p.x = hi.x; v.x = -v.x * restitution; }
    if (p.y > hi.y) { p.y = hi.y; v.y = -v.y * restitution; }
    if (p.z > hi.z) { p.z = hi.z; v.z = -v.z * restitution; }

    positions[gid]  = float4(p, 1.0);
    velocities[gid] = float4(v, 0.0);
}
