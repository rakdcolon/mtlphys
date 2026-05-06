// Constraints.metal
//
// PBF (Position-Based Fluids, Macklin & Müller 2013) constraint solver.
//
// Per simulation step:
//   1. predict                       — x* = x + (v + g·dt)·dt; save prevPositions
//   2. (spatial hash on x*, in Spatial.metal — also gathers sortedPositions)
//   3. for k iterations:
//        densityLambdaSorted         — ρ_i, λ_i in *sorted* slot layout
//        applyDeltaSorted            — Δx, write to sortedPositionsNext (ping-pong)
//   4. scatterPositionsToOriginal    — sortedPositions → positions[origIdx]
//   5. finalize                      — clamp, v = (x_new - x_prev)/dt, walls, commit
//
// Sorted-order (Design B) optimization: each thread handles one sorted slot
// rather than one original particle index. Adjacent threads in a simdgroup
// process particles in the same/adjacent cells — same 27-cell stencil, same
// neighbor data. Inner-loop reads of sortedPositions and sortedLambdas are
// fully sequential, so cache hit rates skyrocket vs. the random-gather form.

#include <metal_stdlib>
#include "Shared.h"
using namespace metal;
using namespace mtlphys;

// ---------------------------------------------------------------------------
// SPH kernels
// ---------------------------------------------------------------------------

static inline float poly6(float r2, float h2, float poly6Norm) {
    if (r2 >= h2) return 0.0f;
    const float diff = h2 - r2;
    return poly6Norm * diff * diff * diff;
}

static inline float3 spikyGrad(float3 r_vec, float r_len, float h, float spikyGradNorm) {
    if (r_len <= 1e-6f || r_len >= h) return float3(0);
    const float diff = h - r_len;
    return -spikyGradNorm * diff * diff * (r_vec / r_len);
}

static inline uint3 cellCoordOf(float3 p, constant SpatialParams& sp) {
    float3 q = (p - sp.gridOrigin) * sp.invCellSize;
    int3 c = int3(floor(q));
    return uint3(clamp(c, int3(0), int3(sp.gridDim) - 1));
}
static inline uint cellHashOf(uint3 c, constant SpatialParams& sp) {
    return c.x + c.y * sp.gridDim.x + c.z * sp.gridDim.x * sp.gridDim.y;
}

// ---------------------------------------------------------------------------
// 1. Predict
// ---------------------------------------------------------------------------

kernel void predictPositions(device float4*           positions      [[ buffer(0) ]],
                             device const float4*     velocities     [[ buffer(1) ]],
                             device float4*           prevPositions  [[ buffer(2) ]],
                             constant PBFParams&      pp             [[ buffer(7) ]],
                             uint                     gid            [[ thread_position_in_grid ]])
{
    if (gid >= pp.particleCount) return;
    const float4 p = positions[gid];
    const float3 v = velocities[gid].xyz;
    prevPositions[gid] = p;
    const float3 vNew = v + pp.gravity * pp.dt;
    positions[gid] = float4(p.xyz + vNew * pp.dt, p.w);
}

// NOTE: A threadgroup-per-cell cooperative-loading variant was tried here and
// regressed 20× because our spatial hash is sparse (~3 particles/cell on
// average → ~324K threadgroups, mostly empty). Cache-friendly sequential
// reads in the sorted-order kernels below already capture ~88% of theoretical
// memory bandwidth; that's the floor without restructuring the data layout.

// ---------------------------------------------------------------------------
// 3a. densityLambdaSorted: density + Lagrange multiplier per sorted slot
// ---------------------------------------------------------------------------
// Reads sortedPositions[slot] for self, sortedPositions[i] for neighbors —
// fully sequential within each cell range. Self-skip via slot comparison
// (no per-iteration sortedIndex lookup). Writes sortedLambdas[slot] for the
// next solver kernel and lambdas[origIdx] for read-back / tests.

kernel void densityLambdaSorted(device const float4*       sortedPositions  [[ buffer(8) ]],
                                device const uint*         cellStart        [[ buffer(3) ]],
                                device const uint*         sortedIndex      [[ buffer(5) ]],
                                device float*              sortedLambdas    [[ buffer(11) ]],
                                device float*              lambdas          [[ buffer(9) ]],
                                constant SpatialParams&    sp               [[ buffer(7) ]],
                                constant PBFParams&        pp               [[ buffer(10) ]],
                                uint                       slot             [[ thread_position_in_grid ]])
{
    if (slot >= sp.particleCount) return;

    const float3 p_i = sortedPositions[slot].xyz;
    const uint3  myC = cellCoordOf(p_i, sp);

    const int zLo = max(int(myC.z) - 1, 0), zHi = min(int(myC.z) + 1, int(sp.gridDim.z) - 1);
    const int yLo = max(int(myC.y) - 1, 0), yHi = min(int(myC.y) + 1, int(sp.gridDim.y) - 1);
    const int xLo = max(int(myC.x) - 1, 0), xHi = min(int(myC.x) + 1, int(sp.gridDim.x) - 1);

    float  density   = poly6(0.0f, pp.h2, pp.poly6Norm);
    float3 gradSelf  = float3(0);
    float  gradSqSum = 0.0f;

    for (int z = zLo; z <= zHi; ++z) {
        for (int y = yLo; y <= yHi; ++y) {
            for (int x = xLo; x <= xHi; ++x) {
                const uint c = cellHashOf(uint3(uint(x), uint(y), uint(z)), sp);
                const uint s = cellStart[c];
                const uint e = cellStart[c + 1];
                for (uint i = s; i < e; ++i) {
                    if (i == slot) continue;
                    const float3 r  = p_i - sortedPositions[i].xyz;
                    const float  r2 = dot(r, r);
                    if (r2 >= pp.h2) continue;
                    density += poly6(r2, pp.h2, pp.poly6Norm);
                    const float3 grad = spikyGrad(r, sqrt(r2), pp.h, pp.spikyGradNorm);
                    gradSelf  += grad;
                    gradSqSum += dot(grad, grad);
                }
            }
        }
    }

    const float constraint = density * pp.invRestDensity - 1.0f;
    const float denom      = pp.invRestDensity * pp.invRestDensity *
                             (dot(gradSelf, gradSelf) + gradSqSum) + pp.epsilon;
    const float lambda     = -constraint / denom;

    sortedLambdas[slot]              = lambda;
    lambdas[sortedIndex[slot]]       = lambda;  // for read-back / tests
}

// ---------------------------------------------------------------------------
// 3b. applyDeltaSorted: position correction; ping-pong write to "next"
// ---------------------------------------------------------------------------
// Reads sortedPositions[slot] / sortedLambdas[slot] for self, neighbor data
// for slots i — all sequential. Writes corrected position to
// sortedPositionsNext[slot]; caller swaps the two buffers each iter.

kernel void applyDeltaSorted(device const float4*         sortedPositions     [[ buffer(8) ]],
                             device const uint*           cellStart           [[ buffer(3) ]],
                             device const float*          sortedLambdas       [[ buffer(11) ]],
                             device float4*               sortedPositionsNext [[ buffer(12) ]],
                             constant SpatialParams&      sp                  [[ buffer(7) ]],
                             constant PBFParams&          pp                  [[ buffer(10) ]],
                             uint                         slot                [[ thread_position_in_grid ]])
{
    if (slot >= sp.particleCount) return;

    const float3 p_i      = sortedPositions[slot].xyz;
    const float  lambda_i = sortedLambdas[slot];
    const uint3  myC      = cellCoordOf(p_i, sp);

    const int zLo = max(int(myC.z) - 1, 0), zHi = min(int(myC.z) + 1, int(sp.gridDim.z) - 1);
    const int yLo = max(int(myC.y) - 1, 0), yHi = min(int(myC.y) + 1, int(sp.gridDim.y) - 1);
    const int xLo = max(int(myC.x) - 1, 0), xHi = min(int(myC.x) + 1, int(sp.gridDim.x) - 1);

    constexpr float kSCorrK = 1e-4f;
    const     float dq2     = (0.2f * pp.h) * (0.2f * pp.h);
    const     float invWq   = 1.0f / poly6(dq2, pp.h2, pp.poly6Norm);

    float3 delta = float3(0);

    for (int z = zLo; z <= zHi; ++z) {
        for (int y = yLo; y <= yHi; ++y) {
            for (int x = xLo; x <= xHi; ++x) {
                const uint c = cellHashOf(uint3(uint(x), uint(y), uint(z)), sp);
                const uint s = cellStart[c];
                const uint e = cellStart[c + 1];
                for (uint i = s; i < e; ++i) {
                    if (i == slot) continue;
                    const float3 r  = p_i - sortedPositions[i].xyz;
                    const float  r2 = dot(r, r);
                    if (r2 >= pp.h2) continue;
                    const float  lambda_j = sortedLambdas[i];
                    const float3 grad     = spikyGrad(r, sqrt(r2), pp.h, pp.spikyGradNorm);
                    const float  wij      = poly6(r2, pp.h2, pp.poly6Norm);
                    const float  ratio    = wij * invWq;
                    const float  ratio2   = ratio * ratio;
                    const float  s_corr   = -kSCorrK * ratio2 * ratio2;
                    delta += (lambda_i + lambda_j + s_corr) * grad;
                }
            }
        }
    }

    delta *= pp.invRestDensity;

    // CFL-like clamp
    const float maxStep = 0.3f * pp.h;
    const float dlen    = length(delta);
    if (dlen > maxStep) delta *= maxStep / dlen;

    // Clamp to bounds inside the solver so particles don't get shoved past
    // walls (causing artificial pressure buildup that finalize then has to
    // dissipate via friction → wall sticking).
    const float3 newPos = clamp(p_i + delta, pp.boundsMin, pp.boundsMax);
    sortedPositionsNext[slot] = float4(newPos, 1.0f);
}

// ---------------------------------------------------------------------------
// 3c. scatterPositionsToOriginal: write sortedPositions back to positions[origIdx]
// ---------------------------------------------------------------------------

kernel void scatterPositionsToOriginal(device const float4*    sortedPositions [[ buffer(8) ]],
                                       device const uint*      sortedIndex     [[ buffer(5) ]],
                                       device float4*          positions       [[ buffer(0) ]],
                                       constant SpatialParams& sp              [[ buffer(7) ]],
                                       uint                    slot            [[ thread_position_in_grid ]])
{
    if (slot >= sp.particleCount) return;
    positions[sortedIndex[slot]] = sortedPositions[slot];
}

// ---------------------------------------------------------------------------
// Foam intensity per particle (post-finalize, uses final velocities)
// ---------------------------------------------------------------------------
// Foam appears where the fluid is BOTH moving fast AND near the surface.
// Surface-ness comes from the spatial hash (neighborCount low → surface).
// Speed comes from the just-computed velocity. Output is in [0, 1] and is
// read by the foam render pass to set sprite brightness.

kernel void computeFoam(device const float4*  velocities    [[ buffer(0) ]],
                        device const uint*    neighborCounts[[ buffer(1) ]],
                        device float*         foamIntensity [[ buffer(2) ]],
                        constant uint&        count         [[ buffer(3) ]],
                        constant float&       maxNeighbors  [[ buffer(4) ]],
                        uint                  gid           [[ thread_position_in_grid ]])
{
    if (gid >= count) return;
    const float speed   = length(velocities[gid].xyz);
    const float surface = 1.0 - clamp(float(neighborCounts[gid]) / maxNeighbors, 0.0, 1.0);
    const float speedF  = smoothstep(1.2, 4.5, speed);   // foam threshold ~1.2 m/s
    foamIntensity[gid]  = surface * speedF;
}

// ---------------------------------------------------------------------------
// 4. Finalize: box clamp, velocity recompute, wall absorption + friction
// ---------------------------------------------------------------------------

kernel void finalizeStep(device float4*           positions       [[ buffer(0) ]],
                         device float4*           velocities      [[ buffer(1) ]],
                         device const float4*     prevPositions   [[ buffer(2) ]],
                         constant PBFParams&      pp              [[ buffer(7) ]],
                         uint                     gid             [[ thread_position_in_grid ]])
{
    if (gid >= pp.particleCount) return;
    float3 p     = positions[gid].xyz;
    float3 pPrev = prevPositions[gid].xyz;

    p = clamp(p, pp.boundsMin, pp.boundsMax);

    float3 v = (p - pPrev) * pp.invDt;
    v *= pp.damping;

    // Wall response: zero the velocity component INTO each wall (absorption),
    // then apply ANISOTROPIC friction:
    //   - floor: strong friction kills tangential skating
    //   - vertical walls: light friction so gravity drags particles down
    //     instead of trapping them like honey on a window
    const float weps = 1e-3f;
    const bool xMin = p.x <= pp.boundsMin.x + weps;
    const bool xMax = p.x >= pp.boundsMax.x - weps;
    const bool yMin = p.y <= pp.boundsMin.y + weps;
    const bool yMax = p.y >= pp.boundsMax.y - weps;
    const bool zMin = p.z <= pp.boundsMin.z + weps;
    const bool zMax = p.z >= pp.boundsMax.z - weps;

    if (xMin) v.x = max(v.x, 0.0f);
    if (xMax) v.x = min(v.x, 0.0f);
    if (yMin) v.y = max(v.y, 0.0f);
    if (yMax) v.y = min(v.y, 0.0f);
    if (zMin) v.z = max(v.z, 0.0f);
    if (zMax) v.z = min(v.z, 0.0f);

    if (yMin) {
        v.x *= 0.55f;
        v.z *= 0.55f;
    } else if (xMin || xMax || zMin || zMax) {
        v *= 0.97f;
    }

    const float vMax = 1.0f * pp.h * pp.invDt;
    const float vLen = length(v);
    if (vLen > vMax) v *= vMax / vLen;

    positions[gid]  = float4(p, 1.0f);
    velocities[gid] = float4(v, 0.0f);
}
