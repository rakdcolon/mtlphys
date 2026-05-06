// Constraints.metal
//
// PBF (Position-Based Fluids, Macklin & Müller 2013) constraint solver.
//
// Per simulation step:
//   1. predict          — x* = x + (v + g·dt)·dt; save prevPositions
//   2. (spatial hash on x*, in Spatial.metal)
//   3. for k iterations:
//        densityLambda  — ρ_i, then λ_i = -(ρ_i/ρ_0 - 1) / (Σ|∇C|² + ε)
//        gatherLambdas  — λ in sorted layout (sequential reads in applyDelta)
//        applyDelta     — Δx_i = (1/ρ_0) Σ_j (λ_i + λ_j) ∇W_spiky(r_ij, h)
//   4. finalize         — clamp to bounds, v = (x_new - x_prev)/dt, commit

#include <metal_stdlib>
#include "Shared.h"
using namespace metal;
using namespace mtlphys;

// ---------------------------------------------------------------------------
// SPH kernels
// ---------------------------------------------------------------------------

// W_poly6(r, h) = poly6Norm * (h^2 - r^2)^3   for r < h, else 0.
static inline float poly6(float r2, float h2, float poly6Norm) {
    if (r2 >= h2) return 0.0f;
    const float diff = h2 - r2;
    return poly6Norm * diff * diff * diff;
}

// ∇W_spiky(r_vec, h), where r_vec = p_i - p_j and gradient is w.r.t. p_i.
// Magnitude: -spikyGradNorm * (h - |r|)^2 along (r/|r|).
static inline float3 spikyGrad(float3 r_vec, float r_len, float h, float spikyGradNorm) {
    if (r_len <= 1e-6f || r_len >= h) return float3(0);
    const float diff = h - r_len;
    return -spikyGradNorm * diff * diff * (r_vec / r_len);
}

// Cell helpers — duplicated tiny inlines so this TU doesn't depend on Spatial.metal.
static inline uint3 cellCoordOf(float3 p, constant SpatialParams& sp) {
    float3 q = (p - sp.gridOrigin) * sp.invCellSize;
    int3 c = int3(floor(q));
    return uint3(clamp(c, int3(0), int3(sp.gridDim) - 1));
}
static inline uint cellHashOf(uint3 c, constant SpatialParams& sp) {
    return c.x + c.y * sp.gridDim.x + c.z * sp.gridDim.x * sp.gridDim.y;
}

// ---------------------------------------------------------------------------
// 1. Predict: save prev positions, advance by velocity + gravity
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

// ---------------------------------------------------------------------------
// 3a. densityLambda: density + Lagrange multiplier per particle
// ---------------------------------------------------------------------------

kernel void densityLambda(device const float4*       positions        [[ buffer(0) ]],
                          device const uint*         cellStart        [[ buffer(3) ]],
                          device const uint*         sortedIndex      [[ buffer(5) ]],
                          device const float4*       sortedPositions  [[ buffer(8) ]],
                          device float*              lambdas          [[ buffer(9) ]],
                          constant SpatialParams&    sp               [[ buffer(7) ]],
                          constant PBFParams&        pp               [[ buffer(10) ]],
                          uint                       gid              [[ thread_position_in_grid ]])
{
    if (gid >= sp.particleCount) return;

    const float3 p_i = positions[gid].xyz;
    const uint3  myC = cellCoordOf(p_i, sp);

    const int zLo = max(int(myC.z) - 1, 0), zHi = min(int(myC.z) + 1, int(sp.gridDim.z) - 1);
    const int yLo = max(int(myC.y) - 1, 0), yHi = min(int(myC.y) + 1, int(sp.gridDim.y) - 1);
    const int xLo = max(int(myC.x) - 1, 0), xHi = min(int(myC.x) + 1, int(sp.gridDim.x) - 1);

    float  density   = poly6(0.0f, pp.h2, pp.poly6Norm);  // self-contribution
    float3 gradSelf  = float3(0);                          // Σ_{j≠i} ∇W_ij
    float  gradSqSum = 0.0f;                               // Σ_{j≠i} |∇W_ij|^2

    for (int z = zLo; z <= zHi; ++z) {
        for (int y = yLo; y <= yHi; ++y) {
            for (int x = xLo; x <= xHi; ++x) {
                const uint c = cellHashOf(uint3(uint(x), uint(y), uint(z)), sp);
                const uint s = cellStart[c];
                const uint e = cellStart[c + 1];
                for (uint i = s; i < e; ++i) {
                    if (sortedIndex[i] == gid) continue;
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

    const float constraint  = density * pp.invRestDensity - 1.0f;
    const float denom       = pp.invRestDensity * pp.invRestDensity *
                              (dot(gradSelf, gradSelf) + gradSqSum) + pp.epsilon;
    lambdas[gid] = -constraint / denom;
}

// ---------------------------------------------------------------------------
// 3b. gatherLambdas: per-particle scalar -> sorted-slot layout
// ---------------------------------------------------------------------------

kernel void gatherLambdas(device const float*        lambdas        [[ buffer(9) ]],
                          device const uint*         sortedIndex    [[ buffer(5) ]],
                          device float*              sortedLambdas  [[ buffer(11) ]],
                          constant SpatialParams&    sp             [[ buffer(7) ]],
                          uint                       gid            [[ thread_position_in_grid ]])
{
    if (gid >= sp.particleCount) return;
    sortedLambdas[gid] = lambdas[sortedIndex[gid]];
}

// ---------------------------------------------------------------------------
// 3c. applyDelta: position correction from neighbor lambdas
// ---------------------------------------------------------------------------

kernel void applyDelta(device float4*               positions        [[ buffer(0) ]],
                       device const uint*           cellStart        [[ buffer(3) ]],
                       device const uint*           sortedIndex      [[ buffer(5) ]],
                       device const float*          lambdas          [[ buffer(9) ]],
                       device const float4*         sortedPositions  [[ buffer(8) ]],
                       device const float*          sortedLambdas    [[ buffer(11) ]],
                       constant SpatialParams&      sp               [[ buffer(7) ]],
                       constant PBFParams&          pp               [[ buffer(10) ]],
                       uint                         gid              [[ thread_position_in_grid ]])
{
    if (gid >= sp.particleCount) return;

    const float3 p_i      = positions[gid].xyz;
    const float  lambda_i = lambdas[gid];
    const uint3  myC      = cellCoordOf(p_i, sp);

    const int zLo = max(int(myC.z) - 1, 0), zHi = min(int(myC.z) + 1, int(sp.gridDim.z) - 1);
    const int yLo = max(int(myC.y) - 1, 0), yHi = min(int(myC.y) + 1, int(sp.gridDim.y) - 1);
    const int xLo = max(int(myC.x) - 1, 0), xHi = min(int(myC.x) + 1, int(sp.gridDim.x) - 1);

    // s_corr: artificial repulsion that prevents tensile instability (the
    // failure mode where surface particles cluster into clumps).
    // s_corr = -k · (W_poly6(r,h) / W_poly6(Δq·h, h))^n   with k=1e-4, n=4, Δq=0.2
    constexpr float kSCorrK    = 1e-4f;
    const     float dq2        = (0.2f * pp.h) * (0.2f * pp.h);
    const     float invWq      = 1.0f / poly6(dq2, pp.h2, pp.poly6Norm);

    float3 delta = float3(0);

    for (int z = zLo; z <= zHi; ++z) {
        for (int y = yLo; y <= yHi; ++y) {
            for (int x = xLo; x <= xHi; ++x) {
                const uint c = cellHashOf(uint3(uint(x), uint(y), uint(z)), sp);
                const uint s = cellStart[c];
                const uint e = cellStart[c + 1];
                for (uint i = s; i < e; ++i) {
                    if (sortedIndex[i] == gid) continue;
                    const float3 r  = p_i - sortedPositions[i].xyz;
                    const float  r2 = dot(r, r);
                    if (r2 >= pp.h2) continue;
                    const float  lambda_j = sortedLambdas[i];
                    const float3 grad     = spikyGrad(r, sqrt(r2), pp.h, pp.spikyGradNorm);
                    const float  wij      = poly6(r2, pp.h2, pp.poly6Norm);
                    const float  ratio    = wij * invWq;
                    const float  ratio2   = ratio * ratio;
                    const float  s_corr   = -kSCorrK * ratio2 * ratio2;       // ratio^4
                    delta += (lambda_i + lambda_j + s_corr) * grad;
                }
            }
        }
    }

    delta *= pp.invRestDensity;

    // CFL-like clamp: never move a particle more than 0.3·h in one iter, so
    // the solver can't catastrophically escape its neighborhood.
    const float maxStep = 0.3f * pp.h;
    const float dlen    = length(delta);
    if (dlen > maxStep) delta *= maxStep / dlen;

    positions[gid] = float4(p_i + delta, 1.0f);
}

// ---------------------------------------------------------------------------
// 4. Finalize: box clamp + velocity recompute + damping
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

    // Wall response: kill the velocity component INTO the wall (absorption)
    // and apply tangential friction (no-slip-like). Without friction PBF
    // produces "wall skating" — particles slide along boundaries unboundedly
    // because no neighbor slows their tangential motion.
    const float weps  = 1e-3f;
    const float kFric = 0.85f;  // tangential velocity multiplier at walls
    bool atWall = false;
    if (p.x <= pp.boundsMin.x + weps) { if (v.x < 0.0f) v.x = 0.0f; atWall = true; }
    if (p.y <= pp.boundsMin.y + weps) { if (v.y < 0.0f) v.y = 0.0f; atWall = true; }
    if (p.z <= pp.boundsMin.z + weps) { if (v.z < 0.0f) v.z = 0.0f; atWall = true; }
    if (p.x >= pp.boundsMax.x - weps) { if (v.x > 0.0f) v.x = 0.0f; atWall = true; }
    if (p.y >= pp.boundsMax.y - weps) { if (v.y > 0.0f) v.y = 0.0f; atWall = true; }
    if (p.z >= pp.boundsMax.z - weps) { if (v.z > 0.0f) v.z = 0.0f; atWall = true; }
    if (atWall) v *= kFric;

    // Velocity ceiling — one cell per frame (CFL-tight). Real PBF can tunnel
    // through neighborhoods at higher speeds, which cascades into instability.
    const float vMax = 1.0f * pp.h * pp.invDt;
    const float vLen = length(v);
    if (vLen > vMax) v *= vMax / vLen;

    positions[gid]  = float4(p, 1.0f);
    velocities[gid] = float4(v, 0.0f);
}
