// Spatial.metal
//
// Spatial-hash pipeline for neighbor finding.
//
// Per frame we run six dispatches:
//   1. hashCells           — for each particle: compute its cell index.
//   2. countCells          — atomically tally particles per cell.
//   3. scanChunkExclusive  — exclusive scan within 1024-element chunks of
//                            cellCounts; emits chunk totals.
//   4. scanChunkSums       — single-block exclusive scan over the chunk totals.
//   5. addChunkOffsets     — add scanned chunk-sums back into the per-chunk
//                            scan, producing a global exclusive scan = cellStart.
//   6. scatterParticles    — write each particle's index into its sorted slot.
//   (+ countNeighbors      — diagnostic / visualization pass.)
//
// After this runs:
//   sortedParticleIndex[ cellStart[c] .. cellStart[c+1] ) = particles in cell c.
//   cellStart has size totalCells + 1; cellStart[totalCells] == particleCount.

#include <metal_stdlib>
#include "Shared.h"
using namespace metal;
using namespace mtlphys;

// Two-level scan parameters. Chunk size must be a multiple of simdgroup width
// (32 on Apple GPUs). 1024 keeps tgmem usage tiny and gives enough parallelism.
constant uint kChunkSize = 1024;
constant uint kSimdWidth = 32;
constant uint kSimdsPerChunk = kChunkSize / kSimdWidth;  // = 32

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static inline uint3 cellCoordOf(float3 p, constant SpatialParams& sp) {
    float3 q = (p - sp.gridOrigin) * sp.invCellSize;
    int3 c   = int3(floor(q));
    return uint3(clamp(c, int3(0), int3(sp.gridDim) - 1));
}

static inline uint cellHashOf(uint3 c, constant SpatialParams& sp) {
    return c.x + c.y * sp.gridDim.x + c.z * sp.gridDim.x * sp.gridDim.y;
}

// ---------------------------------------------------------------------------
// 1. Cell hash per particle
// ---------------------------------------------------------------------------

kernel void hashCells(device const float4*       positions   [[ buffer(0) ]],
                      device uint*               cellHashes  [[ buffer(1) ]],
                      constant SpatialParams&    sp          [[ buffer(7) ]],
                      uint                       gid         [[ thread_position_in_grid ]])
{
    if (gid >= sp.particleCount) return;
    const float3 p = positions[gid].xyz;
    cellHashes[gid] = cellHashOf(cellCoordOf(p, sp), sp);
}

// ---------------------------------------------------------------------------
// 2. Atomic per-cell count
// ---------------------------------------------------------------------------

kernel void countCells(device const uint*               cellHashes  [[ buffer(1) ]],
                       device atomic_uint*              cellCounts  [[ buffer(2) ]],
                       constant SpatialParams&          sp          [[ buffer(7) ]],
                       uint                             gid         [[ thread_position_in_grid ]])
{
    if (gid >= sp.particleCount) return;
    atomic_fetch_add_explicit(&cellCounts[cellHashes[gid]], 1u, memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// 3. Per-chunk exclusive scan with simdgroup intrinsics
// ---------------------------------------------------------------------------
//
// Two-level inside the chunk:
//   a) per-simdgroup exclusive scan via simd_prefix_exclusive_sum (32 lanes)
//   b) single-simdgroup scan of the 32 simdgroup totals
//   c) add stage-(b) result back to stage-(a)
// Each chunk also writes its inclusive total into chunkSums[bid].

kernel void scanChunkExclusive(device const uint*  in         [[ buffer(0) ]],
                               device uint*        out        [[ buffer(1) ]],
                               device uint*        chunkSums  [[ buffer(2) ]],
                               constant uint&      n          [[ buffer(3) ]],
                               uint                gid        [[ thread_position_in_grid ]],
                               uint                tid        [[ thread_position_in_threadgroup ]],
                               uint                bid        [[ threadgroup_position_in_grid ]],
                               uint                laneId     [[ thread_index_in_simdgroup ]],
                               uint                sgId       [[ simdgroup_index_in_threadgroup ]])
{
    threadgroup uint sgSums[kSimdsPerChunk];

    uint val    = (gid < n) ? in[gid] : 0u;
    uint sgPref = simd_prefix_exclusive_sum(val);                        // (a)
    uint sgTot  = simd_broadcast(sgPref + val, kSimdWidth - 1);

    if (laneId == 0) sgSums[sgId] = sgTot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgId == 0) {                                                     // (b)
        uint v = (laneId < kSimdsPerChunk) ? sgSums[laneId] : 0u;
        uint p = simd_prefix_exclusive_sum(v);
        if (laneId < kSimdsPerChunk) sgSums[laneId] = p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint chunkPref = sgSums[sgId] + sgPref;                              // (c)
    if (gid < n) out[gid] = chunkPref;

    // Last live thread in this chunk publishes the inclusive total of the chunk.
    if (tid == kChunkSize - 1) {
        chunkSums[bid] = chunkPref + val;
    }
}

// ---------------------------------------------------------------------------
// 4. Single-block exclusive scan over chunkSums
// ---------------------------------------------------------------------------
// chunkSums has at most kChunkSize entries (currently — extend to a third
// level if we ever exceed ~1M cells).

kernel void scanChunkSums(device uint*       chunkSums [[ buffer(0) ]],
                          constant uint&     m         [[ buffer(3) ]],
                          uint               tid       [[ thread_position_in_threadgroup ]],
                          uint               laneId    [[ thread_index_in_simdgroup ]],
                          uint               sgId      [[ simdgroup_index_in_threadgroup ]])
{
    threadgroup uint sgSums[kSimdsPerChunk];

    uint val    = (tid < m) ? chunkSums[tid] : 0u;
    uint sgPref = simd_prefix_exclusive_sum(val);
    uint sgTot  = simd_broadcast(sgPref + val, kSimdWidth - 1);

    if (laneId == 0) sgSums[sgId] = sgTot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgId == 0) {
        uint v = (laneId < kSimdsPerChunk) ? sgSums[laneId] : 0u;
        uint p = simd_prefix_exclusive_sum(v);
        if (laneId < kSimdsPerChunk) sgSums[laneId] = p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint chunkPref = sgSums[sgId] + sgPref;
    if (tid < m) chunkSums[tid] = chunkPref;
}

// ---------------------------------------------------------------------------
// 5. Add scanned chunk-sums back to per-chunk scan results
// ---------------------------------------------------------------------------

kernel void addChunkOffsets(device uint*         data       [[ buffer(0) ]],
                            device const uint*   chunkSums  [[ buffer(2) ]],
                            constant uint&       n          [[ buffer(3) ]],
                            uint                 gid        [[ thread_position_in_grid ]],
                            uint                 bid        [[ threadgroup_position_in_grid ]])
{
    if (gid >= n || bid == 0) return;
    data[gid] += chunkSums[bid];
}

// ---------------------------------------------------------------------------
// 6. Scatter particle indices into sorted layout
// ---------------------------------------------------------------------------
// Atomic-add gives non-deterministic *intra-cell* ordering, which is fine — we
// only require cell grouping. cellCursor must be zeroed before this dispatch.

kernel void scatterParticles(device const uint*       cellHashes        [[ buffer(1) ]],
                             device const uint*       cellStart         [[ buffer(3) ]],
                             device atomic_uint*      cellCursor        [[ buffer(4) ]],
                             device uint*             sortedIndex       [[ buffer(5) ]],
                             constant SpatialParams&  sp                [[ buffer(7) ]],
                             uint                     gid               [[ thread_position_in_grid ]])
{
    if (gid >= sp.particleCount) return;
    const uint c        = cellHashes[gid];
    const uint localIdx = atomic_fetch_add_explicit(&cellCursor[c], 1u, memory_order_relaxed);
    sortedIndex[cellStart[c] + localIdx] = gid;
}

// ---------------------------------------------------------------------------
// 7. Gather: build a positions array indexed by sorted slot
// ---------------------------------------------------------------------------
// `positions[sortedIndex[i]]` is a random-access gather. Doing it once here
// makes the hot inner loop in countNeighbors a sequential read — huge cache
// win because particles in the same cell now land in adjacent memory.
// (Pattern from NVIDIA Flex / MuJoCo MJX.)

kernel void gatherPositions(device const float4*       positions        [[ buffer(0) ]],
                            device const uint*         sortedIndex      [[ buffer(5) ]],
                            device float4*             sortedPositions  [[ buffer(8) ]],
                            constant SpatialParams&    sp               [[ buffer(7) ]],
                            uint                       gid              [[ thread_position_in_grid ]])
{
    if (gid >= sp.particleCount) return;
    sortedPositions[gid] = positions[sortedIndex[gid]];
}

// ---------------------------------------------------------------------------
// 8. Neighbor count (visualization / W3 prep)
// ---------------------------------------------------------------------------
// Inner loop now reads sortedPositions[i] sequentially within each cell. The
// self-position still comes from positions[gid] (one read per thread, free).
// Self-skip is via sortedIndex[i] == gid.

kernel void countNeighbors(device const float4*       positions        [[ buffer(0) ]],
                           device const uint*         cellStart        [[ buffer(3) ]],
                           device const uint*         sortedIndex      [[ buffer(5) ]],
                           device uint*               neighborCounts   [[ buffer(6) ]],
                           constant SpatialParams&    sp               [[ buffer(7) ]],
                           device const float4*       sortedPositions  [[ buffer(8) ]],
                           uint                       gid              [[ thread_position_in_grid ]])
{
    if (gid >= sp.particleCount) return;

    const float3 p     = positions[gid].xyz;
    const uint3  myC   = cellCoordOf(p, sp);
    const int3   loC   = int3(myC) - 1;
    const int3   hiC   = int3(myC) + 1;

    const int zLo = max(loC.z, 0), zHi = min(hiC.z, int(sp.gridDim.z) - 1);
    const int yLo = max(loC.y, 0), yHi = min(hiC.y, int(sp.gridDim.y) - 1);
    const int xLo = max(loC.x, 0), xHi = min(hiC.x, int(sp.gridDim.x) - 1);

    uint count = 0;
    for (int z = zLo; z <= zHi; ++z) {
        for (int y = yLo; y <= yHi; ++y) {
            for (int x = xLo; x <= xHi; ++x) {
                const uint c = cellHashOf(uint3(uint(x), uint(y), uint(z)), sp);
                const uint s = cellStart[c];
                const uint e = cellStart[c + 1];
                for (uint i = s; i < e; ++i) {
                    if (sortedIndex[i] == gid) continue;
                    const float3 d = sortedPositions[i].xyz - p;
                    if (dot(d, d) < sp.neighborRadiusSq) ++count;
                }
            }
        }
    }

    neighborCounts[gid] = count;
}
