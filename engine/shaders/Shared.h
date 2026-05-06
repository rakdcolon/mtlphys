// Shared between C++ host code and Metal Shading Language.
// Keeps struct layouts in sync — change once, both sides see it.
#pragma once

#ifdef __METAL_VERSION__
  #include <metal_stdlib>
  using namespace metal;
  namespace mp {
      using float3   = ::float3;
      using float4   = ::float4;
      using float4x4 = ::float4x4;
      using uint3    = ::uint3;
  }
#else
  #include <simd/simd.h>
  namespace mp {
      using float3   = simd::float3;
      using float4   = simd::float4;
      using float4x4 = simd::float4x4;
      using uint3    = simd::uint3;
  }
#endif

namespace mtlphys {

struct alignas(16) SimParams {
    mp::float3 gravity;        // m/s^2
    float      dt;             // seconds
    mp::float3 boundsMin;      // box lower corner
    float      damping;        // velocity damping per step (0..1, multiplicative)
    mp::float3 boundsMax;      // box upper corner
    uint32_t   particleCount;
};

// Spatial-hash uniforms. The grid is a regular 3D lattice covering
// [gridOrigin, gridOrigin + gridDim * cellSize). Cell index of a point p is
// floor((p - gridOrigin) / cellSize) clamped to [0, gridDim - 1].
struct alignas(16) SpatialParams {
    mp::float3 gridOrigin;
    float      cellSize;
    float      invCellSize;
    float      neighborRadiusSq;  // squared distance threshold for "neighbor"
    uint32_t   particleCount;
    uint32_t   totalCells;
    mp::uint3  gridDim;
    uint32_t   maxNeighborsForViz; // count saturates here for color ramp
};

struct alignas(16) RenderUniforms {
    mp::float4x4 viewProj;
    mp::float3   cameraPos;
    float        particleRadius;
};

} // namespace mtlphys
