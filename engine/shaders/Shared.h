// Shared between C++ host code and Metal Shading Language.
// Keeps struct layouts in sync — change once, both sides see it.
#pragma once

#ifdef __METAL_VERSION__
  #include <metal_stdlib>
  using namespace metal;
  namespace mp { using float3 = ::float3; using float4 = ::float4; using float4x4 = ::float4x4; }
#else
  #include <simd/simd.h>
  namespace mp { using float3 = simd::float3; using float4 = simd::float4; using float4x4 = simd::float4x4; }
#endif

namespace mtlphys {

// 16-byte aligned for vectorized GPU access. We store position and velocity
// as separate buffers (SoA-ish) because most kernels touch one or the other,
// not both — better cache utilization than packing them together.
struct alignas(16) SimParams {
    mp::float3 gravity;        // m/s^2
    float      dt;             // seconds
    mp::float3 boundsMin;      // box lower corner
    float      damping;        // velocity damping per step (0..1, multiplicative)
    mp::float3 boundsMax;      // box upper corner
    uint32_t   particleCount;
};

struct alignas(16) RenderUniforms {
    mp::float4x4 viewProj;
    mp::float3   cameraPos;
    float        particleRadius;
};

} // namespace mtlphys
