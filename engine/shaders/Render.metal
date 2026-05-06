// Render.metal
// Particle rendering as camera-facing billboards. Each particle becomes a
// soft-edged disk via fragment-shader distance-from-center. Color ramps from
// cool (sparse) to warm (dense) based on neighbor count, so the spatial-hash
// data structure is *visible* — not just a number in the HUD.

#include <metal_stdlib>
#include "Shared.h"
using namespace metal;
using namespace mtlphys;

struct VertexOut {
    float4 position [[ position ]];
    float2 uv;
    float3 worldPos;
    float  density;       // 0..1, derived from neighbor count
};

// Two triangles forming a unit quad in clip-space corner offsets.
constant float2 kCorners[6] = {
    float2(-1, -1), float2( 1, -1), float2(-1,  1),
    float2( 1, -1), float2( 1,  1), float2(-1,  1),
};

vertex VertexOut particleVertex(constant float4*         positions      [[ buffer(0) ]],
                                constant RenderUniforms& u              [[ buffer(1) ]],
                                constant uint*           neighborCounts [[ buffer(2) ]],
                                constant uint&           maxNeighbors   [[ buffer(3) ]],
                                uint                     vid            [[ vertex_id ]],
                                uint                     iid            [[ instance_id ]])
{
    const float3 center = positions[iid].xyz;
    const float2 corner = kCorners[vid];

    const float3 toCam = normalize(u.cameraPos - center);
    const float3 worldUp = float3(0, 1, 0);
    const float3 right = normalize(cross(worldUp, toCam));
    const float3 up    = cross(toCam, right);

    const float3 worldPos = center + (right * corner.x + up * corner.y) * u.particleRadius;

    VertexOut o;
    o.position = u.viewProj * float4(worldPos, 1.0);
    o.uv       = corner;
    o.worldPos = worldPos;
    o.density  = saturate(float(neighborCounts[iid]) / float(max(maxNeighbors, 1u)));
    return o;
}

// Three-stop palette: cool blue → green → warm orange/red as density grows.
static inline float3 densityRamp(float t) {
    const float3 c0 = float3(0.18, 0.55, 0.95);
    const float3 c1 = float3(0.30, 0.85, 0.55);
    const float3 c2 = float3(0.95, 0.45, 0.20);
    return (t < 0.5) ? mix(c0, c1, t * 2.0) : mix(c1, c2, (t - 0.5) * 2.0);
}

fragment float4 particleFragment(VertexOut in [[ stage_in ]])
{
    const float r2 = dot(in.uv, in.uv);
    if (r2 > 1.0) discard_fragment();

    const float alpha = smoothstep(1.0, 0.7, r2);
    const float3 base = densityRamp(in.density);
    return float4(base, alpha);
}
