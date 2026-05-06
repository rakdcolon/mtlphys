// Render.metal
// Particle rendering as camera-facing billboards. Each particle becomes a
// soft-edged disk via fragment-shader distance-from-center. No geometry
// shaders / point sprites — we expand 6 vertices per instance from the
// vertex shader, which is the M-series-friendly path.

#include <metal_stdlib>
#include "Shared.h"
using namespace metal;
using namespace mtlphys;

struct VertexOut {
    float4 position [[ position ]];
    float2 uv;
    float3 worldPos;
};

// Two triangles forming a unit quad in clip-space corner offsets.
constant float2 kCorners[6] = {
    float2(-1, -1), float2( 1, -1), float2(-1,  1),
    float2( 1, -1), float2( 1,  1), float2(-1,  1),
};

vertex VertexOut particleVertex(constant float4*         positions [[ buffer(0) ]],
                                constant RenderUniforms& u         [[ buffer(1) ]],
                                uint                     vid       [[ vertex_id ]],
                                uint                     iid       [[ instance_id ]])
{
    const float3 center = positions[iid].xyz;
    const float2 corner = kCorners[vid];

    // Build a camera-facing basis. Cheap version: derive right/up from view direction.
    const float3 toCam = normalize(u.cameraPos - center);
    const float3 worldUp = float3(0, 1, 0);
    const float3 right = normalize(cross(worldUp, toCam));
    const float3 up    = cross(toCam, right);

    const float3 worldPos = center + (right * corner.x + up * corner.y) * u.particleRadius;

    VertexOut o;
    o.position = u.viewProj * float4(worldPos, 1.0);
    o.uv       = corner;
    o.worldPos = worldPos;
    return o;
}

fragment float4 particleFragment(VertexOut in [[ stage_in ]])
{
    // Soft circular falloff — discards corners of the quad.
    const float r2 = dot(in.uv, in.uv);
    if (r2 > 1.0) discard_fragment();

    // Fake lighting via height for some sense of 3D depth in the cloud.
    const float shade = saturate(0.6 + 0.1 * in.worldPos.y);
    const float alpha = smoothstep(1.0, 0.7, r2);

    // Cool blue-white color, looks "physics-y".
    const float3 base = mix(float3(0.20, 0.55, 0.95), float3(0.85, 0.95, 1.0), shade);
    return float4(base, alpha);
}
