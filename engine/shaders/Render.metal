// Render.metal — Screen-Space Fluid (SSF) rendering pipeline.
//
// Three passes:
//   1. fluidDepth*       — render each particle as a sphere; output linear
//                          view-space depth into an R32Float texture, with
//                          a Depth32Float attachment so the closest particle
//                          wins per pixel.
//   2. fluidSmooth*      — fullscreen Gaussian blur over the depth texture;
//                          treats "no fluid" pixels (depth==0) as a mask so
//                          fluid edges stay sharp instead of bleeding into
//                          the background.
//   3. fluidComposite*   — fullscreen shader that reads smoothed depth,
//                          reconstructs view-space position per pixel, derives
//                          a surface normal via dFdx/dFdy, and shades with
//                          Lambert diffuse + Blinn-Phong specular + Fresnel.

#include <metal_stdlib>
#include "Shared.h"
using namespace metal;
using namespace mtlphys;

// ---------------------------------------------------------------------------
// Pass 1: per-particle sphere depth
// ---------------------------------------------------------------------------

struct DepthVOut {
    float4 position    [[ position ]];
    float2 uv;            // billboard corner offset, [-1, 1]
    float3 viewCenter;    // particle center in view space
};

constant float2 kCorners[6] = {
    float2(-1, -1), float2( 1, -1), float2(-1,  1),
    float2( 1, -1), float2( 1,  1), float2(-1,  1),
};

vertex DepthVOut fluidDepthVertex(constant float4*         positions [[ buffer(0) ]],
                                  constant RenderUniforms& u         [[ buffer(1) ]],
                                  uint                     vid       [[ vertex_id ]],
                                  uint                     iid       [[ instance_id ]])
{
    const float3 worldCenter = positions[iid].xyz;
    const float3 viewCenter  = (u.view * float4(worldCenter, 1.0)).xyz;
    const float2 corner      = kCorners[vid];

    // Camera-aligned billboard: just offset in view-space XY by the radius.
    const float3 viewPos = viewCenter + float3(corner.x * u.particleRadius,
                                               corner.y * u.particleRadius,
                                               0);
    DepthVOut o;
    o.position   = u.proj * float4(viewPos, 1.0);
    o.uv         = corner;
    o.viewCenter = viewCenter;
    return o;
}

struct DepthFOut {
    float color [[ color(0) ]];   // R32Float: linear view-space distance
    float depth [[ depth(less) ]]; // for proper sphere-sphere occlusion
};

fragment DepthFOut fluidDepthFragment(DepthVOut                in [[ stage_in ]],
                                      constant RenderUniforms& u  [[ buffer(1) ]])
{
    const float r2 = dot(in.uv, in.uv);
    if (r2 > 1.0) discard_fragment();

    // Sphere surface offset toward camera (positive z in view space).
    const float zOff = sqrt(1.0 - r2);

    // View-space position of this fragment on the sphere surface.
    const float3 viewPos = in.viewCenter + float3(in.uv.x * u.particleRadius,
                                                  in.uv.y * u.particleRadius,
                                                  u.particleRadius * zOff);

    // Re-project to clip space so the depth attachment gets the correct
    // sphere depth (not the flat billboard depth).
    const float4 clipPos = u.proj * float4(viewPos, 1.0);
    const float  ndcZ    = clipPos.z / clipPos.w;

    DepthFOut o;
    o.color = -viewPos.z;   // linear view distance: positive, larger = farther
    o.depth = ndcZ;
    return o;
}

// ---------------------------------------------------------------------------
// Fullscreen helpers
// ---------------------------------------------------------------------------

struct FSOut {
    float4 position [[ position ]];
    float2 uv;
};

vertex FSOut fullscreenVertex(uint vid [[ vertex_id ]]) {
    // Single triangle covering the screen. uv ∈ [0,1] over the visible area.
    constexpr float2 verts[3] = { float2(-1, -1), float2(3, -1), float2(-1, 3) };
    constexpr float2 uvs[3]   = { float2( 0,  1), float2(2,  1), float2( 0, -1) };
    FSOut o;
    o.position = float4(verts[vid], 0, 1);
    o.uv       = uvs[vid];
    return o;
}

// ---------------------------------------------------------------------------
// Pass 2: depth smoothing (masked Gaussian)
// ---------------------------------------------------------------------------

constant float kSmoothSigma = 2.0;
constant int   kSmoothRadius = 3;

fragment float4 fluidSmoothFragment(FSOut                    in        [[ stage_in ]],
                                    texture2d<float>         depthTex  [[ texture(0) ]],
                                    constant RenderUniforms& u         [[ buffer(1) ]])
{
    constexpr sampler smp(coord::normalized, filter::nearest, address::clamp_to_edge);

    const float center = depthTex.sample(smp, in.uv).r;
    if (center <= 0.0) return float4(0);   // pass background through unchanged

    float sum    = 0.0;
    float weight = 0.0;
    const float twoSigmaSq = 2.0 * kSmoothSigma * kSmoothSigma;

    for (int dy = -kSmoothRadius; dy <= kSmoothRadius; ++dy) {
        for (int dx = -kSmoothRadius; dx <= kSmoothRadius; ++dx) {
            const float2 off = float2(dx, dy) * u.invScreen;
            const float  d   = depthTex.sample(smp, in.uv + off).r;
            if (d > 0.0) {
                const float w = exp(-(dx*dx + dy*dy) / twoSigmaSq);
                sum    += d * w;
                weight += w;
            }
        }
    }

    return float4(sum / max(weight, 1e-6), 0, 0, 1);
}

// ---------------------------------------------------------------------------
// Pass 3: composite — reconstruct surface and shade
// ---------------------------------------------------------------------------

// Reconstruct view-space position from a UV and a linear view-distance.
static inline float3 viewPosFromDepth(float2 uv, float depth, constant RenderUniforms& u) {
    // UV ∈ [0,1] → NDC ∈ [-1,1]; flip Y because Metal UV is top-down but NDC is bottom-up.
    float2 ndc = uv * 2.0 - 1.0;
    ndc.y      = -ndc.y;
    return float3(ndc.x * u.tanHalfFovY * u.aspect * depth,
                  ndc.y * u.tanHalfFovY * depth,
                  -depth);
}

fragment float4 fluidCompositeFragment(FSOut                    in            [[ stage_in ]],
                                       texture2d<float>         smoothedDepth [[ texture(0) ]],
                                       constant RenderUniforms& u             [[ buffer(1) ]])
{
    constexpr sampler smp(coord::normalized, filter::linear, address::clamp_to_edge);

    const float depth = smoothedDepth.sample(smp, in.uv).r;
    if (depth <= 0.0) discard_fragment();   // let MTKView clear color show through

    // Reconstruct view-space position; use derivatives across the 2x2 quad to
    // build the surface normal. Fast and exactly what we want — derivatives of
    // the smoothed surface field.
    const float3 viewPos = viewPosFromDepth(in.uv, depth, u);
    const float3 dx      = dfdx(viewPos);
    const float3 dy      = dfdy(viewPos);
    const float3 N       = normalize(cross(dy, dx));

    // Shading. Light direction expressed in view space (camera frame),
    // mostly downward and slightly to the right — looks plausible for a
    // top-lit scene without needing a light gizmo.
    const float3 L = normalize(float3(0.4, 0.7, 0.6));
    const float3 V = float3(0, 0, 1);
    const float3 H = normalize(L + V);

    const float  diff   = max(dot(N, L), 0.0);
    const float  spec   = pow(max(dot(N, H), 0.0), 64.0);
    const float  fres   = pow(1.0 - max(dot(N, V), 0.0), 3.0);

    const float3 base   = float3(0.18, 0.45, 0.78);
    const float3 sky    = float3(0.55, 0.78, 0.95);

    const float3 color  = base * (0.35 + 0.65 * diff)
                        + float3(1, 1, 1) * spec * 0.9
                        + sky * fres * 0.55;

    return float4(color, 1.0);
}
