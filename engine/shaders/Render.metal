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
// Pass 1b: per-particle thickness (additive Gaussian)
// ---------------------------------------------------------------------------
// Reuses the depth pass's vertex shader (same billboards). Each fragment
// contributes a small Gaussian to the thickness buffer; with additive
// blending, summing over depth gives an estimate of how much fluid each
// view ray traverses. The composite uses this for Beer-Lambert absorption.

fragment float fluidThicknessFragment(DepthVOut                in [[ stage_in ]])
{
    const float r2 = dot(in.uv, in.uv);
    if (r2 > 1.0) discard_fragment();
    // Soft Gaussian; the constant scale (0.05) is the per-particle thickness
    // contribution at the center, tuned to make ~25 stacked particles look
    // visibly deep while leaving thin film transparent.
    return 0.05 * exp(-r2 * 4.0);
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

constant float kSmoothSigma  = 2.8;
constant int   kSmoothRadius = 4;

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

// World-space camera basis extracted from the view matrix.
// View maps world→view; world camera-axes are the rows of view's 3x3 block.
static inline float3 camRightWorld(constant RenderUniforms& u)   { return float3(u.view[0].x, u.view[1].x, u.view[2].x); }
static inline float3 camUpWorld(constant RenderUniforms& u)      { return float3(u.view[0].y, u.view[1].y, u.view[2].y); }
static inline float3 camForwardWorld(constant RenderUniforms& u) { return -float3(u.view[0].z, u.view[1].z, u.view[2].z); }

// World-space view ray for a screen-space UV.
static inline float3 worldViewRay(float2 uv, constant RenderUniforms& u) {
    float2 ndc = uv * 2.0 - 1.0;
    ndc.y      = -ndc.y;
    return normalize(camForwardWorld(u)
                   + camRightWorld(u) * (ndc.x * u.tanHalfFovY * u.aspect)
                   + camUpWorld(u)    * (ndc.y * u.tanHalfFovY));
}

// Sun direction in world space — shared by skyColor and the fluid lighting
// so the sun's specular reflection actually hits where you'd expect.
static inline float3 sunDirWorld() { return normalize(float3(0.45, 0.75, 0.50)); }

// Procedural sky: vertical gradient from horizon→zenith above, horizon→ground
// below; soft sun glow around the shared sun direction.
static inline float3 skyColor(float3 rayDir) {
    constexpr float3 zenith  = float3(0.18, 0.42, 0.78);
    constexpr float3 horizon = float3(0.85, 0.91, 0.97);
    constexpr float3 ground  = float3(0.08, 0.09, 0.11);

    const float h = rayDir.y;
    float3 col = (h >= 0)
        ? mix(horizon, zenith, pow(h, 0.55))
        : mix(horizon, ground, pow(-h, 0.45));

    const float sd   = max(dot(rayDir, sunDirWorld()), 0.0);
    const float core = pow(sd, 380.0);
    const float halo = pow(sd, 6.0) * 0.18;
    col += float3(1.0, 0.96, 0.86) * (core + halo);

    return col;
}

struct CompositeOut {
    float4 color [[ color(0) ]];
    float  depth [[ depth(any) ]];   // proper fluid surface depth so the box draw can depth-test against us
};

fragment CompositeOut fluidCompositeFragment(FSOut                    in            [[ stage_in ]],
                                             texture2d<float>         smoothedDepth [[ texture(0) ]],
                                             texture2d<float>         thicknessTex  [[ texture(1) ]],
                                             constant RenderUniforms& u             [[ buffer(1) ]])
{
    constexpr sampler smp(coord::normalized, filter::linear, address::clamp_to_edge);

    const float3 rayWorld = worldViewRay(in.uv, u);

    const float depth = smoothedDepth.sample(smp, in.uv).r;
    if (depth <= 0.0) {
        CompositeOut o;
        o.color = float4(skyColor(rayWorld), 1.0);
        o.depth = 1.0;
        return o;
    }

    const float3 viewPos = viewPosFromDepth(in.uv, depth, u);
    const float3 dx      = dfdx(viewPos);
    const float3 dy      = dfdy(viewPos);
    const float3 N_view  = normalize(cross(dy, dx));
    const float3 N_world = normalize(camRightWorld(u)   * N_view.x
                                   + camUpWorld(u)      * N_view.y
                                   + camForwardWorld(u) * (-N_view.z));

    // ---- Reflection & refraction (PBR-style water) ------------------------
    // Reflected ray: looks at the sky bouncing off the surface.
    const float3 reflWorld = reflect(rayWorld, N_world);
    const float3 reflSky   = skyColor(reflWorld);

    // Refracted ray: looks at what's THROUGH the surface (n_water = 1.33).
    // Going air→water so total internal reflection isn't possible, but we
    // guard the zero-vector case anyway.
    constexpr float eta      = 1.0 / 1.33;
    const     float3 refrWorld = refract(rayWorld, N_world, eta);
    const     float3 refrSky   = skyColor(length(refrWorld) < 1e-4 ? reflWorld : refrWorld);

    // ---- Beer-Lambert absorption applied to the transmitted light ---------
    // Per-channel extinction: water absorbs red strongly, green moderately,
    // blue least. The transmitted sky color gets multiplied by exp(-α·t),
    // so thin films stay near sky color and thick fluid darkens into deep
    // blue-green.
    const     float  thickness   = thicknessTex.sample(smp, in.uv).r;
    constexpr float3 alpha       = float3(2.5, 0.55, 0.25);
    const     float3 attenuation = exp(-alpha * thickness);
    const     float3 transmitted = refrSky * attenuation;

    // ---- Schlick Fresnel (F0=0.02 is water/air at normal incidence) ------
    const float cosI    = max(dot(-rayWorld, N_world), 0.0);
    const float schlick = 0.02 + 0.98 * pow(1.0 - cosI, 5.0);

    // ---- Hemispherical sky ambient ---------------------------------------
    // Sample the sky in the surface-normal direction — surfaces facing up
    // pick up zenith blue, surfaces facing the horizon pick up brighter
    // horizon color, etc. Provides natural fill light that varies across
    // the surface so flat regions don't look uniformly flat. Weighted by
    // (1 - schlick) so we don't over-brighten the already-mirrored
    // grazing-angle pixels.
    const float3 skyFill = skyColor(N_world) * 0.18 * (1.0 - schlick);

    // ---- Explicit sun specular --------------------------------------------
    // The sun is in skyColor() but it's a wide halo. Add a sharp glisten on
    // top: when the reflection ray aligns with the sun, we get a tight
    // bright highlight — the "glint" you see on real water.
    const float sunAlign = max(dot(reflWorld, sunDirWorld()), 0.0);
    const float3 sunGlint = float3(1.0, 0.96, 0.84) * pow(sunAlign, 320.0) * 2.5;

    const float3 color = mix(transmitted, reflSky, schlick)
                       + skyFill
                       + sunGlint;

    const float4 clip = u.proj * float4(viewPos, 1.0);
    CompositeOut o;
    o.color = float4(color, 1.0);
    o.depth = clip.z / clip.w;
    return o;
}

// ---------------------------------------------------------------------------
// Bounding box wireframe — drawn AFTER composite in the same encoder, with
// Less depth test so lines are hidden where they sit behind fluid.
// ---------------------------------------------------------------------------
//
// 12 edges × 2 vertices = 24-vertex line list. Cube corners are indexed
// 0..7 with bit 0 = x, bit 1 = y, bit 2 = z (0 = boundsMin, 1 = boundsMax).

struct WireOut {
    float4 position [[ position ]];
};

constant uint kEdgeCorners[24] = {
    // bottom (y = min)
    0,1,  1,3,  3,2,  2,0,
    // top (y = max — NB this is bit-2 actually; see corner-bit mapping above)
    4,5,  5,7,  7,6,  6,4,
    // verticals
    0,4,  1,5,  2,6,  3,7,
};

vertex WireOut wireBoxVertex(constant RenderUniforms& u         [[ buffer(1) ]],
                             constant float3&         boundsMin [[ buffer(2) ]],
                             constant float3&         boundsMax [[ buffer(3) ]],
                             uint                     vid       [[ vertex_id ]])
{
    const uint c = kEdgeCorners[vid];
    const float3 p = float3((c & 1u) ? boundsMax.x : boundsMin.x,
                            (c & 2u) ? boundsMax.y : boundsMin.y,
                            (c & 4u) ? boundsMax.z : boundsMin.z);
    WireOut o;
    o.position = u.proj * u.view * float4(p, 1.0);
    return o;
}

fragment float4 wireBoxFragment() {
    // Cool blue-grey, slightly translucent. Subtle so it reads as a guide,
    // not a solid frame.
    return float4(0.62, 0.72, 0.86, 0.85);
}

// ---------------------------------------------------------------------------
// Foam — additive bright sprites for fast surface particles
// ---------------------------------------------------------------------------

struct FoamOut {
    float4 position  [[ position ]];
    float2 uv;
    float  intensity;
};

vertex FoamOut foamVertex(constant float4*         positions     [[ buffer(0) ]],
                          constant RenderUniforms& u             [[ buffer(1) ]],
                          constant float*          foamIntensity [[ buffer(2) ]],
                          uint                     vid           [[ vertex_id ]],
                          uint                     iid           [[ instance_id ]])
{
    const float intensity = foamIntensity[iid];

    // Cull below threshold by collapsing the quad off-screen — saves
    // fragment work for the vast majority of particles which never foam.
    if (intensity < 0.05) {
        FoamOut o;
        o.position  = float4(2, 2, 2, 1);
        o.uv        = float2(0);
        o.intensity = 0;
        return o;
    }

    const float3 worldCenter = positions[iid].xyz;
    const float3 viewCenter  = (u.view * float4(worldCenter, 1)).xyz;
    const float2 corner      = kCorners[vid];

    // Very tight sprite — smaller than the visible SSF particle so foam reads
    // as a bright dot rather than a halo extending around the particle.
    const float  radius      = u.particleRadius * 0.25;
    const float3 viewPos     = viewCenter + float3(corner.x * radius,
                                                   corner.y * radius, 0);
    FoamOut o;
    o.position  = u.proj * float4(viewPos, 1);
    o.uv        = corner;
    o.intensity = intensity;
    return o;
}

fragment float4 foamFragment(FoamOut in [[ stage_in ]])
{
    if (in.intensity < 0.05) discard_fragment();
    const float r2 = dot(in.uv, in.uv);
    if (r2 > 1.0) discard_fragment();
    // Sharp Gaussian (exp(-7r²)) so the foam stays inside the sprite quad
    // and doesn't bleed outward as a halo around isolated particles.
    const float falloff = exp(-r2 * 7.0) * in.intensity * 0.65;
    return float4(falloff, falloff, falloff, falloff);
}

// ---------------------------------------------------------------------------
// Rigid sphere render — one camera-facing billboard, sphere depth + Phong
// shading in the fragment. Drawn after the fluid composite so the fluid
// surface depth-tests properly against it (sphere visible above water,
// occluded below).
// ---------------------------------------------------------------------------

struct SphereVOut {
    float4 position    [[ position ]];
    float2 uv;
    float3 viewCenter;
};

vertex SphereVOut sphereDrawVertex(constant RenderUniforms& u      [[ buffer(1) ]],
                                   constant SphereParams&   sphere [[ buffer(13) ]],
                                   uint                     vid    [[ vertex_id ]])
{
    const float2 corner = kCorners[vid];
    const float3 viewCenter = (u.view * float4(sphere.center, 1.0)).xyz;
    const float3 viewPos = viewCenter + float3(corner.x * sphere.radius,
                                               corner.y * sphere.radius, 0);
    SphereVOut o;
    o.position   = u.proj * float4(viewPos, 1.0);
    o.uv         = corner;
    o.viewCenter = viewCenter;
    return o;
}

struct SphereFOut {
    float4 color [[ color(0) ]];
    float  depth [[ depth(less) ]];
};

fragment SphereFOut sphereDrawFragment(SphereVOut               in     [[ stage_in ]],
                                       constant RenderUniforms& u      [[ buffer(1) ]],
                                       constant SphereParams&   sphere [[ buffer(13) ]])
{
    const float r2 = dot(in.uv, in.uv);
    if (r2 > 1.0) discard_fragment();

    // Sphere-surface offset toward camera; reconstruct view-space surface position.
    const float  zOff    = sqrt(1.0 - r2);
    const float3 viewPos = in.viewCenter + float3(in.uv.x * sphere.radius,
                                                  in.uv.y * sphere.radius,
                                                  sphere.radius * zOff);
    const float3 N       = normalize(viewPos - in.viewCenter);

    // Light: same world-space sun direction as the sky, transformed to view.
    const float3 sunV  = normalize((u.view * float4(sunDirWorld(), 0)).xyz);
    const float3 V     = float3(0, 0, 1);
    const float3 H     = normalize(sunV + V);

    const float diff   = max(dot(N, sunV), 0.0);
    const float spec   = pow(max(dot(N, H), 0.0), 80.0);
    const float fres   = pow(1.0 - max(dot(N, V), 0.0), 3.0);

    constexpr float3 base = float3(0.85, 0.32, 0.22);   // warm red sphere
    const     float3 col  = base * (0.25 + 0.75 * diff)
                          + float3(1, 1, 1) * spec * 0.7
                          + float3(1, 0.95, 0.88) * fres * 0.18;

    const float4 clip = u.proj * float4(viewPos, 1.0);

    SphereFOut o;
    o.color = float4(col, 1.0);
    o.depth = clip.z / clip.w;
    return o;
}
