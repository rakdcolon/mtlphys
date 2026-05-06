#include "mtlphys/Engine.hpp"
#include "Shared.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <simd/simd.h>

using namespace mtlphys;

namespace {

// Round particle count up to a multiple of the threadgroup width so dispatch
// is clean and we don't need bounds-checks inside the integrator kernel.
constexpr uint32_t kThreadgroupSize = 256;

uint32_t roundUp(uint32_t n, uint32_t mult) {
    return ((n + mult - 1) / mult) * mult;
}

simd::float4x4 perspective(float fovY, float aspect, float zNear, float zFar) {
    const float ys = 1.0f / std::tan(fovY * 0.5f);
    const float xs = ys / aspect;
    const float zs = zFar / (zNear - zFar);
    return simd::float4x4{
        simd::float4{ xs,  0,  0,           0 },
        simd::float4{  0, ys,  0,           0 },
        simd::float4{  0,  0, zs,          -1 },
        simd::float4{  0,  0, zNear * zs,   0 },
    };
}

simd::float4x4 lookAt(simd::float3 eye, simd::float3 center, simd::float3 up) {
    const simd::float3 f = simd::normalize(center - eye);
    const simd::float3 s = simd::normalize(simd::cross(f, up));
    const simd::float3 u = simd::cross(s, f);
    return simd::float4x4{
        simd::float4{ s.x, u.x, -f.x, 0 },
        simd::float4{ s.y, u.y, -f.y, 0 },
        simd::float4{ s.z, u.z, -f.z, 0 },
        simd::float4{ -simd::dot(s, eye), -simd::dot(u, eye), simd::dot(f, eye), 1 },
    };
}

} // namespace

Engine::Engine(MTL::Device* device) : _device(device) {
    assert(device);
    _queue = _device->newCommandQueue();
    buildPipelines();
    reset(1'000'000);
}

Engine::~Engine() {
    if (_positions)    _positions->release();
    if (_velocities)   _velocities->release();
    if (_params)       _params->release();
    if (_integratePSO) _integratePSO->release();
    if (_renderPSO)    _renderPSO->release();
    if (_library)      _library->release();
    if (_queue)        _queue->release();
}

void Engine::reset(uint32_t particleCount) {
    buildBuffers(particleCount);
    seedParticles(particleCount);
}

void Engine::buildPipelines() {
    NS::Error* err = nullptr;
    _library = _device->newDefaultLibrary();
    if (!_library) {
        std::fprintf(stderr, "mtlphys: newDefaultLibrary() returned null — is the .metallib bundled?\n");
        std::abort();
    }

    // Integrate kernel
    auto integrateName = NS::String::string("integrate", NS::UTF8StringEncoding);
    auto integrateFn   = _library->newFunction(integrateName);
    _integratePSO = _device->newComputePipelineState(integrateFn, &err);
    integrateFn->release();
    if (!_integratePSO) {
        std::fprintf(stderr, "mtlphys: failed to build integrate PSO: %s\n",
                     err ? err->localizedDescription()->utf8String() : "unknown");
        std::abort();
    }

    // Render pipeline
    auto vertName = NS::String::string("particleVertex",   NS::UTF8StringEncoding);
    auto fragName = NS::String::string("particleFragment", NS::UTF8StringEncoding);
    auto vertFn = _library->newFunction(vertName);
    auto fragFn = _library->newFunction(fragName);

    auto desc = MTL::RenderPipelineDescriptor::alloc()->init();
    desc->setVertexFunction(vertFn);
    desc->setFragmentFunction(fragFn);
    desc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatBGRA8Unorm_sRGB);
    desc->colorAttachments()->object(0)->setBlendingEnabled(true);
    desc->colorAttachments()->object(0)->setRgbBlendOperation(MTL::BlendOperationAdd);
    desc->colorAttachments()->object(0)->setAlphaBlendOperation(MTL::BlendOperationAdd);
    desc->colorAttachments()->object(0)->setSourceRGBBlendFactor(MTL::BlendFactorSourceAlpha);
    desc->colorAttachments()->object(0)->setSourceAlphaBlendFactor(MTL::BlendFactorSourceAlpha);
    desc->colorAttachments()->object(0)->setDestinationRGBBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
    desc->colorAttachments()->object(0)->setDestinationAlphaBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
    desc->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);

    _renderPSO = _device->newRenderPipelineState(desc, &err);
    desc->release();
    vertFn->release();
    fragFn->release();
    if (!_renderPSO) {
        std::fprintf(stderr, "mtlphys: failed to build render PSO: %s\n",
                     err ? err->localizedDescription()->utf8String() : "unknown");
        std::abort();
    }
}

void Engine::buildBuffers(uint32_t particleCount) {
    _particleCount = particleCount;
    const size_t paddedCount = roundUp(particleCount, kThreadgroupSize);
    const size_t vecBytes    = paddedCount * sizeof(simd::float4);

    if (_positions)  { _positions->release();  _positions  = nullptr; }
    if (_velocities) { _velocities->release(); _velocities = nullptr; }
    if (_params)     { _params->release();     _params     = nullptr; }

    _positions  = _device->newBuffer(vecBytes,         MTL::ResourceStorageModeShared);
    _velocities = _device->newBuffer(vecBytes,         MTL::ResourceStorageModeShared);
    _params     = _device->newBuffer(sizeof(SimParams), MTL::ResourceStorageModeShared);
}

void Engine::seedParticles(uint32_t particleCount) {
    auto* pos = static_cast<simd::float4*>(_positions->contents());
    auto* vel = static_cast<simd::float4*>(_velocities->contents());

    std::mt19937 rng(0xC0FFEE);
    std::uniform_real_distribution<float> ux(-2.0f, 2.0f);
    std::uniform_real_distribution<float> uy( 1.0f, 5.0f);
    std::uniform_real_distribution<float> uz(-2.0f, 2.0f);

    for (uint32_t i = 0; i < particleCount; ++i) {
        pos[i] = simd::float4{ ux(rng), uy(rng), uz(rng), 1.0f };
        vel[i] = simd::float4{ 0, 0, 0, 0 };
    }
}

void Engine::step(MTL::CommandBuffer* cmd, float dt) {
    // Update uniform params (CPU writes are visible to GPU because shared storage).
    auto* p = static_cast<SimParams*>(_params->contents());
    p->gravity       = mp::float3{ 0.0f, -9.81f, 0.0f };
    p->dt            = dt;
    p->boundsMin     = mp::float3{ -3.0f, -3.0f, -3.0f };
    p->boundsMax     = mp::float3{  3.0f,  6.0f,  3.0f };
    p->damping       = 0.999f;
    p->particleCount = _particleCount;

    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(_integratePSO);
    enc->setBuffer(_positions,  0, 0);
    enc->setBuffer(_velocities, 0, 1);
    enc->setBuffer(_params,     0, 2);

    const uint32_t padded = roundUp(_particleCount, kThreadgroupSize);
    enc->dispatchThreads(MTL::Size(padded, 1, 1),
                         MTL::Size(kThreadgroupSize, 1, 1));
    enc->endEncoding();
}

void Engine::render(MTL::CommandBuffer* cmd,
                    MTL::RenderPassDescriptor* passDesc,
                    float aspectRatio,
                    float timeSeconds) {
    // Slow camera orbit so the 3D-ness of the sim is obvious in the demo.
    const float r = 9.0f;
    const float a = timeSeconds * 0.3f;
    const simd::float3 eye{ r * std::cos(a), 3.0f, r * std::sin(a) };
    const simd::float3 center{ 0, 0, 0 };
    const simd::float3 up{ 0, 1, 0 };

    RenderUniforms u{};
    u.viewProj       = perspective(1.0f, aspectRatio, 0.1f, 100.0f) * lookAt(eye, center, up);
    u.cameraPos      = mp::float3{ eye.x, eye.y, eye.z };
    u.particleRadius = 0.04f;

    auto* enc = cmd->renderCommandEncoder(passDesc);
    enc->setRenderPipelineState(_renderPSO);
    enc->setVertexBuffer(_positions, 0, 0);
    enc->setVertexBytes(&u, sizeof(u), 1);
    enc->setFragmentBytes(&u, sizeof(u), 0);
    // 6 vertices per particle (two triangles forming a billboard quad).
    enc->drawPrimitives(MTL::PrimitiveTypeTriangle,
                        NS::UInteger(0), NS::UInteger(6),
                        NS::UInteger(_particleCount));
    enc->endEncoding();
}
