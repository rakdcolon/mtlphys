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

// Compute / dispatch tunables. kThreadgroupSize is the per-particle dispatch
// width. The scan-chunk size lives on Engine (so tests can see it) and must
// match the kChunkSize constant in Spatial.metal — they cannot diverge.
constexpr uint32_t kThreadgroupSize = 256;
constexpr uint32_t kScanChunkSize   = mtlphys::Engine::kScanChunkSize;

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

// Sim/spatial constants (kept in one place so they can't disagree).
constexpr simd::float3 kBoundsMin{ -3.0f, -3.0f, -3.0f };
constexpr simd::float3 kBoundsMax{  3.0f,  6.0f,  3.0f };
constexpr float        kCellSize       = 0.08f;   // = 2 * particle radius
constexpr float        kParticleRadius = 0.04f;
constexpr uint32_t     kMaxNeighborsForViz = 60;  // density saturation point

MTL::ComputePipelineState* makeComputePSO(MTL::Device* dev, MTL::Library* lib, const char* name) {
    auto fnName = NS::String::string(name, NS::UTF8StringEncoding);
    auto fn     = lib->newFunction(fnName);
    if (!fn) {
        std::fprintf(stderr, "mtlphys: missing kernel '%s' in default library\n", name);
        std::abort();
    }
    NS::Error* err = nullptr;
    auto pso = dev->newComputePipelineState(fn, &err);
    fn->release();
    if (!pso) {
        std::fprintf(stderr, "mtlphys: failed to build PSO '%s': %s\n", name,
                     err ? err->localizedDescription()->utf8String() : "unknown");
        std::abort();
    }
    return pso;
}

void zeroBuffer(MTL::CommandBuffer* cmd, MTL::Buffer* buf, size_t bytes) {
    auto* blit = cmd->blitCommandEncoder();
    blit->fillBuffer(buf, NS::Range(0, bytes), 0);
    blit->endEncoding();
}

} // namespace

Engine::Engine(MTL::Device* device) : _device(device) {
    assert(device);
    _queue = _device->newCommandQueue();
    buildPipelines();
    reset(1'000'000);
}

Engine::~Engine() {
    auto rel = [](auto*& p) { if (p) { p->release(); p = nullptr; } };
    rel(_positions); rel(_velocities); rel(_params);
    rel(_cellHashes); rel(_cellCounts); rel(_cellStart); rel(_cellCursor);
    rel(_chunkSums); rel(_sortedIndex); rel(_sortedPositions);
    rel(_neighborCounts); rel(_spatialParams);
    rel(_integratePSO); rel(_hashCellsPSO); rel(_countCellsPSO);
    rel(_scanChunkPSO); rel(_scanChunkSumsPSO); rel(_addChunkOffsetsPSO);
    rel(_scatterParticlesPSO); rel(_gatherPositionsPSO);
    rel(_countNeighborsPSO); rel(_renderPSO);
    rel(_library); rel(_queue);
}

void Engine::reset(uint32_t particleCount) {
    buildBuffers(particleCount);
    buildSpatialBuffers();
    seedParticles(particleCount);
}

void Engine::buildPipelines() {
    NS::Error* err = nullptr;
    _library = _device->newDefaultLibrary();
    if (!_library) {
        std::fprintf(stderr, "mtlphys: newDefaultLibrary() returned null — is the .metallib bundled?\n");
        std::abort();
    }

    _integratePSO        = makeComputePSO(_device, _library, "integrate");
    _hashCellsPSO        = makeComputePSO(_device, _library, "hashCells");
    _countCellsPSO       = makeComputePSO(_device, _library, "countCells");
    _scanChunkPSO        = makeComputePSO(_device, _library, "scanChunkExclusive");
    _scanChunkSumsPSO    = makeComputePSO(_device, _library, "scanChunkSums");
    _addChunkOffsetsPSO  = makeComputePSO(_device, _library, "addChunkOffsets");
    _scatterParticlesPSO = makeComputePSO(_device, _library, "scatterParticles");
    _gatherPositionsPSO  = makeComputePSO(_device, _library, "gatherPositions");
    _countNeighborsPSO   = makeComputePSO(_device, _library, "countNeighbors");

    auto vertName = NS::String::string("particleVertex",   NS::UTF8StringEncoding);
    auto fragName = NS::String::string("particleFragment", NS::UTF8StringEncoding);
    auto vertFn   = _library->newFunction(vertName);
    auto fragFn   = _library->newFunction(fragName);

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

    auto rel = [](auto*& p) { if (p) { p->release(); p = nullptr; } };
    rel(_positions); rel(_velocities); rel(_params);

    _positions  = _device->newBuffer(vecBytes,         MTL::ResourceStorageModeShared);
    _velocities = _device->newBuffer(vecBytes,         MTL::ResourceStorageModeShared);
    _params     = _device->newBuffer(sizeof(SimParams), MTL::ResourceStorageModeShared);
}

void Engine::buildSpatialBuffers() {
    // Grid sized to the simulation bounds.
    const simd::float3 extent = kBoundsMax - kBoundsMin;
    const uint32_t gx = uint32_t(std::ceil(extent.x / kCellSize));
    const uint32_t gy = uint32_t(std::ceil(extent.y / kCellSize));
    const uint32_t gz = uint32_t(std::ceil(extent.z / kCellSize));
    _totalCells = gx * gy * gz;

    // Scan operates on (totalCells + 1) entries so that cellStart[totalCells] == N.
    const uint32_t scanLen = _totalCells + 1;
    _scanChunkCount = (scanLen + kScanChunkSize - 1) / kScanChunkSize;
    if (_scanChunkCount > kScanChunkSize) {
        std::fprintf(stderr,
                     "mtlphys: scan needs a third level (chunks=%u > %u). "
                     "Bounds or cell size pushed cell count too high.\n",
                     _scanChunkCount, kScanChunkSize);
        std::abort();
    }

    auto rel = [](auto*& p) { if (p) { p->release(); p = nullptr; } };
    rel(_cellHashes); rel(_cellCounts); rel(_cellStart); rel(_cellCursor);
    rel(_chunkSums); rel(_sortedIndex); rel(_sortedPositions);
    rel(_neighborCounts); rel(_spatialParams);

    const size_t particleBytes  = roundUp(_particleCount, kThreadgroupSize) * sizeof(uint32_t);
    const size_t particleVec4Bs = roundUp(_particleCount, kThreadgroupSize) * sizeof(simd::float4);
    const size_t cellBytesP1    = scanLen * sizeof(uint32_t);
    const size_t cellBytes      = _totalCells * sizeof(uint32_t);
    const size_t chunkBytes     = _scanChunkCount * sizeof(uint32_t);

    _cellHashes      = _device->newBuffer(particleBytes,         MTL::ResourceStorageModePrivate);
    _cellCounts      = _device->newBuffer(cellBytesP1,           MTL::ResourceStorageModePrivate);
    _cellStart       = _device->newBuffer(cellBytesP1,           MTL::ResourceStorageModePrivate);
    _cellCursor      = _device->newBuffer(cellBytes,             MTL::ResourceStorageModePrivate);
    _chunkSums       = _device->newBuffer(chunkBytes,            MTL::ResourceStorageModePrivate);
    _sortedIndex     = _device->newBuffer(particleBytes,         MTL::ResourceStorageModePrivate);
    _sortedPositions = _device->newBuffer(particleVec4Bs,        MTL::ResourceStorageModePrivate);
    _neighborCounts  = _device->newBuffer(particleBytes,         MTL::ResourceStorageModePrivate);
    _spatialParams   = _device->newBuffer(sizeof(SpatialParams), MTL::ResourceStorageModeShared);

    // Fill the SpatialParams once — geometry doesn't change between frames yet.
    auto* sp = static_cast<SpatialParams*>(_spatialParams->contents());
    sp->gridOrigin         = mp::float3{ kBoundsMin.x, kBoundsMin.y, kBoundsMin.z };
    sp->cellSize           = kCellSize;
    sp->invCellSize        = 1.0f / kCellSize;
    sp->neighborRadiusSq   = kCellSize * kCellSize;     // ~= (2 * particleRadius)^2
    sp->particleCount      = _particleCount;
    sp->totalCells         = _totalCells;
    sp->gridDim            = mp::uint3{ gx, gy, gz };
    sp->maxNeighborsForViz = kMaxNeighborsForViz;
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
    auto* p = static_cast<SimParams*>(_params->contents());
    p->gravity       = mp::float3{ 0.0f, -9.81f, 0.0f };
    p->dt            = dt;
    p->boundsMin     = mp::float3{ kBoundsMin.x, kBoundsMin.y, kBoundsMin.z };
    p->boundsMax     = mp::float3{ kBoundsMax.x, kBoundsMax.y, kBoundsMax.z };
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

void Engine::buildSpatialHash(MTL::CommandBuffer* cmd) {
    const uint32_t scanLen        = _totalCells + 1;
    const uint32_t paddedScanLen  = _scanChunkCount * kScanChunkSize;
    const uint32_t paddedParticles = roundUp(_particleCount, kThreadgroupSize);

    // Zero the buffers we'll atomically fill this frame.
    zeroBuffer(cmd, _cellCounts, _totalCells * sizeof(uint32_t) + sizeof(uint32_t));
    zeroBuffer(cmd, _cellCursor, _totalCells * sizeof(uint32_t));
    // chunkSums is fully overwritten by the scan; cellStart is fully overwritten
    // by the scan + add-back, so neither needs zeroing.

    // 1. Hash
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(_hashCellsPSO);
        enc->setBuffer(_positions,     0, 0);
        enc->setBuffer(_cellHashes,    0, 1);
        enc->setBuffer(_spatialParams, 0, 7);
        enc->dispatchThreads(MTL::Size(paddedParticles, 1, 1),
                             MTL::Size(kThreadgroupSize, 1, 1));
        enc->endEncoding();
    }

    // 2. Count
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(_countCellsPSO);
        enc->setBuffer(_cellHashes,    0, 1);
        enc->setBuffer(_cellCounts,    0, 2);
        enc->setBuffer(_spatialParams, 0, 7);
        enc->dispatchThreads(MTL::Size(paddedParticles, 1, 1),
                             MTL::Size(kThreadgroupSize, 1, 1));
        enc->endEncoding();
    }

    // 3. scanChunkExclusive: cellCounts -> cellStart, totals -> chunkSums
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(_scanChunkPSO);
        enc->setBuffer(_cellCounts, 0, 0);
        enc->setBuffer(_cellStart,  0, 1);
        enc->setBuffer(_chunkSums,  0, 2);
        enc->setBytes(&scanLen, sizeof(scanLen), 3);
        enc->dispatchThreads(MTL::Size(paddedScanLen, 1, 1),
                             MTL::Size(kScanChunkSize, 1, 1));
        enc->endEncoding();
    }

    // 4. scanChunkSums: in-place scan over chunkSums (single threadgroup)
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(_scanChunkSumsPSO);
        enc->setBuffer(_chunkSums, 0, 0);
        enc->setBytes(&_scanChunkCount, sizeof(_scanChunkCount), 3);
        enc->dispatchThreads(MTL::Size(kScanChunkSize, 1, 1),
                             MTL::Size(kScanChunkSize, 1, 1));
        enc->endEncoding();
    }

    // 5. addChunkOffsets: cellStart += chunkSums[bid] (skipping bid 0)
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(_addChunkOffsetsPSO);
        enc->setBuffer(_cellStart, 0, 0);
        enc->setBuffer(_chunkSums, 0, 2);
        enc->setBytes(&scanLen, sizeof(scanLen), 3);
        enc->dispatchThreads(MTL::Size(paddedScanLen, 1, 1),
                             MTL::Size(kScanChunkSize, 1, 1));
        enc->endEncoding();
    }

    // 6. Scatter particle indices into sorted order
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(_scatterParticlesPSO);
        enc->setBuffer(_cellHashes,    0, 1);
        enc->setBuffer(_cellStart,     0, 3);
        enc->setBuffer(_cellCursor,    0, 4);
        enc->setBuffer(_sortedIndex,   0, 5);
        enc->setBuffer(_spatialParams, 0, 7);
        enc->dispatchThreads(MTL::Size(paddedParticles, 1, 1),
                             MTL::Size(kThreadgroupSize, 1, 1));
        enc->endEncoding();
    }

    // 7. Gather positions into sorted layout — turns the inner-loop random
    //    gather of countNeighbors into a sequential read.
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(_gatherPositionsPSO);
        enc->setBuffer(_positions,        0, 0);
        enc->setBuffer(_sortedIndex,      0, 5);
        enc->setBuffer(_spatialParams,    0, 7);
        enc->setBuffer(_sortedPositions,  0, 8);
        enc->dispatchThreads(MTL::Size(paddedParticles, 1, 1),
                             MTL::Size(kThreadgroupSize, 1, 1));
        enc->endEncoding();
    }

    // 8. Neighbor count (density visualization, prep for W3 constraint solver)
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(_countNeighborsPSO);
        enc->setBuffer(_positions,        0, 0);
        enc->setBuffer(_cellStart,        0, 3);
        enc->setBuffer(_sortedIndex,      0, 5);
        enc->setBuffer(_neighborCounts,   0, 6);
        enc->setBuffer(_spatialParams,    0, 7);
        enc->setBuffer(_sortedPositions,  0, 8);
        enc->dispatchThreads(MTL::Size(paddedParticles, 1, 1),
                             MTL::Size(kThreadgroupSize, 1, 1));
        enc->endEncoding();
    }
}

void Engine::runExclusiveScan(MTL::CommandBuffer* cmd,
                              MTL::Buffer*        in,
                              MTL::Buffer*        out,
                              MTL::Buffer*        chunkSums,
                              uint32_t            n) {
    if (n == 0) return;
    if (n > kMaxScanLength) {
        std::fprintf(stderr,
                     "mtlphys: runExclusiveScan(n=%u) exceeds 2-level cap (%u). "
                     "Add a third hierarchy level.\n", n, kMaxScanLength);
        std::abort();
    }
    const uint32_t chunkCount   = (n + kScanChunkSize - 1) / kScanChunkSize;
    const uint32_t paddedScan   = chunkCount * kScanChunkSize;

    // Stage 1: per-chunk exclusive scan, totals to chunkSums.
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(_scanChunkPSO);
        enc->setBuffer(in,        0, 0);
        enc->setBuffer(out,       0, 1);
        enc->setBuffer(chunkSums, 0, 2);
        enc->setBytes(&n, sizeof(n), 3);
        enc->dispatchThreads(MTL::Size(paddedScan, 1, 1),
                             MTL::Size(kScanChunkSize, 1, 1));
        enc->endEncoding();
    }
    // Stage 2: scan chunkSums in place (single threadgroup).
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(_scanChunkSumsPSO);
        enc->setBuffer(chunkSums, 0, 0);
        enc->setBytes(&chunkCount, sizeof(chunkCount), 3);
        enc->dispatchThreads(MTL::Size(kScanChunkSize, 1, 1),
                             MTL::Size(kScanChunkSize, 1, 1));
        enc->endEncoding();
    }
    // Stage 3: add scanned chunkSums into the per-chunk results.
    {
        auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(_addChunkOffsetsPSO);
        enc->setBuffer(out,       0, 0);
        enc->setBuffer(chunkSums, 0, 2);
        enc->setBytes(&n, sizeof(n), 3);
        enc->dispatchThreads(MTL::Size(paddedScan, 1, 1),
                             MTL::Size(kScanChunkSize, 1, 1));
        enc->endEncoding();
    }
}

void Engine::render(MTL::CommandBuffer* cmd,
                    MTL::RenderPassDescriptor* passDesc,
                    float aspectRatio,
                    float timeSeconds) {
    const float r = 9.0f;
    const float a = timeSeconds * 0.3f;
    const simd::float3 eye{ r * std::cos(a), 3.0f, r * std::sin(a) };
    const simd::float3 center{ 0, 0, 0 };
    const simd::float3 up{ 0, 1, 0 };

    RenderUniforms u{};
    u.viewProj       = perspective(1.0f, aspectRatio, 0.1f, 100.0f) * lookAt(eye, center, up);
    u.cameraPos      = mp::float3{ eye.x, eye.y, eye.z };
    u.particleRadius = kParticleRadius;

    const uint32_t maxN = kMaxNeighborsForViz;

    auto* enc = cmd->renderCommandEncoder(passDesc);
    enc->setRenderPipelineState(_renderPSO);
    enc->setVertexBuffer(_positions, 0, 0);
    enc->setVertexBytes(&u, sizeof(u), 1);
    enc->setVertexBuffer(_neighborCounts, 0, 2);
    enc->setVertexBytes(&maxN, sizeof(maxN), 3);
    enc->drawPrimitives(MTL::PrimitiveTypeTriangle,
                        NS::UInteger(0), NS::UInteger(6),
                        NS::UInteger(_particleCount));
    enc->endEncoding();
}
