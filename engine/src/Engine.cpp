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

constexpr uint32_t kThreadgroupSize = 256;
constexpr uint32_t kScanChunkSize   = mtlphys::Engine::kScanChunkSize;

// ---- Sim/PBF tunables. Edit in one place; both engine and tests read these.
constexpr simd::float3 kBoundsMin{ -3.0f, -3.0f, -3.0f };
constexpr simd::float3 kBoundsMax{  3.0f,  6.0f,  3.0f };
constexpr float        kCellSize       = 0.10f;     // = h
constexpr float        kParticleRadius = 0.04f;
constexpr uint32_t     kMaxNeighborsForViz = 60;

constexpr float        kSmoothingH    = 0.10f;      // SPH smoothing length
// rest density derived analytically: 26 neighbors of a cubic-grid particle at
// spacing 0.5*h all sit inside the kernel, sum of poly6 contributions ≈ 8072.
constexpr float        kRestDensity   = 8000.0f;
// CFM relaxation. Higher ε → more compliant fluid (less rigid). 600 was too
// stiff and made the cube behave like a falling solid.
constexpr float        kEpsilonCFM    = 1500.0f;
constexpr float        kPBFDamping    = 0.992f;    // light damping → waves persist
// 3 iters: enough incompressibility for waves to propagate cleanly without
// stiffening the cube into a solid.
constexpr uint32_t     kSolverIters   = 3;

constexpr float kPi = 3.14159265358979323846f;

uint32_t roundUp(uint32_t n, uint32_t mult) { return ((n + mult - 1) / mult) * mult; }

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

MTL::ComputePipelineState* makeComputePSO(MTL::Device* dev, MTL::Library* lib, const char* name) {
    auto fnName = NS::String::string(name, NS::UTF8StringEncoding);
    auto fn     = lib->newFunction(fnName);
    if (!fn) {
        std::fprintf(stderr, "mtlphys: missing kernel '%s'\n", name);
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

float Engine::cellSize() const noexcept { return kCellSize; }

Engine::Engine(MTL::Device* device) : _device(device) {
    assert(device);
    _queue = _device->newCommandQueue();
    buildPipelines();
    reset(200'000);
}

Engine::~Engine() {
    auto rel = [](auto*& p) { if (p) { p->release(); p = nullptr; } };
    rel(_positions); rel(_velocities); rel(_prevPositions); rel(_params); rel(_pbfParams);
    rel(_cellHashes); rel(_cellCounts); rel(_cellStart); rel(_cellCursor);
    rel(_chunkSums); rel(_sortedIndex); rel(_sortedPositions);
    rel(_neighborCounts); rel(_spatialParams);
    rel(_lambdas); rel(_sortedLambdas);
    rel(_integratePSO); rel(_hashCellsPSO); rel(_countCellsPSO);
    rel(_scanChunkPSO); rel(_scanChunkSumsPSO); rel(_addChunkOffsetsPSO);
    rel(_scatterParticlesPSO); rel(_gatherPositionsPSO); rel(_countNeighborsPSO);
    rel(_predictPSO); rel(_densityLambdaPSO); rel(_gatherLambdasPSO);
    rel(_applyDeltaPSO); rel(_finalizePSO); rel(_renderPSO);
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
    _predictPSO          = makeComputePSO(_device, _library, "predictPositions");
    _densityLambdaPSO    = makeComputePSO(_device, _library, "densityLambda");
    _gatherLambdasPSO    = makeComputePSO(_device, _library, "gatherLambdas");
    _applyDeltaPSO       = makeComputePSO(_device, _library, "applyDelta");
    _finalizePSO         = makeComputePSO(_device, _library, "finalizeStep");

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
    rel(_positions); rel(_velocities); rel(_prevPositions); rel(_params); rel(_pbfParams);

    _positions     = _device->newBuffer(vecBytes,             MTL::ResourceStorageModeShared);
    _velocities    = _device->newBuffer(vecBytes,             MTL::ResourceStorageModeShared);
    _prevPositions = _device->newBuffer(vecBytes,             MTL::ResourceStorageModePrivate);
    _params        = _device->newBuffer(sizeof(SimParams),    MTL::ResourceStorageModeShared);
    _pbfParams     = _device->newBuffer(sizeof(PBFParams),    MTL::ResourceStorageModeShared);
}

void Engine::buildSpatialBuffers() {
    const simd::float3 extent = kBoundsMax - kBoundsMin;
    const uint32_t gx = uint32_t(std::ceil(extent.x / kCellSize));
    const uint32_t gy = uint32_t(std::ceil(extent.y / kCellSize));
    const uint32_t gz = uint32_t(std::ceil(extent.z / kCellSize));
    _totalCells = gx * gy * gz;

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
    rel(_lambdas); rel(_sortedLambdas);

    const size_t particleBytes  = roundUp(_particleCount, kThreadgroupSize) * sizeof(uint32_t);
    const size_t particleFloatBs= roundUp(_particleCount, kThreadgroupSize) * sizeof(float);
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
    _lambdas         = _device->newBuffer(particleFloatBs,       MTL::ResourceStorageModePrivate);
    _sortedLambdas   = _device->newBuffer(particleFloatBs,       MTL::ResourceStorageModePrivate);

    auto* sp = static_cast<SpatialParams*>(_spatialParams->contents());
    sp->gridOrigin         = mp::float3{ kBoundsMin.x, kBoundsMin.y, kBoundsMin.z };
    sp->cellSize           = kCellSize;
    sp->invCellSize        = 1.0f / kCellSize;
    sp->neighborRadiusSq   = kSmoothingH * kSmoothingH;
    sp->particleCount      = _particleCount;
    sp->totalCells         = _totalCells;
    sp->gridDim            = mp::uint3{ gx, gy, gz };
    sp->maxNeighborsForViz = kMaxNeighborsForViz;
}

void Engine::seedParticles(uint32_t particleCount) {
    auto* pos = static_cast<simd::float4*>(_positions->contents());
    auto* vel = static_cast<simd::float4*>(_velocities->contents());

    // PBF requires roughly uniform initial density that matches kRestDensity.
    // Random sampling produces clusters whose density spikes the constraint
    // and blows up the solver. We lay particles in a regular cubic block at
    // half-h spacing — the same packing the rest-density was tuned for.
    const float    spacing  = kSmoothingH * 0.5f;
    const uint32_t side     = uint32_t(std::ceil(std::cbrt(double(particleCount))));
    const float    halfBlk  = 0.5f * (side - 1) * spacing;
    const float    topY     = 2.0f;   // shorter fall = lower impact velocity

    uint32_t i = 0;
    for (uint32_t y = 0; y < side && i < particleCount; ++y) {
        for (uint32_t z = 0; z < side && i < particleCount; ++z) {
            for (uint32_t x = 0; x < side && i < particleCount; ++x) {
                pos[i] = simd::float4{
                    -halfBlk + x * spacing,
                    topY     - y * spacing,
                    -halfBlk + z * spacing,
                    1.0f
                };
                vel[i] = simd::float4{ 0, 0, 0, 0 };
                ++i;
            }
        }
    }
}

void Engine::writePBFParams(float dt) {
    auto* pp = static_cast<PBFParams*>(_pbfParams->contents());
    const float h  = kSmoothingH;
    const float h2 = h * h;
    const float h3 = h2 * h;
    const float h6 = h3 * h3;
    const float h9 = h6 * h3;
    pp->gravity        = mp::float3{ 0.0f, -9.81f, 0.0f };
    pp->dt             = dt;
    pp->boundsMin      = mp::float3{ kBoundsMin.x, kBoundsMin.y, kBoundsMin.z };
    pp->invDt          = 1.0f / dt;
    pp->boundsMax      = mp::float3{ kBoundsMax.x, kBoundsMax.y, kBoundsMax.z };
    pp->h              = h;
    pp->h2             = h2;
    pp->poly6Norm      = 315.0f / (64.0f * kPi * h9);
    pp->spikyGradNorm  = 45.0f  / (kPi * h6);
    pp->restDensity    = kRestDensity;
    pp->invRestDensity = 1.0f / kRestDensity;
    pp->epsilon        = kEpsilonCFM;
    pp->damping        = kPBFDamping;
    pp->particleCount  = _particleCount;
}

// ---------------------------------------------------------------------------
// integrate (legacy semi-implicit Euler — kept for tests)
// ---------------------------------------------------------------------------

void Engine::integrate(MTL::CommandBuffer* cmd, float dt) {
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
    enc->dispatchThreads(MTL::Size(padded, 1, 1), MTL::Size(kThreadgroupSize, 1, 1));
    enc->endEncoding();
}

// ---------------------------------------------------------------------------
// buildSpatialHash (unchanged from W2)
// ---------------------------------------------------------------------------

void Engine::buildSpatialHash(MTL::CommandBuffer* cmd) {
    const uint32_t scanLen         = _totalCells + 1;
    const uint32_t paddedScanLen   = _scanChunkCount * kScanChunkSize;
    const uint32_t paddedParticles = roundUp(_particleCount, kThreadgroupSize);

    zeroBuffer(cmd, _cellCounts, scanLen * sizeof(uint32_t));
    zeroBuffer(cmd, _cellCursor, _totalCells * sizeof(uint32_t));

    {   auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(_hashCellsPSO);
        enc->setBuffer(_positions,     0, 0);
        enc->setBuffer(_cellHashes,    0, 1);
        enc->setBuffer(_spatialParams, 0, 7);
        enc->dispatchThreads(MTL::Size(paddedParticles, 1, 1),
                             MTL::Size(kThreadgroupSize, 1, 1));
        enc->endEncoding(); }
    {   auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(_countCellsPSO);
        enc->setBuffer(_cellHashes,    0, 1);
        enc->setBuffer(_cellCounts,    0, 2);
        enc->setBuffer(_spatialParams, 0, 7);
        enc->dispatchThreads(MTL::Size(paddedParticles, 1, 1),
                             MTL::Size(kThreadgroupSize, 1, 1));
        enc->endEncoding(); }
    {   auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(_scanChunkPSO);
        enc->setBuffer(_cellCounts, 0, 0);
        enc->setBuffer(_cellStart,  0, 1);
        enc->setBuffer(_chunkSums,  0, 2);
        enc->setBytes(&scanLen, sizeof(scanLen), 3);
        enc->dispatchThreads(MTL::Size(paddedScanLen, 1, 1),
                             MTL::Size(kScanChunkSize, 1, 1));
        enc->endEncoding(); }
    {   auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(_scanChunkSumsPSO);
        enc->setBuffer(_chunkSums, 0, 0);
        enc->setBytes(&_scanChunkCount, sizeof(_scanChunkCount), 3);
        enc->dispatchThreads(MTL::Size(kScanChunkSize, 1, 1),
                             MTL::Size(kScanChunkSize, 1, 1));
        enc->endEncoding(); }
    {   auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(_addChunkOffsetsPSO);
        enc->setBuffer(_cellStart, 0, 0);
        enc->setBuffer(_chunkSums, 0, 2);
        enc->setBytes(&scanLen, sizeof(scanLen), 3);
        enc->dispatchThreads(MTL::Size(paddedScanLen, 1, 1),
                             MTL::Size(kScanChunkSize, 1, 1));
        enc->endEncoding(); }
    {   auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(_scatterParticlesPSO);
        enc->setBuffer(_cellHashes,    0, 1);
        enc->setBuffer(_cellStart,     0, 3);
        enc->setBuffer(_cellCursor,    0, 4);
        enc->setBuffer(_sortedIndex,   0, 5);
        enc->setBuffer(_spatialParams, 0, 7);
        enc->dispatchThreads(MTL::Size(paddedParticles, 1, 1),
                             MTL::Size(kThreadgroupSize, 1, 1));
        enc->endEncoding(); }
    {   auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(_gatherPositionsPSO);
        enc->setBuffer(_positions,        0, 0);
        enc->setBuffer(_sortedIndex,      0, 5);
        enc->setBuffer(_spatialParams,    0, 7);
        enc->setBuffer(_sortedPositions,  0, 8);
        enc->dispatchThreads(MTL::Size(paddedParticles, 1, 1),
                             MTL::Size(kThreadgroupSize, 1, 1));
        enc->endEncoding(); }
    {   auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(_countNeighborsPSO);
        enc->setBuffer(_positions,        0, 0);
        enc->setBuffer(_cellStart,        0, 3);
        enc->setBuffer(_sortedIndex,      0, 5);
        enc->setBuffer(_neighborCounts,   0, 6);
        enc->setBuffer(_spatialParams,    0, 7);
        enc->setBuffer(_sortedPositions,  0, 8);
        enc->dispatchThreads(MTL::Size(paddedParticles, 1, 1),
                             MTL::Size(kThreadgroupSize, 1, 1));
        enc->endEncoding(); }
}

// ---------------------------------------------------------------------------
// step — full PBF pipeline
// ---------------------------------------------------------------------------

void Engine::step(MTL::CommandBuffer* cmd, float dt) {
    if (dt <= 0) return;
    writePBFParams(dt);
    const uint32_t padded = roundUp(_particleCount, kThreadgroupSize);

    // 1. Predict
    {   auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(_predictPSO);
        enc->setBuffer(_positions,      0, 0);
        enc->setBuffer(_velocities,     0, 1);
        enc->setBuffer(_prevPositions,  0, 2);
        enc->setBuffer(_pbfParams,      0, 7);
        enc->dispatchThreads(MTL::Size(padded, 1, 1),
                             MTL::Size(kThreadgroupSize, 1, 1));
        enc->endEncoding(); }

    // 2. Spatial hash on predicted positions (also produces sortedPositions)
    buildSpatialHash(cmd);

    // 3. Solver iterations
    for (uint32_t iter = 0; iter < kSolverIters; ++iter) {
        // 3a. density + lambda
        {   auto* enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(_densityLambdaPSO);
            enc->setBuffer(_positions,        0, 0);
            enc->setBuffer(_cellStart,        0, 3);
            enc->setBuffer(_sortedIndex,      0, 5);
            enc->setBuffer(_spatialParams,    0, 7);
            enc->setBuffer(_sortedPositions,  0, 8);
            enc->setBuffer(_lambdas,          0, 9);
            enc->setBuffer(_pbfParams,        0, 10);
            enc->dispatchThreads(MTL::Size(padded, 1, 1),
                                 MTL::Size(kThreadgroupSize, 1, 1));
            enc->endEncoding(); }

        // 3b. gather lambdas → sortedLambdas
        {   auto* enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(_gatherLambdasPSO);
            enc->setBuffer(_sortedIndex,    0, 5);
            enc->setBuffer(_spatialParams,  0, 7);
            enc->setBuffer(_lambdas,        0, 9);
            enc->setBuffer(_sortedLambdas,  0, 11);
            enc->dispatchThreads(MTL::Size(padded, 1, 1),
                                 MTL::Size(kThreadgroupSize, 1, 1));
            enc->endEncoding(); }

        // 3c. apply position correction
        {   auto* enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(_applyDeltaPSO);
            enc->setBuffer(_positions,        0, 0);
            enc->setBuffer(_cellStart,        0, 3);
            enc->setBuffer(_sortedIndex,      0, 5);
            enc->setBuffer(_spatialParams,    0, 7);
            enc->setBuffer(_sortedPositions,  0, 8);
            enc->setBuffer(_lambdas,          0, 9);
            enc->setBuffer(_pbfParams,        0, 10);
            enc->setBuffer(_sortedLambdas,    0, 11);
            enc->dispatchThreads(MTL::Size(padded, 1, 1),
                                 MTL::Size(kThreadgroupSize, 1, 1));
            enc->endEncoding(); }

        // (We do NOT re-gatherPositions per iter — neighbor positions become
        // slightly stale across iters but stability is fine for K=3. Add a
        // re-gather if K grows or instability appears.)
    }

    // 4. Box clamp + velocity recompute + damping
    {   auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(_finalizePSO);
        enc->setBuffer(_positions,      0, 0);
        enc->setBuffer(_velocities,     0, 1);
        enc->setBuffer(_prevPositions,  0, 2);
        enc->setBuffer(_pbfParams,      0, 7);
        enc->dispatchThreads(MTL::Size(padded, 1, 1),
                             MTL::Size(kThreadgroupSize, 1, 1));
        enc->endEncoding(); }
}

void Engine::runExclusiveScan(MTL::CommandBuffer* cmd,
                              MTL::Buffer*        in,
                              MTL::Buffer*        out,
                              MTL::Buffer*        chunkSums,
                              uint32_t            n) {
    if (n == 0) return;
    if (n > kMaxScanLength) {
        std::fprintf(stderr,
                     "mtlphys: runExclusiveScan(n=%u) exceeds 2-level cap (%u).\n",
                     n, kMaxScanLength);
        std::abort();
    }
    const uint32_t chunkCount = (n + kScanChunkSize - 1) / kScanChunkSize;
    const uint32_t paddedScan = chunkCount * kScanChunkSize;

    {   auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(_scanChunkPSO);
        enc->setBuffer(in,        0, 0);
        enc->setBuffer(out,       0, 1);
        enc->setBuffer(chunkSums, 0, 2);
        enc->setBytes(&n, sizeof(n), 3);
        enc->dispatchThreads(MTL::Size(paddedScan, 1, 1),
                             MTL::Size(kScanChunkSize, 1, 1));
        enc->endEncoding(); }
    {   auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(_scanChunkSumsPSO);
        enc->setBuffer(chunkSums, 0, 0);
        enc->setBytes(&chunkCount, sizeof(chunkCount), 3);
        enc->dispatchThreads(MTL::Size(kScanChunkSize, 1, 1),
                             MTL::Size(kScanChunkSize, 1, 1));
        enc->endEncoding(); }
    {   auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(_addChunkOffsetsPSO);
        enc->setBuffer(out,       0, 0);
        enc->setBuffer(chunkSums, 0, 2);
        enc->setBytes(&n, sizeof(n), 3);
        enc->dispatchThreads(MTL::Size(paddedScan, 1, 1),
                             MTL::Size(kScanChunkSize, 1, 1));
        enc->endEncoding(); }
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
