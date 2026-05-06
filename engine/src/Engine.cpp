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
constexpr float        kParticleRadius = 0.035f;
constexpr uint32_t     kMaxNeighborsForViz = 96;

constexpr float        kSmoothingH    = 0.10f;      // SPH smoothing length
// rest density derived analytically: 26 neighbors of a cubic-grid particle at
// spacing 0.5*h all sit inside the kernel, sum of poly6 contributions ≈ 8072.
constexpr float        kRestDensity   = 8000.0f;
// CFM relaxation. Higher ε → more compliant fluid (less rigid). 600 was too
// stiff and made the cube behave like a falling solid.
constexpr float        kEpsilonCFM    = 800.0f;
constexpr float        kPBFDamping    = 0.996f;    // light damping → waves persist
// 3 iters: enough incompressibility for waves to propagate cleanly without
// stiffening the cube into a solid.
constexpr uint32_t     kSolverIters   = 5;

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
    rel(_chunkSums); rel(_sortedIndex); rel(_sortedPositions); rel(_sortedPositionsNext);
    rel(_neighborCounts); rel(_spatialParams);
    rel(_lambdas); rel(_sortedLambdas);
    rel(_integratePSO); rel(_hashCellsPSO); rel(_countCellsPSO);
    rel(_scanChunkPSO); rel(_scanChunkSumsPSO); rel(_addChunkOffsetsPSO);
    rel(_scatterParticlesPSO); rel(_gatherPositionsPSO); rel(_countNeighborsPSO);
    rel(_predictPSO); rel(_densityLambdaSortedPSO); rel(_applyDeltaSortedPSO);
    rel(_scatterPositionsPSO); rel(_finalizePSO);
    rel(_fluidDepthPSO); rel(_fluidThicknessPSO); rel(_fluidSmoothPSO); rel(_fluidCompositePSO);
    rel(_wireBoxPSO);
    rel(_fluidDepthDSS); rel(_compositeDSS); rel(_wireBoxDSS); rel(_passthroughDSS);
    rel(_depthLinearTex); rel(_depthAttachmentTex); rel(_smoothedDepthTex); rel(_thicknessTex);
    rel(_library); rel(_queue);
}

void Engine::reset(uint32_t particleCount) {
    buildBuffers(particleCount);
    buildSpatialBuffers();
    seedParticles(particleCount);
}

void Engine::buildPipelines() {
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
    _predictPSO              = makeComputePSO(_device, _library, "predictPositions");
    _densityLambdaSortedPSO  = makeComputePSO(_device, _library, "densityLambdaSorted");
    _applyDeltaSortedPSO     = makeComputePSO(_device, _library, "applyDeltaSorted");
    _scatterPositionsPSO     = makeComputePSO(_device, _library, "scatterPositionsToOriginal");
    _finalizePSO             = makeComputePSO(_device, _library, "finalizeStep");

    // ---- Build the three SSF render pipelines ------------------------------

    auto buildRenderPSO = [&](const char* vertName, const char* fragName,
                              MTL::PixelFormat colorFmt,
                              MTL::PixelFormat depthFmt) -> MTL::RenderPipelineState* {
        auto vName = NS::String::string(vertName, NS::UTF8StringEncoding);
        auto fName = NS::String::string(fragName, NS::UTF8StringEncoding);
        auto vFn   = _library->newFunction(vName);
        auto fFn   = _library->newFunction(fName);
        if (!vFn || !fFn) {
            std::fprintf(stderr, "mtlphys: missing render shaders %s/%s\n", vertName, fragName);
            std::abort();
        }
        auto desc = MTL::RenderPipelineDescriptor::alloc()->init();
        desc->setVertexFunction(vFn);
        desc->setFragmentFunction(fFn);
        desc->colorAttachments()->object(0)->setPixelFormat(colorFmt);
        if (depthFmt != MTL::PixelFormatInvalid) {
            desc->setDepthAttachmentPixelFormat(depthFmt);
        }
        NS::Error* e = nullptr;
        auto pso = _device->newRenderPipelineState(desc, &e);
        desc->release(); vFn->release(); fFn->release();
        if (!pso) {
            std::fprintf(stderr, "mtlphys: failed to build PSO %s/%s: %s\n", vertName, fragName,
                         e ? e->localizedDescription()->utf8String() : "unknown");
            std::abort();
        }
        return pso;
    };

    _fluidDepthPSO     = buildRenderPSO("fluidDepthVertex",      "fluidDepthFragment",
                                        MTL::PixelFormatR32Float, MTL::PixelFormatDepth32Float);
    _fluidSmoothPSO    = buildRenderPSO("fullscreenVertex",      "fluidSmoothFragment",
                                        MTL::PixelFormatR32Float, MTL::PixelFormatInvalid);
    // Composite renders to the MTKView drawable, which has a Depth32Float
    // attachment configured. The pipeline must declare it too even though the
    // composite shader doesn't read/write depth.
    _fluidCompositePSO = buildRenderPSO("fullscreenVertex",      "fluidCompositeFragment",
                                        MTL::PixelFormatBGRA8Unorm_sRGB, MTL::PixelFormatDepth32Float);

    // Thickness pipeline. Reuses fluidDepthVertex (same billboard geometry).
    // No depth attachment; additive blending sums per-particle Gaussians into
    // the R16Float thickness texture.
    {
        auto vName = NS::String::string("fluidDepthVertex",      NS::UTF8StringEncoding);
        auto fName = NS::String::string("fluidThicknessFragment", NS::UTF8StringEncoding);
        auto vFn   = _library->newFunction(vName);
        auto fFn   = _library->newFunction(fName);
        auto desc  = MTL::RenderPipelineDescriptor::alloc()->init();
        desc->setVertexFunction(vFn);
        desc->setFragmentFunction(fFn);
        auto color = desc->colorAttachments()->object(0);
        color->setPixelFormat(MTL::PixelFormatR16Float);
        color->setBlendingEnabled(true);
        color->setRgbBlendOperation(MTL::BlendOperationAdd);
        color->setAlphaBlendOperation(MTL::BlendOperationAdd);
        color->setSourceRGBBlendFactor(MTL::BlendFactorOne);
        color->setSourceAlphaBlendFactor(MTL::BlendFactorOne);
        color->setDestinationRGBBlendFactor(MTL::BlendFactorOne);
        color->setDestinationAlphaBlendFactor(MTL::BlendFactorOne);
        NS::Error* e = nullptr;
        _fluidThicknessPSO = _device->newRenderPipelineState(desc, &e);
        desc->release(); vFn->release(); fFn->release();
        if (!_fluidThicknessPSO) {
            std::fprintf(stderr, "mtlphys: failed to build thickness PSO: %s\n",
                         e ? e->localizedDescription()->utf8String() : "unknown");
            std::abort();
        }
    }

    // Wireframe-box pipeline. Same color/depth formats as the composite (we
    // share its render pass), with alpha blending so the lines sit softly
    // over the fluid color.
    {
        auto vName = NS::String::string("wireBoxVertex",   NS::UTF8StringEncoding);
        auto fName = NS::String::string("wireBoxFragment", NS::UTF8StringEncoding);
        auto vFn   = _library->newFunction(vName);
        auto fFn   = _library->newFunction(fName);
        auto desc  = MTL::RenderPipelineDescriptor::alloc()->init();
        desc->setVertexFunction(vFn);
        desc->setFragmentFunction(fFn);
        auto color = desc->colorAttachments()->object(0);
        color->setPixelFormat(MTL::PixelFormatBGRA8Unorm_sRGB);
        color->setBlendingEnabled(true);
        color->setRgbBlendOperation(MTL::BlendOperationAdd);
        color->setAlphaBlendOperation(MTL::BlendOperationAdd);
        color->setSourceRGBBlendFactor(MTL::BlendFactorSourceAlpha);
        color->setSourceAlphaBlendFactor(MTL::BlendFactorSourceAlpha);
        color->setDestinationRGBBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
        color->setDestinationAlphaBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
        desc->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);
        NS::Error* e = nullptr;
        _wireBoxPSO = _device->newRenderPipelineState(desc, &e);
        desc->release(); vFn->release(); fFn->release();
        if (!_wireBoxPSO) {
            std::fprintf(stderr, "mtlphys: failed to build wireBox PSO: %s\n",
                         e ? e->localizedDescription()->utf8String() : "unknown");
            std::abort();
        }
    }

    // Depth-stencil states.
    {
        auto d = MTL::DepthStencilDescriptor::alloc()->init();
        d->setDepthCompareFunction(MTL::CompareFunctionLess);
        d->setDepthWriteEnabled(true);
        _fluidDepthDSS = _device->newDepthStencilState(d);
        d->release();
    }
    {   // Composite: write depth (so wire box can occlude correctly), no test.
        auto d = MTL::DepthStencilDescriptor::alloc()->init();
        d->setDepthCompareFunction(MTL::CompareFunctionAlways);
        d->setDepthWriteEnabled(true);
        _compositeDSS = _device->newDepthStencilState(d);
        d->release();
    }
    {   // Wire box: depth-test against composite's fluid depth, don't write.
        auto d = MTL::DepthStencilDescriptor::alloc()->init();
        d->setDepthCompareFunction(MTL::CompareFunctionLessEqual);
        d->setDepthWriteEnabled(false);
        _wireBoxDSS = _device->newDepthStencilState(d);
        d->release();
    }
    {
        auto d = MTL::DepthStencilDescriptor::alloc()->init();
        d->setDepthCompareFunction(MTL::CompareFunctionAlways);
        d->setDepthWriteEnabled(false);
        _passthroughDSS = _device->newDepthStencilState(d);
        d->release();
    }
}

void Engine::ensureRenderTargets(uint32_t w, uint32_t h) {
    if (w == _rtWidth && h == _rtHeight && _depthLinearTex) return;
    auto rel = [](auto*& t) { if (t) { t->release(); t = nullptr; } };
    rel(_depthLinearTex); rel(_depthAttachmentTex); rel(_smoothedDepthTex); rel(_thicknessTex);

    auto desc = MTL::TextureDescriptor::alloc()->init();
    desc->setTextureType(MTL::TextureType2D);
    desc->setWidth(w);
    desc->setHeight(h);
    desc->setStorageMode(MTL::StorageModePrivate);
    desc->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);

    desc->setPixelFormat(MTL::PixelFormatR32Float);
    _depthLinearTex   = _device->newTexture(desc);
    _smoothedDepthTex = _device->newTexture(desc);

    desc->setPixelFormat(MTL::PixelFormatR16Float);
    _thicknessTex     = _device->newTexture(desc);

    desc->setPixelFormat(MTL::PixelFormatDepth32Float);
    desc->setUsage(MTL::TextureUsageRenderTarget);
    _depthAttachmentTex = _device->newTexture(desc);

    desc->release();
    _rtWidth  = w;
    _rtHeight = h;
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
    rel(_chunkSums); rel(_sortedIndex); rel(_sortedPositions); rel(_sortedPositionsNext);
    rel(_neighborCounts); rel(_spatialParams);
    rel(_lambdas); rel(_sortedLambdas);

    const size_t particleBytes  = roundUp(_particleCount, kThreadgroupSize) * sizeof(uint32_t);
    const size_t particleFloatBs= roundUp(_particleCount, kThreadgroupSize) * sizeof(float);
    const size_t particleVec4Bs = roundUp(_particleCount, kThreadgroupSize) * sizeof(simd::float4);
    const size_t cellBytesP1    = scanLen * sizeof(uint32_t);
    const size_t cellBytes      = _totalCells * sizeof(uint32_t);
    const size_t chunkBytes     = _scanChunkCount * sizeof(uint32_t);

    _cellHashes          = _device->newBuffer(particleBytes,         MTL::ResourceStorageModePrivate);
    _cellCounts          = _device->newBuffer(cellBytesP1,           MTL::ResourceStorageModePrivate);
    _cellStart           = _device->newBuffer(cellBytesP1,           MTL::ResourceStorageModePrivate);
    _cellCursor          = _device->newBuffer(cellBytes,             MTL::ResourceStorageModePrivate);
    _chunkSums           = _device->newBuffer(chunkBytes,            MTL::ResourceStorageModePrivate);
    _sortedIndex         = _device->newBuffer(particleBytes,         MTL::ResourceStorageModePrivate);
    _sortedPositions     = _device->newBuffer(particleVec4Bs,        MTL::ResourceStorageModePrivate);
    _sortedPositionsNext = _device->newBuffer(particleVec4Bs,        MTL::ResourceStorageModePrivate);
    _neighborCounts      = _device->newBuffer(particleBytes,         MTL::ResourceStorageModePrivate);
    _spatialParams       = _device->newBuffer(sizeof(SpatialParams), MTL::ResourceStorageModeShared);
    _lambdas             = _device->newBuffer(particleFloatBs,       MTL::ResourceStorageModePrivate);
    _sortedLambdas       = _device->newBuffer(particleFloatBs,       MTL::ResourceStorageModePrivate);

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

    // 3. Solver iterations — sorted-order, ping-pong sortedPositions buffers.
    MTL::Buffer* current = _sortedPositions;
    MTL::Buffer* next    = _sortedPositionsNext;

    for (uint32_t iter = 0; iter < kSolverIters; ++iter) {
        // 3a. density + lambda
        {   auto* enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(_densityLambdaSortedPSO);
            enc->setBuffer(current,           0, 8);
            enc->setBuffer(_cellStart,        0, 3);
            enc->setBuffer(_sortedIndex,      0, 5);
            enc->setBuffer(_sortedLambdas,    0, 11);
            enc->setBuffer(_lambdas,          0, 9);
            enc->setBuffer(_spatialParams,    0, 7);
            enc->setBuffer(_pbfParams,        0, 10);
            enc->dispatchThreads(MTL::Size(padded, 1, 1),
                                 MTL::Size(kThreadgroupSize, 1, 1));
            enc->endEncoding(); }

        // 3b. apply delta
        {   auto* enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(_applyDeltaSortedPSO);
            enc->setBuffer(current,           0, 8);
            enc->setBuffer(_cellStart,        0, 3);
            enc->setBuffer(_sortedLambdas,    0, 11);
            enc->setBuffer(next,              0, 12);
            enc->setBuffer(_spatialParams,    0, 7);
            enc->setBuffer(_pbfParams,        0, 10);
            enc->dispatchThreads(MTL::Size(padded, 1, 1),
                                 MTL::Size(kThreadgroupSize, 1, 1));
            enc->endEncoding(); }

        std::swap(current, next);
    }

    // 4. Scatter sorted positions back to original-index layout
    {   auto* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(_scatterPositionsPSO);
        enc->setBuffer(current,        0, 8);
        enc->setBuffer(_sortedIndex,   0, 5);
        enc->setBuffer(_positions,     0, 0);
        enc->setBuffer(_spatialParams, 0, 7);
        enc->dispatchThreads(MTL::Size(padded, 1, 1),
                             MTL::Size(kThreadgroupSize, 1, 1));
        enc->endEncoding(); }

    // 5. Box clamp + velocity recompute + walls
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
                    MTL::RenderPassDescriptor* finalPassDesc,
                    uint32_t pixelWidth,
                    uint32_t pixelHeight,
                    float aspectRatio,
                    float timeSeconds) {
    ensureRenderTargets(pixelWidth, pixelHeight);

    // ---- Camera ----
    // Frame the entire 6×9×6 simulation box: aim at box center (y=1.5),
    // pull back far enough that the 9-unit vertical extent fits the FOV
    // with margin. Slight downward look from y=4 gives a 3/4 perspective.
    const float fovY = 1.0f;
    const float r    = 13.0f;
    const float a    = timeSeconds * 0.3f;
    const simd::float3 eye{ r * std::cos(a), 4.0f, r * std::sin(a) };
    const simd::float3 center{ 0, 1.5f, 0 };
    const simd::float3 upV{ 0, 1, 0 };

    RenderUniforms u{};
    u.view           = lookAt(eye, center, upV);
    u.proj           = perspective(fovY, aspectRatio, 0.1f, 100.0f);
    u.cameraPos      = mp::float3{ eye.x, eye.y, eye.z };
    u.particleRadius = kParticleRadius * 1.7f;   // bigger than physics radius → smoother surface
    u.invScreen      = mp::float2{ 1.0f / float(pixelWidth), 1.0f / float(pixelHeight) };
    u.tanHalfFovY    = std::tan(fovY * 0.5f);
    u.aspect         = aspectRatio;

    // ---- Pass 1: depth (offscreen) ----
    {
        auto desc = MTL::RenderPassDescriptor::alloc()->init();
        auto color = desc->colorAttachments()->object(0);
        color->setTexture(_depthLinearTex);
        color->setLoadAction(MTL::LoadActionClear);
        color->setStoreAction(MTL::StoreActionStore);
        color->setClearColor(MTL::ClearColor(0, 0, 0, 1));   // 0 = "no fluid"
        auto depth = desc->depthAttachment();
        depth->setTexture(_depthAttachmentTex);
        depth->setLoadAction(MTL::LoadActionClear);
        depth->setStoreAction(MTL::StoreActionDontCare);
        depth->setClearDepth(1.0);

        auto* enc = cmd->renderCommandEncoder(desc);
        desc->release();
        enc->setRenderPipelineState(_fluidDepthPSO);
        enc->setDepthStencilState(_fluidDepthDSS);
        enc->setVertexBuffer(_positions, 0, 0);
        enc->setVertexBytes(&u, sizeof(u), 1);
        enc->setFragmentBytes(&u, sizeof(u), 1);
        enc->drawPrimitives(MTL::PrimitiveTypeTriangle,
                            NS::UInteger(0), NS::UInteger(6),
                            NS::UInteger(_particleCount));
        enc->endEncoding();
    }

    // ---- Pass 2: smooth depth (offscreen → smoothed offscreen) ----
    {
        auto desc = MTL::RenderPassDescriptor::alloc()->init();
        auto color = desc->colorAttachments()->object(0);
        color->setTexture(_smoothedDepthTex);
        color->setLoadAction(MTL::LoadActionClear);
        color->setStoreAction(MTL::StoreActionStore);
        color->setClearColor(MTL::ClearColor(0, 0, 0, 1));

        auto* enc = cmd->renderCommandEncoder(desc);
        desc->release();
        enc->setRenderPipelineState(_fluidSmoothPSO);
        enc->setDepthStencilState(_passthroughDSS);
        enc->setFragmentTexture(_depthLinearTex, 0);
        enc->setFragmentBytes(&u, sizeof(u), 1);
        enc->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
        enc->endEncoding();
    }

    // ---- Pass 2b: thickness (additive Gaussian per particle) ----
    {
        auto desc = MTL::RenderPassDescriptor::alloc()->init();
        auto color = desc->colorAttachments()->object(0);
        color->setTexture(_thicknessTex);
        color->setLoadAction(MTL::LoadActionClear);
        color->setStoreAction(MTL::StoreActionStore);
        color->setClearColor(MTL::ClearColor(0, 0, 0, 1));

        auto* enc = cmd->renderCommandEncoder(desc);
        desc->release();
        enc->setRenderPipelineState(_fluidThicknessPSO);
        enc->setVertexBuffer(_positions, 0, 0);
        enc->setVertexBytes(&u, sizeof(u), 1);
        enc->drawPrimitives(MTL::PrimitiveTypeTriangle,
                            NS::UInteger(0), NS::UInteger(6),
                            NS::UInteger(_particleCount));
        enc->endEncoding();
    }

    // ---- Pass 3: composite + bounding box (single encoder) ----
    {
        auto* enc = cmd->renderCommandEncoder(finalPassDesc);

        // 3a. Fullscreen composite — writes color and fluid-surface NDC depth.
        enc->setRenderPipelineState(_fluidCompositePSO);
        enc->setDepthStencilState(_compositeDSS);
        enc->setFragmentTexture(_smoothedDepthTex, 0);
        enc->setFragmentTexture(_thicknessTex,     1);
        enc->setFragmentBytes(&u, sizeof(u), 1);
        enc->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));

        // 3b. Bounding-box wireframe — 12 lines, depth-tested against the
        // fluid surface. Lines are visible everywhere except where they sit
        // behind the fluid (then occluded), giving a "fluid in a glass tank" feel.
        const simd::float3 bMin = kBoundsMin;
        const simd::float3 bMax = kBoundsMax;
        enc->setRenderPipelineState(_wireBoxPSO);
        enc->setDepthStencilState(_wireBoxDSS);
        enc->setVertexBytes(&u,    sizeof(u),    1);
        enc->setVertexBytes(&bMin, sizeof(bMin), 2);
        enc->setVertexBytes(&bMax, sizeof(bMax), 3);
        enc->drawPrimitives(MTL::PrimitiveTypeLine, NS::UInteger(0), NS::UInteger(24));

        enc->endEncoding();
    }
}
