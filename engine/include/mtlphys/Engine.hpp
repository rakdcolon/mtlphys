// Engine — owns the simulation state on the GPU and drives one PBF step per
// frame.
//
// Threading model: not thread-safe. The host must call from one thread (the
// MTKView delegate's main thread). GPU work is encoded onto a command buffer
// and dispatched async.
#pragma once

#include <cstdint>
#include <cstddef>

namespace MTL {
    class Device;
    class CommandQueue;
    class Library;
    class ComputePipelineState;
    class RenderPipelineState;
    class DepthStencilState;
    class Buffer;
    class Texture;
    class CommandBuffer;
    class RenderPassDescriptor;
}

namespace mtlphys {

class Engine {
public:
    explicit Engine(MTL::Device* device);
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    void reset(uint32_t particleCount);

    // Old semi-implicit Euler integrator with axis-aligned box bounds. Kept
    // for backward compatibility (tests use it as a known-good reference).
    void integrate(MTL::CommandBuffer* cmd, float dt);

    // Spatial-hash rebuild: hash → count → 3-stage scan → scatter →
    // gatherPositions → countNeighbors. Idempotent given current positions.
    void buildSpatialHash(MTL::CommandBuffer* cmd);

    // Full PBF step: predict → spatial hash → 3 solver iterations → finalize.
    // This is the headline simulation entry point.
    void step(MTL::CommandBuffer* cmd, float dt);

    // Render the particle field as a fluid surface (screen-space fluid: depth
    // pass → masked Gaussian blur → composite with normal reconstruction).
    // Caller passes the MTKView final pass descriptor + drawable size.
    void render(MTL::CommandBuffer* cmd,
                MTL::RenderPassDescriptor* finalPassDesc,
                uint32_t pixelWidth,
                uint32_t pixelHeight,
                float aspectRatio,
                float timeSeconds);

    uint32_t particleCount() const noexcept { return _particleCount; }
    uint32_t totalCells()    const noexcept { return _totalCells; }
    float    cellSize()      const noexcept;

    // ---- Test / diagnostic accessors ----------------------------------------
    MTL::Buffer* positionsBuffer()      const noexcept { return _positions; }
    MTL::Buffer* velocitiesBuffer()     const noexcept { return _velocities; }
    MTL::Buffer* prevPositionsBuffer()  const noexcept { return _prevPositions; }
    MTL::Buffer* cellHashesBuffer()     const noexcept { return _cellHashes; }
    MTL::Buffer* cellStartBuffer()      const noexcept { return _cellStart; }
    MTL::Buffer* sortedIndexBuffer()    const noexcept { return _sortedIndex; }
    MTL::Buffer* neighborCountsBuffer() const noexcept { return _neighborCounts; }
    MTL::Buffer* lambdasBuffer()        const noexcept { return _lambdas; }

    static constexpr uint32_t kScanChunkSize = 1024;
    static constexpr uint32_t kMaxScanLength = kScanChunkSize * kScanChunkSize;

    // Run only the 3-stage exclusive prefix scan on caller-provided buffers.
    void runExclusiveScan(MTL::CommandBuffer* cmd,
                          MTL::Buffer*        in,
                          MTL::Buffer*        out,
                          MTL::Buffer*        chunkSums,
                          uint32_t            n);

private:
    void buildPipelines();
    void buildBuffers(uint32_t particleCount);
    void buildSpatialBuffers();
    void seedParticles(uint32_t particleCount);
    void writePBFParams(float dt);
    void ensureRenderTargets(uint32_t w, uint32_t h);

    MTL::Device*               _device              = nullptr;
    MTL::CommandQueue*         _queue               = nullptr;
    MTL::Library*              _library             = nullptr;

    // Compute pipelines
    MTL::ComputePipelineState* _integratePSO        = nullptr;
    MTL::ComputePipelineState* _hashCellsPSO        = nullptr;
    MTL::ComputePipelineState* _countCellsPSO       = nullptr;
    MTL::ComputePipelineState* _scanChunkPSO        = nullptr;
    MTL::ComputePipelineState* _scanChunkSumsPSO    = nullptr;
    MTL::ComputePipelineState* _addChunkOffsetsPSO  = nullptr;
    MTL::ComputePipelineState* _scatterParticlesPSO = nullptr;
    MTL::ComputePipelineState* _gatherPositionsPSO  = nullptr;
    MTL::ComputePipelineState* _countNeighborsPSO   = nullptr;
    MTL::ComputePipelineState* _predictPSO              = nullptr;
    MTL::ComputePipelineState* _densityLambdaSortedPSO  = nullptr;
    MTL::ComputePipelineState* _applyDeltaSortedPSO     = nullptr;
    MTL::ComputePipelineState* _scatterPositionsPSO     = nullptr;
    MTL::ComputePipelineState* _finalizePSO             = nullptr;

    // Render pipelines for screen-space fluid rendering
    MTL::RenderPipelineState*  _fluidDepthPSO       = nullptr;
    MTL::RenderPipelineState*  _fluidThicknessPSO   = nullptr;
    MTL::RenderPipelineState*  _fluidSmoothPSO      = nullptr;
    MTL::RenderPipelineState*  _fluidCompositePSO   = nullptr;
    MTL::RenderPipelineState*  _wireBoxPSO          = nullptr;
    MTL::DepthStencilState*    _fluidDepthDSS       = nullptr;
    MTL::DepthStencilState*    _compositeDSS        = nullptr;  // depthWrite=true, always
    MTL::DepthStencilState*    _wireBoxDSS          = nullptr;  // depthWrite=false, less
    MTL::DepthStencilState*    _passthroughDSS      = nullptr;

    // Offscreen textures for SSF passes (allocated lazily on resize)
    MTL::Texture* _depthLinearTex     = nullptr;  // R32Float, view-space depth
    MTL::Texture* _depthAttachmentTex = nullptr;  // Depth32Float, sphere occlusion
    MTL::Texture* _smoothedDepthTex   = nullptr;  // R32Float, smoothed depth
    MTL::Texture* _thicknessTex       = nullptr;  // R16Float, additive thickness
    uint32_t      _rtWidth  = 0;
    uint32_t      _rtHeight = 0;

    // Particle state
    MTL::Buffer* _positions      = nullptr;
    MTL::Buffer* _velocities     = nullptr;
    MTL::Buffer* _prevPositions  = nullptr;   // saved at predict, used in finalize
    MTL::Buffer* _params         = nullptr;   // SimParams uniform (for integrate)
    MTL::Buffer* _pbfParams      = nullptr;   // PBFParams uniform

    // Spatial-hash state
    MTL::Buffer* _cellHashes     = nullptr;
    MTL::Buffer* _cellCounts     = nullptr;
    MTL::Buffer* _cellStart      = nullptr;
    MTL::Buffer* _cellCursor     = nullptr;
    MTL::Buffer* _chunkSums      = nullptr;
    MTL::Buffer* _sortedIndex         = nullptr;
    MTL::Buffer* _sortedPositions     = nullptr;
    MTL::Buffer* _sortedPositionsNext = nullptr;   // ping-pong target for solver
    MTL::Buffer* _neighborCounts      = nullptr;
    MTL::Buffer* _spatialParams       = nullptr;

    // PBF solver scratch
    MTL::Buffer* _lambdas        = nullptr;   // float per particle (orig-index, for read-back)
    MTL::Buffer* _sortedLambdas  = nullptr;   // float per sorted slot

    uint32_t _particleCount   = 0;
    uint32_t _totalCells      = 0;
    uint32_t _scanChunkCount  = 0;
};

} // namespace mtlphys
