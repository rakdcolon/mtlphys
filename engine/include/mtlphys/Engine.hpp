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
    class Buffer;
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

    // Render the particle field, colored by neighbor density.
    void render(MTL::CommandBuffer* cmd,
                MTL::RenderPassDescriptor* passDesc,
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
    MTL::ComputePipelineState* _predictPSO          = nullptr;
    MTL::ComputePipelineState* _densityLambdaPSO    = nullptr;
    MTL::ComputePipelineState* _gatherLambdasPSO    = nullptr;
    MTL::ComputePipelineState* _applyDeltaPSO       = nullptr;
    MTL::ComputePipelineState* _finalizePSO         = nullptr;
    MTL::RenderPipelineState*  _renderPSO           = nullptr;

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
    MTL::Buffer* _sortedIndex    = nullptr;
    MTL::Buffer* _sortedPositions= nullptr;
    MTL::Buffer* _neighborCounts = nullptr;
    MTL::Buffer* _spatialParams  = nullptr;

    // PBF solver scratch
    MTL::Buffer* _lambdas        = nullptr;   // float per particle
    MTL::Buffer* _sortedLambdas  = nullptr;   // float per sorted slot

    uint32_t _particleCount   = 0;
    uint32_t _totalCells      = 0;
    uint32_t _scanChunkCount  = 0;
};

} // namespace mtlphys
