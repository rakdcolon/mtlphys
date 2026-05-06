// Engine — owns the simulation state on the GPU and drives one step per frame.
//
// Threading model: not thread-safe. The host must call step() / hash() / render()
// from the same thread (typically the MTKView delegate's main thread).
// The GPU work itself is encoded onto a command buffer and dispatched async.
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

    // Reset particle state — fills positions inside the bounds, velocity zero.
    void reset(uint32_t particleCount);

    // Encode one simulation step onto the given command buffer. Does NOT commit.
    void step(MTL::CommandBuffer* cmd, float dt);

    // Encode one full spatial-hash rebuild + neighbor-count pass onto cmd.
    // After this, neighborCounts buffer is populated and used by render().
    void buildSpatialHash(MTL::CommandBuffer* cmd);

    // Encode the particle render pass. Does NOT commit or present.
    void render(MTL::CommandBuffer* cmd,
                MTL::RenderPassDescriptor* passDesc,
                float aspectRatio,
                float timeSeconds);

    uint32_t particleCount() const noexcept { return _particleCount; }

    // ---- Test / diagnostic accessors ----------------------------------------
    // Read-only handles to the internal GPU buffers. Most are Private storage;
    // tests must blit-to-Shared before reading on the CPU.
    MTL::Buffer* positionsBuffer()      const noexcept { return _positions; }
    MTL::Buffer* velocitiesBuffer()     const noexcept { return _velocities; }
    MTL::Buffer* cellHashesBuffer()     const noexcept { return _cellHashes; }
    MTL::Buffer* cellStartBuffer()      const noexcept { return _cellStart; }
    MTL::Buffer* sortedIndexBuffer()    const noexcept { return _sortedIndex; }
    MTL::Buffer* neighborCountsBuffer() const noexcept { return _neighborCounts; }

    uint32_t totalCells()    const noexcept { return _totalCells; }
    static constexpr uint32_t kScanChunkSize    = 1024;
    static constexpr uint32_t kMaxScanLength    = kScanChunkSize * kScanChunkSize;  // 2-level scan cap

    // Run only the 3-stage exclusive prefix scan on caller-provided buffers.
    // chunkSums must hold at least ceil(n / kScanChunkSize) uints.
    // Encodes onto cmd; caller commits and waits.
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

    MTL::Device*               _device              = nullptr;  // not owned
    MTL::CommandQueue*         _queue               = nullptr;  // we own everything below
    MTL::Library*              _library             = nullptr;

    // Pipelines
    MTL::ComputePipelineState* _integratePSO        = nullptr;
    MTL::ComputePipelineState* _hashCellsPSO        = nullptr;
    MTL::ComputePipelineState* _countCellsPSO       = nullptr;
    MTL::ComputePipelineState* _scanChunkPSO        = nullptr;
    MTL::ComputePipelineState* _scanChunkSumsPSO    = nullptr;
    MTL::ComputePipelineState* _addChunkOffsetsPSO  = nullptr;
    MTL::ComputePipelineState* _scatterParticlesPSO = nullptr;
    MTL::ComputePipelineState* _gatherPositionsPSO  = nullptr;
    MTL::ComputePipelineState* _countNeighborsPSO   = nullptr;
    MTL::RenderPipelineState*  _renderPSO           = nullptr;

    // Particle state
    MTL::Buffer* _positions      = nullptr;  // float4 per particle (xyz + pad)
    MTL::Buffer* _velocities     = nullptr;  // float4 per particle (xyz + pad)
    MTL::Buffer* _params         = nullptr;  // SimParams uniform

    // Spatial-hash state
    MTL::Buffer* _cellHashes     = nullptr;  // uint per particle
    MTL::Buffer* _cellCounts     = nullptr;  // uint per cell + 1 (last slot stays 0)
    MTL::Buffer* _cellStart      = nullptr;  // uint per cell + 1 (exclusive scan)
    MTL::Buffer* _cellCursor     = nullptr;  // uint per cell, atomic counter for scatter
    MTL::Buffer* _chunkSums      = nullptr;  // uint per scan-chunk
    MTL::Buffer* _sortedIndex     = nullptr;  // uint per particle
    MTL::Buffer* _sortedPositions = nullptr;  // float4 per particle, sorted-slot indexed
    MTL::Buffer* _neighborCounts  = nullptr;  // uint per particle (visualization)
    MTL::Buffer* _spatialParams  = nullptr;  // SpatialParams uniform

    uint32_t _particleCount  = 0;
    uint32_t _totalCells     = 0;
    uint32_t _scanChunkCount = 0;     // ceil((totalCells+1) / kChunkSize)
};

} // namespace mtlphys
