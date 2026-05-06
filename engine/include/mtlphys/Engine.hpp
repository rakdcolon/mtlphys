// Engine — owns the simulation state on the GPU and drives one step per frame.
//
// Threading model: not thread-safe. The host must call step() and render()
// from the same thread (typically the MTKView delegate's main thread for now).
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

    // Encode the particle render pass. Does NOT commit or present.
    void render(MTL::CommandBuffer* cmd,
                MTL::RenderPassDescriptor* passDesc,
                float aspectRatio,
                float timeSeconds);

    uint32_t particleCount() const noexcept { return _particleCount; }

private:
    void buildPipelines();
    void buildBuffers(uint32_t particleCount);
    void seedParticles(uint32_t particleCount);

    MTL::Device*               _device       = nullptr;  // not owned
    MTL::CommandQueue*         _queue        = nullptr;  // we own
    MTL::Library*              _library      = nullptr;
    MTL::ComputePipelineState* _integratePSO = nullptr;
    MTL::RenderPipelineState*  _renderPSO    = nullptr;

    MTL::Buffer* _positions  = nullptr;  // float4 per particle (xyz + pad)
    MTL::Buffer* _velocities = nullptr;  // float4 per particle (xyz + pad)
    MTL::Buffer* _params     = nullptr;  // SimParams uniform

    uint32_t _particleCount = 0;
};

} // namespace mtlphys
