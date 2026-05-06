#import "EngineBridge.h"

#include <Metal/Metal.hpp>
#include "mtlphys/Engine.hpp"

#include <memory>

@implementation MPEngineBridge {
    std::unique_ptr<mtlphys::Engine> _engine;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device {
    if ((self = [super init])) {
        // Bridging an Obj-C MTLDevice handle to metal-cpp's MTL::Device is
        // just a reinterpret — they wrap the same underlying object.
        auto* mtlDevice = (__bridge MTL::Device*)device;
        _engine = std::make_unique<mtlphys::Engine>(mtlDevice);
    }
    return self;
}

- (void)stepWithCommandBuffer:(id<MTLCommandBuffer>)cmd dt:(float)dt {
    auto* mtlCmd = (__bridge MTL::CommandBuffer*)cmd;
    _engine->step(mtlCmd, dt);
}

- (void)buildSpatialHashWithCommandBuffer:(id<MTLCommandBuffer>)cmd {
    auto* mtlCmd = (__bridge MTL::CommandBuffer*)cmd;
    _engine->buildSpatialHash(mtlCmd);
}

- (void)renderWithCommandBuffer:(id<MTLCommandBuffer>)cmd
            renderPassDescriptor:(MTLRenderPassDescriptor*)passDesc
                     aspectRatio:(float)aspect
                            time:(float)t {
    auto* mtlCmd  = (__bridge MTL::CommandBuffer*)cmd;
    auto* mtlPass = (__bridge MTL::RenderPassDescriptor*)passDesc;
    _engine->render(mtlCmd, mtlPass, aspect, t);
}

- (void)resetWithParticleCount:(uint32_t)count {
    _engine->reset(count);
}

- (uint32_t)particleCount {
    return _engine->particleCount();
}

@end
