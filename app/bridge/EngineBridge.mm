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
                      pixelWidth:(uint32_t)pixelWidth
                     pixelHeight:(uint32_t)pixelHeight
                     aspectRatio:(float)aspect
                                eyeX:(float)eyeX eyeY:(float)eyeY eyeZ:(float)eyeZ
                          targetX:(float)targetX targetY:(float)targetY targetZ:(float)targetZ {
    auto* mtlCmd  = (__bridge MTL::CommandBuffer*)cmd;
    auto* mtlPass = (__bridge MTL::RenderPassDescriptor*)passDesc;
    const float eye[3]    = { eyeX, eyeY, eyeZ };
    const float target[3] = { targetX, targetY, targetZ };
    _engine->render(mtlCmd, mtlPass, pixelWidth, pixelHeight, aspect, eye, target);
}

- (void)applyMousePulseWithCommandBuffer:(id<MTLCommandBuffer>)cmd
                              originX:(float)ox originY:(float)oy originZ:(float)oz
                                 dirX:(float)dx dirY:(float)dy dirZ:(float)dz
                               forceX:(float)fx forceY:(float)fy forceZ:(float)fz
                               radius:(float)radius {
    auto* mtlCmd = (__bridge MTL::CommandBuffer*)cmd;
    const float o[3] = { ox, oy, oz };
    const float d[3] = { dx, dy, dz };
    const float f[3] = { fx, fy, fz };
    _engine->applyMousePulse(mtlCmd, o, d, f, radius);
}

- (void)resetWithParticleCount:(uint32_t)count {
    _engine->reset(count);
}

- (void)resetWithParticleCount:(uint32_t)count scene:(uint32_t)sceneIdx {
    _engine->reset(count, sceneIdx);
}

- (uint32_t)currentScene {
    return _engine->currentScene();
}

- (uint32_t)particleCount {
    return _engine->particleCount();
}

@end
