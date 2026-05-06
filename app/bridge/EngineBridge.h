// EngineBridge — minimal Obj-C wrapper around the C++ Engine so SwiftUI/Metal
// code can talk to it without seeing C++ types directly.
#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

@interface MPEngineBridge : NSObject

- (instancetype)initWithDevice:(id<MTLDevice>)device;
- (void)stepWithCommandBuffer:(id<MTLCommandBuffer>)cmd dt:(float)dt;
- (void)buildSpatialHashWithCommandBuffer:(id<MTLCommandBuffer>)cmd;
- (void)renderWithCommandBuffer:(id<MTLCommandBuffer>)cmd
              renderPassDescriptor:(MTLRenderPassDescriptor*)passDesc
                        pixelWidth:(uint32_t)pixelWidth
                       pixelHeight:(uint32_t)pixelHeight
                       aspectRatio:(float)aspect
                                eyeX:(float)eyeX eyeY:(float)eyeY eyeZ:(float)eyeZ
                          targetX:(float)targetX targetY:(float)targetY targetZ:(float)targetZ;
- (void)resetWithParticleCount:(uint32_t)count;

@property (readonly) uint32_t particleCount;

@end

NS_ASSUME_NONNULL_END
