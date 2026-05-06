// EngineBridge — minimal Obj-C wrapper around the C++ Engine so SwiftUI/Metal
// code can talk to it without seeing C++ types directly.
#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

@interface MPEngineBridge : NSObject

- (instancetype)initWithDevice:(id<MTLDevice>)device;
- (void)stepWithCommandBuffer:(id<MTLCommandBuffer>)cmd dt:(float)dt;
- (void)renderWithCommandBuffer:(id<MTLCommandBuffer>)cmd
              renderPassDescriptor:(MTLRenderPassDescriptor*)passDesc
                       aspectRatio:(float)aspect
                              time:(float)t;
- (void)resetWithParticleCount:(uint32_t)count;

@property (readonly) uint32_t particleCount;

@end

NS_ASSUME_NONNULL_END
