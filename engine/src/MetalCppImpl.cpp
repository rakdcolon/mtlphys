// metal-cpp is header-only, but its headers declare a large pile of selectors
// and class pointers as extern. Each must be DEFINED in exactly one
// translation unit. These three macros emit those definitions. This file does
// nothing else — keep it free of other code so the macro contract stays clear.
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
