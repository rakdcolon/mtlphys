// Test/bench entry point.
//
//   ./mtlphys_tests                       # unit tests + benchmarks
//   ./mtlphys_tests -ts-exclude=bench     # unit tests only
//   ./mtlphys_tests -ts=bench             # benchmarks only
//   ./mtlphys_tests --help                # all doctest flags
//
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "TestUtils.hpp"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <cstdio>

int main(int argc, char** argv) {
    // Initialize Metal lazily here (not in static-init) so any failure shows
    // up with a real stack trace instead of a silent SIGSEGV.
    auto* dev = mtlphys::test::device();
    std::printf("mtlphys-tests | device: %s\n\n", dev->name()->utf8String());

    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    return ctx.run();
}
