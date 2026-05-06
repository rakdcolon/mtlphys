#include "TestUtils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace mtlphys::test {

namespace {
MTL::Device*       g_device = nullptr;
MTL::CommandQueue* g_queue  = nullptr;
} // namespace

MTL::Device* device() {
    if (!g_device) {
        g_device = MTL::CreateSystemDefaultDevice();
        if (!g_device) {
            std::fprintf(stderr, "test setup: no Metal device available\n");
            std::abort();
        }
    }
    return g_device;
}

MTL::CommandQueue* queue() {
    if (!g_queue) g_queue = device()->newCommandQueue();
    return g_queue;
}

uint32_t cpuCellHash(simd::float3 pos,
                     simd::float3 origin,
                     float        cellSize,
                     simd::uint3  gridDim) {
    const float invCs = 1.0f / cellSize;
    int ix = int(std::floor((pos.x - origin.x) * invCs));
    int iy = int(std::floor((pos.y - origin.y) * invCs));
    int iz = int(std::floor((pos.z - origin.z) * invCs));
    ix = std::clamp(ix, 0, int(gridDim.x) - 1);
    iy = std::clamp(iy, 0, int(gridDim.y) - 1);
    iz = std::clamp(iz, 0, int(gridDim.z) - 1);
    return uint32_t(ix) + uint32_t(iy) * gridDim.x + uint32_t(iz) * gridDim.x * gridDim.y;
}

Stats computeStats(std::vector<double>& seconds) {
    if (seconds.empty()) return {};
    std::sort(seconds.begin(), seconds.end());
    auto pct = [&](double p) {
        size_t idx = std::min(seconds.size() - 1,
                              size_t(std::floor(p * (seconds.size() - 1))));
        return seconds[idx] * 1000.0;
    };
    Stats s;
    s.p50_ms  = pct(0.50);
    s.p95_ms  = pct(0.95);
    s.p99_ms  = pct(0.99);
    s.min_ms  = seconds.front() * 1000.0;
    s.max_ms  = seconds.back()  * 1000.0;
    double sum = 0;
    for (double v : seconds) sum += v;
    s.mean_ms = (sum / seconds.size()) * 1000.0;
    return s;
}

void printBenchHeader() {
    std::printf("\n%-32s %10s %9s %9s %9s %9s\n",
                "benchmark", "N", "p50_ms", "p95_ms", "p99_ms", "mean_ms");
    std::printf("%-32s %10s %9s %9s %9s %9s\n",
                "--------------------------------",
                "----------", "---------", "---------", "---------", "---------");
}

void printBenchRow(const std::string& name, uint32_t n, const Stats& s) {
    std::printf("%-32s %10u %9.3f %9.3f %9.3f %9.3f\n",
                name.c_str(), n, s.p50_ms, s.p95_ms, s.p99_ms, s.mean_ms);
}

} // namespace mtlphys::test
