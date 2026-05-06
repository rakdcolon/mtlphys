# mtlphys

A GPU-accelerated 3D physics engine for Apple Silicon, built on Metal.

> **Status:** Week 1 — pipeline scaffolding. Goal: prove the end-to-end path (C++ engine → Metal compute → Metal render → SwiftUI window) before optimizing anything.

## What it is

`mtlphys` is a unified-particle physics solver based on **XPBD** (Extended Position-Based Dynamics). Rigid bodies, soft bodies, cloth, and fluids are all expressed as constraints over particles, solved on the GPU each frame. The engine targets M-series Macs and uses Metal directly via `metal-cpp`.

The headline goal: **1M+ interacting particles, 3D, 60+ fps on M-series**, with benchmarks against a CPU baseline.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  SwiftUI app (mtlphys target)                            │
│  ├─ ContentView / MetalView  (UI, scene picker, HUD)     │
│  └─ Renderer (Swift)         (drives MTKView frame loop) │
└──────────────────────┬───────────────────────────────────┘
                       │  Obj-C++ bridge (.mm)
┌──────────────────────▼───────────────────────────────────┐
│  MtlPhysCore static library (C++20 + metal-cpp)          │
│  ├─ Engine          (owns particle state, step loop)     │
│  ├─ MetalContext    (device, command queue, libraries)   │
│  └─ kernels.metal   (integrator, neighbors, constraints) │
└──────────────────────────────────────────────────────────┘
```

## Getting started

### Prerequisites

- macOS 14+ on Apple Silicon
- Xcode 15+
- [XcodeGen](https://github.com/yonoz/XcodeGen) — `brew install xcodegen`

### One-time setup

1. **Download metal-cpp** from <https://developer.apple.com/metal/cpp/>.
   Unzip the contents (the folder containing `Foundation/`, `Metal/`, `QuartzCore/`) into `vendor/metal-cpp/`.

   ```sh
   ./scripts/fetch_metal_cpp.sh   # opens the download page in your browser
   ```

2. **Generate the Xcode project:**

   ```sh
   xcodegen generate
   open mtlphys.xcodeproj
   ```

3. Build and run the `mtlphys` scheme. You should see ~10k particles falling under gravity in a 3D Metal-rendered window.

## Tests & benchmarks

Every change to the engine should be validated against the test/bench suite. They live in `tests/` and run as a single command-line binary built by the `mtlphys_tests` target.

```sh
xcodegen generate

# Build the test binary (once per source change)
xcodebuild -project mtlphys.xcodeproj -scheme mtlphys_tests \
           -configuration Release build

# Locate it (path varies by Xcode build dir)
TESTBIN=$(xcodebuild -project mtlphys.xcodeproj -scheme mtlphys_tests \
                     -configuration Release -showBuildSettings 2>/dev/null \
          | awk -F' = ' '/^[[:space:]]*BUILT_PRODUCTS_DIR /{print $2}')/mtlphys_tests

# Unit tests only (~1 s)
"$TESTBIN" -tse=bench

# Benchmarks only (~10 s)
"$TESTBIN" -ts=bench

# Everything
"$TESTBIN"
```

Benchmarks print a p50 / p95 / p99 / mean table per workload — paste into a PR description or commit message when a change moves the numbers.

## Layout

```
mtlphys/
├── project.yml             XcodeGen spec — single source of truth for the build
├── engine/
│   ├── include/mtlphys/    Public headers
│   ├── src/                C++ engine (metal-cpp)
│   └── shaders/            Metal Shading Language compute + render kernels
├── app/                    SwiftUI macOS app + Obj-C++ bridge
├── vendor/metal-cpp/       Apple's Metal C++ headers (download separately)
└── scripts/                Setup helpers
```

## Roadmap

- [x] **W1** Scaffolding, integrator kernel, basic point renderer
- [ ] **W2** GPU spatial hash + neighbor finding
- [ ] **W3** XPBD constraint solver (distance + density)
- [ ] **W4** Rigid bodies via shape matching, demo scenes
- [ ] **W5** Screen-space fluid rendering, benchmarks vs Bullet, polish

## License

MIT
