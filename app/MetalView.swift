import SwiftUI
import MetalKit

struct MetalView: NSViewRepresentable {
    @ObservedObject var hud: PerfHUD

    func makeCoordinator() -> Renderer {
        Renderer(hud: hud)
    }

    func makeNSView(context: Context) -> MTKView {
        let view = MTKView()
        view.device = context.coordinator.device
        view.colorPixelFormat = .bgra8Unorm_srgb
        view.depthStencilPixelFormat = .depth32Float
        view.clearColor = MTLClearColor(red: 0.04, green: 0.05, blue: 0.07, alpha: 1.0)
        view.preferredFramesPerSecond = 120
        view.delegate = context.coordinator
        context.coordinator.attach(view: view)
        return view
    }

    func updateNSView(_ nsView: MTKView, context: Context) {}
}

final class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let engine: MPEngineBridge
    private weak var view: MTKView?

    private var lastFrameTime: CFTimeInterval = CACurrentMediaTime()
    private var startTime:    CFTimeInterval = CACurrentMediaTime()
    private var frameCounter = 0
    private var fpsAccum:     CFTimeInterval = 0

    private let timingLock = NSLock()
    private var simAccum:    CFTimeInterval = 0
    private var renderAccum: CFTimeInterval = 0
    private var timingCount: Int = 0

    private let hud: PerfHUD

    init(hud: PerfHUD) {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            fatalError("mtlphys: no Metal device available")
        }
        guard let q = dev.makeCommandQueue() else {
            fatalError("mtlphys: failed to create command queue")
        }
        self.device = dev
        self.commandQueue = q
        self.engine = MPEngineBridge(device: dev)
        self.hud = hud
        super.init()
        Task { @MainActor in self.hud.particleCount = Int(self.engine.particleCount) }
    }

    func attach(view: MTKView) { self.view = view }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let passDesc = view.currentRenderPassDescriptor else { return }

        let now    = CACurrentMediaTime()
        let dt     = Float(min(now - lastFrameTime, 1.0/30.0))   // clamp giant first frame
        lastFrameTime = now
        let t      = Float(now - startTime)
        let aspect = Float(view.drawableSize.width / max(view.drawableSize.height, 1))

        // ---- PBF step CB: predict → spatial hash → 3 solver iters → finalize ----
        // Clamp dt for stability — early frames or stalls can spike dt past the
        // safe SPH timestep, which causes blow-up. 1/120s ceiling.
        let stepDt = min(dt, 1.0 / 120.0)
        guard let simCmd = commandQueue.makeCommandBuffer() else { return }
        engine.step(with: simCmd, dt: stepDt)
        simCmd.addCompletedHandler { [weak self] cb in
            guard let self else { return }
            let elapsed = cb.gpuEndTime - cb.gpuStartTime
            self.timingLock.lock()
            self.simAccum += elapsed
            self.timingLock.unlock()
        }
        simCmd.commit()

        // ---- Render CB ----
        guard let renderCmd = commandQueue.makeCommandBuffer() else { return }
        let w = UInt32(view.drawableSize.width)
        let h = UInt32(view.drawableSize.height)
        engine.render(with: renderCmd, renderPassDescriptor: passDesc,
                      pixelWidth: w, pixelHeight: h,
                      aspectRatio: aspect, time: t)
        renderCmd.present(drawable)
        renderCmd.addCompletedHandler { [weak self] cb in
            guard let self else { return }
            let elapsed = cb.gpuEndTime - cb.gpuStartTime
            self.timingLock.lock()
            self.renderAccum += elapsed
            self.timingCount += 1
            self.timingLock.unlock()
        }
        renderCmd.commit()

        // Rolling FPS + GPU-time over ~0.5s windows.
        frameCounter += 1
        fpsAccum += Double(dt)
        if fpsAccum >= 0.5 {
            let measuredFps = Double(frameCounter) / fpsAccum
            let count = Int(engine.particleCount)

            timingLock.lock()
            let n = max(1, timingCount)
            let simMs    = (simAccum    / Double(n)) * 1000.0
            let renderMs = (renderAccum / Double(n)) * 1000.0
            simAccum = 0; renderAccum = 0; timingCount = 0
            timingLock.unlock()

            Task { @MainActor in
                self.hud.fps           = measuredFps
                self.hud.particleCount = count
                self.hud.simMs         = simMs
                self.hud.renderMs      = renderMs
            }
            frameCounter = 0
            fpsAccum = 0
        }
    }
}
