import SwiftUI
import MetalKit
import simd

// MARK: - Custom MTKView with mouse + scroll input

protocol CameraInputDelegate: AnyObject {
    func dragOrbit(deltaX: CGFloat, deltaY: CGFloat)
    func scrollZoom(delta: CGFloat)
    func pulseStart(at viewPoint: CGPoint, viewSize: CGSize)
    func pulseDrag(to viewPoint: CGPoint, deltaX: CGFloat, deltaY: CGFloat, viewSize: CGSize)
    func pulseEnd()
}

final class FluidMTKView: MTKView {
    weak var inputDelegate: CameraInputDelegate?

    override var acceptsFirstResponder: Bool { true }
    override func becomeFirstResponder() -> Bool { true }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
    }
    override func mouseDragged(with event: NSEvent) {
        inputDelegate?.dragOrbit(deltaX: event.deltaX, deltaY: event.deltaY)
    }
    override func scrollWheel(with event: NSEvent) {
        inputDelegate?.scrollZoom(delta: event.scrollingDeltaY)
    }
    // Right mouse → fluid pulse
    override func rightMouseDown(with event: NSEvent) {
        let local = convert(event.locationInWindow, from: nil)
        inputDelegate?.pulseStart(at: local, viewSize: bounds.size)
    }
    override func rightMouseDragged(with event: NSEvent) {
        let local = convert(event.locationInWindow, from: nil)
        inputDelegate?.pulseDrag(to: local,
                                 deltaX: event.deltaX, deltaY: event.deltaY,
                                 viewSize: bounds.size)
    }
    override func rightMouseUp(with event: NSEvent) {
        inputDelegate?.pulseEnd()
    }
}

// MARK: - SwiftUI wrapper

struct MetalView: NSViewRepresentable {
    @ObservedObject var hud: PerfHUD
    @ObservedObject var sceneCtrl: SceneController

    func makeCoordinator() -> Renderer {
        let r = Renderer(hud: hud)
        sceneCtrl.onSelect = { [weak r] kind in r?.changeScene(kind) }
        return r
    }

    func makeNSView(context: Context) -> FluidMTKView {
        let view = FluidMTKView()
        view.device = context.coordinator.device
        view.colorPixelFormat = .bgra8Unorm_srgb
        view.depthStencilPixelFormat = .depth32Float
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1.0)
        view.preferredFramesPerSecond = 120
        view.delegate = context.coordinator
        view.inputDelegate = context.coordinator
        context.coordinator.attach(view: view)
        return view
    }

    func updateNSView(_ nsView: FluidMTKView, context: Context) {}
}

// MARK: - Renderer (MTKView delegate + camera state + input)

final class Renderer: NSObject, MTKViewDelegate, CameraInputDelegate {
    let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let engine: MPEngineBridge
    private weak var view: MTKView?

    // Camera state — orbit around `target` at radius `distance`, defined by
    // azimuth (around y) and elevation (above horizontal).
    private var azimuth:   Float = .pi * 0.25
    private var elevation: Float = 0.40
    private var distance:  Float = 13.0
    private var target = SIMD3<Float>(0, 1.5, 0)

    private var lastFrameTime: CFTimeInterval = CACurrentMediaTime()
    private var frameCounter = 0
    private var fpsAccum: CFTimeInterval = 0

    private let timingLock = NSLock()
    private var simAccum: CFTimeInterval = 0
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

    // MARK: scene changes

    func changeScene(_ kind: SceneKind) {
        engine.reset(withParticleCount: engine.particleCount, scene: kind.rawValue)
    }

    // MARK: input

    func dragOrbit(deltaX: CGFloat, deltaY: CGFloat) {
        let sens: Float = 0.005
        azimuth   -= Float(deltaX) * sens
        elevation += Float(deltaY) * sens
        // Clamp elevation so we don't flip past the poles
        elevation  = max(-1.4, min(1.4, elevation))
    }

    func scrollZoom(delta: CGFloat) {
        let sens: Float = 0.04
        distance *= (1.0 - Float(delta) * sens)
        distance  = max(4.0, min(40.0, distance))
    }

    // MARK: pulse — right-click drag to push fluid

    private var pulseHeld     = false
    private var pulsePoint    = CGPoint.zero
    private var pulseViewSize = CGSize.zero
    private var pulseDX: Float = 0
    private var pulseDY: Float = 0

    func pulseStart(at viewPoint: CGPoint, viewSize: CGSize) {
        pulseHeld = true
        pulsePoint = viewPoint
        pulseViewSize = viewSize
        pulseDX = 0; pulseDY = 0
    }
    func pulseDrag(to viewPoint: CGPoint, deltaX: CGFloat, deltaY: CGFloat, viewSize: CGSize) {
        pulsePoint = viewPoint
        pulseViewSize = viewSize
        pulseDX += Float(deltaX)
        pulseDY += Float(deltaY)
    }
    func pulseEnd() { pulseHeld = false }

    // Compute the current pulse: cursor RAY (origin + dir) and the world-space
    // force the user just dragged. Cylinder pulse on the GPU side picks up any
    // particle within radius of this ray, anywhere along it — so we don't need
    // a hit-point. Returns nil if the user isn't actively dragging.
    private func computePulse() -> (origin: SIMD3<Float>, dir: SIMD3<Float>, force: SIMD3<Float>)? {
        guard pulseHeld, (pulseDX != 0 || pulseDY != 0),
              pulseViewSize.width > 0, pulseViewSize.height > 0 else { return nil }

        let ndcX = Float(pulsePoint.x / pulseViewSize.width  * 2.0 - 1.0)
        let ndcY = Float(pulsePoint.y / pulseViewSize.height * 2.0 - 1.0)

        let eye    = eyePosition()
        let fwd    = simd_normalize(target - eye)
        let right  = simd_normalize(simd_cross(fwd, SIMD3<Float>(0, 1, 0)))
        let upV    = simd_cross(right, fwd)

        let fovY: Float = 1.0
        let tanHalfFov  = tan(fovY * 0.5)
        let aspect      = Float(pulseViewSize.width / pulseViewSize.height)
        let dir = simd_normalize(fwd
                               + right * (ndcX * tanHalfFov * aspect)
                               + upV   * (ndcY * tanHalfFov))

        let scale: Float = 0.6
        let force = (right * pulseDX - upV * pulseDY) * scale

        pulseDX = 0; pulseDY = 0
        return (eye, dir, force)
    }

    private func eyePosition() -> SIMD3<Float> {
        let cy = cos(elevation), sy = sin(elevation)
        let cx = cos(azimuth),   sx = sin(azimuth)
        return target + SIMD3<Float>(distance * cy * cx,
                                     distance * sy,
                                     distance * cy * sx)
    }

    // MARK: MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let passDesc = view.currentRenderPassDescriptor else { return }

        let now = CACurrentMediaTime()
        let dt  = Float(min(now - lastFrameTime, 1.0/30.0))
        lastFrameTime = now

        let aspect = Float(view.drawableSize.width / max(view.drawableSize.height, 1))
        let stepDt = min(dt, 1.0/120.0)

        // ---- Sim ----
        guard let simCmd = commandQueue.makeCommandBuffer() else { return }
        if let pulse = computePulse() {
            engine.applyMousePulse(with: simCmd,
                                   originX: pulse.origin.x, originY: pulse.origin.y, originZ: pulse.origin.z,
                                   dirX:    pulse.dir.x,    dirY:    pulse.dir.y,    dirZ:    pulse.dir.z,
                                   forceX:  pulse.force.x,  forceY:  pulse.force.y,  forceZ:  pulse.force.z,
                                   radius: 0.5)
        }
        engine.step(with: simCmd, dt: stepDt)
        simCmd.addCompletedHandler { [weak self] cb in
            guard let self else { return }
            let elapsed = cb.gpuEndTime - cb.gpuStartTime
            self.timingLock.lock()
            self.simAccum += elapsed
            self.timingLock.unlock()
        }
        simCmd.commit()

        // ---- Render ----
        guard let renderCmd = commandQueue.makeCommandBuffer() else { return }
        let eye = eyePosition()
        let w = UInt32(view.drawableSize.width)
        let h = UInt32(view.drawableSize.height)
        engine.render(with: renderCmd,
                      renderPassDescriptor: passDesc,
                      pixelWidth: w, pixelHeight: h,
                      aspectRatio: aspect,
                      eyeX: eye.x, eyeY: eye.y, eyeZ: eye.z,
                      targetX: target.x, targetY: target.y, targetZ: target.z)
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

        // ---- Rolling HUD over ~0.5 s ----
        frameCounter += 1
        fpsAccum += Double(dt)
        if fpsAccum >= 0.5 {
            let fps = Double(frameCounter) / fpsAccum
            let count = Int(engine.particleCount)
            timingLock.lock()
            let n = max(1, timingCount)
            let simMs = (simAccum / Double(n)) * 1000
            let renderMs = (renderAccum / Double(n)) * 1000
            simAccum = 0; renderAccum = 0; timingCount = 0
            timingLock.unlock()
            Task { @MainActor in
                self.hud.fps = fps
                self.hud.particleCount = count
                self.hud.simMs = simMs
                self.hud.renderMs = renderMs
            }
            frameCounter = 0; fpsAccum = 0
        }
    }
}
