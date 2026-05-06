import SwiftUI

// Scene IDs mirror Engine::kScene* constants in Engine.hpp.
enum SceneKind: UInt32, CaseIterable, Identifiable {
    case cubeDrop   = 0
    case damBreak   = 1
    case sphereDrop = 2

    var id: UInt32 { rawValue }
    var label: String {
        switch self {
        case .cubeDrop:   return "Cube"
        case .damBreak:   return "Dam"
        case .sphereDrop: return "Sphere"
        }
    }
}

// Single source of truth for all interactive sim state. The Renderer holds a
// reference and reads pause/speed each frame; UI mutations call back into the
// Renderer via onConfigChange (for re-seeding) so scene/count switches actually
// rebuild the engine state.
@MainActor
final class SimController: ObservableObject {
    @Published var scene: SceneKind   = .cubeDrop
    @Published var paused: Bool       = false
    @Published var speed: Float       = 1.0
    @Published var particleCount: UInt32 = 500_000

    var onConfigChange: (() -> Void)?

    func selectScene(_ s: SceneKind) {
        guard s != scene else { return }
        scene = s
        onConfigChange?()
    }
    func selectCount(_ c: UInt32) {
        guard c != particleCount else { return }
        particleCount = c
        onConfigChange?()
    }
    func togglePause() { paused.toggle() }
    func setSpeed(_ s: Float) { speed = s }
    func requestReset() { onConfigChange?() }
}

struct ContentView: View {
    @StateObject private var hud = PerfHUD()
    @StateObject private var sim = SimController()

    var body: some View {
        ZStack(alignment: .topLeading) {
            MetalView(hud: hud, sim: sim)
                .ignoresSafeArea()

            VStack(alignment: .leading, spacing: 10) {
                HUDOverlay(hud: hud)
                ControlPanel(sim: sim)
            }
            .padding(16)
        }
        .background(Color.black)
    }
}

// MARK: - Control panel

private struct ControlPanel: View {
    @ObservedObject var sim: SimController

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            SegmentRow(label: "Scene",
                       options: SceneKind.allCases.map { ($0, $0.label) },
                       selected: sim.scene,
                       onSelect: sim.selectScene)

            SegmentRow(label: "Count",
                       options: [
                            (UInt32(50_000),  "50K"),
                            (UInt32(200_000), "200K"),
                            (UInt32(500_000), "500K"),
                            (UInt32(1_000_000), "1M"),
                       ],
                       selected: sim.particleCount,
                       onSelect: sim.selectCount)

            SegmentRow(label: "Speed",
                       options: [
                            (Float(0.25), "¼×"),
                            (Float(0.5),  "½×"),
                            (Float(1.0),  "1×"),
                            (Float(2.0),  "2×"),
                       ],
                       selected: sim.speed,
                       onSelect: sim.setSpeed)

            HStack(spacing: 6) {
                Text("       ")
                    .font(.system(size: 11, design: .monospaced))
                Button(sim.paused ? "▶ Play" : "⏸ Pause") { sim.togglePause() }
                    .buttonStyle(.borderedProminent)
                    .tint(sim.paused ? .green : .gray)
                    .controlSize(.small)
                Button("↻ Reset") { sim.requestReset() }
                    .buttonStyle(.borderedProminent)
                    .tint(.gray)
                    .controlSize(.small)
                Text("(space / R)")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.5))
            }
        }
        .padding(8)
        .background(.black.opacity(0.45), in: RoundedRectangle(cornerRadius: 8))
    }
}

private struct SegmentRow<T: Hashable>: View {
    let label: String
    let options: [(T, String)]
    let selected: T
    let onSelect: (T) -> Void

    var body: some View {
        HStack(spacing: 4) {
            Text(label.padding(toLength: 6, withPad: " ", startingAt: 0))
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(.white.opacity(0.65))
            ForEach(options.indices, id: \.self) { idx in
                let (value, lbl) = options[idx]
                Button(lbl) { onSelect(value) }
                    .buttonStyle(.borderedProminent)
                    .tint(selected == value ? .blue : .gray.opacity(0.6))
                    .controlSize(.small)
            }
        }
    }
}

private struct HUDOverlay: View {
    @ObservedObject var hud: PerfHUD

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("mtlphys").font(.system(size: 14, weight: .bold, design: .monospaced))
            Text(String(format: "%6.1f fps", hud.fps))
                .font(.system(size: 12, design: .monospaced))
            Text("\(hud.particleCount) particles")
                .font(.system(size: 12, design: .monospaced))
            Divider().frame(width: 180).background(.white.opacity(0.2))
            Text(String(format: "pbf    %6.2f ms", hud.simMs))
                .font(.system(size: 12, design: .monospaced))
            Text(String(format: "render %6.2f ms", hud.renderMs))
                .font(.system(size: 12, design: .monospaced))
            Text(String(format: "total  %6.2f ms / 8.33 budget", hud.simMs + hud.renderMs))
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(.white.opacity(0.6))
        }
        .foregroundStyle(.white.opacity(0.85))
        .padding(10)
        .background(.black.opacity(0.45), in: RoundedRectangle(cornerRadius: 8))
    }
}

@MainActor
final class PerfHUD: ObservableObject {
    @Published var fps: Double = 0
    @Published var particleCount: Int = 0
    @Published var simMs: Double = 0
    @Published var renderMs: Double = 0
}
