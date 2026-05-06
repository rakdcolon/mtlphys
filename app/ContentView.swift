import SwiftUI

// Scene IDs mirror Engine::kScene* constants in Engine.hpp.
enum SceneKind: UInt32, CaseIterable, Identifiable {
    case cubeDrop   = 0
    case damBreak   = 1
    case sphereDrop = 2

    var id: UInt32 { rawValue }
    var label: String {
        switch self {
        case .cubeDrop:   return "Cube Drop"
        case .damBreak:   return "Dam Break"
        case .sphereDrop: return "Sphere Drop"
        }
    }
}

@MainActor
final class SceneController: ObservableObject {
    @Published var current: SceneKind = .cubeDrop
    var onSelect: ((SceneKind) -> Void)?

    func select(_ s: SceneKind) {
        guard s != current else { return }
        current = s
        onSelect?(s)
    }
}

struct ContentView: View {
    @StateObject private var hud = PerfHUD()
    @StateObject private var sceneCtrl = SceneController()

    var body: some View {
        ZStack(alignment: .topLeading) {
            MetalView(hud: hud, sceneCtrl: sceneCtrl)
                .ignoresSafeArea()

            VStack(alignment: .leading, spacing: 12) {
                HUDOverlay(hud: hud)
                ScenePicker(ctrl: sceneCtrl)
            }
            .padding(16)
        }
        .background(Color.black)
    }
}

private struct ScenePicker: View {
    @ObservedObject var ctrl: SceneController

    var body: some View {
        HStack(spacing: 6) {
            ForEach(SceneKind.allCases) { kind in
                Button(kind.label) { ctrl.select(kind) }
                    .buttonStyle(.borderedProminent)
                    .tint(ctrl.current == kind ? .blue : .gray)
                    .controlSize(.small)
            }
        }
        .padding(8)
        .background(.black.opacity(0.45), in: RoundedRectangle(cornerRadius: 8))
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
