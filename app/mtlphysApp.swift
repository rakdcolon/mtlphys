import SwiftUI

@main
struct MtlPhysApp: App {
    var body: some Scene {
        WindowGroup("mtlphys") {
            ContentView()
                .frame(minWidth: 900, minHeight: 600)
        }
        .windowStyle(.hiddenTitleBar)
    }
}
