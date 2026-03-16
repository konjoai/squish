// SquishBar — macOS 13+ menu bar extra for the squish local inference server.
//
// Build & run (requires Xcode 15+ or swift 5.9 toolchain):
//   swift build -c release
//   .build/release/SquishBar
//
// Or open in Xcode:
//   open Package.swift

import SwiftUI

@main
struct SquishBarApp: App {
    @StateObject private var engine = SquishEngine()

    var body: some Scene {
        MenuBarExtra {
            SquishMenuView()
                .environmentObject(engine)
        } label: {
            MenuBarLabel(engine: engine)
        }
        .menuBarExtraStyle(.window)
    }
}
