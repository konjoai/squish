import SwiftUI
import AppKit

@main
struct SquishBarApp: App {
    @StateObject private var engine = SquishEngine()
    @NSApplicationDelegateAdaptor(SquishBarAppDelegate.self) private var appDelegate

    var body: some Scene {
        MenuBarExtra {
            MiniMenuView()
                .environmentObject(engine)
        } label: {
            MenuBarLabel(engine: engine)
        }
        .menuBarExtraStyle(.window)

        Window("Squish", id: "squish-main") {
            SquishWindowView()
                .environmentObject(engine)
        }
        .commandsRemoved()
        .windowResizability(.contentSize)
        .defaultSize(width: 920, height: 640)

        // Internal web-UI window — single instance. openWindow(id:"squish-webui")
        // brings the existing window to front instead of spawning a new one.
        Window("Squish Web Chat", id: "squish-webui") {
            WebUIWindow()
                .environmentObject(engine)
        }
        .commandsRemoved()
        .defaultSize(width: 980, height: 720)
    }
}

// macOS lets the spawned `squish run` subprocess survive its parent (it's
// reparented to launchd, PID 1). Without explicit cleanup the port stays
// bound and the next launch fails with EADDRINUSE. This delegate posts a
// notification on terminate that the engine listens for, and the engine
// kills any spawned server before AppKit finishes shutting us down.
final class SquishBarAppDelegate: NSObject, NSApplicationDelegate {
    func applicationWillTerminate(_ notification: Notification) {
        NotificationCenter.default.post(name: .squishBarWillQuit, object: nil)
    }
}

extension Notification.Name {
    static let squishBarWillQuit = Notification.Name("squishBarWillQuit")
}

// A hidden helper view that bridges the global-hotkey notification into the
// SwiftUI openWindow API. Embedded in the menu-bar label since that's always
// alive once the app is running.
struct HotkeyBridge: View {
    @Environment(\.openWindow) private var openWindow
    var body: some View {
        Color.clear
            .frame(width: 0, height: 0)
            .onReceive(NotificationCenter.default.publisher(for: .openSquishWebUI)) { _ in
                openWindow(id: "squish-webui")
                NSApp.activate(ignoringOtherApps: true)
            }
    }
}
