import SwiftUI

struct MenuBarLabel: View {
    @ObservedObject var engine: SquishEngine

    // Show throughput only when actively generating. Idle squish reports
    // avg_tps ≈ 0 because no tokens have flowed recently — we don't want
    // a permanent "0.0 tok/s" sitting in the menu bar.
    private var activeTPS: Double? {
        guard engine.serverRunning,
              let h = engine.health, h.loaded,
              let tps = h.avg_tps, tps >= 0.1
        else { return nil }
        return tps
    }

    private var dotColor: Color {
        // Gray when offline (calm), green when healthy. Red was too alarming
        // for what's usually a benign "server isn't running yet" state.
        engine.serverRunning ? .green : .gray
    }

    var body: some View {
        HStack(spacing: 3) {
            Image("SquishMenuBar")
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(width: 18, height: 18)
                // Overlay the status dot INSIDE the frame's bottom-right
                // corner so it can't be clipped by NSStatusItem's content
                // rect. SF Symbol + .palette rendering keeps the color from
                // being template-tinted to monochrome.
                .overlay(alignment: .bottomTrailing) {
                    Image(systemName: "circle.fill")
                        .symbolRenderingMode(.palette)
                        .foregroundStyle(dotColor)
                        .font(.system(size: 7))
                        // Tiny white halo so the dot stays visible against
                        // both the dark squish character and lighter menu
                        // bar gradients.
                        .background(
                            Circle()
                                .fill(Color(NSColor.windowBackgroundColor))
                                .frame(width: 9, height: 9)
                        )
                }

            if let tps = activeTPS {
                Text(String(format: "%.1f tok/s", tps))
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
            }

            // Zero-size bridge that listens for .openSquishWebUI notifications
            // (posted by the global ⌘⌥S hotkey) and converts them into a
            // SwiftUI openWindow call — the hotkey handler can't reach
            // @Environment(\.openWindow) directly.
            HotkeyBridge()
        }
        .help(engine.statusLabel)
    }
}
