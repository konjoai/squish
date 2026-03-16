// MenuBarLabel.swift — The icon + text displayed in the macOS menu bar.

import SwiftUI

struct MenuBarLabel: View {
    @ObservedObject var engine: SquishEngine

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: engine.statusSymbol)
                .imageScale(.small)
            if engine.serverRunning, let h = engine.health, h.loaded, let tps = h.avg_tps {
                Text(String(format: "%.1f tok/s", tps))
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
            }
        }
        .help(engine.statusLabel)
    }
}
