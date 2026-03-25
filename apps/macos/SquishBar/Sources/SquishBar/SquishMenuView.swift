// SquishMenuView.swift — Menu window displayed when the status bar item is clicked.

import SwiftUI

struct SquishMenuView: View {
    @EnvironmentObject var engine: SquishEngine

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {

            // ── Header ────────────────────────────────────────────────────────
            HStack {
                Image(systemName: engine.statusSymbol)
                    .foregroundStyle(engine.serverRunning ? .green : .secondary)
                VStack(alignment: .leading, spacing: 1) {
                    Text("squish")
                        .font(.headline)
                    Text(statusLine)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if let v = engine.health?.version {
                    Text("v\(v)")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)

            Divider()

            // ── Model picker ──────────────────────────────────────────────────
            if !engine.models.isEmpty {
                VStack(alignment: .leading, spacing: 0) {
                    Menu("Switch Model") {
                        ForEach(engine.models, id: \.self) { modelId in
                            Button {
                                engine.switchModel(modelId)
                            } label: {
                                Label(
                                    modelId,
                                    systemImage: modelId == (engine.health?.model ?? engine.preferredModel)
                                        ? "checkmark" : "cpu"
                                )
                            }
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 4)
                }
            }

            // ── Pull Model ────────────────────────────────────────────────────
            MenuButton(title: "Pull Model…", systemImage: "arrow.down.circle") {
                engine.promptPullModel()
            }

            // ── Compression progress ──────────────────────────────────────────
            if let progress = engine.compressionProgress {
                VStack(alignment: .leading, spacing: 4) {
                    ProgressView(value: max(0, min(progress, 1)))
                        .progressViewStyle(.linear)
                    Text(engine.compressionStatus)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 6)
            }

            Divider()

            // ── Chat button ───────────────────────────────────────────────────
            MenuButton(title: "Open Chat UI", systemImage: "bubble.left.and.bubble.right") {
                NSWorkspace.shared.open(engine.chatURL)
            }
            .disabled(!engine.serverRunning)

            // ── Model info ────────────────────────────────────────────────────
            if let h = engine.health, let model = h.model {
                HStack {
                    Image(systemName: "cpu")
                        .foregroundStyle(.secondary)
                        .frame(width: 18)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(model)
                            .font(.body)
                        if let uptime = h.uptime_s {
                            Text("up \(formatUptime(uptime))")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    Spacer()
                    if let reqs = h.requests {
                        Text("\(reqs) req")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 6)
            }

            Divider()

            // ── Server controls ───────────────────────────────────────────────
            if engine.serverRunning {
                MenuButton(title: "Stop Server", systemImage: "stop.circle") {
                    engine.stopServer()
                }
            } else {
                MenuButton(title: "Start Server  (\(engine.preferredModel))",
                           systemImage: "play.circle") {
                    engine.startServer()
                }
            }

            // ── Settings ──────────────────────────────────────────────────────
            Divider()

            SettingsSection(engine: engine)

            Divider()

            // ── Quit ──────────────────────────────────────────────────────────
            MenuButton(title: "Quit SquishBar", systemImage: "power") {
                NSApplication.shared.terminate(nil)
            }
            .padding(.bottom, 4)
        }
        .frame(width: 300)
        .task { await engine.fetchModels() }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private var statusLine: String {
        guard engine.serverRunning, let h = engine.health else {
            return engine.lastError != nil ? "connection error" : "server offline"
        }
        guard h.loaded else { return "loading model…" }
        if let tps = h.avg_tps {
            return String(format: "%.1f tok/s  •  http://\(engine.host):\(engine.port)", tps)
        }
        return "http://\(engine.host):\(engine.port)"
    }

    private func formatUptime(_ seconds: Double) -> String {
        let s = Int(seconds)
        if s < 60 { return "\(s)s" }
        let m = s / 60; let rs = s % 60
        if m < 60 { return "\(m)m \(rs)s" }
        return "\(m / 60)h \(m % 60)m"
    }
}


// ── Reusable menu row ─────────────────────────────────────────────────────────

struct MenuButton: View {
    let title: String
    let systemImage: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Label(title, systemImage: systemImage)
        }
        .buttonStyle(.plain)
        .padding(.horizontal, 16)
        .padding(.vertical, 6)
        .frame(maxWidth: .infinity, alignment: .leading)
        .contentShape(Rectangle())
        .hoverHighlight()
    }
}


// ── Settings section ──────────────────────────────────────────────────────────

struct SettingsSection: View {
    @ObservedObject var engine: SquishEngine
    @State private var editing = false

    var body: some View {
        if editing {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Host").frame(width: 44, alignment: .leading)
                    TextField("127.0.0.1", text: $engine.host)
                        .textFieldStyle(.roundedBorder)
                }
                HStack {
                    Text("Port").frame(width: 44, alignment: .leading)
                    TextField("11435", value: $engine.port, format: .number)
                        .textFieldStyle(.roundedBorder)
                }
                HStack {
                    Text("Model").frame(width: 44, alignment: .leading)
                    TextField("qwen3:8b", text: $engine.preferredModel)
                        .textFieldStyle(.roundedBorder)
                }
                HStack {
                    Text("Hotkey").frame(width: 44, alignment: .leading)
                    TextField("⌘⌥S", text: $engine.hotkey)
                        .textFieldStyle(.roundedBorder)
                }
                Button("Done") { editing = false }
                    .buttonStyle(.plain)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
        } else {
            MenuButton(title: "Settings…", systemImage: "gear") {
                editing = true
            }
        }
    }
}


// ── Hover highlight modifier ──────────────────────────────────────────────────

private struct HoverHighlightModifier: ViewModifier {
    @State private var hovering = false
    func body(content: Content) -> some View {
        content
            .background(hovering ? Color.accentColor.opacity(0.15) : Color.clear)
            .cornerRadius(6)
            .onHover { hovering = $0 }
    }
}

extension View {
    func hoverHighlight() -> some View {
        modifier(HoverHighlightModifier())
    }
}
