import SwiftUI

struct MiniMenuView: View {
    @EnvironmentObject var engine: SquishEngine
    @Environment(\.openWindow) var openWindow

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack(spacing: 10) {
                ZStack(alignment: .bottomTrailing) {
                    Image("SquishLogo")
                        .resizable().scaledToFit()
                        .frame(width: 36, height: 36)
                    Circle()
                        .fill(engine.serverRunning ? Color.green : Color.gray.opacity(0.7))
                        .frame(width: 10, height: 10)
                        .overlay(Circle().stroke(Color(NSColor.windowBackgroundColor), lineWidth: 1.5))
                }
                VStack(alignment: .leading, spacing: 1) {
                    Text("squish").font(.headline)
                    Text(engine.statusLabel).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                if let v = engine.health?.version {
                    Text("v\(v)").font(.caption2).foregroundStyle(.tertiary)
                }
            }
            .padding(.horizontal, 14).padding(.vertical, 10)

            // Spinning indicator while a model switch is in flight — gives
            // the user clear "yes, it's working" feedback while the new
            // model loads (which can take 10s–60s for larger weights).
            if engine.isSwitching {
                HStack(spacing: 8) {
                    SpinnerIcon()
                    VStack(alignment: .leading, spacing: 1) {
                        Text("Switching model").font(.caption).bold()
                        Text(engine.compressionStatus.isEmpty
                             ? "Loading \(engine.preferredModel)…"
                             : engine.compressionStatus)
                            .font(.caption2).foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                    Spacer()
                }
                .padding(.horizontal, 14).padding(.vertical, 8)
                Divider()
            }

            // Show recent spawn error — usually "binary not found" or a
            // squish-CLI rejection like "Unknown model: 'TinyLlama:1.1b'".
            // This is the visible signal users need when Start Server seems
            // to do nothing.
            if let err = engine.lastError, !engine.isSwitching {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 6) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange).font(.system(size: 11))
                        Text("Server error").font(.caption).bold()
                        Spacer()
                        Button("Log") { engine.revealServerLog() }
                            .buttonStyle(.borderless).font(.caption)
                    }
                    Text(err).font(.caption2).foregroundStyle(.secondary)
                        .lineLimit(3).fixedSize(horizontal: false, vertical: true)
                }
                .padding(.horizontal, 14).padding(.vertical, 6)
                Divider()
            }

            Divider()

            MiniBtn(title: "Open Squish", icon: "rectangle.expand.vertical") {
                openWindow(id: "squish-main")
                NSApp.activate(ignoringOtherApps: true)
            }

            MiniBtn(title: "Open Web Chat", icon: "globe") {
                openWindow(id: "squish-webui")
                NSApp.activate(ignoringOtherApps: true)
            }

            Divider()

            if let model = engine.health?.model {
                HStack {
                    Image(systemName: "cpu")
                        .foregroundStyle(.secondary).frame(width: 18)
                    Text(engine.friendlyModelName(for: model)).font(.system(size: 13))
                    Spacer()
                    if let tps = engine.health?.avg_tps {
                        Text(String(format: "%.1f tok/s", tps))
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                .padding(.horizontal, 14).padding(.vertical, 7)
                Divider()
            }

            if !engine.models.isEmpty {
                let active = engine.health?.model.map { engine.friendlyModelName(for: $0) }
                    ?? engine.preferredModel
                Menu("Switch Model") {
                    ForEach(engine.models, id: \.self) { m in
                        Button {
                            engine.switchModel(m)
                        } label: {
                            Label(m, systemImage: m == active ? "checkmark" : "cpu")
                        }
                    }
                }
                .padding(.horizontal, 14).padding(.vertical, 4)
            }

            MiniBtn(title: "Pull Model…", icon: "arrow.down.circle") { engine.promptPullModel() }

            if let p = engine.compressionProgress {
                VStack(alignment: .leading, spacing: 3) {
                    ProgressView(value: max(0, min(p, 1))).progressViewStyle(.linear)
                    Text(engine.compressionStatus).font(.caption)
                        .foregroundStyle(.secondary).lineLimit(1)
                }
                .padding(.horizontal, 14).padding(.vertical, 6)
            }

            Divider()

            if engine.serverRunning {
                MiniBtn(title: "Stop Server", icon: "stop.circle") { engine.stopServer() }
            } else {
                MiniBtn(title: "Start Server (\(engine.preferredModel))", icon: "play.circle") {
                    engine.startServer()
                }
            }

            Divider()

            MiniBtn(title: "Quit SquishBar", icon: "power") {
                NSApplication.shared.terminate(nil)
            }
            .padding(.bottom, 4)
        }
        .frame(width: 280)
        .task { await engine.fetchModels() }
    }
}

struct SpinnerIcon: View {
    @State private var angle: Double = 0
    var body: some View {
        Image(systemName: "arrow.triangle.2.circlepath")
            .font(.system(size: 14, weight: .semibold))
            .foregroundStyle(.tint)
            .rotationEffect(.degrees(angle))
            .onAppear {
                withAnimation(.linear(duration: 1.1).repeatForever(autoreverses: false)) {
                    angle = 360
                }
            }
    }
}

private struct MiniBtn: View {
    let title: String
    let icon: String
    let action: () -> Void
    @State private var hovering = false
    var body: some View {
        Button(action: action) {
            Label(title, systemImage: icon)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 14).padding(.vertical, 6)
                .background(hovering ? Color.accentColor.opacity(0.12) : Color.clear)
                .cornerRadius(5)
        }
        .buttonStyle(.plain)
        .contentShape(Rectangle())
        .onHover { hovering = $0 }
    }
}
