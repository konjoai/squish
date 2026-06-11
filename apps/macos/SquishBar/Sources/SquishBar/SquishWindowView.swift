import SwiftUI

struct SquishWindowView: View {
    @EnvironmentObject var engine: SquishEngine
    @State private var activePanel: Panel = .chat

    enum Panel: String, CaseIterable {
        case chat = "Chat"
        case models = "Models"
    }

    var body: some View {
        HStack(spacing: 0) {
            SidebarView(activePanel: $activePanel)
                .frame(width: 220)
                .background(SQ.surface)

            Divider().background(SQ.border)

            Group {
                switch activePanel {
                case .chat:   ChatView()
                case .models: ModelBrowserView()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(SQ.bg)
        }
        .background(SQ.bg)
    }
}

struct SidebarView: View {
    @EnvironmentObject var engine: SquishEngine
    @Binding var activePanel: SquishWindowView.Panel

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                Image("SquishLogo")
                    .resizable().scaledToFit()
                    .frame(width: 32, height: 32)
                Text("squish")
                    .font(.system(size: 16, weight: .bold))
                    .foregroundStyle(SQ.textPrimary)
                Spacer()
                Circle()
                    .fill(engine.serverRunning ? SQ.green : Color.gray.opacity(0.6))
                    .frame(width: 8, height: 8)
            }
            .padding(.horizontal, 16).padding(.vertical, 14)
            .background(SQ.surface2)

            Divider().background(SQ.border)

            if engine.isSwitching {
                HStack(spacing: 8) {
                    SpinnerIcon()
                    VStack(alignment: .leading, spacing: 1) {
                        Text("Switching")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(SQ.accentBright)
                        Text(engine.preferredModel)
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(SQ.textSecond)
                            .lineLimit(1)
                    }
                    Spacer()
                }
                .padding(.horizontal, 14).padding(.vertical, 8)
                Divider().background(SQ.border)
            } else if engine.serverRunning, let h = engine.health {
                ServerStatsRow(health: h)
                Divider().background(SQ.border)
            }

            ForEach(SquishWindowView.Panel.allCases, id: \.self) { panel in
                SidebarNavItem(
                    label: panel.rawValue,
                    icon: panel == .chat ? "bubble.left.and.bubble.right" : "server.rack",
                    isActive: activePanel == panel
                ) { activePanel = panel }
            }

            Divider().background(SQ.border).padding(.top, 4)

            VStack(alignment: .leading, spacing: 4) {
                Text("MODEL").font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(SQ.textSecond).kerning(1.5)
                if let m = engine.health?.model {
                    Text(engine.friendlyModelName(for: m))
                        .font(.system(size: 12)).foregroundStyle(SQ.accentBright)
                } else {
                    Text(engine.preferredModel).font(.system(size: 12))
                        .foregroundStyle(SQ.textSecond)
                }
            }
            .padding(.horizontal, 16).padding(.vertical, 10)

            Spacer()

            Divider().background(SQ.border)
            HStack(spacing: 8) {
                if engine.serverRunning {
                    SidebarIconBtn(icon: "stop.circle", color: SQ.red) { engine.stopServer() }
                } else {
                    SidebarIconBtn(icon: "play.circle", color: SQ.green) { engine.startServer() }
                }
                SidebarIconBtn(icon: "arrow.down.circle", color: SQ.textSecond) {
                    engine.promptPullModel()
                }
                SidebarIconBtn(icon: "arrow.counterclockwise", color: SQ.textSecond) {
                    Task { await engine.fetchModels() }
                }
                Spacer()
                if let v = engine.health?.version {
                    Text("v\(v)").font(.system(size: 10))
                        .foregroundStyle(SQ.textSecond)
                }
            }
            .padding(.horizontal, 14).padding(.vertical, 10)
        }
    }
}

struct ServerStatsRow: View {
    let health: SquishHealth
    var body: some View {
        HStack(spacing: 0) {
            StatPill(label: "tok/s", value: health.avg_tps.map { String(format:"%.1f",$0) } ?? "—")
            Divider().frame(height: 24).background(SQ.border)
            StatPill(label: "req", value: health.requests.map(String.init) ?? "0")
            if let u = health.uptime_s {
                Divider().frame(height: 24).background(SQ.border)
                StatPill(label: "up", value: formatUptime(u))
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
    }
    private func formatUptime(_ s: Double) -> String {
        let i = Int(s); if i < 60 { return "\(i)s" }
        let m = i/60; return m < 60 ? "\(m)m" : "\(m/60)h\(m%60)m"
    }
}

struct StatPill: View {
    let label: String
    let value: String
    var body: some View {
        VStack(spacing: 1) {
            Text(value).font(.system(size: 13, weight: .semibold, design: .monospaced))
                .foregroundStyle(SQ.accentBright)
            Text(label).font(.system(size: 9)).foregroundStyle(SQ.textSecond).kerning(1)
        }
        .frame(maxWidth: .infinity)
    }
}

struct SidebarNavItem: View {
    let label: String
    let icon: String
    let isActive: Bool
    let action: () -> Void
    @State private var hover = false
    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Image(systemName: icon)
                    .foregroundStyle(isActive ? SQ.accent : SQ.textSecond)
                    .frame(width: 18)
                Text(label).font(.system(size: 13, weight: isActive ? .semibold : .regular))
                    .foregroundStyle(isActive ? SQ.textPrimary : SQ.textSecond)
                Spacer()
            }
            .padding(.horizontal, 14).padding(.vertical, 7)
            .background(
                isActive ? SQ.accent.opacity(0.15) :
                hover    ? SQ.accent.opacity(0.07) : Color.clear
            )
            .cornerRadius(6)
            .padding(.horizontal, 6)
        }
        .buttonStyle(.plain)
        .onHover { hover = $0 }
    }
}

struct SidebarIconBtn: View {
    let icon: String
    let color: Color
    let action: () -> Void
    var body: some View {
        Button(action: action) {
            Image(systemName: icon).foregroundStyle(color)
                .font(.system(size: 16))
        }
        .buttonStyle(.plain)
    }
}
