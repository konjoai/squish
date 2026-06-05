import SwiftUI

struct ModelBrowserView: View {
    @EnvironmentObject var engine: SquishEngine
    @State private var searchText = ""

    var filtered: [String] {
        searchText.isEmpty ? engine.models
        : engine.models.filter { $0.localizedCaseInsensitiveContains(searchText) }
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Models").font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(SQ.textPrimary)
                Spacer()
                Text("\(engine.models.count) available")
                    .font(.system(size: 11)).foregroundStyle(SQ.textSecond)
                Button(action: { Task { await engine.fetchModels() } }) {
                    Image(systemName: "arrow.clockwise")
                        .font(.system(size: 12))
                        .foregroundStyle(SQ.textSecond)
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 20).padding(.vertical, 12)
            .background(SQ.surface)

            Divider().background(SQ.border)

            HStack(spacing: 8) {
                Image(systemName: "magnifyingglass").foregroundStyle(SQ.textSecond)
                TextField("Search models…", text: $searchText)
                    .textFieldStyle(.plain)
                    .font(.system(size: 13))
                    .foregroundStyle(SQ.textPrimary)
            }
            .padding(.horizontal, 16).padding(.vertical, 10)
            .background(SQ.surface2)

            Divider().background(SQ.border)

            if engine.models.isEmpty && !engine.serverRunning {
                VStack(spacing: 12) {
                    Spacer()
                    Image(systemName: "server.rack")
                        .font(.system(size: 36)).foregroundStyle(SQ.textSecond)
                    Text("No models cached yet")
                        .font(.system(size: 14)).foregroundStyle(SQ.textSecond)
                    Text("Start the server once to detect available models.\nThey'll be remembered after that.")
                        .font(.system(size: 12)).foregroundStyle(SQ.textSecond)
                        .multilineTextAlignment(.center)
                    Button("Start Server") { engine.startServer() }
                        .foregroundStyle(SQ.accent).buttonStyle(.plain)
                        .font(.system(size: 13, weight: .semibold))
                    Spacer()
                }
                .frame(maxWidth: .infinity)
            } else if engine.models.isEmpty {
                VStack(spacing: 12) {
                    Spacer()
                    Image(systemName: "server.rack")
                        .font(.system(size: 36)).foregroundStyle(SQ.textSecond)
                    Text("No models detected")
                        .font(.system(size: 14)).foregroundStyle(SQ.textSecond)
                    Spacer()
                }
                .frame(maxWidth: .infinity)
            } else {
                ScrollView {
                    let active = engine.health?.model.map { engine.friendlyModelName(for: $0) }
                        ?? engine.preferredModel
                    LazyVStack(spacing: 6) {
                        ForEach(filtered, id: \.self) { model in
                            ModelRow(
                                model: model,
                                isCurrent: model == active
                            ) {
                                engine.switchModel(model)
                            }
                        }
                    }
                    .padding(12)
                }
            }

            Divider().background(SQ.border)

            Button(action: { engine.promptPullModel() }) {
                Label("Pull New Model…", systemImage: "arrow.down.circle")
                    .font(.system(size: 13))
                    .foregroundStyle(SQ.accentBright)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
            }
            .buttonStyle(.plain)
            .background(SQ.surface)
        }
    }
}

struct ModelRow: View {
    let model: String
    let isCurrent: Bool
    let onSelect: () -> Void
    @State private var hover = false

    var body: some View {
        Button(action: onSelect) {
            HStack(spacing: 12) {
                Image(systemName: isCurrent ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(isCurrent ? SQ.green : SQ.textSecond)
                    .font(.system(size: 16))
                VStack(alignment: .leading, spacing: 2) {
                    Text(model).font(.system(size: 13, weight: isCurrent ? .semibold : .regular))
                        .foregroundStyle(isCurrent ? SQ.textPrimary : SQ.textSecond)
                    Text(modelFamily(model)).font(.system(size: 11))
                        .foregroundStyle(SQ.textSecond)
                }
                Spacer()
                if isCurrent {
                    Text("active").font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(SQ.green)
                        .padding(.horizontal, 8).padding(.vertical, 3)
                        .background(SQ.green.opacity(0.12))
                        .cornerRadius(4)
                }
            }
            .padding(.horizontal, 14).padding(.vertical, 10)
            .background(
                isCurrent ? SQ.accent.opacity(0.1) :
                hover      ? SQ.accent.opacity(0.06) : SQ.surface
            )
            .cornerRadius(8)
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(isCurrent ? SQ.accent.opacity(0.3) : SQ.border, lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
        .onHover { hover = $0 }
    }

    private func modelFamily(_ m: String) -> String {
        if m.hasPrefix("qwen3")  { return "Alibaba · Qwen3 family" }
        if m.hasPrefix("qwen2")  { return "Alibaba · Qwen2.5 family" }
        if m.hasPrefix("llama")  { return "Meta · Llama family" }
        if m.hasPrefix("gemma")  { return "Google · Gemma family" }
        if m.hasPrefix("mistral"){ return "Mistral AI" }
        if m.hasPrefix("phi")    { return "Microsoft · Phi family" }
        return "Local model"
    }
}
