import SwiftUI

struct ChatView: View {
    @EnvironmentObject var engine: SquishEngine
    @State private var input: String = ""

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 1) {
                    Text("Chat").font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(SQ.textPrimary)
                    Text(engine.health?.model.map { engine.friendlyModelName(for: $0) }
                         ?? engine.preferredModel)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(SQ.textSecond)
                }
                Spacer()
                Button {
                    engine.agentMode.toggle()
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: engine.agentMode ? "bolt.fill" : "bolt")
                            .font(.system(size: 10))
                        Text("agent")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .foregroundStyle(engine.agentMode ? SQ.accentBright : SQ.textSecond)
                    .padding(.horizontal, 9).padding(.vertical, 4)
                    .background(engine.agentMode ? SQ.accent.opacity(0.18) : Color.clear)
                    .overlay(
                        RoundedRectangle(cornerRadius: 7)
                            .stroke(engine.agentMode ? SQ.accent.opacity(0.5) : SQ.border, lineWidth: 1)
                    )
                    .cornerRadius(7)
                }
                .buttonStyle(.plain)
                .help(engine.agentMode
                      ? "Agent mode: prompts call tools via /v1/agent/run"
                      : "Chat mode: plain completions")
                if !engine.messages.isEmpty {
                    Button("Clear") { engine.clearChat() }
                        .font(.system(size: 11))
                        .foregroundStyle(SQ.textSecond)
                        .buttonStyle(.plain)
                        .padding(.leading, 4)
                }
            }
            .padding(.horizontal, 20).padding(.vertical, 12)
            .background(SQ.surface)

            Divider().background(SQ.border)

            if engine.messages.isEmpty {
                EmptyStateView()
            } else {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(spacing: 0) {
                            ForEach(engine.messages) { msg in
                                MessageBubble(message: msg)
                                    .id(msg.id)
                            }
                        }
                        .padding(.vertical, 16)
                    }
                    .onChange(of: engine.messages.count) { _ in
                        if let last = engine.messages.last {
                            withAnimation(.easeOut(duration: 0.2)) {
                                proxy.scrollTo(last.id, anchor: .bottom)
                            }
                        }
                    }
                    .onChange(of: engine.messages.last?.content) { _ in
                        if let last = engine.messages.last {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                }
            }

            Divider().background(SQ.border)
            InputBar(input: $input) {
                let text = input
                input = ""
                engine.sendMessage(text)
            } onStop: {
                engine.stopGeneration()
            }
        }
    }
}

struct MessageBubble: View {
    let message: ChatMessage
    private var isUser: Bool { message.role == "user" }

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            if isUser { Spacer(minLength: 60) }

            if !isUser {
                Image("SquishLogo")
                    .resizable().scaledToFit()
                    .frame(width: 28, height: 28)
                    .padding(.top, 2)
            }

            VStack(alignment: isUser ? .trailing : .leading, spacing: 4) {
                Text(message.role == "user" ? "You" : "squish")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(SQ.textSecond)
                    .kerning(0.5)

                // Agentic turns: render each tool call as a live card.
                if !isUser && !message.toolCalls.isEmpty {
                    ForEach(message.toolCalls) { call in
                        ToolCallCardView(call: call)
                    }
                }

                if message.content.isEmpty && message.isStreaming && message.toolCalls.isEmpty {
                    TypingIndicator()
                        .padding(.horizontal, 14).padding(.vertical, 10)
                        .background(SQ.asstBubble)
                        .cornerRadius(12)
                        .overlay(RoundedRectangle(cornerRadius: 12).stroke(SQ.border, lineWidth: 1))
                } else if !message.content.isEmpty {
                    Group {
                        if isUser {
                            // User messages stay plain — no point parsing markdown
                            // out of what the user just typed.
                            Text(message.content)
                                .font(.system(size: 14))
                                .foregroundStyle(SQ.textPrimary)
                                .textSelection(.enabled)
                        } else {
                            // Assistant output is markdown — render structure
                            // (lists, headings, code blocks) plus inline bold/italic.
                            MarkdownView(content: message.content)
                                .textSelection(.enabled)
                        }
                    }
                    .padding(.horizontal, 14).padding(.vertical, 10)
                    .background(isUser ? SQ.userBubble : SQ.asstBubble)
                    .cornerRadius(12)
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(isUser ? SQ.accent.opacity(0.3) : SQ.border, lineWidth: 1)
                    )
                }
            }

            if isUser {
                Circle()
                    .fill(SQ.accent.opacity(0.2))
                    .frame(width: 28, height: 28)
                    .overlay(
                        Text("U").font(.system(size: 11, weight: .bold))
                            .foregroundStyle(SQ.accentBright)
                    )
                    .padding(.top, 2)
            } else {
                Spacer(minLength: 60)
            }
        }
        .padding(.horizontal, 16).padding(.vertical, 4)
    }
}

/// A single agent tool invocation, rendered live as it executes.
struct ToolCallCardView: View {
    let call: ToolCallRecord

    private var accent: Color {
        if call.done && call.error != nil { return SQ.red }
        if call.done { return SQ.green }
        return SQ.accentBright
    }

    /// SF Symbol + friendly label per built-in tool.
    private var meta: (icon: String, label: String) {
        switch call.name {
        case "squish_list_dir":    return ("folder", "List directory")
        case "squish_read_file":   return ("doc.text", "Read file")
        case "squish_read_document": return ("doc.richtext", "Read document")
        case "squish_write_file":  return ("square.and.pencil", "Write file")
        case "squish_create_file": return ("doc.badge.plus", "Create file")
        case "squish_delete_file": return ("trash", "Delete file")
        case "squish_move_file":   return ("arrow.right.doc.on.clipboard", "Move file")
        case "squish_apply_edit":  return ("pencil.line", "Edit file")
        case "squish_run_shell":   return ("terminal", "Run shell")
        case "squish_python_repl": return ("chevron.left.forwardslash.chevron.right", "Python")
        case "squish_fetch_url":   return ("globe", "Fetch URL")
        case "squish_web_search":  return ("magnifyingglass", "Web search")
        default:
            let clean = call.name
                .replacingOccurrences(of: "squish_", with: "")
                .replacingOccurrences(of: "_", with: " ")
            return ("wrench.and.screwdriver", clean.capitalized)
        }
    }

    /// The most telling argument (path/command/url/query) shown inline.
    private var argHint: String {
        guard let d = call.arguments.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: d) as? [String: Any]
        else { return "" }
        for k in ["path", "command", "url", "query", "src", "code"] {
            if let v = obj[k] { return String(describing: v) }
        }
        return obj.values.first.map { String(describing: $0) } ?? ""
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 7) {
                Image(systemName: meta.icon)
                    .font(.system(size: 11))
                    .foregroundStyle(accent)
                Text(meta.label)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(SQ.textPrimary)
                if !argHint.isEmpty {
                    Text(argHint)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(accent.opacity(0.9))
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                Spacer()
                if !call.done {
                    ProgressView().controlSize(.small).scaleEffect(0.6)
                } else if let ms = call.elapsedMs {
                    Text(String(format: "%.0f ms", ms))
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(SQ.textSecond)
                }
                if call.done {
                    Image(systemName: call.error != nil ? "xmark.circle.fill" : "checkmark.circle.fill")
                        .font(.system(size: 11))
                        .foregroundStyle(call.error != nil ? SQ.red : SQ.green)
                }
            }
            if !call.arguments.isEmpty {
                Text(call.arguments)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(SQ.textSecond)
                    .textSelection(.enabled)
                    .lineLimit(3)
            }
            if call.done, let out = call.error ?? call.result, !out.isEmpty {
                Text(out)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(call.error != nil ? SQ.red : SQ.textPrimary)
                    .textSelection(.enabled)
                    .lineLimit(8)
                    .padding(.top, 4)
                    .overlay(Rectangle().fill(SQ.border).frame(height: 1), alignment: .top)
            }
        }
        .padding(.horizontal, 12).padding(.vertical, 9)
        .background(SQ.asstBubble)
        .cornerRadius(10)
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(accent.opacity(0.35), lineWidth: 1)
        )
    }
}

struct TypingIndicator: View {
    @State private var dotPhase = 0
    let timer = Timer.publish(every: 0.4, on: .main, in: .common).autoconnect()

    var body: some View {
        HStack(spacing: 5) {
            ForEach(0..<3) { i in
                Circle()
                    .fill(SQ.accentBright)
                    .frame(width: 7, height: 7)
                    .opacity(dotPhase == i ? 1.0 : 0.3)
                    .animation(.easeInOut(duration: 0.3), value: dotPhase)
            }
        }
        .padding(.vertical, 4)
        .onReceive(timer) { _ in dotPhase = (dotPhase + 1) % 3 }
    }
}

struct InputBar: View {
    @Binding var input: String
    let onSend: () -> Void
    let onStop: () -> Void
    @EnvironmentObject var engine: SquishEngine
    @FocusState private var focused: Bool

    var body: some View {
        HStack(spacing: 10) {
            TextField("Message squish…", text: $input, axis: .vertical)
                .font(.system(size: 14))
                .foregroundStyle(SQ.textPrimary)
                .textFieldStyle(.plain)
                .lineLimit(1...6)
                .focused($focused)
                .onSubmit {
                    if !engine.isGenerating && !input.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty {
                        onSend()
                    }
                }

            if engine.isGenerating {
                Button(action: onStop) {
                    Image(systemName: "stop.fill")
                        .font(.system(size: 14))
                        .foregroundStyle(SQ.red)
                }
                .buttonStyle(.plain)
            } else {
                Button(action: onSend) {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.system(size: 26))
                        .foregroundStyle(
                            input.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty
                            ? SQ.textSecond : SQ.accent
                        )
                }
                .buttonStyle(.plain)
                .disabled(input.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty)
                .keyboardShortcut(.return, modifiers: .command)
            }
        }
        .padding(.horizontal, 16).padding(.vertical, 12)
        .background(SQ.surface)
        .onAppear { focused = true }
    }
}

struct EmptyStateView: View {
    @EnvironmentObject var engine: SquishEngine
    var body: some View {
        VStack(spacing: 16) {
            Spacer()
            Image("SquishLogo")
                .resizable().scaledToFit()
                .frame(width: 72, height: 72)
                .opacity(0.7)
            Text("squish")
                .font(.system(size: 24, weight: .bold))
                .foregroundStyle(SQ.textPrimary)
            Text(engine.serverRunning
                 ? "Start chatting with \(engine.health?.model.map { engine.friendlyModelName(for: $0) } ?? engine.preferredModel)"
                 : "Start the squish server to begin")
                .font(.system(size: 14))
                .foregroundStyle(SQ.textSecond)
                .multilineTextAlignment(.center)
            if !engine.serverRunning {
                Button("Start Server") { engine.startServer() }
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 20).padding(.vertical, 8)
                    .background(SQ.accent)
                    .cornerRadius(8)
                    .buttonStyle(.plain)
            }
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
}
