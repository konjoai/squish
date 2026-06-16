// SquishEngine.swift

import Foundation
import SwiftUI
import AppKit

// ── Data models ───────────────────────────────────────────────────────
struct SquishHealth: Codable {
    var status:   String
    var version:  String?
    var model:    String?
    var loaded:   Bool
    var avg_tps:  Double?
    var requests: Int?
    var uptime_s: Double?
}

/// One tool invocation inside an agentic assistant turn — rendered as a card
/// in the chat so the user can watch the agent call tools live.
struct ToolCallRecord: Identifiable {
    let id: String          // call_id from the server
    let name: String
    var arguments: String   // compact JSON
    var result: String? = nil
    var error: String? = nil
    var elapsedMs: Double? = nil
    var done: Bool = false
}

struct ChatMessage: Identifiable {
    let id = UUID()
    let role: String
    var content: String
    var isStreaming: Bool = false
    /// Tool calls made during an agentic turn (agent mode), in execution order.
    var toolCalls: [ToolCallRecord] = []
}

// ── Engine ────────────────────────────────────────────────────────────
@MainActor
final class SquishEngine: ObservableObject {
    @AppStorage("squish.host")   var host: String = "127.0.0.1"
    @AppStorage("squish.port")   var port: Int    = 11435
    @AppStorage("squish.apiKey") var apiKey: String = "squish"
    @AppStorage("squish.model")  var preferredModel: String = "qwen3:8b"
    @AppStorage("squish.hotkey") var hotkey: String = "⌘⌥S"
    /// When on, chat prompts run through the tool-calling agent loop
    /// (POST /v1/agent/run) instead of plain chat completions.
    @AppStorage("squish.agentMode") var agentMode: Bool = false
    @AppStorage("squish.cachedModels") private var cachedModelsJSON: String = ""

    @Published var health:              SquishHealth? = nil
    @Published var models:              [String]      = []
    @Published var isPolling:           Bool          = false
    @Published var serverRunning:       Bool          = false
    @Published var lastError:           String?       = nil
    @Published var compressionProgress: Double?       = nil
    @Published var compressionStatus:   String        = ""
    @Published var messages:            [ChatMessage] = []
    @Published var isGenerating:        Bool          = false
    @Published var isSwitching:         Bool          = false

    private var pollTask:   Task<Void, Never>? = nil
    private var serverProc: Process?           = nil
    private var streamTask: Task<Void, Never>? = nil

    init() {
        startPolling()
        loadCachedModels()
        _registerGlobalHotkey()
        // Refresh the model list from the filesystem at launch so a stale
        // `preferredModel` persisted in UserDefaults (e.g. "meta-llama-3.1:8b"
        // — an Ollama-format name `squish run` rejects) gets healed BEFORE
        // any user interaction triggers a spawn.
        Task { await self.fetchModels() }
        // Clean up our spawned squish subprocess on app quit so it doesn't
        // get orphaned to launchd and leave the port bound.
        NotificationCenter.default.addObserver(
            forName: .squishBarWillQuit, object: nil, queue: .main
        ) { [weak self] _ in
            self?.serverProc?.terminate()
            // Belt-and-suspenders: if the spawned process ignored SIGTERM,
            // kill anything still holding our port. quitDeadline is small
            // because AppKit gives us only a few seconds before SIGKILL.
            if let port = self?.port {
                let pids = self?.pidsListening(on: port) ?? []
                for pid in pids { kill(pid, SIGTERM) }
                usleep(200_000)
                let remaining = self?.pidsListening(on: port) ?? []
                for pid in remaining { kill(pid, SIGKILL) }
            }
        }
    }
    deinit { pollTask?.cancel() }

    // ── Polling ───────────────────────────────────────────────────────
    func startPolling() {
        pollTask?.cancel()
        pollTask = Task {
            while !Task.isCancelled {
                await fetchHealth()
                try? await Task.sleep(for: .seconds(5))
            }
        }
        isPolling = true
    }

    func stopPolling() { pollTask?.cancel(); pollTask = nil; isPolling = false }

    private func fetchHealth() async {
        guard let url = URL(string: "http://\(host):\(port)/health") else { return }
        var req = URLRequest(url: url, timeoutInterval: 4)
        req.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        do {
            let (data, _) = try await URLSession.shared.data(for: req)
            let h = try JSONDecoder().decode(SquishHealth.self, from: data)
            health = h; serverRunning = true; lastError = nil
            // Auto-refresh model list when server becomes available
            if models.isEmpty { Task { await self.fetchModels() } }
        } catch {
            health = nil; serverRunning = false
        }
    }

    func fetchModels() async {
        // Strategy:
        //   • Scan ~/.squish/models and ~/models for directory names.
        //   • Convert each directory name to ollama-style catalog shorthand
        //     (e.g. "Qwen2.5-1.5B-Instruct-int3" → "qwen2.5:1.5b-int3").
        //   • That shorthand IS what gets passed to `squish run` — the CLI
        //     rejects bare directory names but accepts catalog IDs and
        //     catalog IDs with quant suffixes.
        //   • Dedupe by shorthand so users don't see both "qwen2.5:1.5b"
        //     (from the bf16 anchor) and "qwen2.5:1.5b-int4" (from the int4
        //     sidecar) — they're the same default combo.
        let installed = installedModelDirectoryNames()
        if !installed.isEmpty {
            var seen = Set<String>()
            var friendly: [String] = []
            for dir in installed {
                let name = friendlyModelName(for: dir)
                if seen.insert(name).inserted { friendly.append(name) }
            }
            friendly.sort()
            models = friendly
            if let encoded = try? JSONEncoder().encode(friendly) {
                cachedModelsJSON = String(data: encoded, encoding: .utf8) ?? ""
            }
            if !friendly.contains(preferredModel) {
                let liveModel = health?.model.flatMap { friendlyModelName(for: $0) }
                let fallback = liveModel.flatMap { friendly.contains($0) ? $0 : nil }
                    ?? friendly.first { $0.hasPrefix("qwen3:") && !$0.contains("-int") }
                    ?? friendly.first { $0.hasPrefix("qwen3:") }
                    ?? friendly.first
                if let f = fallback { preferredModel = f }
            }
            return
        }
        await fetchModelsV1Fallback()
    }

    /// Convert a model directory name into ollama-style catalog shorthand.
    /// Examples:
    ///   Qwen2.5-1.5B-Instruct-bf16          → qwen2.5:1.5b
    ///   Qwen2.5-1.5B-Instruct-int4          → qwen2.5:1.5b
    ///   Qwen2.5-1.5B-Instruct-int3          → qwen2.5:1.5b-int3
    ///   Qwen2.5-1.5B-Instruct-int4-awq      → qwen2.5:1.5b-awq
    ///   Qwen2.5-1.5B-Instruct-mixed-attn    → qwen2.5:1.5b-mixed-attn
    ///   Meta-Llama-3.1-8B-Instruct-bf16     → llama3.1:8b
    ///   Llama-3.2-1B-Instruct-int3          → llama3.2:1b-int3
    ///   Mistral-7B-Instruct-v0.3-bf16       → mistral:7b
    ///   gemma-3-4b-it-int4                  → gemma3:4b
    ///   Qwen1.5-MoE-A2.7B-Chat-int3         → qwen1.5-moe:a2.7b-int3
    ///   Qwen3-0.6B-int4                     → qwen3:0.6b
    /// The returned string IS the load argument we'll pass to `squish run`.
    static func friendlyModelName(for dir: String) -> String {
        var s = dir.lowercased()
        if s.hasPrefix("meta-") { s = String(s.dropFirst(5)) }

        // Drop variant tags that don't affect the catalog identity.
        for tag in ["-instruct", "-chat", "-it"] {
            s = s.replacingOccurrences(of: tag, with: "")
        }
        // Drop trailing version suffix like "-v0.3" (Mistral, OLMoE).
        s = s.replacingOccurrences(
            of: #"-v\d+(?:\.\d+)?"#, with: "", options: .regularExpression
        )

        // Rewrite int4-awq → awq BEFORE the generic -int4 strip below.
        s = s.replacingOccurrences(of: "-int4-awq", with: "-awq")
        // Default precision suffixes don't appear in the friendly name —
        // bf16 and int4 are both "the default loadout" for catalog shorthand.
        s = s.replacingOccurrences(
            of: #"-(bf16|int4)$"#, with: "", options: .regularExpression
        )

        // Family / version / size → catalog colon form.
        let rewrites: [(String, String)] = [
            (#"^qwen3-(\d+(?:\.\d+)?b)"#,            "qwen3:$1"),
            (#"^qwen2\.5-(\d+(?:\.\d+)?b)"#,         "qwen2.5:$1"),
            (#"^qwen1\.5-moe-a(\d+(?:\.\d+)?b)"#,    "qwen1.5-moe:a$1"),
            (#"^llama-(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?b)"#, "llama$1:$2"),
            (#"^mistral-(\d+(?:\.\d+)?b)"#,          "mistral:$1"),
            (#"^gemma-(\d+)-(\d+(?:\.\d+)?b)"#,      "gemma$1:$2"),
            (#"^smollm2-(\d+(?:\.\d+)?[mb])"#,       "smollm2:$1"),
            (#"^deepseek-r1-(\d+(?:\.\d+)?b)"#,      "deepseek-r1:$1"),
            (#"^phi(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?b)"#, "phi$1:$2"),
        ]
        for (pattern, replacement) in rewrites {
            s = s.replacingOccurrences(
                of: pattern, with: replacement, options: .regularExpression
            )
        }
        return s
    }

    /// Instance accessor so SwiftUI views can call engine.friendlyModelName(for:).
    func friendlyModelName(for dir: String) -> String {
        Self.friendlyModelName(for: dir)
    }

    /// Lists every entry in the squish models directory.
    /// Search order matches `squish/cli.py:_resolve_models_dir`:
    ///   1. $SQUISH_MODELS_DIR
    ///   2. ~/.squish/models   (canonical)
    ///   3. ~/models           (legacy)
    private func installedModelDirectoryNames() -> [String] {
        let fm = FileManager.default
        var roots: [URL] = []
        if let envPath = ProcessInfo.processInfo.environment["SQUISH_MODELS_DIR"],
           !envPath.isEmpty {
            roots.append(URL(fileURLWithPath: (envPath as NSString).expandingTildeInPath))
        }
        let home = fm.homeDirectoryForCurrentUser
        roots.append(home.appendingPathComponent(".squish/models", isDirectory: true))
        roots.append(home.appendingPathComponent("models", isDirectory: true))

        var seen = Set<String>()
        var out: [String] = []
        var diag: [String] = []
        for root in roots {
            let exists = fm.fileExists(atPath: root.path)
            diag.append("• \(root.path) — \(exists ? "exists" : "missing")")
            guard exists,
                  let entries = try? fm.contentsOfDirectory(
                    at: root, includingPropertiesForKeys: [.isDirectoryKey]
                  )
            else { continue }
            for entry in entries {
                let name = entry.lastPathComponent
                if name.hasPrefix(".") { continue }
                if seen.insert(name).inserted { out.append(name) }
            }
            diag.append("  ↳ \(entries.count) entries")
        }
        writeDiagnostic("Model dir scan: \(out.count) found\n" + diag.joined(separator: "\n"))
        return out.sorted()
    }

    private func writeDiagnostic(_ msg: String) {
        let logs = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/SquishBar", isDirectory: true)
        try? FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true)
        let url = logs.appendingPathComponent("squishbar.log")
        let line = "[\(ISO8601DateFormatter().string(from: Date()))] \(msg)\n"
        if let data = line.data(using: .utf8) {
            if let handle = try? FileHandle(forWritingTo: url) {
                _ = try? handle.seekToEnd()
                try? handle.write(contentsOf: data)
                try? handle.close()
            } else {
                try? data.write(to: url)
            }
        }
    }

    private func fetchModelsV1Fallback() async {
        guard let url = URL(string: "http://\(host):\(port)/v1/models") else { return }
        var req = URLRequest(url: url, timeoutInterval: 4)
        req.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        do {
            let (data, _) = try await URLSession.shared.data(for: req)
            struct Resp: Decodable { struct M: Decodable { let id: String }; let data: [M] }
            let r = try JSONDecoder().decode(Resp.self, from: data)
            let fetched = r.data.map(\.id).sorted()
            if !fetched.isEmpty {
                models = fetched
                if let encoded = try? JSONEncoder().encode(fetched) {
                    cachedModelsJSON = String(data: encoded, encoding: .utf8) ?? ""
                }
            }
        } catch { }
    }

    func loadCachedModels() {
        guard !cachedModelsJSON.isEmpty,
              let data = cachedModelsJSON.data(using: .utf8),
              let cached = try? JSONDecoder().decode([String].self, from: data)
        else { return }
        if models.isEmpty { models = cached }
    }

    // ── Chat streaming ────────────────────────────────────────────────
    func sendMessage(_ text: String) {
        if agentMode { sendAgentMessage(text); return }
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        guard serverRunning else { return }

        messages.append(ChatMessage(role: "user", content: text))
        let reply = ChatMessage(role: "assistant", content: "", isStreaming: true)
        messages.append(reply)
        let replyIdx = messages.count - 1
        isGenerating = true
        streamTask?.cancel()

        let history = messages.dropLast().map { ["role": $0.role, "content": $0.content] }
        let model = health?.model ?? preferredModel
        let urlStr = "http://\(host):\(port)/v1/chat/completions"
        let key = apiKey

        streamTask = Task {
            defer {
                Task { @MainActor in
                    self.isGenerating = false
                    if replyIdx < self.messages.count {
                        self.messages[replyIdx].isStreaming = false
                    }
                }
            }
            guard let url = URL(string: urlStr) else { return }
            var req = URLRequest(url: url)
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
            let body: [String: Any] = ["model": model, "messages": history, "stream": true]
            req.httpBody = try? JSONSerialization.data(withJSONObject: body)

            do {
                let (bytes, _) = try await URLSession.shared.bytes(for: req)
                for try await line in bytes.lines {
                    if Task.isCancelled { break }
                    guard line.hasPrefix("data: ") else { continue }
                    let chunk = String(line.dropFirst(6))
                    if chunk == "[DONE]" { break }
                    guard let d = chunk.data(using: .utf8),
                          let json = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
                          let choices = json["choices"] as? [[String: Any]],
                          let delta = choices.first?["delta"] as? [String: Any],
                          let token = delta["content"] as? String else { continue }
                    await MainActor.run {
                        if replyIdx < self.messages.count {
                            self.messages[replyIdx].content += token
                        }
                    }
                }
            } catch { }
        }
    }

    func stopGeneration() {
        streamTask?.cancel()
        isGenerating = false
        if let i = messages.indices.last { messages[i].isStreaming = false }
    }

    func clearChat() { messages = [] }

    // ── Agentic chat (tool calling) ───────────────────────────────────
    /// Run the prompt through the server's multi-step agent loop
    /// (POST /v1/agent/run, SSE). The model can call tools (read/write files,
    /// run shell, search the web …); each call streams into the assistant
    /// turn as a live tool-call card.
    func sendAgentMessage(_ text: String) {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        guard serverRunning else { return }

        messages.append(ChatMessage(role: "user", content: text))
        let reply = ChatMessage(role: "assistant", content: "", isStreaming: true)
        messages.append(reply)
        let replyIdx = messages.count - 1
        isGenerating = true
        streamTask?.cancel()

        let history = messages.dropLast().map { ["role": $0.role, "content": $0.content] }
        let urlStr = "http://\(host):\(port)/v1/agent/run"
        let key = apiKey

        streamTask = Task {
            defer {
                Task { @MainActor in
                    self.isGenerating = false
                    if replyIdx < self.messages.count {
                        self.messages[replyIdx].isStreaming = false
                    }
                }
            }
            guard let url = URL(string: urlStr) else { return }
            var req = URLRequest(url: url)
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.setValue("text/event-stream", forHTTPHeaderField: "Accept")
            req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
            let body: [String: Any] = [
                "messages": history,
                "max_steps": 8,
                "max_tokens": 768,
                "temperature": 0.4,
            ]
            req.httpBody = try? JSONSerialization.data(withJSONObject: body)

            do {
                let (bytes, _) = try await URLSession.shared.bytes(for: req)
                for try await line in bytes.lines {
                    if Task.isCancelled { break }
                    guard line.hasPrefix("data: ") else { continue }
                    let chunk = String(line.dropFirst(6))
                    guard let d = chunk.data(using: .utf8),
                          let json = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
                          let type = json["type"] as? String else { continue }
                    await MainActor.run { self.applyAgentEvent(type: type, json: json, replyIdx: replyIdx) }
                    if type == "done" || type == "error" { break }
                }
            } catch { }
        }
    }

    /// Fold one /v1/agent/run SSE event into the assistant message at `replyIdx`.
    private func applyAgentEvent(type: String, json: [String: Any], replyIdx: Int) {
        guard replyIdx < messages.count else { return }
        switch type {
        case "text_delta":
            if let delta = json["delta"] as? String { messages[replyIdx].content += delta }
        case "tool_call_start":
            // The streamed text up to here was the tool-call syntax; replace it
            // with a structured tool card so the bubble stays readable.
            messages[replyIdx].content = ""
            let callId = json["call_id"] as? String ?? UUID().uuidString
            let name = json["tool_name"] as? String ?? "tool"
            var argStr = ""
            if let args = json["arguments"],
               let data = try? JSONSerialization.data(withJSONObject: args),
               let s = String(data: data, encoding: .utf8) { argStr = s }
            messages[replyIdx].toolCalls.append(
                ToolCallRecord(id: callId, name: name, arguments: argStr)
            )
        case "tool_call_result":
            let callId = json["call_id"] as? String ?? ""
            if let i = messages[replyIdx].toolCalls.firstIndex(where: { $0.id == callId }) {
                messages[replyIdx].toolCalls[i].result = json["result"] as? String
                messages[replyIdx].toolCalls[i].error = json["error"] as? String
                messages[replyIdx].toolCalls[i].elapsedMs = json["elapsed_ms"] as? Double
                messages[replyIdx].toolCalls[i].done = true
            }
        case "error":
            let msg = json["message"] as? String ?? "agent error"
            if messages[replyIdx].content.isEmpty {
                messages[replyIdx].content = "⚠️ \(msg)"
            }
        default:
            break  // step_complete / done: no visible state change
        }
    }

    // ── Server management ─────────────────────────────────────────────

    /// Searches well-known install locations for the `squish` binary.
    /// `.app` bundles launched via `open` inherit a sparse PATH from launchd
    /// (`/usr/bin:/bin:/usr/sbin:/sbin`) — Homebrew's `/opt/homebrew/bin` is
    /// NOT on it, so we can't rely on PATH lookup alone.
    private func locateSquishBinary() -> String? {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let candidates = [
            "/opt/homebrew/bin/squish",   // Homebrew on Apple Silicon
            "/usr/local/bin/squish",      // Homebrew on Intel / pip --user
            "\(home)/.local/bin/squish",  // pipx / pip --user
            "\(home)/bin/squish",
            "/opt/local/bin/squish",      // MacPorts
        ]
        for c in candidates where FileManager.default.isExecutableFile(atPath: c) {
            return c
        }
        return which("squish")
    }

    func revealServerLog() {
        let url = URL(fileURLWithPath: serverLogPath)
        if FileManager.default.fileExists(atPath: serverLogPath) {
            NSWorkspace.shared.activateFileViewerSelecting([url])
        } else {
            NSWorkspace.shared.open(url.deletingLastPathComponent())
        }
    }

    private var serverLogPath: String {
        let logs = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/SquishBar", isDirectory: true)
        try? FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true)
        return logs.appendingPathComponent("server.log").path
    }

    func startServer() {
        guard serverProc == nil else { return }
        guard let squishBin = locateSquishBinary() else {
            lastError = "squish binary not found. Install with `brew install squish` " +
                        "or `pip install squish-ai`."
            return
        }

        // If something is already on the port (orphaned squish from a previous
        // session, a manual `squish run` the user forgot about, etc.) the
        // spawn would die with EADDRINUSE. SquishBar owns this port, so kill
        // the squatter and proceed.
        let squatters = pidsListening(on: port)
        if !squatters.isEmpty {
            Task { [port] in
                await self.killProcessesOnPort(port)
                await self.waitForPortFree(port, timeout: 5.0)
                await MainActor.run { self._spawnServerProcess(squishBin: squishBin) }
            }
            return
        }
        _spawnServerProcess(squishBin: squishBin)
    }

    private func _spawnServerProcess(squishBin: String) {

        // Truncate the log on each fresh start so the user sees only the
        // current attempt. Use a FileHandle so the child inherits writability.
        FileManager.default.createFile(atPath: serverLogPath, contents: nil)
        guard let logHandle = FileHandle(forWritingAtPath: serverLogPath) else {
            lastError = "Could not open SquishBar log file"
            return
        }

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: squishBin)
        proc.arguments = ["run", preferredModel,
                          "--host", host,
                          "--port", String(port)]
        proc.standardOutput = logHandle
        proc.standardError  = logHandle

        // Give the child a PATH that includes Homebrew so its own scripts
        // (the squish shim is a Python entry-point) can find sub-binaries.
        var env = ProcessInfo.processInfo.environment
        let extraPaths = "/opt/homebrew/bin:/usr/local/bin:/opt/local/bin"
        if let existing = env["PATH"], !existing.contains("/opt/homebrew/bin") {
            env["PATH"] = "\(extraPaths):\(existing)"
        } else if env["PATH"] == nil {
            env["PATH"] = "\(extraPaths):/usr/bin:/bin:/usr/sbin:/sbin"
        }
        // Force unbuffered Python output so the log streams immediately.
        env["PYTHONUNBUFFERED"] = "1"
        env["SQUISH_API_KEY"]   = apiKey
        proc.environment = env

        proc.terminationHandler = { [weak self] p in
            Task { @MainActor [weak self] in
                self?.serverProc = nil
                if p.terminationStatus != 0 {
                    self?.lastError = "Server exited (status \(p.terminationStatus)). " +
                                      "See ~/Library/Logs/SquishBar/server.log"
                }
            }
        }

        do {
            try proc.run()
            serverProc = proc
            lastError  = nil
        } catch {
            lastError = "Failed to start squish: \(error.localizedDescription)"
        }
    }

    func stopServer() {
        serverProc?.terminate()
        serverProc = nil
        health = nil; serverRunning = false
    }

    func switchModel(_ modelId: String) {
        guard modelId != (health?.model ?? preferredModel) else { return }
        preferredModel = modelId
        compressionStatus = "Switching to \(modelId)…"
        isSwitching = true
        lastError = nil

        Task {
            defer { Task { @MainActor in self.isSwitching = false } }
            stopServer()
            await killProcessesOnPort(port)
            await waitForPortFree(port, timeout: 8.0)
            startServer()
            await waitForServerUp(timeout: 120.0)
            compressionStatus = ""
        }
    }

    /// Send SIGTERM to every process listening on `port`. Falls back to SIGKILL
    /// after a short grace period for processes that ignore SIGTERM (e.g.
    /// daemonized uvicorn workers).
    private func killProcessesOnPort(_ port: Int) async {
        let pids = pidsListening(on: port)
        guard !pids.isEmpty else { return }
        for pid in pids { kill(pid, SIGTERM) }
        // Grace period for clean shutdown.
        for _ in 0..<20 {
            if pidsListening(on: port).isEmpty { return }
            try? await Task.sleep(for: .milliseconds(100))
        }
        // Hard-kill any survivors.
        for pid in pidsListening(on: port) { kill(pid, SIGKILL) }
    }

    private func pidsListening(on port: Int) -> [pid_t] {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/sbin/lsof")
        proc.arguments = ["-ti", "tcp:\(port)", "-sTCP:LISTEN"]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = Pipe()
        do {
            try proc.run()
            proc.waitUntilExit()
        } catch { return [] }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let out = String(data: data, encoding: .utf8) ?? ""
        return out.split(whereSeparator: \.isNewline).compactMap { pid_t($0) }
    }

    private func waitForPortFree(_ port: Int, timeout: Double) async {
        let deadline = Double(Int(timeout * 10))
        for _ in 0..<Int(deadline) {
            if pidsListening(on: port).isEmpty { return }
            try? await Task.sleep(for: .milliseconds(100))
        }
    }

    private func waitForServerUp(timeout: Double) async {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            await fetchHealth()
            if serverRunning, health?.loaded == true {
                await fetchModels()
                return
            }
            try? await Task.sleep(for: .milliseconds(500))
        }
    }

    func promptPullModel() {
        let alert = NSAlert()
        alert.messageText = "Pull Model"
        alert.informativeText = "Enter model ID (e.g. qwen3:8b)"
        alert.addButton(withTitle: "Pull"); alert.addButton(withTitle: "Cancel")
        let tf = NSTextField(frame: NSRect(x:0, y:0, width:260, height:24))
        tf.placeholderString = "qwen3:8b"; alert.accessoryView = tf
        if alert.runModal() == .alertFirstButtonReturn {
            let id = tf.stringValue.trimmingCharacters(in:.whitespacesAndNewlines)
            if !id.isEmpty { _startPull(modelId: id) }
        }
    }

    private func _startPull(modelId: String) {
        compressionProgress = 0
        compressionStatus = "Pulling \(modelId)…"
        guard let squishBin = locateSquishBinary() else {
            lastError = "squish binary not found"
            compressionStatus = ""
            compressionProgress = nil
            return
        }
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: squishBin)
        proc.arguments = ["pull", modelId]
        var env = ProcessInfo.processInfo.environment
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/opt/local/bin:\(env["PATH"] ?? "/usr/bin:/bin")"
        env["PYTHONUNBUFFERED"] = "1"
        proc.environment = env
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError  = pipe
        proc.terminationHandler = { [weak self] p in
            Task { @MainActor [weak self] in
                if p.terminationStatus == 0 {
                    self?.compressionStatus = "Pull complete: \(modelId)"
                } else {
                    self?.compressionStatus = "Pull failed (exit \(p.terminationStatus))"
                }
                try? await Task.sleep(for: .seconds(3))
                self?.compressionProgress = nil
                self?.compressionStatus   = ""
                await self?.fetchModels()
            }
        }
        pipe.fileHandleForReading.readabilityHandler = { [weak self] fh in
            let line = String(data: fh.availableData, encoding: .utf8) ?? ""
            let progress: Double? = {
                let pattern = #"(\d+(?:\.\d+)?)\s*(?:MB|GB)\s*/\s*(\d+(?:\.\d+)?)\s*(?:MB|GB)"#
                guard let regex = try? NSRegularExpression(pattern: pattern),
                      let match = regex.firstMatch(in: line, range: NSRange(line.startIndex..., in: line)),
                      let r1 = Range(match.range(at: 1), in: line),
                      let r2 = Range(match.range(at: 2), in: line),
                      let cur = Double(line[r1]), let total = Double(line[r2]),
                      total > 0 else { return nil }
                return min(cur / total, 1.0)
            }()
            Task { @MainActor [weak self] in
                if let p = progress { self?.compressionProgress = p }
                if !line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    self?.compressionStatus = line.trimmingCharacters(in: .whitespacesAndNewlines)
                }
            }
        }
        try? proc.run()
    }

    var chatURL: URL { URL(string: "http://\(host):\(port)/chat")! }

    var statusLabel: String {
        guard serverRunning, let h = health else {
            return lastError != nil ? "connection error" : "squish: offline"
        }
        guard h.loaded else { return "loading model…" }
        if let tps = h.avg_tps { return String(format:"%.1f tok/s", tps) }
        return "squish: ready"
    }

    var statusSymbol: String {
        serverRunning ? (health?.loaded == true ? "brain" : "hourglass") : "circle.slash"
    }

    private func which(_ cmd: String) -> String? {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/which")
        proc.arguments = [cmd]
        // Inject a rich PATH so /usr/bin/which can see Homebrew binaries
        // when the app was launched via `open` (which gives us only
        // /usr/bin:/bin:/usr/sbin:/sbin).
        var env = ProcessInfo.processInfo.environment
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/opt/local/bin:\(env["PATH"] ?? "/usr/bin:/bin")"
        proc.environment = env
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError  = Pipe()
        try? proc.run()
        proc.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let path = String(data: data, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return (path?.isEmpty == false) ? path : nil
    }

    private func _registerGlobalHotkey() {
        guard AXIsProcessTrusted() else {
            let opts: NSDictionary = [kAXTrustedCheckOptionPrompt.takeUnretainedValue(): true]
            AXIsProcessTrustedWithOptions(opts)
            return
        }
        NSEvent.addGlobalMonitorForEvents(matching: .keyDown) { event in
            if event.modifierFlags.contains([.command, .option]),
               event.charactersIgnoringModifiers == "s" {
                // Notify the App scene so it can openWindow(id:"squish-webui")
                // — opening via SwiftUI keeps a single window instance instead
                // of NSWorkspace.shared.open() which spawns browser tabs.
                NotificationCenter.default.post(name: .openSquishWebUI, object: nil)
            }
        }
    }
}

extension Notification.Name {
    /// Posted when the user wants the in-app web chat window opened.
    static let openSquishWebUI = Notification.Name("openSquishWebUI")
}
