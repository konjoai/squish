// SquishEngine.swift — polls /health every 5s, manages squish server process.

import Foundation
import SwiftUI
import AppKit

// ── Health response model ─────────────────────────────────────────────────────

struct SquishHealth: Codable {
    var status:     String
    var version:    String?
    var model:      String?
    var loaded:     Bool
    var avg_tps:    Double?
    var requests:   Int?
    var uptime_s:   Double?
}

// ── Engine ────────────────────────────────────────────────────────────────────

@MainActor
final class SquishEngine: ObservableObject {

    // Connection settings (reads from UserDefaults so the user can change them)
    @AppStorage("squish.host")   var host: String = "127.0.0.1"
    @AppStorage("squish.port")   var port: Int    = 11435
    @AppStorage("squish.apiKey") var apiKey: String = "squish"
    @AppStorage("squish.model")  var preferredModel: String = "qwen3:8b"

    // Hotkey preference
    @AppStorage("squish.hotkey") var hotkey: String = "⌘⌥S"

    // Published state
    @Published var health:              SquishHealth? = nil
    @Published var models:              [String]      = []
    @Published var isPolling:           Bool          = false
    @Published var serverRunning:       Bool          = false
    @Published var lastError:           String?       = nil
    @Published var compressionProgress: Double?       = nil
    @Published var compressionStatus:   String        = ""

    private var pollTask:   Task<Void, Never>? = nil
    private var serverProc: Process?           = nil

    init() {
        startPolling()
        _registerGlobalHotkey()
    }

    deinit { pollTask?.cancel() }

    // ── Polling ───────────────────────────────────────────────────────────────

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

    func stopPolling() {
        pollTask?.cancel()
        isPolling = false
    }

    private func fetchHealth() async {
        guard let url = URL(string: "http://\(host):\(port)/health") else { return }
        var req = URLRequest(url: url, timeoutInterval: 4)
        req.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        do {
            let (data, _) = try await URLSession.shared.data(for: req)
            let h = try JSONDecoder().decode(SquishHealth.self, from: data)
            health        = h
            serverRunning = true
            lastError     = nil
        } catch {
            health        = nil
            serverRunning = false
            lastError     = error.localizedDescription
        }
    }

    // ── Model list ────────────────────────────────────────────────────────────

    func fetchModels() async {
        guard let url = URL(string: "http://\(host):\(port)/v1/models") else { return }
        var req = URLRequest(url: url, timeoutInterval: 4)
        req.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        struct Resp: Codable { struct M: Codable { var id: String }; var data: [M] }
        do {
            let (data, _) = try await URLSession.shared.data(for: req)
            let resp = try JSONDecoder().decode(Resp.self, from: data)
            models = resp.data.map(\.id)
        } catch {}
    }

    // ── Server lifecycle ──────────────────────────────────────────────────────

    func startServer() {
        guard serverProc == nil else { return }
        // Find `squish` on $PATH; fall back to ~/.local/bin/squish
        let squishBin: String = {
            if let p = which("squish") { return p }
            let candidate = (FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".local/bin/squish")).path
            if FileManager.default.isExecutableFile(atPath: candidate) { return candidate }
            return "squish"
        }()

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        proc.arguments = [squishBin, "run", preferredModel]
        proc.terminationHandler = { [weak self] _ in
            Task { @MainActor in self?.serverProc = nil }
        }
        do {
            try proc.run()
            serverProc = proc
        } catch {
            lastError = "Failed to start squish: \(error.localizedDescription)"
        }
    }

    func stopServer() {
        serverProc?.terminate()
        serverProc = nil
    }

    // ── Convenience ───────────────────────────────────────────────────────────

    var chatURL: URL {
        URL(string: "http://\(host):\(port)/chat")!
    }

    var statusLabel: String {
        guard serverRunning, let h = health else { return "squish: offline" }
        if h.loaded, let tps = h.avg_tps {
            return String(format: "squish: %.1f tok/s", tps)
        }
        return "squish: loading…"
    }

    var statusSymbol: String {
        serverRunning ? (health?.loaded == true ? "brain" : "hourglass") : "circle.slash"
    }

    // ── Model switching ───────────────────────────────────────────────────────

    func switchModel(_ modelId: String) {
        guard modelId != preferredModel else { return }
        preferredModel = modelId
        if serverRunning {
            stopServer()
            Task {
                try? await Task.sleep(for: .seconds(1))
                startServer()
            }
        }
    }

    // ── Pull / compression progress ───────────────────────────────────────────

    func promptPullModel() {
        let panel = NSAlert()
        panel.messageText = "Pull a Model"
        panel.informativeText = "Enter a catalog model ID to download and pre-compress (e.g. qwen3:14b):"
        panel.addButton(withTitle: "Pull")
        panel.addButton(withTitle: "Cancel")
        let input = NSTextField(frame: NSRect(x: 0, y: 0, width: 260, height: 24))
        input.placeholderString = "qwen3:8b"
        panel.accessoryView = input
        guard panel.runModal() == .alertFirstButtonReturn else { return }
        let modelId = input.stringValue.trimmingCharacters(in: .whitespaces)
        guard !modelId.isEmpty else { return }
        _startPull(modelId: modelId)
    }

    private func _startPull(modelId: String) {
        compressionProgress = 0
        compressionStatus = "Pulling \(modelId)…"

        let squishBin: String = which("squish") ?? "squish"
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        proc.arguments = [squishBin, "pull", modelId]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError  = pipe
        proc.terminationHandler = { [weak self] p in
            Task { @MainActor [weak self] in
                if p.terminationStatus == 0 {
                    self?.compressionStatus   = "Pull complete: \(modelId)"
                } else {
                    self?.compressionStatus   = "Pull failed (exit \(p.terminationStatus))"
                }
                try? await Task.sleep(for: .seconds(3))
                self?.compressionProgress = nil
                self?.compressionStatus   = ""
            }
        }
        pipe.fileHandleForReading.readabilityHandler = { [weak self] fh in
            let line = String(data: fh.availableData, encoding: .utf8) ?? ""
            // Parse rough progress: "X MB / Y MB" style lines
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

    // ── Private helpers ───────────────────────────────────────────────────────

    private func which(_ cmd: String) -> String? {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/which")
        proc.arguments = [cmd]
        let pipe = Pipe()
        proc.standardOutput = pipe
        try? proc.run()
        proc.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let path = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
        return (path?.isEmpty == false) ? path : nil
    }

    private func _registerGlobalHotkey() {
        // Require Accessibility permission; prompt if not yet granted
        guard AXIsProcessTrusted() else {
            let opts: NSDictionary = [kAXTrustedCheckOptionPrompt.takeUnretainedValue(): true]
            AXIsProcessTrustedWithOptions(opts)
            return
        }
        NSEvent.addGlobalMonitorForEvents(matching: .keyDown) { [weak self] event in
            guard let self else { return }
            // ⌘⌥S (default hotkey) — open chat URL
            if event.modifierFlags.contains([.command, .option]),
               event.charactersIgnoringModifiers == "s" {
                NSWorkspace.shared.open(self.chatURL)
            }
        }
    }
}
