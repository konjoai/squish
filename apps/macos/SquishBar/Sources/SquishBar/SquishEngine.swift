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

    // Published state
    @Published var health:   SquishHealth? = nil
    @Published var models:   [String]      = []
    @Published var isPolling:  Bool        = false
    @Published var serverRunning: Bool     = false
    @Published var lastError:  String?     = nil

    private var pollTask:   Task<Void, Never>? = nil
    private var serverProc: Process?           = nil

    init() { startPolling() }

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
}
