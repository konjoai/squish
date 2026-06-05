import SwiftUI
import WebKit

// ── Web-UI window backed by WKWebView ─────────────────────────────────────
//
// Hosting the squish chat page inside our own NSWindow has two payoffs:
//   1. openWindow(id:"squish-webui") brings the existing window to front
//      and reloads its contents — opening the page from the menubar twice
//      no longer spawns a duplicate browser tab.
//   2. We don't depend on the user's default browser being open, configured,
//      or trustworthy with the injected API key (the key is only injected
//      for loopback requests anyway).

struct WebUIWindow: View {
    @EnvironmentObject var engine: SquishEngine
    @State private var canGoBack = false
    @State private var canGoForward = false
    @State private var isLoading = false
    @State private var coordinator = WebViewCoordinator()

    var body: some View {
        VStack(spacing: 0) {
            // Compact navigation bar — back/forward/reload + URL display.
            HStack(spacing: 8) {
                Button(action: { coordinator.webView?.goBack() }) {
                    Image(systemName: "chevron.left")
                }
                .buttonStyle(.borderless)
                .disabled(!canGoBack)

                Button(action: { coordinator.webView?.goForward() }) {
                    Image(systemName: "chevron.right")
                }
                .buttonStyle(.borderless)
                .disabled(!canGoForward)

                Button(action: { coordinator.webView?.reload() }) {
                    Image(systemName: isLoading ? "xmark" : "arrow.clockwise")
                }
                .buttonStyle(.borderless)

                Text(engine.chatURL.absoluteString)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(SQ.textSecond)
                    .lineLimit(1)
                    .truncationMode(.middle)

                Spacer()

                Circle()
                    .fill(engine.serverRunning ? SQ.green : Color.gray.opacity(0.6))
                    .frame(width: 7, height: 7)
                Text(engine.serverRunning ? "Connected" : "Offline")
                    .font(.system(size: 10))
                    .foregroundStyle(SQ.textSecond)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(SQ.surface)

            Divider().background(SQ.border)

            WebViewContainer(
                url: engine.chatURL,
                coordinator: coordinator,
                canGoBack: $canGoBack,
                canGoForward: $canGoForward,
                isLoading: $isLoading
            )
        }
        .background(SQ.bg)
    }
}

// ── WKWebView wrapper ─────────────────────────────────────────────────────

final class WebViewCoordinator: NSObject, WKNavigationDelegate, ObservableObject {
    weak var webView: WKWebView?

    var onCanGoBack: ((Bool) -> Void)?
    var onCanGoForward: ((Bool) -> Void)?
    var onIsLoading: ((Bool) -> Void)?

    func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!) {
        onIsLoading?(true)
    }
    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        onIsLoading?(false)
        onCanGoBack?(webView.canGoBack)
        onCanGoForward?(webView.canGoForward)
    }
    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        onIsLoading?(false)
    }
    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        onIsLoading?(false)
    }
}

struct WebViewContainer: NSViewRepresentable {
    let url: URL
    let coordinator: WebViewCoordinator
    @Binding var canGoBack: Bool
    @Binding var canGoForward: Bool
    @Binding var isLoading: Bool

    func makeNSView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.websiteDataStore = .default()  // share cookies/localStorage with itself across launches
        let wv = WKWebView(frame: .zero, configuration: config)
        wv.navigationDelegate = coordinator
        coordinator.webView = wv
        coordinator.onCanGoBack    = { canGoBack    = $0 }
        coordinator.onCanGoForward = { canGoForward = $0 }
        coordinator.onIsLoading    = { isLoading    = $0 }
        wv.load(URLRequest(url: url))
        return wv
    }

    func updateNSView(_ nsView: WKWebView, context: Context) {
        // If the engine's chatURL has changed (e.g. user changed port), navigate
        // there. Otherwise leave the current page alone so the chat history
        // survives focus changes.
        if nsView.url?.host != url.host || nsView.url?.port != url.port {
            nsView.load(URLRequest(url: url))
        }
    }
}
