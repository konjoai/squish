/**
 * squish-vscode/src/chatPanel.ts
 *
 * Implements the sidebar WebviewView.  The extension host streams SSE chunks
 * from the Squish server and relays them to the webview via postMessage —
 * keeping the webview's CSP valid (no direct localhost fetch from webview).
 */
import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { SquishClient, ChatMessage } from './squishClient';

export class ChatPanel implements vscode.WebviewViewProvider {
    public static readonly viewType = 'squish.chatView';

    private _view?: vscode.WebviewView;
    private _history: ChatMessage[] = [];

    constructor(private readonly _extensionUri: vscode.Uri) {}

    // ── WebviewViewProvider ───────────────────────────────────────────────

    resolveWebviewView(
        webviewView: vscode.WebviewView,
        _context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken,
    ): void {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [
                vscode.Uri.joinPath(this._extensionUri, 'media'),
            ],
        };

        webviewView.webview.html = this._buildHtml(webviewView.webview);

        webviewView.webview.onDidReceiveMessage(async (msg) => {
            switch (msg.type) {
                case 'userMessage':
                    await this._handleUserMessage(msg.text as string);
                    break;
                case 'clearHistory':
                    this.clearHistory();
                    break;
            }
        });
    }

    // ── Public API ────────────────────────────────────────────────────────

    clearHistory(): void {
        this._history = [];
        this._view?.webview.postMessage({ type: 'clearHistory' });
    }

    // ── Internal ──────────────────────────────────────────────────────────

    private async _handleUserMessage(text: string): Promise<void> {
        if (!this._view) {
            return;
        }

        const cfg = vscode.workspace.getConfiguration('squish');
        const host: string = cfg.get('host', '127.0.0.1');
        const port: number = cfg.get('port', 11435);
        const apiKey: string = cfg.get('apiKey', 'squish');
        const model: string = cfg.get('model', '7b');
        const maxTokens: number = cfg.get('maxTokens', 1024);
        const temperature: number = cfg.get('temperature', 0.7);
        const systemPrompt: string = cfg.get('systemPrompt', '');

        // Build message list
        const messages: ChatMessage[] = [];
        if (systemPrompt) {
            messages.push({ role: 'system', content: systemPrompt });
        }
        messages.push(...this._history);
        messages.push({ role: 'user', content: text });

        // Optimistically add user message to history
        this._history.push({ role: 'user', content: text });

        const client = new SquishClient(host, port, apiKey);
        let assistantReply = '';

        // Signal start of assistant turn
        this._view.webview.postMessage({ type: 'streamStart' });

        client.streamChat(
            messages,
            maxTokens,
            temperature,
            model,
            (chunk) => {
                if (chunk.delta) {
                    assistantReply += chunk.delta;
                    this._view?.webview.postMessage({
                        type: 'streamChunk',
                        delta: chunk.delta,
                    });
                }
                if (chunk.done) {
                    this._view?.webview.postMessage({ type: 'streamEnd' });
                    if (assistantReply) {
                        this._history.push({ role: 'assistant', content: assistantReply });
                    }
                }
            },
            (err) => {
                this._view?.webview.postMessage({
                    type: 'streamError',
                    message: err.message,
                });
            },
        );
    }

    private _buildHtml(webview: vscode.Webview): string {
        const chatJsUri = webview.asWebviewUri(
            vscode.Uri.joinPath(this._extensionUri, 'media', 'chat.js'),
        );
        const styleUri = webview.asWebviewUri(
            vscode.Uri.joinPath(this._extensionUri, 'media', 'style.css'),
        );

        // Content Security Policy: only allow scripts and styles from the
        // extension's own media directory, plus inline styles for VS Code theming.
        const nonce = _nonce();

        return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy"
    content="default-src 'none';
             style-src ${webview.cspSource} 'unsafe-inline';
             script-src 'nonce-${nonce}';">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="${styleUri}">
  <title>Squish Chat</title>
</head>
<body>
  <div id="toolbar">
    <button id="btn-clear" title="Clear history">Clear</button>
  </div>
  <div id="messages" aria-live="polite"></div>
  <div id="input-area">
    <textarea
      id="user-input"
      placeholder="Ask something… (Enter to send, Shift+Enter for newline)"
      rows="3"
      autofocus
    ></textarea>
    <button id="btn-send">Send</button>
  </div>
  <script nonce="${nonce}" src="${chatJsUri}"></script>
</body>
</html>`;
    }
}

function _nonce(): string {
    let text = '';
    const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    for (let i = 0; i < 32; i++) {
        text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
}
