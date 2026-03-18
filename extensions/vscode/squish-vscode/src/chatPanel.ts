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

    // State for filtering <think>...</think> blocks out of the stream
    private _inThink = false;
    private _thinkBuf = '';

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

        // Reset think-block filter state for this turn
        this._inThink = false;
        this._thinkBuf = '';

        // Signal start of assistant turn
        this._view.webview.postMessage({ type: 'streamStart' });

        client.streamChat(
            messages,
            maxTokens,
            temperature,
            model,
            (chunk) => {
                if (chunk.delta) {
                    const visible = this._filterThink(chunk.delta);
                    assistantReply += visible;
                    if (visible) {
                        this._view?.webview.postMessage({
                            type: 'streamChunk',
                            delta: visible,
                        });
                    }
                }
                if (chunk.done) {
                    // Flush remaining buffered content (no trimming — whitespace is intentional)
                    const flushed = this._thinkBuf;
                    if (!this._inThink && flushed) {
                        assistantReply += flushed;
                        this._view?.webview.postMessage({ type: 'streamChunk', delta: flushed });
                    }
                    this._thinkBuf = '';
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

    /**
     * Strip <think>...</think> blocks from streaming delta text.
     * Buffers up to the tag length so splits across chunks are handled correctly.
     */
    private _filterThink(delta: string): string {
        this._thinkBuf += delta;
        let out = '';
        const OPEN = '<think>';
        const CLOSE = '</think>';

        while (this._thinkBuf.length > 0) {
            if (this._inThink) {
                const ci = this._thinkBuf.indexOf(CLOSE);
                if (ci >= 0) {
                    this._inThink = false;
                    // Skip whitespace immediately after closing tag
                    this._thinkBuf = this._thinkBuf.slice(ci + CLOSE.length).replace(/^\n+/, '');
                } else {
                    // Inside think block — discard but keep tail in case CLOSE is split
                    if (this._thinkBuf.length > CLOSE.length) {
                        this._thinkBuf = this._thinkBuf.slice(-CLOSE.length);
                    }
                    break;
                }
            } else {
                const oi = this._thinkBuf.indexOf(OPEN);
                if (oi >= 0) {
                    out += this._thinkBuf.slice(0, oi);
                    this._inThink = true;
                    this._thinkBuf = this._thinkBuf.slice(oi + OPEN.length);
                } else {
                    // No open tag — emit safely, hold tail in case OPEN is split
                    const safe = Math.max(0, this._thinkBuf.length - OPEN.length);
                    out += this._thinkBuf.slice(0, safe);
                    this._thinkBuf = this._thinkBuf.slice(safe);
                    break;
                }
            }
        }
        return out;
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
  <div id="header">
    <div class="header-brand">
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="28" height="28" aria-hidden="true">
        <defs>
          <linearGradient id="sq-hd" x1="20%" y1="0%" x2="80%" y2="100%">
            <stop offset="0%" stop-color="#C4B5FD"/>
            <stop offset="100%" stop-color="#7C3AED"/>
          </linearGradient>
        </defs>
        <ellipse cx="3.8" cy="12.5" rx="2.5" ry="3.5" fill="url(#sq-hd)" stroke="#3730A3" stroke-width="0.8" transform="rotate(-12,3.8,12.5)"/>
        <ellipse cx="20.2" cy="12.5" rx="2.5" ry="3.5" fill="url(#sq-hd)" stroke="#3730A3" stroke-width="0.8" transform="rotate(12,20.2,12.5)"/>
        <rect x="5.5" y="1.5" width="13" height="17" rx="4.5" fill="url(#sq-hd)" stroke="#3730A3" stroke-width="1.2"/>
        <ellipse cx="9" cy="21.2" rx="3" ry="2" fill="url(#sq-hd)" stroke="#3730A3" stroke-width="0.8"/>
        <ellipse cx="15" cy="21.2" rx="3" ry="2" fill="url(#sq-hd)" stroke="#3730A3" stroke-width="0.8"/>
        <circle cx="9.5" cy="9" r="2.1" fill="white"/>
        <circle cx="14.5" cy="9" r="2.1" fill="white"/>
        <circle cx="10" cy="9.6" r="1.2" fill="#1E1B4B"/>
        <circle cx="15" cy="9.6" r="1.2" fill="#1E1B4B"/>
        <circle cx="10.6" cy="8.8" r="0.45" fill="white"/>
        <circle cx="15.6" cy="8.8" r="0.45" fill="white"/>
        <circle cx="7.8" cy="12.5" r="1.4" fill="#FB7185" fill-opacity="0.55"/>
        <circle cx="16.2" cy="12.5" r="1.4" fill="#FB7185" fill-opacity="0.55"/>
        <path d="M8.5 13.5 Q12 16.2 15.5 13.5" fill="none" stroke="#1E1B4B" stroke-width="1.1" stroke-linecap="round"/>
        <path d="M8 4 Q12 2.5 16 5" fill="none" stroke="white" stroke-width="0.8" stroke-linecap="round" stroke-opacity="0.45"/>
      </svg>
      <span class="header-name">Squish AI</span>
    </div>
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
