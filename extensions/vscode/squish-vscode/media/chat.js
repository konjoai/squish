/**
 * squish-vscode/media/chat.js
 *
 * Webview script.  Receives messages from the extension host and renders
 * the conversation UI.  Communicates back via acquireVsCodeApi().
 *
 * Message protocol (extension → webview):
 *   { type: 'streamStart' }                — assistant turn begins
 *   { type: 'streamChunk', delta: string } — partial token
 *   { type: 'streamEnd' }                  — assistant turn complete
 *   { type: 'streamError', message: str }  — error
 *   { type: 'clearHistory' }               — wipe the UI
 *
 * Message protocol (webview → extension):
 *   { type: 'userMessage', text: string }
 *   { type: 'clearHistory' }
 */
(function () {
    'use strict';

    const vscode = acquireVsCodeApi();

    const messagesEl = document.getElementById('messages');
    const inputEl    = document.getElementById('user-input');
    const sendBtn    = document.getElementById('btn-send');
    const clearBtn   = document.getElementById('btn-clear');

    let _currentAssistantEl = null;
    let _generating = false;

    // ── Send ──────────────────────────────────────────────────────────────

    function send() {
        const text = inputEl.value.trim();
        if (!text || _generating) { return; }
        inputEl.value = '';
        _generating = true;
        sendBtn.disabled = true;

        _appendMessage('user', text);

        vscode.postMessage({ type: 'userMessage', text });
    }

    // ── Event listeners ───────────────────────────────────────────────────

    sendBtn.addEventListener('click', send);

    inputEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            send();
        }
    });

    clearBtn.addEventListener('click', () => {
        messagesEl.innerHTML = '';
        vscode.postMessage({ type: 'clearHistory' });
    });

    // ── Extension → webview messages ──────────────────────────────────────

    window.addEventListener('message', (event) => {
        const msg = event.data;
        switch (msg.type) {
            case 'streamStart':
                _currentAssistantEl = _appendMessage('assistant', '');
                break;

            case 'streamChunk':
                if (_currentAssistantEl && msg.delta) {
                    // Append raw text as a text node to avoid XSS
                    _currentAssistantEl
                        .querySelector('.content')
                        .appendChild(document.createTextNode(msg.delta));
                    // Auto-scroll
                    messagesEl.scrollTop = messagesEl.scrollHeight;
                }
                break;

            case 'streamEnd':
                _generating = false;
                sendBtn.disabled = false;
                _currentAssistantEl = null;
                inputEl.focus();
                break;

            case 'streamError':
                _appendMessage('error', '⚠ ' + (msg.message || 'Unknown error'));
                _generating = false;
                sendBtn.disabled = false;
                _currentAssistantEl = null;
                inputEl.focus();
                break;

            case 'clearHistory':
                messagesEl.innerHTML = '';
                break;
        }
    });

    // ── Helpers ───────────────────────────────────────────────────────────

    function _appendMessage(role, text) {
        const row = document.createElement('div');
        row.className = 'message ' + role;

        const label = document.createElement('span');
        label.className = 'label';
        label.textContent = role === 'user' ? 'You' : role === 'assistant' ? 'Squish' : '';

        const content = document.createElement('span');
        content.className = 'content';
        if (text) {
            content.appendChild(document.createTextNode(text));
        }

        row.appendChild(label);
        row.appendChild(content);
        messagesEl.appendChild(row);
        messagesEl.scrollTop = messagesEl.scrollHeight;

        return row;
    }
}());
