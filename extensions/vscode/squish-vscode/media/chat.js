/**
 * squish-vscode/media/chat.js
 *
 * Squish Agent v0.2 — webview script.
 *
 * Extension → webview protocol:
 *   streamStart                          — assistant turn begins
 *   streamChunk  { delta }               — partial token
 *   streamEnd                            — assistant turn complete
 *   streamError  { message }             — error during generation
 *   clearHistory                         — wipe UI + start fresh
 *   toolCallStart { id, name, args }     — tool invocation starting
 *   toolCallEnd   { id, result }         — tool invocation complete
 *   sessionList   { sessions, activeId } — session list update
 *   sessionLoaded { session }            — session loaded (replay messages)
 *   agentTask     { text, context }      — pre-fill input and submit
 *
 * Webview → extension protocol:
 *   userMessage     { text }
 *   stopGeneration
 *   clearHistory
 *   newSession
 *   loadSession     { id }
 *   deleteSession   { id }
 *   renameSession   { id, title }
 *   requestSessionList
 */
(function () {
    'use strict';

    const vscode = acquireVsCodeApi();

    // ── DOM refs ──────────────────────────────────────────────────────────
    const messagesEl  = /** @type {HTMLDivElement}  */ (document.getElementById('messages'));
    const inputEl     = /** @type {HTMLTextAreaElement} */ (document.getElementById('user-input'));
    const sendBtn     = /** @type {HTMLButtonElement} */ (document.getElementById('btn-send'));
    const stopBtn     = /** @type {HTMLButtonElement} */ (document.getElementById('btn-stop'));
    const regenBtn    = /** @type {HTMLButtonElement} */ (document.getElementById('btn-regen'));
    const histPanel   = /** @type {HTMLDivElement}  */ (document.getElementById('hist-panel'));
    const histOverlay = /** @type {HTMLDivElement}  */ (document.getElementById('hist-overlay'));
    const histList    = /** @type {HTMLDivElement}  */ (document.getElementById('hist-list'));
    const histToggle  = /** @type {HTMLButtonElement} */ (document.getElementById('hist-toggle'));
    const histClose   = /** @type {HTMLButtonElement} */ (document.getElementById('hist-close'));
    const histNewBtn  = /** @type {HTMLButtonElement} */ (document.getElementById('hist-new-btn'));
    const newChatBtn  = /** @type {HTMLButtonElement} */ (document.getElementById('btn-new-chat'));

    // ── State ─────────────────────────────────────────────────────────────
    let _generating         = false;
    let _lastUserText       = '';
    let _activeSessionId    = '';

    // Current assistant turn elements
    let _currentBubble      = /** @type {HTMLElement|null} */ (null);
    let _currentContentEl   = /** @type {HTMLElement|null} */ (null);
    let _currentIndicatorEl = /** @type {HTMLElement|null} */ (null);
    let _currentAckEl       = /** @type {HTMLElement|null} */ (null);

    let _ackTimerId         = /** @type {number|null} */ (null);
    let _ackWords           = /** @type {string[]} */ ([]);
    let _ackWordIdx         = 0;
    let _realStarted        = false;

    // Smooth character render queue (~14 chars/frame ≈ 840 chars/sec @ 60fps)
    const _queue            = /** @type {string[]} */ ([]);
    let   _rafId            = /** @type {number|null} */ (null);
    const CHARS_PER_FRAME   = 14;

    const _toolCards        = new Map();

    // ── History sidebar ───────────────────────────────────────────────────

    function _openHist() {
        histPanel?.classList.add('open');
        histOverlay?.classList.add('open');
        vscode.postMessage({ type: 'requestSessionList' });
    }

    function _closeHist() {
        histPanel?.classList.remove('open');
        histOverlay?.classList.remove('open');
    }

    histToggle?.addEventListener('click', () => {
        histPanel?.classList.contains('open') ? _closeHist() : _openHist();
    });
    histClose?.addEventListener('click', _closeHist);
    histOverlay?.addEventListener('click', _closeHist);

    histNewBtn?.addEventListener('click', () => {
        _closeHist();
        vscode.postMessage({ type: 'newSession' });
    });

    newChatBtn?.addEventListener('click', () => {
        vscode.postMessage({ type: 'newSession' });
    });

    function _renderSessionList(sessions, activeId) {
        _activeSessionId = activeId ?? '';
        if (!histList) { return; }
        histList.innerHTML = '';
        if (!sessions || sessions.length === 0) {
            const empty = document.createElement('div');
            empty.style.cssText = 'padding:12px 8px;color:var(--text-dim);font-size:12px;';
            empty.textContent = 'No saved conversations yet.';
            histList.appendChild(empty);
            return;
        }
        for (const s of sessions) {
            const item = document.createElement('div');
            item.className = 'hist-item' + (s.id === activeId ? ' active' : '');
            item.setAttribute('role', 'listitem');
            item.dataset.id = s.id;

            const title = document.createElement('span');
            title.className = 'hist-item-title';
            title.textContent = s.title || 'Untitled';

            const del = document.createElement('button');
            del.className = 'hist-item-del';
            del.title = 'Delete conversation';
            del.textContent = '\u2715';
            del.setAttribute('aria-label', 'Delete');

            del.addEventListener('click', (e) => {
                e.stopPropagation();
                vscode.postMessage({ type: 'deleteSession', id: s.id });
            });

            item.addEventListener('click', () => {
                _closeHist();
                vscode.postMessage({ type: 'loadSession', id: s.id });
            });

            item.appendChild(title);
            item.appendChild(del);
            histList.appendChild(item);
        }
    }

    // ── Send ──────────────────────────────────────────────────────────────

    function send() {
        const text = inputEl.value.trim();
        if (!text || _generating) { return; }
        _lastUserText = text;
        inputEl.value = '';
        _setGenerating(true);

        _appendUserMessage(text);
        vscode.postMessage({ type: 'userMessage', text });
    }

    function _setGenerating(state) {
        _generating = state;
        sendBtn.disabled = state;
        if (state) {
            stopBtn?.removeAttribute('hidden');
            regenBtn?.setAttribute('hidden', '');
        } else {
            stopBtn?.setAttribute('hidden', '');
            if (_lastUserText) { regenBtn?.removeAttribute('hidden'); }
        }
    }

    // ── Event listeners ───────────────────────────────────────────────────

    sendBtn.addEventListener('click', send);

    inputEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
    });

    stopBtn?.addEventListener('click', () => {
        vscode.postMessage({ type: 'stopGeneration' });
    });

    regenBtn?.addEventListener('click', () => {
        if (_lastUserText && !_generating) {
            const text = _lastUserText;
            // Remove the last assistant message from the UI before regenerating
            const messages = messagesEl.querySelectorAll('.message-wrap');
            const last = messages[messages.length - 1];
            if (last?.classList.contains('role-assistant')) { last.remove(); }
            _setGenerating(true);
            vscode.postMessage({ type: 'userMessage', text });
        }
    });

    // Delegated copy-code handler
    messagesEl.addEventListener('click', (e) => {
        const btn = e.target?.closest?.('.copy-code-btn');
        if (!btn) { return; }
        const id = btn.dataset.id;
        const codeEl = id ? document.getElementById(id) : null;
        if (!codeEl) { return; }
        navigator.clipboard?.writeText(codeEl.textContent ?? '').then(() => {
            btn.textContent = 'Copied!';
            setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
        });
    });

    // ── Extension → webview messages ──────────────────────────────────────

    window.addEventListener('message', (event) => {
        const msg = event.data;
        switch (msg.type) {

            case 'streamStart':
                _startTurn();
                break;

            case 'streamChunk':
                if (msg.delta) { _onRealChunk(msg.delta); }
                break;

            case 'streamEnd':
                _endTurn();
                break;

            case 'streamError':
                _cancelAck();
                _flushQueue();
                _appendErrorMessage('\u26a0 ' + (msg.message || 'Unknown error'));
                _finalizeTurn();
                break;

            case 'clearHistory':
                messagesEl.innerHTML = '';
                _resetTurnState();
                regenBtn?.setAttribute('hidden', '');
                _lastUserText = '';
                break;

            case 'toolCallStart':
                _onToolCallStart(msg.id, msg.name, msg.args);
                break;

            case 'toolCallEnd':
                _onToolCallEnd(msg.id, msg.result);
                break;

            case 'sessionList':
                _renderSessionList(msg.sessions, msg.activeId);
                break;

            case 'sessionLoaded': {
                const session = msg.session;
                if (!session) { break; }
                _activeSessionId = session.id;
                messagesEl.innerHTML = '';
                regenBtn?.setAttribute('hidden', '');
                _lastUserText = '';
                for (const m of (session.messages ?? [])) {
                    if (m.role === 'user') {
                        _appendUserMessage(m.content ?? '');
                        _lastUserText = m.content ?? '';
                    } else if (m.role === 'assistant') {
                        _appendAssistantMessage(m.content ?? '');
                    }
                }
                if (_lastUserText) { regenBtn?.removeAttribute('hidden'); }
                messagesEl.scrollTop = messagesEl.scrollHeight;
                break;
            }

            case 'agentTask':
                // Pre-fill the input box with a task and submit immediately
                inputEl.value = (msg.context ? `[Context: ${msg.context}]\n\n` : '') + (msg.text || '');
                send();
                break;
        }
    });

    // ── Turn lifecycle ────────────────────────────────────────────────────

    function _startTurn() {
        _realStarted = false;
        _queue.length = 0;
        if (_rafId) { cancelAnimationFrame(_rafId); _rafId = null; }
        _toolCards.clear();

        const wrap = document.createElement('div');
        wrap.className = 'message-wrap role-assistant';

        const label = document.createElement('div');
        label.className = 'message-role-label';
        label.textContent = 'Squish Agent';

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';

        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            dot.style.cssText = 'display:inline-block;width:5px;height:5px;border-radius:50%;background:var(--accent);margin:0 2px;animation:blink 1.2s ease-in-out infinite;';
            dot.style.animationDelay = `${i * 0.2}s`;
            indicator.appendChild(dot);
        }

        const ack = document.createElement('span');
        ack.className = 'ack-text';
        ack.style.color = 'var(--text-dim)';

        const content = document.createElement('span');
        content.className = 'content';

        bubble.appendChild(indicator);
        bubble.appendChild(ack);
        bubble.appendChild(content);
        wrap.appendChild(label);
        wrap.appendChild(bubble);
        messagesEl.appendChild(wrap);
        messagesEl.scrollTop = messagesEl.scrollHeight;

        _currentBubble      = bubble;
        _currentContentEl   = content;
        _currentIndicatorEl = indicator;
        _currentAckEl       = ack;

        _ackWords   = _buildAckWords(_lastUserText);
        _ackWordIdx = 0;
        _ackTimerId = setTimeout(_typeNextAckWord, 320);
    }

    function _typeNextAckWord() {
        if (_realStarted || !_currentAckEl || _ackWordIdx >= _ackWords.length) {
            _ackTimerId = null;
            return;
        }
        const sep  = _ackWordIdx === 0 ? '' : ' ';
        _currentAckEl.appendChild(document.createTextNode(sep + _ackWords[_ackWordIdx++]));
        messagesEl.scrollTop = messagesEl.scrollHeight;
        _ackTimerId = setTimeout(_typeNextAckWord, 115);
    }

    function _onRealChunk(delta) {
        if (!_realStarted) {
            _realStarted = true;
            _cancelAck();
            if (_currentIndicatorEl) { _currentIndicatorEl.remove(); _currentIndicatorEl = null; }
            if (_currentAckEl)       { _currentAckEl.remove(); _currentAckEl = null; }
        }
        for (const ch of delta) { _queue.push(ch); }
        if (!_rafId) { _rafId = requestAnimationFrame(_drainQueue); }
    }

    function _drainQueue() {
        if (!_currentContentEl || _queue.length === 0) { _rafId = null; return; }
        const text = _queue.splice(0, CHARS_PER_FRAME).join('');
        _currentContentEl.appendChild(document.createTextNode(text));
        messagesEl.scrollTop = messagesEl.scrollHeight;
        _rafId = requestAnimationFrame(_drainQueue);
    }

    function _endTurn() {
        _cancelAck();
        _flushQueue();
        if (_currentIndicatorEl) { _currentIndicatorEl.remove(); _currentIndicatorEl = null; }
        if (_currentAckEl)       { _currentAckEl.remove(); _currentAckEl = null; }
        if (_currentContentEl) {
            _currentContentEl.classList.remove('streaming');
            const plain = _currentContentEl.textContent || '';
            _currentContentEl.innerHTML = _renderMarkdown(plain);
        }
        _finalizeTurn();
    }

    function _flushQueue() {
        if (_rafId) { cancelAnimationFrame(_rafId); _rafId = null; }
        if (_currentContentEl && _queue.length > 0) {
            _currentContentEl.appendChild(document.createTextNode(_queue.splice(0).join('')));
            messagesEl.scrollTop = messagesEl.scrollHeight;
        }
    }

    function _cancelAck() {
        if (_ackTimerId !== null) { clearTimeout(_ackTimerId); _ackTimerId = null; }
    }

    function _finalizeTurn() {
        _setGenerating(false);
        _resetTurnState();
        inputEl.focus();
    }

    function _resetTurnState() {
        _cancelAck();
        if (_rafId) { cancelAnimationFrame(_rafId); _rafId = null; }
        _queue.length = 0;
        _toolCards.clear();
        _currentBubble      = null;
        _currentContentEl   = null;
        _currentIndicatorEl = null;
        _currentAckEl       = null;
        _realStarted        = false;
    }

    // ── Tool call UI ──────────────────────────────────────────────────────

    function _onToolCallStart(id, name, argsJson) {
        if (!_currentBubble) { return; }
        _cancelAck();
        if (_currentIndicatorEl) { _currentIndicatorEl.remove(); _currentIndicatorEl = null; }
        if (_currentAckEl)       { _currentAckEl.remove(); _currentAckEl = null; }
        _flushQueue();

        const card = document.createElement('div');
        card.className = 'tool-card';

        const header = document.createElement('div');
        header.className = 'tool-header';

        const spinner = document.createElement('div');
        spinner.className = 'tool-spinner';

        const nameEl = document.createElement('span');
        nameEl.textContent = name;

        const statusEl = document.createElement('span');
        statusEl.style.cssText = 'margin-left:auto;font-size:10px;color:var(--text-dim)';
        statusEl.textContent = 'running\u2026';

        header.appendChild(spinner);
        header.appendChild(nameEl);
        header.appendChild(statusEl);
        card.appendChild(header);

        let args;
        try { args = JSON.parse(argsJson || '{}'); } catch { args = {}; }
        const argsText = Object.entries(args)
            .filter(([, v]) => v !== undefined && v !== null)
            .map(([k, v]) => {
                const s = typeof v === 'string' ? v : JSON.stringify(v);
                return `${k}: ${s.length > 80 ? s.slice(0, 80) + '\u2026' : s}`;
            })
            .join('\n');

        if (argsText) {
            const pre = document.createElement('pre');
            pre.className = 'tool-result';
            pre.textContent = argsText;
            card.appendChild(pre);
        }

        const resultEl = document.createElement('pre');
        resultEl.className = 'tool-result';
        resultEl.style.display = 'none';
        card.appendChild(resultEl);

        _currentBubble.appendChild(card);
        messagesEl.scrollTop = messagesEl.scrollHeight;
        _toolCards.set(id, { card, spinner, statusEl, resultEl });
    }

    function _onToolCallEnd(id, result) {
        const entry = _toolCards.get(id);
        if (!entry) { return; }
        const { spinner, statusEl, resultEl } = entry;
        spinner.style.display = 'none';
        statusEl.textContent = '\u2714 done';
        statusEl.style.color = 'var(--success)';
        if (result != null && String(result).length > 0) {
            const s = String(result);
            resultEl.textContent = s.length > 600 ? s.slice(0, 600) + '\n\u2026 (truncated)' : s;
            resultEl.style.display = '';
        }
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    // ── Markdown renderer ─────────────────────────────────────────────────

    function _esc(s) {
        return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    function _renderMarkdown(text) {
        const out = [];
        const parts = text.split(/(```[\s\S]*?```)/g);
        for (const part of parts) {
            const fenceMatch = part.match(/^```(\w*)\n?([\s\S]*?)```$/);
            if (fenceMatch) {
                const lang = fenceMatch[1] || '';
                const code = fenceMatch[2] || '';
                const id = 'cb_' + Math.random().toString(36).slice(2, 9);
                out.push(
                    '<div class="code-block">'
                    + '<div class="code-header">'
                    + '<span class="code-lang">' + _esc(lang) + '</span>'
                    + '<button class="copy-code-btn" data-id="' + id + '">Copy</button>'
                    + '</div>'
                    + '<pre><code id="' + id + '">' + _esc(code) + '</code></pre>'
                    + '</div>',
                );
            } else {
                const paragraphs = part.split(/\n{2,}/);
                for (const para of paragraphs) {
                    let s = _esc(para.trim());
                    if (!s) { continue; }
                    s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
                    s = s.replace(/\*([^*]+?)\*/g, '<em>$1</em>');
                    s = s.replace(/`([^`]+?)`/g, '<code>$1</code>');
                    s = s.replace(/\n/g, '<br>');
                    out.push('<p>' + s + '</p>');
                }
            }
        }
        return out.join('');
    }

    // ── Acknowledgement phrase generator ──────────────────────────────────

    function _buildAckWords(text) {
        const t = (text || '').toLowerCase();
        let phrase;
        if      (/\b(fix|debug|bug|error|broken|crash|issue)\b/.test(t)) { phrase = 'Let me debug that\u2026'; }
        else if (/\b(write|create|generate|make|build|scaffold)\b/.test(t)) { phrase = "I'll create that for you\u2026"; }
        else if (/\b(test|spec|coverage|unittest)\b/.test(t))              { phrase = 'Writing those tests\u2026'; }
        else if (/\b(refactor|improve|optimize|clean|rewrite)\b/.test(t))  { phrase = 'Working on that refactor\u2026'; }
        else if (/\b(explain|what\s+is|how\s+does|why|when|where)\b/.test(t)) { phrase = 'Let me explain\u2026'; }
        else if (/\b(review|check|look\s+at|analyze|audit)\b/.test(t))     { phrase = 'Reviewing that now\u2026'; }
        else if (/\?/.test(text))                                           { phrase = 'Let me answer that\u2026'; }
        else                                                                { phrase = 'On it\u2026'; }
        return phrase.split(' ');
    }

    // ── DOM helpers ───────────────────────────────────────────────────────

    function _appendUserMessage(text) {
        const wrap = document.createElement('div');
        wrap.className = 'message-wrap role-user';
        const label = document.createElement('div');
        label.className = 'message-role-label';
        label.textContent = 'You';
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.appendChild(document.createTextNode(text));
        wrap.appendChild(label);
        wrap.appendChild(bubble);
        messagesEl.appendChild(wrap);
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function _appendAssistantMessage(text) {
        const wrap = document.createElement('div');
        wrap.className = 'message-wrap role-assistant';
        const label = document.createElement('div');
        label.className = 'message-role-label';
        label.textContent = 'Squish Agent';
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.innerHTML = _renderMarkdown(text);
        wrap.appendChild(label);
        wrap.appendChild(bubble);
        messagesEl.appendChild(wrap);
    }

    function _appendErrorMessage(text) {
        const wrap = document.createElement('div');
        wrap.className = 'message-wrap role-assistant';
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.style.color = 'var(--danger)';
        bubble.textContent = text;
        wrap.appendChild(bubble);
        messagesEl.appendChild(wrap);
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    // ── Init ──────────────────────────────────────────────────────────────
    vscode.postMessage({ type: 'requestSessionList' });
    inputEl.focus();

}());
