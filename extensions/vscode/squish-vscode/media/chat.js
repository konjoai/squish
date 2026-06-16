/**
 * squish-vscode/media/chat.js
 *
 * Squish Agent v0.2 — webview script.
 *
 * Extension → webview protocol:
 *   streamStart                          — assistant turn begins
 *   streamChunk  { delta }               — partial token
 *   streamEnd                            — assistant turn complete
 *   streamClear                          — erase buffered content (text-mode tool call detected)
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

            case 'streamClear':
                // The extension detected a text-mode tool call after the model
                // had already streamed JSON args as plain text.  Erase them.
                _cancelAck();
                if (_rafId) { cancelAnimationFrame(_rafId); _rafId = null; }
                _queue.length = 0;
                if (_currentContentEl) { _currentContentEl.innerHTML = ''; }
                _realStarted = false;
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

    // Friendly label + icon per VS Code agent tool (mirrors the web UI cards).
    const TOOL_META = {
        read_file:           ['\ud83d\udcc4', 'Read file'],
        get_selection:       ['\u2702\ufe0f', 'Get selection'],
        get_open_files:      ['\ud83d\udcd1', 'Open files'],
        run_terminal:        ['\u26a1', 'Run terminal'],
        insert_at_cursor:    ['\u2328\ufe0f', 'Insert at cursor'],
        write_file:          ['\u270f\ufe0f', 'Write file'],
        list_directory:      ['\ud83d\udcc2', 'List directory'],
        get_diagnostics:     ['\ud83d\udd0d', 'Diagnostics'],
        apply_edit:          ['\u270f\ufe0f', 'Edit file'],
        search_workspace:    ['\ud83d\udd0e', 'Search workspace'],
        create_file:         ['\u2728', 'Create file'],
        delete_file:         ['\ud83d\uddd1\ufe0f', 'Delete file'],
        get_git_status:      ['\ud83c\udf3f', 'Git status'],
        get_symbol_at_cursor:['\ud83d\udd23', 'Symbol at cursor'],
        web_search:          ['\ud83c\udf10', 'Web search'],
    };
    function _toolMeta(name) {
        const m = TOOL_META[name];
        if (m) { return { icon: m[0], label: m[1] }; }
        const clean = String(name || 'tool').replace(/_/g, ' ');
        return { icon: '\ud83d\udd27', label: clean.charAt(0).toUpperCase() + clean.slice(1) };
    }
    // The most telling argument to show inline (path, command, query, url\u2026).
    function _primaryArgHint(args) {
        if (!args || typeof args !== 'object') { return ''; }
        const v = args.path ?? args.relativePath ?? args.command ?? args.url
            ?? args.query ?? args.oldText ?? args.symbol ?? Object.values(args)[0];
        if (v == null) { return ''; }
        let s = typeof v === 'string' ? v : JSON.stringify(v);
        s = s.replace(/\s+/g, ' ').trim();
        return s.length > 60 ? s.slice(0, 58) + '\u2026' : s;
    }

    function _onToolCallStart(id, name, argsJson) {
        if (!_currentBubble) { return; }
        _cancelAck();
        if (_currentIndicatorEl) { _currentIndicatorEl.remove(); _currentIndicatorEl = null; }
        if (_currentAckEl)       { _currentAckEl.remove(); _currentAckEl = null; }
        _flushQueue();

        let args;
        try { args = JSON.parse(argsJson || '{}'); } catch { args = {}; }
        const meta = _toolMeta(name);
        const argsStr = Object.keys(args).length ? JSON.stringify(args, null, 2) : '';

        const card = document.createElement('div');
        card.className = 'tool-card pending';

        const header = document.createElement('div');
        header.className = 'tool-header';
        header.addEventListener('click', () => card.classList.toggle('open'));
        header.innerHTML =
            '<span class="tool-icon">' + meta.icon + '</span>'
            + '<span class="tool-name">' + _esc(meta.label) + '</span>'
            + '<span class="tool-arg-hint">' + _esc(_primaryArgHint(args)) + '</span>'
            + '<span class="tool-chevron">\u25b6</span>'
            + '<span class="tool-status"><span class="tool-spinner"></span></span>';
        card.appendChild(header);

        const preview = document.createElement('div');
        preview.className = 'tool-preview';
        card.appendChild(preview);

        const body = document.createElement('div');
        body.className = 'tool-body';
        body.innerHTML =
            '<div class="tool-section-label">Arguments</div>'
            + '<pre class="tool-args">' + _esc(argsStr || '\u2014') + '</pre>'
            + '<div class="tool-section-label tool-result-label" style="display:none">Result</div>'
            + '<pre class="tool-result-pre" style="display:none"></pre>';
        card.appendChild(body);

        _currentBubble.appendChild(card);
        messagesEl.scrollTop = messagesEl.scrollHeight;
        _toolCards.set(id, { card, preview });
    }

    function _onToolCallEnd(id, result) {
        const entry = _toolCards.get(id);
        if (!entry) { return; }
        const { card, preview } = entry;
        const resultStr = result == null ? '' : String(result);
        const errored = /^\s*(\[ERROR\]|Error:)/.test(resultStr);

        card.classList.remove('pending');
        card.classList.add(errored ? 'error' : 'done');
        const statusEl = card.querySelector('.tool-status');
        if (statusEl) {
            statusEl.innerHTML = errored
                ? '<span class="tool-x">\u2717</span>'
                : '<span class="tool-check">\u2714</span>';
        }

        if (resultStr.length > 0) {
            const label = card.querySelector('.tool-result-label');
            const pre = card.querySelector('.tool-result-pre');
            if (label) { label.style.display = ''; }
            if (pre) {
                pre.textContent = resultStr.length > 8192
                    ? resultStr.slice(0, 8192) + '\n\u2026 (truncated)'
                    : resultStr;
                pre.style.display = '';
            }
            // Always-visible one-line preview, so output is discoverable un-clicked.
            const firstLine = (resultStr.split('\n').find((l) => l.trim()) || '').trim();
            preview.textContent = firstLine.length > 96 ? firstLine.slice(0, 94) + '\u2026' : firstLine;
            if (errored) { preview.style.color = 'var(--danger, #f87171)'; }
        }
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    // ── Markdown renderer ─────────────────────────────────────────────────

    function _esc(s) {
        return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    function _codeBlock(lang, code) {
        const id = 'cb_' + Math.random().toString(36).slice(2, 9);
        return '<div class="code-block">'
            + '<div class="code-header">'
            + '<span class="code-lang">' + _esc(lang || 'text') + '</span>'
            + '<button class="copy-code-btn" data-id="' + id + '">Copy</button>'
            + '</div>'
            + '<pre><code id="' + id + '">' + _esc(code) + '</code></pre>'
            + '</div>';
    }

    // If a block is bare JSON (an array/object the model emitted as plain text,
    // like a tool list), pretty-print it so it renders as a formatted code block
    // instead of an unreadable wrapped line.
    function _tryPrettyJson(s) {
        if (!/^[[{]/.test(s) || s.length > 50000) { return null; }
        try {
            const v = JSON.parse(s);
            if (v && typeof v === 'object') { return JSON.stringify(v, null, 2); }
        } catch { /* not JSON — fall through */ }
        return null;
    }

    function _inline(escaped) {
        return escaped
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*([^*]+?)\*/g, '<em>$1</em>')
            .replace(/`([^`]+?)`/g, '<code>$1</code>');
    }

    // Render a non-fenced block: headings, ordered/unordered lists, paragraphs.
    function _renderLines(block) {
        const lines = block.split('\n');
        const out = [];
        let listType = null;
        let items = [];
        let para = [];
        const flushList = () => {
            if (items.length) { out.push('<' + listType + '>' + items.join('') + '</' + listType + '>'); items = []; }
            listType = null;
        };
        const flushPara = () => {
            if (para.length) { out.push('<p>' + _inline(_esc(para.join('\n'))).replace(/\n/g, '<br>') + '</p>'); para = []; }
        };
        for (const line of lines) {
            const h = line.match(/^(#{1,6})\s+(.+)$/);
            const ul = line.match(/^\s*[-*]\s+(.+)$/);
            const ol = line.match(/^\s*\d+[.)]\s+(.+)$/);
            if (h) {
                flushList(); flushPara();
                const lvl = Math.min(6, h[1].length + 2);
                out.push('<h' + lvl + '>' + _inline(_esc(h[2])) + '</h' + lvl + '>');
            } else if (ul) {
                flushPara(); if (listType !== 'ul') { flushList(); } listType = 'ul';
                items.push('<li>' + _inline(_esc(ul[1])) + '</li>');
            } else if (ol) {
                flushPara(); if (listType !== 'ol') { flushList(); } listType = 'ol';
                items.push('<li>' + _inline(_esc(ol[1])) + '</li>');
            } else {
                flushList(); para.push(line);
            }
        }
        flushList(); flushPara();
        return out.join('');
    }

    function _renderMarkdown(text) {
        const out = [];
        const parts = text.split(/(```[\s\S]*?```)/g);
        for (const part of parts) {
            const fenceMatch = part.match(/^```(\w*)\n?([\s\S]*?)```$/);
            if (fenceMatch) {
                out.push(_codeBlock(fenceMatch[1] || '', fenceMatch[2] || ''));
                continue;
            }
            for (const raw of part.split(/\n{2,}/)) {
                const block = raw.replace(/^\n+|\s+$/g, '');
                if (!block.trim()) { continue; }
                const pretty = _tryPrettyJson(block.trim());
                if (pretty !== null) { out.push(_codeBlock('json', pretty)); continue; }
                out.push(_renderLines(block));
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
