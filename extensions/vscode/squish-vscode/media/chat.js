/**
 * squish-vscode/media/chat.js
 *
 * Webview script. Receives messages from the extension host and renders
 * the conversation UI.
 *
 * Message protocol (extension → webview):
 *   { type: 'streamStart' }                          — assistant turn begins
 *   { type: 'streamChunk', delta: string }           — partial token
 *   { type: 'streamEnd' }                            — assistant turn complete
 *   { type: 'streamError', message: str }            — error
 *   { type: 'clearHistory' }                         — wipe the UI
 *   { type: 'toolCallStart', id, name, args }        — tool invocation beginning
 *   { type: 'toolCallEnd',   id, name, result }      — tool invocation complete
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

    // ── State ─────────────────────────────────────────────────────────────

    let _generating          = false;
    let _lastUserText        = '';

    // Current assistant turn elements
    let _currentBubble       = null;  // .message.assistant div
    let _currentContentEl    = null;  // .content span
    let _currentIndicatorEl  = null;  // .typing-indicator div
    let _currentAckEl        = null;  // .ack-text span

    // Ack typing state
    let _ackTimerId          = null;
    let _ackWords            = [];
    let _ackWordIdx          = 0;
    let _realStarted         = false; // has first real streamChunk arrived?

    // Smooth character render queue — 14 chars/frame ≈ 840 chars/sec @ 60 fps,
    // fast enough to feel instant while remaining smooth during token bursts.
    const _queue             = [];
    let   _rafId             = null;
    const CHARS_PER_FRAME    = 14;

    // Active tool call cards keyed by call id
    const _toolCards         = new Map();

    // ── Send ──────────────────────────────────────────────────────────────

    function send() {
        const text = inputEl.value.trim();
        if (!text || _generating) { return; }
        _lastUserText = text;
        inputEl.value = '';
        _generating = true;
        sendBtn.disabled = true;

        _appendUserMessage(text);
        vscode.postMessage({ type: 'userMessage', text });
    }

    // ── Event listeners ───────────────────────────────────────────────────

    sendBtn.addEventListener('click', send);

    inputEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
    });

    clearBtn.addEventListener('click', () => {
        messagesEl.innerHTML = '';
        _resetTurnState();
        vscode.postMessage({ type: 'clearHistory' });
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
                _appendErrorMessage('⚠ ' + (msg.message || 'Unknown error'));
                _finalizeTurn();
                break;

            case 'clearHistory':
                messagesEl.innerHTML = '';
                _resetTurnState();
                break;

            case 'toolCallStart':
                _onToolCallStart(msg.id, msg.name, msg.args);
                break;

            case 'toolCallEnd':
                _onToolCallEnd(msg.id, msg.result);
                break;
        }
    });

    // ── Turn lifecycle ────────────────────────────────────────────────────

    function _startTurn() {
        _realStarted = false;
        _queue.length = 0;
        if (_rafId) { cancelAnimationFrame(_rafId); _rafId = null; }
        _toolCards.clear();

        // Build assistant bubble: label + indicator + ack span + content span
        const row = document.createElement('div');
        row.className = 'message assistant';

        const label = document.createElement('span');
        label.className = 'label';
        label.textContent = 'Squish';

        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        for (let i = 0; i < 3; i++) {
            indicator.appendChild(document.createElement('span'));
        }

        const ack = document.createElement('span');
        ack.className = 'ack-text';

        const content = document.createElement('span');
        content.className = 'content';

        row.appendChild(label);
        row.appendChild(indicator);
        row.appendChild(ack);
        row.appendChild(content);
        messagesEl.appendChild(row);
        messagesEl.scrollTop = messagesEl.scrollHeight;

        _currentBubble      = row;
        _currentContentEl   = content;
        _currentIndicatorEl = indicator;
        _currentAckEl       = ack;

        // After a brief pause, start typing the acknowledgement
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
        const word = _ackWords[_ackWordIdx++];
        _currentAckEl.appendChild(document.createTextNode(sep + word));
        messagesEl.scrollTop = messagesEl.scrollHeight;
        _ackTimerId = setTimeout(_typeNextAckWord, 115);
    }

    function _onRealChunk(delta) {
        if (!_realStarted) {
            _realStarted = true;
            _cancelAck();

            // Swap out the waiting state for the streaming state
            if (_currentIndicatorEl) { _currentIndicatorEl.style.display = 'none'; }
            if (_currentAckEl)      { _currentAckEl.remove(); _currentAckEl = null; }
            if (_currentContentEl)  { _currentContentEl.classList.add('streaming'); }
        }

        for (const ch of delta) { _queue.push(ch); }
        if (!_rafId) { _rafId = requestAnimationFrame(_drainQueue); }
    }

    function _drainQueue() {
        if (!_currentContentEl || _queue.length === 0) {
            _rafId = null;
            return;
        }
        const n    = Math.min(CHARS_PER_FRAME, _queue.length);
        const text = _queue.splice(0, n).join('');
        // Batch the text node append via DocumentFragment to minimise reflows
        const frag = document.createDocumentFragment();
        frag.appendChild(document.createTextNode(text));
        _currentContentEl.appendChild(frag);
        messagesEl.scrollTop = messagesEl.scrollHeight;
        _rafId = requestAnimationFrame(_drainQueue);
    }

    function _endTurn() {
        _cancelAck();
        _flushQueue();
        if (_currentIndicatorEl) { _currentIndicatorEl.style.display = 'none'; }
        if (_currentAckEl)       { _currentAckEl.remove(); _currentAckEl = null; }
        if (_currentContentEl)   {
            _currentContentEl.classList.remove('streaming');
            // Re-render the accumulated plain text as inline markdown
            const plain = _currentContentEl.textContent || '';
            _currentContentEl.innerHTML = _renderMarkdown(plain);
        }
        _finalizeTurn();
    }

    function _flushQueue() {
        if (_rafId) { cancelAnimationFrame(_rafId); _rafId = null; }
        if (_currentContentEl && _queue.length > 0) {
            const frag = document.createDocumentFragment();
            frag.appendChild(document.createTextNode(_queue.splice(0).join('')));
            _currentContentEl.appendChild(frag);
            messagesEl.scrollTop = messagesEl.scrollHeight;
        }
    }

    function _cancelAck() {
        if (_ackTimerId !== null) { clearTimeout(_ackTimerId); _ackTimerId = null; }
    }

    function _finalizeTurn() {
        _generating = false;
        sendBtn.disabled = false;
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

        // Ensure any pending ack/indicator is cleared before showing tool cards
        _cancelAck();
        if (_currentIndicatorEl) { _currentIndicatorEl.style.display = 'none'; }
        if (_currentAckEl)      { _currentAckEl.remove(); _currentAckEl = null; }
        // Flush any streamed text before the tool card (precautionary)
        _flushQueue();

        const card = document.createElement('div');
        card.className = 'tool-call-card';
        card.setAttribute('data-id', id);

        const header = document.createElement('div');
        header.className = 'tool-call-header';

        const iconEl = document.createElement('span');
        iconEl.className = 'tool-call-icon';
        iconEl.textContent = '⚙';

        const nameEl = document.createElement('span');
        nameEl.className = 'tool-call-name';
        nameEl.textContent = name;

        const statusEl = document.createElement('span');
        statusEl.className = 'tool-call-status running';
        statusEl.textContent = 'running…';

        header.appendChild(iconEl);
        header.appendChild(nameEl);
        header.appendChild(statusEl);
        card.appendChild(header);

        // Collapsible args section
        let args;
        try { args = JSON.parse(argsJson || '{}'); } catch { args = {}; }
        const argsText = Object.entries(args)
            .map(([k, v]) => `${k}: ${JSON.stringify(v)}`)
            .join('\n');

        if (argsText) {
            const details = document.createElement('details');
            const summary = document.createElement('summary');
            summary.textContent = 'Arguments';
            const pre = document.createElement('pre');
            pre.className = 'tool-call-args';
            pre.textContent = argsText;
            details.appendChild(summary);
            details.appendChild(pre);
            card.appendChild(details);
        }

        // Placeholder for result (filled in on toolCallEnd)
        const resultEl = document.createElement('pre');
        resultEl.className = 'tool-call-result hidden';
        card.appendChild(resultEl);

        _currentBubble.appendChild(card);
        messagesEl.scrollTop = messagesEl.scrollHeight;

        _toolCards.set(id, { card, statusEl, resultEl });
    }

    function _onToolCallEnd(id, result) {
        const entry = _toolCards.get(id);
        if (!entry) { return; }

        const { statusEl, resultEl } = entry;
        statusEl.textContent = 'done';
        statusEl.className = 'tool-call-status done';

        if (result !== undefined && result !== null && String(result).length > 0) {
            const truncated = String(result).length > 500
                ? String(result).slice(0, 500) + '\n… (truncated)'
                : String(result);
            resultEl.textContent = truncated;
            resultEl.classList.remove('hidden');
        }

        messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    // ── Inline markdown renderer ───────────────────────────────────────────
    // Supports: **bold**, *italic*, `code`, and paragraph breaks (blank lines).
    // All text is HTML-escaped before processing to prevent XSS.

    function _esc(s) {
        return s
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    function _renderMarkdown(text) {
        // Split into paragraphs on blank lines
        const paragraphs = text.split(/\n{2,}/);
        return paragraphs.map((para) => {
            // Escape the raw paragraph text
            let s = _esc(para.trim());
            if (!s) { return ''; }
            // Bold: **...**  (must come before italic)
            s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
            // Italic: *...*  (single asterisk, non-greedy)
            s = s.replace(/\*([^*]+?)\*/g, '<em>$1</em>');
            // Inline code: `...`
            s = s.replace(/`([^`]+?)`/g, '<code>$1</code>');
            // Preserve single newlines as <br>
            s = s.replace(/\n/g, '<br>');
            return '<p>' + s + '</p>';
        }).filter(Boolean).join('');
    }

    // ── Acknowledgement phrase generator ─────────────────────────────────

    function _buildAckWords(text) {
        const t = (text || '').toLowerCase();
        let phrase;
        if      (/\b(fix|debug|bug|error|broken|crash|issue)\b/.test(t)) phrase = "Let me debug that...";
        else if (/\b(write|create|generate|make|build|scaffold)\b/.test(t)) phrase = "I'll create that for you...";
        else if (/\b(test|spec|coverage|unittest)\b/.test(t))            phrase = "Writing those tests...";
        else if (/\b(refactor|improve|optimize|clean|rewrite)\b/.test(t)) phrase = "Working on that refactor...";
        else if (/\b(explain|what\s+is|how\s+does|why|when|where)\b/.test(t)) phrase = "Let me explain...";
        else if (/\b(review|check|look\s+at|analyze|audit)\b/.test(t))  phrase = "Reviewing that now...";
        else if (/\b(help|assist|show|guide)\b/.test(t))                 phrase = "I'll help with that...";
        else if (/\?/.test(text))                                        phrase = "Let me answer that...";
        else                                                             phrase = "On it...";
        return phrase.split(' ');
    }

    // ── DOM helpers ───────────────────────────────────────────────────────

    function _appendUserMessage(text) {
        const row = document.createElement('div');
        row.className = 'message user';
        const label = document.createElement('span');
        label.className = 'label';
        label.textContent = 'You';
        const content = document.createElement('span');
        content.className = 'content';
        content.appendChild(document.createTextNode(text));
        row.appendChild(label);
        row.appendChild(content);
        messagesEl.appendChild(row);
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function _appendErrorMessage(text) {
        const row = document.createElement('div');
        row.className = 'message error';
        const content = document.createElement('span');
        content.className = 'content';
        content.appendChild(document.createTextNode(text));
        row.appendChild(content);
        messagesEl.appendChild(row);
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }
}());

