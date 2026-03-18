/**
 * squish-vscode/media/chat.js
 *
 * Webview script. Receives messages from the extension host and renders
 * the conversation UI.
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

    // Smooth character render queue
    const _queue             = [];
    let   _rafId             = null;
    const CHARS_PER_FRAME    = 4;  // ~240 chars/sec at 60 fps; smooths token bursts

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
        }
    });

    // ── Turn lifecycle ────────────────────────────────────────────────────

    function _startTurn() {
        _realStarted = false;
        _queue.length = 0;
        if (_rafId) { cancelAnimationFrame(_rafId); _rafId = null; }

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
        _currentContentEl.appendChild(document.createTextNode(text));
        messagesEl.scrollTop = messagesEl.scrollHeight;
        _rafId = requestAnimationFrame(_drainQueue);
    }

    function _endTurn() {
        _cancelAck();
        _flushQueue();
        if (_currentIndicatorEl) { _currentIndicatorEl.style.display = 'none'; }
        if (_currentAckEl)       { _currentAckEl.remove(); _currentAckEl = null; }
        if (_currentContentEl)   { _currentContentEl.classList.remove('streaming'); }
        _finalizeTurn();
    }

    function _flushQueue() {
        if (_rafId) { cancelAnimationFrame(_rafId); _rafId = null; }
        if (_currentContentEl && _queue.length > 0) {
            _currentContentEl.appendChild(
                document.createTextNode(_queue.splice(0).join(''))
            );
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
        _currentBubble      = null;
        _currentContentEl   = null;
        _currentIndicatorEl = null;
        _currentAckEl       = null;
        _realStarted        = false;
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
