/**
 * squish-vscode/src/inlineCompletion.ts
 *
 * InlineCompletionItemProvider that triggers when the cursor is at the end of
 * a line matching `// squish:` or `# squish:` (case-insensitive), behaving
 * like a code-suggestion ghost-text provider similar to GitHub Copilot.
 *
 * Also provides a debounced "complete-at-cursor" trigger that fires after the
 * user types 3+ characters in a language we support.
 */
import * as vscode from 'vscode';
import { SquishClient, ChatMessage } from './squishClient';

const TRIGGER_RE = /[/]{2}\s*squish:\s*$|#\s*squish:\s*$/i;

// Languages for "fill-in-the-middle" completions (beyond explicit trigger)
const FIM_LANGUAGES = new Set([
    'typescript', 'javascript', 'python', 'rust', 'go', 'cpp', 'c', 'java',
]);

export class SquishInlineCompletionProvider implements vscode.InlineCompletionItemProvider {
    private _lastRequest?: AbortController;

    async provideInlineCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position,
        _context: vscode.InlineCompletionContext,
        token: vscode.CancellationToken,
    ): Promise<vscode.InlineCompletionList | undefined> {
        const cfg = vscode.workspace.getConfiguration('squish');
        if (!cfg.get<boolean>('enableInlineCompletion', true)) {
            return undefined;
        }

        const lineText = document.lineAt(position.line).text;
        const isExplicit = TRIGGER_RE.test(lineText);
        const isFim      = FIM_LANGUAGES.has(document.languageId);

        if (!isExplicit && !isFim) {
            return undefined;
        }

        // Abort previous in-flight request
        this._lastRequest?.abort();
        const controller = new AbortController();
        this._lastRequest = controller;

        token.onCancellationRequested(() => controller.abort());

        // ── Build context ─────────────────────────────────────────────────
        const prefix = _buildPrefix(document, position, isExplicit);
        const suffix = _buildSuffix(document, position);

        if (!prefix.trim()) {
            return undefined;
        }

        // ── Call Squish ───────────────────────────────────────────────────
        const host: string  = cfg.get('host', '127.0.0.1');
        const port: number  = cfg.get('port', 11435);
        const apiKey: string = cfg.get('apiKey', 'squish');
        const model: string  = cfg.get('model', '7b');

        const client = new SquishClient(host, port, apiKey);

        const systemMsg: ChatMessage = {
            role: 'system',
            content: isExplicit
                ? 'You are a code completion assistant. The user has a code comment starting with "squish:". Complete the code they are describing after the comment. Return ONLY the code, no explanation, no markdown fences.'
                : 'You are an inline code completion assistant. Complete the code at the cursor position. Return ONLY the completion text — no explanation, no markdown fences. Keep it concise.',
        };

        const userMsg: ChatMessage = {
            role: 'user',
            content: isExplicit
                ? `File: ${document.fileName}\nLanguage: ${document.languageId}\n\n${prefix}`
                : `File: ${document.fileName}\nLanguage: ${document.languageId}\n\nCODE BEFORE CURSOR:\n${prefix}\n\nCODE AFTER CURSOR:\n${suffix}\n\nProvide the completion that fits between the prefix and suffix.`,
        };

        let completion = '';
        try {
            completion = await _fetchCompletion(client, [systemMsg, userMsg], model, controller.signal);
        } catch {
            return undefined;
        }

        if (!completion || token.isCancellationRequested) {
            return undefined;
        }

        // ── Build the inline item ─────────────────────────────────────────
        let insertText = completion;

        // For explicit "squish:" trigger — replace from the trigger comment forward
        let insertRange: vscode.Range | undefined;
        if (isExplicit) {
            const triggerMatch = lineText.match(TRIGGER_RE);
            if (triggerMatch) {
                const commentStart = lineText.search(TRIGGER_RE);
                const replaceStart = new vscode.Position(position.line, commentStart);
                // Replace to end of current line
                const replaceEnd   = new vscode.Position(position.line, lineText.length);
                insertRange = new vscode.Range(replaceStart, replaceEnd);
                // Prepend a newline so it starts on next line (cleaner UX)
                insertText = '\n' + completion;
            }
        }

        const item = new vscode.InlineCompletionItem(insertText);
        if (insertRange) {
            item.range = insertRange;
        }

        return { items: [item] };
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

function _buildPrefix(
    document: vscode.TextDocument,
    position: vscode.Position,
    isExplicit: boolean,
): string {
    // For explicit trigger, take full file up to cursor (max 100 lines around cursor)
    // For FIM, take last 60 lines before cursor
    const start = Math.max(0, position.line - (isExplicit ? 100 : 60));
    const end   = position.line;
    const lines: string[] = [];
    for (let i = start; i <= end; i++) {
        lines.push(document.lineAt(i).text);
    }
    return lines.join('\n');
}

function _buildSuffix(document: vscode.TextDocument, position: vscode.Position): string {
    const start = position.line + 1;
    const end   = Math.min(document.lineCount - 1, position.line + 30);
    const lines: string[] = [];
    for (let i = start; i <= end; i++) {
        lines.push(document.lineAt(i).text);
    }
    return lines.join('\n');
}

function _fetchCompletion(
    client: SquishClient,
    messages: ChatMessage[],
    model: string,
    signal: AbortSignal,
): Promise<string> {
    return new Promise<string>((resolve, reject) => {
        if (signal.aborted) { reject(new Error('aborted')); return; }

        let text = '';
        signal.addEventListener('abort', () => { client.abort(); reject(new Error('aborted')); });

        client.streamChat(
            messages,
            256,     // maxTokens — keep completions short
            0.2,     // temperature — deterministic completions
            model,
            (chunk) => {
                if (chunk.delta) text += chunk.delta;
                if (chunk.done)  resolve(text.trim());
            },
            (err) => reject(err),
        );
    });
}
