/**
 * squish-vscode/src/chatPanel.ts
 *
 * Implements the sidebar WebviewView for Squish Agent.
 * Streams SSE chunks from the Squish server to the webview via postMessage,
 * manages sessions via HistoryManager, and runs a multi-round agentic tool loop.
 */
import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import { exec, spawn } from 'child_process';
import { promisify } from 'util';
import { SquishClient, ChatMessage, ToolDefinition, ToolCall } from './squishClient';
import { HistoryManager, Session } from './historyManager';

const execAsync = promisify(exec);

export class ChatPanel implements vscode.WebviewViewProvider {
    public static readonly viewType = 'squish.chatView';

    private _view?: vscode.WebviewView;
    private _session: Session;
    private _historyManager: HistoryManager;
    private _activeClient?: SquishClient;

    // Full conversation history (source of truth — kept in sync with _session.messages)
    private _history: ChatMessage[] = [];

    // State for filtering <think>...</think> blocks out of the stream
    private _inThink = false;
    private _thinkBuf = '';

    // ── Tool definitions (passed to the model) ────────────────────────────
    private static readonly TOOLS: ToolDefinition[] = [
        {
            type: 'function',
            function: {
                name: 'read_file',
                description: 'Read the contents of a file in the current workspace.',
                parameters: {
                    type: 'object',
                    properties: {
                        path: {
                            type: 'string',
                            description: 'Relative path from the workspace root, e.g. "src/index.ts"',
                        },
                    },
                    required: ['path'],
                },
            },
        },
        {
            type: 'function',
            function: {
                name: 'get_selection',
                description: 'Get the text currently selected in the active editor.',
                parameters: { type: 'object', properties: {} },
            },
        },
        {
            type: 'function',
            function: {
                name: 'get_open_files',
                description: 'List the relative paths of all files currently open in the editor.',
                parameters: { type: 'object', properties: {} },
            },
        },
        {
            type: 'function',
            function: {
                name: 'run_terminal',
                description: 'Run a shell command in the VS Code integrated terminal and return its output.',
                parameters: {
                    type: 'object',
                    properties: {
                        command: {
                            type: 'string',
                            description: 'The shell command to execute.',
                        },
                    },
                    required: ['command'],
                },
            },
        },
        {
            type: 'function',
            function: {
                name: 'insert_at_cursor',
                description: 'Insert text at the current cursor position in the active editor.',
                parameters: {
                    type: 'object',
                    properties: {
                        text: {
                            type: 'string',
                            description: 'The text to insert at the cursor.',
                        },
                    },
                    required: ['text'],
                },
            },
        },
        {
            type: 'function',
            function: {
                name: 'write_file',
                description: 'Write or overwrite a file in the workspace. Shows a confirmation dialog before overwriting an existing file.',
                parameters: {
                    type: 'object',
                    properties: {
                        path: {
                            type: 'string',
                            description: 'Relative path from the workspace root, e.g. "src/index.ts"',
                        },
                        content: {
                            type: 'string',
                            description: 'The full content to write to the file.',
                        },
                    },
                    required: ['path', 'content'],
                },
            },
        },
        {
            type: 'function',
            function: {
                name: 'list_directory',
                description: 'List files and subdirectories at a path in the workspace.',
                parameters: {
                    type: 'object',
                    properties: {
                        path: {
                            type: 'string',
                            description: 'Relative path from the workspace root (omit or use "." for the root).',
                        },
                    },
                    required: [],
                },
            },
        },
        {
            type: 'function',
            function: {
                name: 'get_diagnostics',
                description: 'Get all VS Code errors and warnings (diagnostics) for the current workspace.',
                parameters: { type: 'object', properties: {} },
            },
        },
        {
            type: 'function',
            function: {
                name: 'apply_edit',
                description: 'Apply a surgical text replacement in a file. Replaces old_text with new_text. Safer than write_file for targeted edits.',
                parameters: {
                    type: 'object',
                    properties: {
                        path:     { type: 'string', description: 'Relative path from workspace root.' },
                        old_text: { type: 'string', description: 'The exact text to replace. Must be unique in the file.' },
                        new_text: { type: 'string', description: 'The replacement text.' },
                    },
                    required: ['path', 'old_text', 'new_text'],
                },
            },
        },
        {
            type: 'function',
            function: {
                name: 'search_workspace',
                description: 'Search all files in the workspace for a text pattern (case-insensitive substring or regex). Returns matching file paths and line snippets.',
                parameters: {
                    type: 'object',
                    properties: {
                        query:   { type: 'string', description: 'The text or regex pattern to search for.' },
                        glob:    { type: 'string', description: 'Optional glob pattern to scope the search, e.g. "**/*.ts".' },
                        is_regex:{ type: 'string', description: '"true" to treat query as a regex, otherwise plain string.' },
                    },
                    required: ['query'],
                },
            },
        },
        {
            type: 'function',
            function: {
                name: 'create_file',
                description: 'Create a new file in the workspace with the given content. Fails if file already exists (use write_file to overwrite).',
                parameters: {
                    type: 'object',
                    properties: {
                        path:    { type: 'string', description: 'Relative path for the new file.' },
                        content: { type: 'string', description: 'File content.' },
                    },
                    required: ['path', 'content'],
                },
            },
        },
        {
            type: 'function',
            function: {
                name: 'delete_file',
                description: 'Delete a file from the workspace. Always asks the user for confirmation.',
                parameters: {
                    type: 'object',
                    properties: {
                        path: { type: 'string', description: 'Relative path of the file to delete.' },
                    },
                    required: ['path'],
                },
            },
        },
        {
            type: 'function',
            function: {
                name: 'get_git_status',
                description: 'Get the current git status and recent commit log for the workspace.',
                parameters: { type: 'object', properties: {} },
            },
        },
        {
            type: 'function',
            function: {
                name: 'get_symbol_at_cursor',
                description: 'Get the symbol (function, class, variable) currently under the editor cursor, including its definition location.',
                parameters: { type: 'object', properties: {} },
            },
        },
    ];

    constructor(extensionUri: vscode.Uri, historyManager: HistoryManager) {
        this._extensionUri = extensionUri;
        this._historyManager = historyManager;
        this._session = historyManager.createSession();
    }

    private readonly _extensionUri: vscode.Uri;

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
                    this.newSession();
                    break;
                case 'stopGeneration':
                    this._abortGeneration();
                    break;
                case 'newSession':
                    this.newSession();
                    break;
                case 'loadSession':
                    this._loadSession(msg.id as string);
                    break;
                case 'deleteSession':
                    this._deleteSession(msg.id as string);
                    break;
                case 'renameSession':
                    this._historyManager.rename(msg.id as string, msg.title as string);
                    this._sendSessionList();
                    break;
                case 'requestSessionList':
                    this._sendSessionList();
                    break;
            }
        });

        // Push initial session list once webview is ready
        setTimeout(() => this._sendSessionList(), 300);
    }

    // ── Public API ────────────────────────────────────────────────────────

    /** Start a new blank session (called from command or webview). */
    newSession(): void {
        this._session = this._historyManager.createSession();
        this._history = [];
        this._view?.webview.postMessage({ type: 'clearHistory' });
        this._sendSessionList();
    }

    /** Push a user message into the active session, then run the agent.
     *  Called from CodeLens / context menu commands. */
    startAgentTask(prompt: string, contextNote?: string): void {
        const fullPrompt = contextNote
            ? `${contextNote}\n\n${prompt}`
            : prompt;
        // Reveal the panel first
        vscode.commands.executeCommand('squish.chatView.focus');
        // Schedule so the webview has time to appear
        setTimeout(() => {
            this._handleUserMessage(fullPrompt).catch(() => {/* already surfaced in webview */});
        }, 200);
    }

    /** Legacy alias kept for backward compat. */
    clearHistory(): void {
        this.newSession();
    }

    // ── Private helpers ───────────────────────────────────────────────────

    private _abortGeneration(): void {
        this._activeClient?.abort();
        this._view?.webview.postMessage({ type: 'streamEnd' });
    }

    private _sendSessionList(): void {
        const sessions = this._historyManager.list();
        this._view?.webview.postMessage({
            type: 'sessionList',
            sessions: sessions.map(s => ({
                id: s.id,
                title: s.title,
                updatedAt: s.updatedAt,
            })),
            activeId: this._session.id,
        });
    }

    private _loadSession(id: string): void {
        const session = this._historyManager.load(id);
        if (!session) return;
        this._session = session;
        this._history = session.messages.map((m) => ({ ...m }));
        this._view?.webview.postMessage({
            type: 'sessionLoaded',
            session: {
                id: session.id,
                title: session.title,
                messages: session.messages
                    .filter(m => m.role === 'user' || m.role === 'assistant')
                    .map(m => ({ role: m.role, content: m.content ?? '' })),
            },
        });
    }

    private _deleteSession(id: string): void {
        this._historyManager.delete(id);
        if (this._session.id === id) {
            this._session = this._historyManager.createSession();
            this._view?.webview.postMessage({ type: 'clearHistory' });
        }
        this._sendSessionList();
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

        // Update session title from first user message, then sync history
        if (this._session.messages.length === 0) {
            this._session.title = text.length > 60 ? text.slice(0, 60) + '\u2026' : text;
        }
        this._history.push({ role: 'user', content: text });
        this._session.messages = [...this._history];

        this._activeClient = new SquishClient(host, port, apiKey);
        const client = this._activeClient;

        // Reset think-block filter state for this turn
        this._inThink = false;
        this._thinkBuf = '';

        // Signal start of assistant turn
        this._view.webview.postMessage({ type: 'streamStart' });

        await this._runToolLoop(client, messages, maxTokens, temperature, model);
    }

    /**
     * Execute the full agentic tool-calling loop.
     * The model may request 0 or more tool calls before producing a final answer.
     */
    private _runToolLoop(
        client: SquishClient,
        messages: ChatMessage[],
        maxTokens: number,
        temperature: number,
        model: string,
    ): Promise<void> {
        return new Promise<void>((resolve) => {
            const MAX_TOOL_ROUNDS = 10;
            let toolRound = 0;

            const runOnce = (msgs: ChatMessage[]) => {
                let assistantReply = '';

                client.streamChat(
                    msgs,
                    maxTokens,
                    temperature,
                    model,
                    async (chunk) => {
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
                            // Flush buffered think content
                            const flushed = this._thinkBuf;
                            if (!this._inThink && flushed) {
                                assistantReply += flushed;
                                this._view?.webview.postMessage({ type: 'streamChunk', delta: flushed });
                            }
                            this._thinkBuf = '';

                            const toolCalls = chunk.toolCalls;
                            const wantsTools = chunk.finishReason === 'tool_calls' && toolCalls && toolCalls.length > 0;

                            if (wantsTools && toolRound < MAX_TOOL_ROUNDS) {
                                toolRound++;

                                // Append the assistant message (with tool_calls) to history
                                const assistantMsg: ChatMessage = {
                                    role: 'assistant',
                                    content: assistantReply || null,
                                    tool_calls: toolCalls,
                                };
                                this._history.push(assistantMsg);
                                const nextMsgs = [...msgs, assistantMsg];

                                // Execute each tool call and collect results
                                for (const tc of toolCalls!) {
                                    this._view?.webview.postMessage({
                                        type: 'toolCallStart',
                                        id: tc.id,
                                        name: tc.function.name,
                                        args: tc.function.arguments,
                                    });

                                    let result: string;
                                    try {
                                        result = await this._executeToolCall(tc);
                                    } catch (e) {
                                        result = `Error: ${(e as Error).message}`;
                                    }

                                    this._view?.webview.postMessage({
                                        type: 'toolCallEnd',
                                        id: tc.id,
                                        name: tc.function.name,
                                        result,
                                    });

                                    const toolResultMsg: ChatMessage = {
                                        role: 'tool',
                                        content: result,
                                        tool_call_id: tc.id,
                                        name: tc.function.name,
                                    };
                                    this._history.push(toolResultMsg);
                                    nextMsgs.push(toolResultMsg);
                                }

                                // Reset think filter and continue
                                this._inThink = false;
                                this._thinkBuf = '';
                                runOnce(nextMsgs);
                            } else {
                                // Final answer — commit to history and signal done
                                this._view?.webview.postMessage({ type: 'streamEnd' });
                                if (assistantReply) {
                                    this._history.push({ role: 'assistant', content: assistantReply });
                                }
                                // Persist session to disk
                                this._session.messages = [...this._history];
                                this._historyManager.save(this._session);
                                this._sendSessionList();
                                resolve();
                            }
                        }
                    },
                    (err) => {
                        this._view?.webview.postMessage({
                            type: 'streamError',
                            message: err.message,
                        });
                        resolve();
                    },
                    ChatPanel.TOOLS,
                );
            };

            runOnce(messages);
        });
    }

    /**
     * Execute a single tool call and return the string result.
     */
    private async _executeToolCall(tc: ToolCall): Promise<string> {
        const args = this._parseToolArgs(tc.function.arguments);

        switch (tc.function.name) {
            case 'read_file':
                return this._toolReadFile(args.path as string);

            case 'get_selection':
                return this._toolGetSelection();

            case 'get_open_files':
                return this._toolGetOpenFiles();

            case 'run_terminal':
                return this._toolRunTerminal(args.command as string);

            case 'insert_at_cursor':
                return this._toolInsertAtCursor(args.text as string);

            case 'write_file':
                return this._toolWriteFile(args.path as string, args.content as string);

            case 'list_directory':
                return this._toolListDirectory((args.path as string | undefined) ?? '.');

            case 'get_diagnostics':
                return this._toolGetDiagnostics();

            case 'apply_edit':
                return this._toolApplyEdit(
                    args.path as string,
                    args.old_text as string,
                    args.new_text as string,
                );

            case 'search_workspace':
                return this._toolSearchWorkspace(
                    args.query as string,
                    args.glob as string | undefined,
                    (args.is_regex as string | undefined) === 'true',
                );

            case 'create_file':
                return this._toolCreateFile(args.path as string, args.content as string);

            case 'delete_file':
                return this._toolDeleteFile(args.path as string);

            case 'get_git_status':
                return this._toolGetGitStatus();

            case 'get_symbol_at_cursor':
                return this._toolGetSymbolAtCursor();

            default:
                throw new Error(`Unknown tool: ${tc.function.name}`);
        }
    }

    private _parseToolArgs(argsJson: string): Record<string, unknown> {
        try {
            return JSON.parse(argsJson) as Record<string, unknown>;
        } catch {
            return {};
        }
    }

    // ── Tool implementations ──────────────────────────────────────────────

    private async _toolReadFile(relativePath: string): Promise<string> {
        const folders = vscode.workspace.workspaceFolders;
        if (!folders || folders.length === 0) {
            throw new Error('No workspace open');
        }
        const abs = vscode.Uri.joinPath(folders[0].uri, relativePath);
        const bytes = await vscode.workspace.fs.readFile(abs);
        return Buffer.from(bytes).toString('utf8');
    }

    private _toolGetSelection(): string {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return '';
        }
        return editor.document.getText(editor.selection);
    }

    private _toolGetOpenFiles(): string {
        const folders = vscode.workspace.workspaceFolders;
        const rootPath = folders?.[0]?.uri.fsPath ?? '';
        const openDocs = vscode.workspace.textDocuments
            .filter((d) => !d.isUntitled && d.uri.scheme === 'file')
            .map((d) => (rootPath ? path.relative(rootPath, d.uri.fsPath) : d.uri.fsPath));
        return JSON.stringify(openDocs);
    }

    private _toolRunTerminal(command: string): Promise<string> {
        return new Promise((resolve) => {
            // Run in a new unnamed terminal, capture output via a temp file
            const tmpFile = path.join(
                os.tmpdir(),
                `squish_tool_${Date.now()}.txt`,
            );
            const shell = process.platform === 'win32' ? 'cmd.exe' : '/bin/sh';
            const shellArgs = process.platform === 'win32'
                ? ['/c', `${command} > "${tmpFile}" 2>&1`]
                : ['-c', `${command} > "${tmpFile}" 2>&1`];
            const child = spawn(shell, shellArgs, { shell: false });
            child.on('close', () => {
                try {
                    resolve(fs.readFileSync(tmpFile, 'utf8'));
                } catch {
                    resolve('');
                } finally {
                    try { fs.unlinkSync(tmpFile); } catch { /* ignore */ }
                }
            });
            child.on('error', (e: Error) => resolve(`Error: ${e.message}`));
        });
    }

    private async _toolInsertAtCursor(text: string): Promise<string> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return 'No active editor';
        }
        await editor.edit((editBuilder) => {
            editBuilder.replace(editor.selection, text);
        });
        return 'Inserted';
    }

    private async _toolWriteFile(relativePath: string, content: string): Promise<string> {
        const folders = vscode.workspace.workspaceFolders;
        if (!folders || folders.length === 0) {
            throw new Error('No workspace open');
        }
        const abs = vscode.Uri.joinPath(folders[0].uri, relativePath);
        let exists = false;
        try {
            await vscode.workspace.fs.stat(abs);
            exists = true;
        } catch {
            // file does not exist — new file, no confirmation needed
        }
        if (exists) {
            const answer = await vscode.window.showWarningMessage(
                `Squish wants to overwrite "${relativePath}". Proceed?`,
                'Overwrite',
                'Cancel',
            );
            if (answer !== 'Overwrite') {
                return 'Cancelled.';
            }
        }
        await vscode.workspace.fs.writeFile(abs, Buffer.from(content, 'utf8'));
        return exists ? `Updated: ${relativePath}` : `Created: ${relativePath}`;
    }

    private async _toolListDirectory(relativePath: string): Promise<string> {
        const folders = vscode.workspace.workspaceFolders;
        if (!folders || folders.length === 0) {
            throw new Error('No workspace open');
        }
        const dir = (!relativePath || relativePath === '.')
            ? folders[0].uri
            : vscode.Uri.joinPath(folders[0].uri, relativePath);
        const entries = await vscode.workspace.fs.readDirectory(dir);
        const lines = entries.map(([name, type]) =>
            type === vscode.FileType.Directory ? `${name}/` : name,
        );
        return lines.length > 0 ? lines.join('\n') : '(empty)';
    }

    private async _toolApplyEdit(relativePath: string, oldText: string, newText: string): Promise<string> {
        const folders = vscode.workspace.workspaceFolders;
        if (!folders || folders.length === 0) {
            throw new Error('No workspace open');
        }
        const abs = vscode.Uri.joinPath(folders[0].uri, relativePath);
        const bytes = await vscode.workspace.fs.readFile(abs);
        const src = Buffer.from(bytes).toString('utf8');
        const count = src.split(oldText).length - 1;
        if (count === 0) {
            throw new Error(`old_text not found in ${relativePath}`);
        }
        if (count > 1) {
            throw new Error(`old_text is ambiguous — found ${count} occurrences in ${relativePath}`);
        }
        const next = src.replace(oldText, newText);
        await vscode.workspace.fs.writeFile(abs, Buffer.from(next, 'utf8'));
        return `Applied edit to ${relativePath}`;
    }

    private async _toolSearchWorkspace(query: string, glob?: string, isRegex = false): Promise<string> {
        const pattern = glob ?? '**/*';
        const uris = await vscode.workspace.findFiles(pattern, '**/node_modules/**', 200);
        const folders = vscode.workspace.workspaceFolders;
        const root = folders?.[0]?.uri.fsPath ?? '';
        const lines: string[] = [];
        const re = isRegex ? new RegExp(query, 'i') : null;
        for (const uri of uris) {
            try {
                const bytes = await vscode.workspace.fs.readFile(uri);
                const text = Buffer.from(bytes).toString('utf8');
                const fileLines = text.split('\n');
                for (let ln = 0; ln < fileLines.length; ln++) {
                    const lineText = fileLines[ln];
                    const hit = re ? re.test(lineText) : lineText.toLowerCase().includes(query.toLowerCase());
                    if (hit) {
                        const rel = root ? path.relative(root, uri.fsPath) : uri.fsPath;
                        lines.push(`${rel}:${ln + 1}: ${lineText.trim()}`);
                        if (lines.length >= 50) { break; }
                    }
                }
                if (lines.length >= 50) { break; }
            } catch { /* skip unreadable files */ }
        }
        return lines.length > 0 ? lines.join('\n') : 'No matches found.';
    }

    private async _toolCreateFile(relativePath: string, content: string): Promise<string> {
        const folders = vscode.workspace.workspaceFolders;
        if (!folders || folders.length === 0) {
            throw new Error('No workspace open');
        }
        const abs = vscode.Uri.joinPath(folders[0].uri, relativePath);
        let exists = false;
        try { await vscode.workspace.fs.stat(abs); exists = true; } catch { /* new file */ }
        if (exists) {
            throw new Error(`File already exists: ${relativePath}. Use write_file to overwrite.`);
        }
        const cfg = vscode.workspace.getConfiguration('squish');
        if (cfg.get<boolean>('requireApprovalForWrite', true)) {
            const ans = await vscode.window.showInformationMessage(
                `Squish wants to create "${relativePath}". Proceed?`,
                'Create', 'Cancel',
            );
            if (ans !== 'Create') { return 'Cancelled.'; }
        }
        await vscode.workspace.fs.writeFile(abs, Buffer.from(content, 'utf8'));
        return `Created: ${relativePath}`;
    }

    private async _toolDeleteFile(relativePath: string): Promise<string> {
        const folders = vscode.workspace.workspaceFolders;
        if (!folders || folders.length === 0) {
            throw new Error('No workspace open');
        }
        const abs = vscode.Uri.joinPath(folders[0].uri, relativePath);
        const ans = await vscode.window.showWarningMessage(
            `Squish wants to DELETE "${relativePath}". This cannot be undone.`,
            'Delete', 'Cancel',
        );
        if (ans !== 'Delete') { return 'Cancelled.'; }
        await vscode.workspace.fs.delete(abs, { recursive: false, useTrash: true });
        return `Deleted: ${relativePath}`;
    }

    private async _toolGetGitStatus(): Promise<string> {
        const root = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
        if (!root) { return 'No workspace open.'; }
        try {
            const { stdout: status } = await execAsync('git status --short', { cwd: root, timeout: 5000 });
            const { stdout: log } = await execAsync(
                'git log --oneline -5', { cwd: root, timeout: 5000 },
            ).catch(() => ({ stdout: '' }));
            return `Git status:\n${status || '(clean)'}\n\nRecent commits:\n${log || '(none)'}`;
        } catch (e) {
            return `git error: ${(e as Error).message}`;
        }
    }

    private async _toolGetSymbolAtCursor(): Promise<string> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) { return 'No active editor.'; }
        const position = editor.selection.active;
        const wordRange = editor.document.getWordRangeAtPosition(position);
        if (!wordRange) { return 'No symbol at cursor.'; }
        const word = editor.document.getText(wordRange);
        // Try to get definition via VS Code language service
        try {
            const defs = await vscode.commands.executeCommand<vscode.Location[]>(
                'vscode.executeDefinitionProvider',
                editor.document.uri,
                position,
            );
            if (defs && defs.length > 0) {
                const d = defs[0];
                const folders = vscode.workspace.workspaceFolders;
                const root = folders?.[0]?.uri.fsPath ?? '';
                const relPath = root ? path.relative(root, d.uri.fsPath) : d.uri.fsPath;
                return `Symbol: ${word}\nDefined at: ${relPath}:${d.range.start.line + 1}`;
            }
        } catch { /* language service unavailable */ }
        return `Symbol at cursor: ${word}`;
    }

    private _toolGetDiagnostics(): string {
        const all = vscode.languages.getDiagnostics();
        const folders = vscode.workspace.workspaceFolders;
        const root = folders?.[0]?.uri.fsPath ?? '';
        const lines: string[] = [];
        for (const [uri, diags] of all) {
            for (const d of diags) {
                const sev = d.severity === vscode.DiagnosticSeverity.Error   ? 'error'
                          : d.severity === vscode.DiagnosticSeverity.Warning ? 'warning'
                          : 'info';
                const rel = root ? path.relative(root, uri.fsPath) : uri.fsPath;
                lines.push(`${rel}:${d.range.start.line + 1} [${sev}] ${d.message}`);
            }
        }
        return lines.length > 0 ? lines.join('\n') : 'No diagnostics.';
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
        const nonce = _nonce();

        // SVG logo path (same squish character as before)
        const logoSvg = `<svg viewBox="12 23 106 94" xmlns="http://www.w3.org/2000/svg" width="28" height="28">
  <ellipse cx="22" cy="65" rx="8" ry="12" fill="#8B5CF6" stroke="#1A0B40" stroke-width="3" transform="rotate(15 22 65)"/>
  <ellipse cx="108" cy="65" rx="8" ry="12" fill="#8B5CF6" stroke="#1A0B40" stroke-width="3" transform="rotate(-15 108 65)"/>
  <ellipse cx="45" cy="105" rx="6" ry="10" fill="#8B5CF6" stroke="#1A0B40" stroke-width="3"/>
  <ellipse cx="85" cy="105" rx="6" ry="10" fill="#8B5CF6" stroke="#1A0B40" stroke-width="3"/>
  <path d="M35 25C25 25 20 35 20 45V85C20 95 25 105 35 105H95C105 105 110 95 110 85V45C110 35 105 25 95 25H35Z" fill="#8B5CF6" stroke="#1A0B40" stroke-width="4" stroke-linejoin="round"/>
  <circle cx="48" cy="55" r="6" fill="black"/><circle cx="46" cy="53" r="2" fill="white"/>
  <circle cx="80" cy="55" r="6" fill="black"/><circle cx="78" cy="53" r="2" fill="white"/>
  <path d="M58 70C58 75 70 75 70 70H58Z" fill="#FF8A8A" stroke="#1A0B40" stroke-width="2"/>
  <circle cx="40" cy="68" r="4" fill="#FFBABA" opacity="0.6"/>
  <circle cx="88" cy="68" r="4" fill="#FFBABA" opacity="0.6"/>
</svg>`;

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
  <title>Squish Agent</title>
</head>
<body>

<!-- History slide-in panel -->
<div id="hist-panel" aria-hidden="true">
  <div id="hist-panel-header">
    <span id="hist-panel-title">History</span>
    <button id="hist-close" title="Close history">&#x2715;</button>
  </div>
  <button id="hist-new-btn">+ New Chat</button>
  <div id="hist-list" role="list"></div>
</div>
<div id="hist-overlay"></div>

<!-- Main shell -->
<div id="shell">

  <!-- Header -->
  <div id="header">
    <button id="hist-toggle" title="Chat history" aria-label="Toggle history">&#x2630;</button>
    <div class="header-brand">
      ${logoSvg}
      <span class="header-name">Squish <span class="header-accent">Agent</span></span>
    </div>
    <div class="header-actions">
      <button id="btn-new-chat" title="New conversation" aria-label="New chat">&#x2b;</button>
    </div>
  </div>

  <!-- Messages -->
  <div id="messages" aria-live="polite" role="log"></div>

  <!-- Input -->
  <div id="input-area">
    <div id="input-row">
      <textarea
        id="user-input"
        placeholder="Ask Squish Agent… (Enter to send, Shift+Enter for newline)"
        rows="3"
        aria-label="Message input"
      ></textarea>
    </div>
    <div id="input-btns">
      <button id="btn-regen" class="btn-secondary" hidden title="Regenerate last response">&#x21ba; Regenerate</button>
      <div class="input-btns-right">
        <button id="btn-stop" class="btn-danger" hidden>&#x25a0; Stop</button>
        <button id="btn-send">Send &#x21b5;</button>
      </div>
    </div>
  </div>

</div><!-- /shell -->

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
