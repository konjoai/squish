/**
 * squish-vscode/src/extension.ts
 *
 * Extension entry point for Squish Agent v0.2.
 * Registers all commands, creates the status bar, wires up the sidebar
 * WebviewViews, inline-completion provider, CodeLens provider, and
 * health-poll loop.
 */
import * as vscode from 'vscode';
import { SquishClient } from './squishClient';
import { ChatPanel } from './chatPanel';
import { ServerManager } from './serverManager';
import { HistoryManager } from './historyManager';
import { MonitorPanel } from './monitorPanel';
import { SquishInlineCompletionProvider } from './inlineCompletion';
import { SquishCodeLensProvider } from './codeLens';
import { ContextCollector } from './contextCollector';

let statusBar: vscode.StatusBarItem;
let chatPanel: ChatPanel | undefined;
let serverManager: ServerManager;
let pollTimer: NodeJS.Timeout | undefined;

export function activate(context: vscode.ExtensionContext): void {
    // ── Server manager ────────────────────────────────────────────────────
    serverManager = new ServerManager(context);

    // ── History manager ───────────────────────────────────────────────────
    const cfg = vscode.workspace.getConfiguration('squish');
    const historyPath: string | undefined = cfg.get('historyPath');
    const historyManager = new HistoryManager(historyPath);

    // ── Status bar ────────────────────────────────────────────────────────
    statusBar = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Right,
        100,
    );
    statusBar.command = 'squish.openChat';
    statusBar.text = '$(hubot) squish: offline';
    statusBar.tooltip = 'Squish Agent — click to open chat';
    statusBar.show();
    context.subscriptions.push(statusBar);

    // ── Chat panel (sidebar WebviewView) ──────────────────────────────────
    chatPanel = new ChatPanel(context.extensionUri, historyManager);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('squish.chatView', chatPanel, {
            webviewOptions: { retainContextWhenHidden: true },
        }),
    );

    // ── Monitor panel (WebviewView) ───────────────────────────────────────
    const monitorPanel = new MonitorPanel(context.extensionUri);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('squish.monitorView', monitorPanel),
    );

    // ── Context collector ─────────────────────────────────────────────────
    const contextCollector = new ContextCollector();

    // ── Inline completion provider ────────────────────────────────────────
    const inlineProvider = new SquishInlineCompletionProvider();
    context.subscriptions.push(
        vscode.languages.registerInlineCompletionItemProvider(
            { pattern: '**' },
            inlineProvider,
        ),
    );

    // ── CodeLens provider ─────────────────────────────────────────────────
    const codeLensProvider = new SquishCodeLensProvider();
    const codeLensSelector: vscode.DocumentSelector = [
        { language: 'typescript' },
        { language: 'javascript' },
        { language: 'python' },
        { language: 'rust' },
        { language: 'go' },
        { language: 'cpp' },
        { language: 'c' },
    ];
    context.subscriptions.push(
        vscode.languages.registerCodeLensProvider(codeLensSelector, codeLensProvider),
    );

    // ── Commands ──────────────────────────────────────────────────────────
    context.subscriptions.push(

        // Focus chat view
        vscode.commands.registerCommand('squish.openChat', () => {
            vscode.commands.executeCommand('squish.chatView.focus');
        }),

        // Focus monitor view
        vscode.commands.registerCommand('squish.openMonitor', () => {
            vscode.commands.executeCommand('squish.monitorView.focus');
        }),

        // Start a new blank conversation
        vscode.commands.registerCommand('squish.newChat', () => {
            chatPanel?.newSession();
            vscode.commands.executeCommand('squish.chatView.focus');
        }),

        // Legacy: clear chat history (new session)
        vscode.commands.registerCommand('squish.clearHistory', () => {
            chatPanel?.newSession();
        }),

        // Start Squish server
        vscode.commands.registerCommand('squish.startServer', async () => {
            const serverCfg = vscode.workspace.getConfiguration('squish');
            const model: string = serverCfg.get('model', '7b');
            await serverManager.start(model);
        }),

        // Stop Squish server
        vscode.commands.registerCommand('squish.stopServer', async () => {
            await serverManager.stop();
        }),

        // Select model from running server
        vscode.commands.registerCommand('squish.selectModel', async () => {
            const modelCfg = vscode.workspace.getConfiguration('squish');
            const host: string = modelCfg.get('host', '127.0.0.1');
            const port: number = modelCfg.get('port', 11435);
            const apiKey: string = modelCfg.get('apiKey', 'squish');
            const client = new SquishClient(host, port, apiKey);
            try {
                const models = await client.models();
                const pick = await vscode.window.showQuickPick(models, {
                    placeHolder: 'Select a Squish model',
                });
                if (pick) {
                    await modelCfg.update('model', pick, vscode.ConfigurationTarget.Global);
                    vscode.window.showInformationMessage(`Squish model set to: ${pick}`);
                }
            } catch {
                vscode.window.showErrorMessage('Could not fetch models — is the server running?');
            }
        }),

        // ── Code-action commands (all call chatPanel.startAgentTask) ──────

        vscode.commands.registerCommand('squish.explainSelection', async () => {
            const ctx = await contextCollector.collect();
            const sel = ctx.selection;
            if (!sel) {
                vscode.window.showWarningMessage('Select some code first.');
                return;
            }
            chatPanel?.startAgentTask(
                `Explain the following code:\n\`\`\`\n${sel}\n\`\`\``,
                `File: ${ctx.activeFile ?? 'unknown'}, Language: ${ctx.language ?? 'unknown'}`,
            );
            vscode.commands.executeCommand('squish.chatView.focus');
        }),

        vscode.commands.registerCommand('squish.fixDiagnostic', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showWarningMessage('No active editor.');
                return;
            }
            const diagnostics = vscode.languages.getDiagnostics(editor.document.uri);
            if (diagnostics.length === 0) {
                vscode.window.showInformationMessage('No diagnostics in the current file.');
                return;
            }
            const diag = diagnostics[0];
            const lineText = editor.document.lineAt(diag.range.start.line).text.trim();
            chatPanel?.startAgentTask(
                `Fix this diagnostic in ${editor.document.fileName}:\n\n` +
                `Error (line ${diag.range.start.line + 1}): ${diag.message}\n` +
                `Code: \`${lineText}\`\n\n` +
                `Read the file, understand the context, and apply the fix.`,
                `File: ${editor.document.fileName}`,
            );
            vscode.commands.executeCommand('squish.chatView.focus');
        }),

        vscode.commands.registerCommand('squish.refactorSelection', async () => {
            const ctx = await contextCollector.collect();
            const sel = ctx.selection;
            if (!sel) {
                vscode.window.showWarningMessage('Select some code first.');
                return;
            }
            chatPanel?.startAgentTask(
                `Refactor the following code for clarity and idiomatic style:\n\`\`\`\n${sel}\n\`\`\`\n\n` +
                `Apply the change directly to the file.`,
                `File: ${ctx.activeFile ?? 'unknown'}, Language: ${ctx.language ?? 'unknown'}`,
            );
            vscode.commands.executeCommand('squish.chatView.focus');
        }),

        vscode.commands.registerCommand('squish.documentFunction', async () => {
            const ctx = await contextCollector.collect();
            const sel = ctx.selection;
            if (!sel) {
                vscode.window.showWarningMessage('Select a function to document.');
                return;
            }
            chatPanel?.startAgentTask(
                `Add a complete docstring/doc-comment to this ${ctx.language ?? 'unknown'} function:\n\`\`\`\n${sel}\n\`\`\`\n\n` +
                `Apply the change directly to the file.`,
                `File: ${ctx.activeFile ?? 'unknown'}`,
            );
            vscode.commands.executeCommand('squish.chatView.focus');
        }),

        vscode.commands.registerCommand('squish.generateTests', async () => {
            const ctx = await contextCollector.collect();
            const sel = ctx.selection;
            if (!sel) {
                vscode.window.showWarningMessage('Select a function to test.');
                return;
            }
            chatPanel?.startAgentTask(
                `Generate comprehensive unit tests for this ${ctx.language ?? 'unknown'} function:\n\`\`\`\n${sel}\n\`\`\`\n\n` +
                `Create a test file using the project's existing test conventions.`,
                `File: ${ctx.activeFile ?? 'unknown'}`,
            );
            vscode.commands.executeCommand('squish.chatView.focus');
        }),

        // Open the current file in the editor panel (no-op; handled by VS Code)
        vscode.commands.registerCommand('squish.openInPanel', () => {
            vscode.commands.executeCommand('squish.chatView.focus');
        }),
    );

    // ── Health poll ───────────────────────────────────────────────────────
    _startHealthPoll(context);

    // ── Auto-start ────────────────────────────────────────────────────────
    if (cfg.get<boolean>('autoStart', false)) {
        const model: string = cfg.get('model', '7b');
        serverManager.start(model).catch(() => {/* already running is fine */});
    }
}

export function deactivate(): void {
    if (pollTimer) {
        clearInterval(pollTimer);
    }
}

// ── Internal ──────────────────────────────────────────────────────────────

function _startHealthPoll(context: vscode.ExtensionContext): void {
    const _poll = async () => {
        const cfg = vscode.workspace.getConfiguration('squish');
        const host: string = cfg.get('host', '127.0.0.1');
        const port: number = cfg.get('port', 11435);
        const apiKey: string = cfg.get('apiKey', 'squish');

        const portUp = await serverManager.portOpen(host, port);
        if (!portUp) {
            statusBar.text = '$(error) squish: offline';
            statusBar.backgroundColor = new vscode.ThemeColor('statusBarItem.errorBackground');
            return;
        }

        const client = new SquishClient(host, port, apiKey);
        try {
            const info = await client.health();
            if (info.loaded) {
                statusBar.text = `$(hubot) squish: ${info.model ?? 'ready'} | ${info.tps?.toFixed(1) ?? '—'} tok/s`;
                statusBar.backgroundColor = undefined;
            } else {
                statusBar.text = '$(loading~spin) squish: loading\u2026';
                statusBar.backgroundColor = undefined;
            }
        } catch (err: unknown) {
            const code = (err as NodeJS.ErrnoException)?.code ?? (err as Error)?.message ?? '';
            if (code.includes('ETIMEDOUT') || code.includes('timed out')) {
                statusBar.text = '$(sync~spin) squish: busy\u2026';
                statusBar.backgroundColor = undefined;
            } else {
                statusBar.text = '$(error) squish: offline';
                statusBar.backgroundColor = new vscode.ThemeColor('statusBarItem.errorBackground');
            }
        }
    };

    _poll();
    pollTimer = setInterval(_poll, 8000);
    context.subscriptions.push({ dispose: () => clearInterval(pollTimer!) });
}
