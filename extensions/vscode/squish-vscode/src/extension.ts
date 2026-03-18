/**
 * squish-vscode/src/extension.ts
 *
 * Extension entry point.  Registers commands, creates the status bar item,
 * and wires up the sidebar chat WebviewView.
 */
import * as vscode from 'vscode';
import { SquishClient } from './squishClient';
import { ChatPanel } from './chatPanel';
import { ServerManager } from './serverManager';

let statusBar: vscode.StatusBarItem;
let chatPanel: ChatPanel | undefined;
let serverManager: ServerManager;
let pollTimer: NodeJS.Timeout | undefined;

export function activate(context: vscode.ExtensionContext): void {
    // ── Server manager ────────────────────────────────────────────────────
    serverManager = new ServerManager(context);

    // ── Status bar ────────────────────────────────────────────────────────
    statusBar = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Right,
        100,
    );
    statusBar.command = 'squish.openChat';
    statusBar.text = '$(hubot) squish: offline';
    statusBar.tooltip = 'Squish local AI — click to open chat';
    statusBar.show();
    context.subscriptions.push(statusBar);

    // ── Chat panel (sidebar WebviewView) ──────────────────────────────────
    chatPanel = new ChatPanel(context.extensionUri);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('squish.chatView', chatPanel),
    );

    // ── Commands ──────────────────────────────────────────────────────────
    context.subscriptions.push(
        vscode.commands.registerCommand('squish.openChat', () => {
            vscode.commands.executeCommand('squish.chatView.focus');
        }),

        vscode.commands.registerCommand('squish.startServer', async () => {
            const cfg = vscode.workspace.getConfiguration('squish');
            const model: string = cfg.get('model', '7b');
            await serverManager.start(model);
        }),

        vscode.commands.registerCommand('squish.stopServer', async () => {
            await serverManager.stop();
        }),

        vscode.commands.registerCommand('squish.selectModel', async () => {
            const cfg = vscode.workspace.getConfiguration('squish');
            const host: string = cfg.get('host', '127.0.0.1');
            const port: number = cfg.get('port', 11435);
            const apiKey: string = cfg.get('apiKey', 'squish');
            const client = new SquishClient(host, port, apiKey);
            try {
                const models = await client.models();
                const pick = await vscode.window.showQuickPick(models, {
                    placeHolder: 'Select a Squish model',
                });
                if (pick) {
                    await cfg.update('model', pick, vscode.ConfigurationTarget.Global);
                    vscode.window.showInformationMessage(`Squish model set to: ${pick}`);
                }
            } catch {
                vscode.window.showErrorMessage('Could not fetch models — is the server running?');
            }
        }),

        vscode.commands.registerCommand('squish.clearHistory', () => {
            chatPanel?.clearHistory();
        }),
    );

    // ── Health poll ───────────────────────────────────────────────────────
    _startHealthPoll(context);

    // ── Auto-start ────────────────────────────────────────────────────────
    const cfg = vscode.workspace.getConfiguration('squish');
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

        // Fast TCP check first — if port is closed the server is definitely offline
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
                statusBar.text = '$(loading~spin) squish: loading…';
                statusBar.backgroundColor = undefined;
            }
        } catch (err: unknown) {
            // Port is open but HTTP timed out — server is running but busy with inference
            const code = (err as NodeJS.ErrnoException)?.code ?? (err as Error)?.message ?? '';
            if (code.includes('ETIMEDOUT') || code.includes('timed out')) {
                statusBar.text = '$(sync~spin) squish: busy…';
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
