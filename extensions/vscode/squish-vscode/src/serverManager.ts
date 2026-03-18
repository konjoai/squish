/**
 * squish-vscode/src/serverManager.ts
 *
 * Spawn and stop the squish daemon process.
 * Checks port liveness before starting (avoids double-start).
 */
import * as vscode from 'vscode';
import * as child_process from 'child_process';
import * as net from 'net';

export class ServerManager {
    private _proc: child_process.ChildProcess | undefined;

    constructor(private readonly _context: vscode.ExtensionContext) {
        _context.subscriptions.push({ dispose: () => this.stop() });
    }

    // ── Public API ────────────────────────────────────────────────────────

    async start(model: string): Promise<void> {
        const cfg = vscode.workspace.getConfiguration('squish');
        const host: string = cfg.get('host', '127.0.0.1');
        const port: number = cfg.get('port', 11435);
        const thinkingBudget: number = cfg.get('thinkingBudget', 0);

        if (await this.portOpen(host, port)) {
            vscode.window.showInformationMessage(
                `Squish server already running on ${host}:${port}.`,
            );
            return;
        }

        // Find `squish` binary — prefer the one on PATH, fallback to pip
        const squishBin = this._findSquishBin();
        if (!squishBin) {
            const install = await vscode.window.showErrorMessage(
                'squish binary not found. Install with: pip install squish',
                'Copy command',
            );
            if (install) {
                await vscode.env.clipboard.writeText('pip install squish');
            }
            return;
        }

        const args = [
            'run', model,
            '--host', host,
            '--port', String(port),
            '--thinking-budget', String(thinkingBudget),
        ];

        this._proc = child_process.spawn(squishBin, args, {
            stdio: ['ignore', 'pipe', 'pipe'],
            detached: false,
        });

        this._proc.stdout?.on('data', (data: Buffer) => {
            console.log('[squish]', data.toString().trim());
        });
        this._proc.stderr?.on('data', (data: Buffer) => {
            console.error('[squish]', data.toString().trim());
        });
        this._proc.on('exit', (code) => {
            if (code && code !== 0) {
                vscode.window.showWarningMessage(
                    `Squish server exited with code ${code}.`,
                );
            }
            this._proc = undefined;
        });

        vscode.window.showInformationMessage(
            `Starting Squish server (model: ${model}) on ${host}:${port}…`,
        );
    }

    async stop(): Promise<void> {
        if (this._proc) {
            this._proc.kill('SIGTERM');
            this._proc = undefined;
            vscode.window.showInformationMessage('Squish server stopped.');
        }
    }

    isRunning(): boolean {
        return this._proc != null && !this._proc.killed;
    }

    // ── Internal ──────────────────────────────────────────────────────────

    private _findSquishBin(): string | undefined {
        const candidates = ['squish', 'squish3'];
        for (const c of candidates) {
            try {
                child_process.execSync(`which ${c}`, { stdio: 'ignore' });
                return c;
            } catch {
                // not found
            }
        }
        return undefined;
    }

    portOpen(host: string, port: number): Promise<boolean> {
        return new Promise((resolve) => {
            const sock = new net.Socket();
            sock.setTimeout(1000);
            sock.once('connect', () => { sock.destroy(); resolve(true); });
            sock.once('error', () => { sock.destroy(); resolve(false); });
            sock.once('timeout', () => { sock.destroy(); resolve(false); });
            sock.connect(port, host);
        });
    }
}
