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
                'Squish binary not found. Set "squish.venvPath" in settings to your venv ' +
                '(e.g. /Users/you/squish/.venv), or install globally with: pip install squish',
                'Open Settings',
                'Copy pip command',
            );
            if (install === 'Open Settings') {
                await vscode.commands.executeCommand('workbench.action.openSettings', 'squish.venvPath');
            } else if (install === 'Copy pip command') {
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
        const cfg = vscode.workspace.getConfiguration('squish');
        const venvPath: string = cfg.get('venvPath', '').trim();

        // 1. User-configured path: can be the binary itself or a venv dir
        if (venvPath) {
            const fs = require('fs') as typeof import('fs');
            const path = require('path') as typeof import('path');
            // If it points directly to an executable file, use it
            if (fs.existsSync(venvPath) && !fs.statSync(venvPath).isDirectory()) {
                return venvPath;
            }
            // If it points to a directory, look for bin/squish inside
            const binInVenv = path.join(venvPath, 'bin', 'squish');
            if (fs.existsSync(binInVenv)) {
                return binInVenv;
            }
        }

        // 2. Binary on PATH — works when VS Code inherits the user's PATH
        //    (e.g. launched from a terminal that has the venv activated)
        const candidates = ['squish', 'squish3'];
        for (const c of candidates) {
            try {
                child_process.execSync(`which ${c}`, { stdio: 'ignore' });
                return c;
            } catch {
                // not on PATH
            }
        }

        // 3. Well-known venv locations relative to each open workspace folder
        //    Covers the typical `git clone squish && cd squish && python -m venv .venv`
        const fs = require('fs') as typeof import('fs');
        const path = require('path') as typeof import('path');
        const os = require('os') as typeof import('os');
        const workspaceFolders = vscode.workspace.workspaceFolders ?? [];
        const knownRelative = ['.venv/bin/squish', 'venv/bin/squish', 'env/bin/squish'];
        for (const folder of workspaceFolders) {
            for (const rel of knownRelative) {
                const abs = path.join(folder.uri.fsPath, rel);
                if (fs.existsSync(abs)) {
                    return abs;
                }
            }
        }

        // 4. Common global install locations: pipx, user pip, homebrew
        const homeDir = os.homedir();
        const globalPaths = [
            path.join(homeDir, '.local', 'bin', 'squish'),      // pip install --user
            path.join(homeDir, '.local', 'pipx', 'venvs', 'squish', 'bin', 'squish'), // pipx
            '/opt/homebrew/bin/squish',                          // homebrew arm64
            '/usr/local/bin/squish',                             // homebrew x86 / pip
        ];
        for (const p of globalPaths) {
            if (fs.existsSync(p)) {
                return p;
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
