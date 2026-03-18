/**
 * __tests__/serverManager.test.ts
 *
 * Unit tests for ServerManager.
 * Mocks vscode (via __mocks__/vscode.ts), child_process, net, and fs.
 */
import * as child_process from 'child_process';
import * as net from 'net';
import * as fs from 'fs';
import * as os from 'os';
import * as vscode from 'vscode';

jest.mock('child_process');
jest.mock('net');
jest.mock('fs');
jest.mock('os');

const mockCp = child_process as jest.Mocked<typeof child_process>;
const mockNet = net as jest.Mocked<typeof net>;
const mockFs = fs as jest.Mocked<typeof fs>;
const mockOs = os as jest.Mocked<typeof os>;

// Import AFTER mocks are set up
import { ServerManager } from '../src/serverManager';

// ── Helpers ───────────────────────────────────────────────────────────────────

function makeContext(): vscode.ExtensionContext {
    return {
        subscriptions: [],
        extensionUri: vscode.Uri.file('/ext'),
    } as unknown as vscode.ExtensionContext;
}

function stubPortClosed(): void {
    const sock = Object.assign(
        Object.create(require('events').EventEmitter.prototype),
        { setTimeout: jest.fn(), destroy: jest.fn(), connect: jest.fn() },
    );
    mockNet.Socket.mockImplementation(() => {
        // Emit 'error' immediately → port not open
        setTimeout(() => sock.emit('error', new Error('ECONNREFUSED')), 0);
        return sock as unknown as net.Socket;
    });
}

function stubPortOpen(): void {
    const sock = Object.assign(
        Object.create(require('events').EventEmitter.prototype),
        { setTimeout: jest.fn(), destroy: jest.fn(), connect: jest.fn() },
    );
    mockNet.Socket.mockImplementation(() => {
        setTimeout(() => sock.emit('connect'), 0);
        return sock as unknown as net.Socket;
    });
}

function stubExecSync(found: boolean): void {
    if (found) {
        mockCp.execSync.mockReturnValue(Buffer.from('/usr/local/bin/squish\n'));
    } else {
        mockCp.execSync.mockImplementation(() => { throw new Error('not found'); });
    }
}

function makeSpawnMock(): {
    proc: child_process.ChildProcess;
    stdout: { on: jest.Mock };
    stderr: { on: jest.Mock };
    onExit: (code: number) => void;
} {
    const stdout = { on: jest.fn() };
    const stderr = { on: jest.fn() };
    let exitCb: ((code: number) => void) | undefined;
    const proc = {
        stdout,
        stderr,
        killed: false,
        kill: jest.fn((sig: string) => { proc.killed = true; }),
        on: jest.fn((event: string, cb: unknown) => {
            if (event === 'exit') exitCb = cb as (code: number) => void;
        }),
    };
    mockCp.spawn.mockReturnValue(proc as unknown as child_process.ChildProcess);
    return {
        proc: proc as unknown as child_process.ChildProcess,
        stdout,
        stderr,
        onExit: (code: number) => exitCb?.(code),
    };
}

// ── Tests ─────────────────────────────────────────────────────────────────────

// Default fs/os stubs — no files exist, homedir is /home/test
function stubFsDefault(): void {
    mockFs.existsSync.mockReturnValue(false);
    mockFs.statSync.mockReturnValue({ isDirectory: () => false } as fs.Stats);
    mockOs.homedir.mockReturnValue('/home/test');
}

describe('ServerManager.start()', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        stubFsDefault();
        (vscode.workspace as unknown as {
            _setConfig: (k: string, v: unknown) => void;
            _setWorkspaceFolders: (f: Array<{ uri: { fsPath: string } }>) => void;
        })._setConfig('venvPath', '');
        (vscode.workspace as unknown as {
            _setWorkspaceFolders: (f: Array<{ uri: { fsPath: string } }>) => void;
        })._setWorkspaceFolders([]);
    });

    test('shows "already running" when port is open', async () => {
        stubPortOpen();
        const mgr = new ServerManager(makeContext());
        await mgr.start('7b');
        expect(vscode.window.showInformationMessage).toHaveBeenCalledWith(
            expect.stringContaining('already running'),
        );
        expect(mockCp.spawn).not.toHaveBeenCalled();
    });

    test('shows error when squish binary not found', async () => {
        stubPortClosed();
        stubExecSync(false);
        const mgr = new ServerManager(makeContext());
        await mgr.start('7b');
        expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
            expect.stringContaining('not found'),
            expect.anything(),
            expect.anything(),
        );
    });

    test('spawns squish run when binary found and port closed', async () => {
        stubPortClosed();
        stubExecSync(true);
        makeSpawnMock();

        const mgr = new ServerManager(makeContext());
        await mgr.start('7b');

        expect(mockCp.spawn).toHaveBeenCalledWith(
            expect.stringMatching(/squish/),
            expect.arrayContaining(['run', '7b']),
            expect.anything(),
        );
        expect(vscode.window.showInformationMessage).toHaveBeenCalledWith(
            expect.stringContaining('Starting'),
        );
    });

    test('passes --thinking-budget from config', async () => {
        stubPortClosed();
        stubExecSync(true);
        makeSpawnMock();
        (vscode.workspace as unknown as { _setConfig: (k: string, v: unknown) => void })
            ._setConfig('thinkingBudget', 0);

        const mgr = new ServerManager(makeContext());
        await mgr.start('qwen3:8b');

        expect(mockCp.spawn).toHaveBeenCalledWith(
            expect.stringMatching(/squish/),
            expect.arrayContaining(['--thinking-budget', '0']),
            expect.anything(),
        );
    });

    test('isRunning() reflects active process', async () => {
        stubPortClosed();
        stubExecSync(true);
        const { proc } = makeSpawnMock();

        const mgr = new ServerManager(makeContext());
        await mgr.start('14b');
        expect(mgr.isRunning()).toBe(true);
    });

    test('wires stdout/stderr listeners', async () => {
        stubPortClosed();
        stubExecSync(true);
        const { stdout, stderr } = makeSpawnMock();

        const mgr = new ServerManager(makeContext());
        await mgr.start('7b');

        expect(stdout.on).toHaveBeenCalledWith('data', expect.any(Function));
        expect(stderr.on).toHaveBeenCalledWith('data', expect.any(Function));
    });

    test('shows warning on non-zero exit', async () => {
        stubPortClosed();
        stubExecSync(true);
        const { onExit } = makeSpawnMock();

        const mgr = new ServerManager(makeContext());
        await mgr.start('7b');
        onExit(1);

        expect(vscode.window.showWarningMessage).toHaveBeenCalledWith(
            expect.stringContaining('exited with code 1'),
        );
    });
});

describe('ServerManager.stop()', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        stubFsDefault();
        (vscode.workspace as unknown as {
            _setConfig: (k: string, v: unknown) => void;
            _setWorkspaceFolders: (f: Array<{ uri: { fsPath: string } }>) => void;
        })._setConfig('venvPath', '');
        (vscode.workspace as unknown as {
            _setWorkspaceFolders: (f: Array<{ uri: { fsPath: string } }>) => void;
        })._setWorkspaceFolders([]);
    });

    test('kills process and shows message', async () => {
        stubPortClosed();
        stubExecSync(true);
        const { proc } = makeSpawnMock();

        const mgr = new ServerManager(makeContext());
        await mgr.start('7b');
        await mgr.stop();

        expect(proc.kill).toHaveBeenCalledWith('SIGTERM');
        expect(vscode.window.showInformationMessage).toHaveBeenCalledWith(
            expect.stringContaining('stopped'),
        );
    });

    test('no-op when not running', async () => {
        const mgr = new ServerManager(makeContext());
        await expect(mgr.stop()).resolves.toBeUndefined();
        expect(vscode.window.showInformationMessage).not.toHaveBeenCalledWith(
            expect.stringContaining('stopped'),
        );
    });

    test('isRunning() false after stop', async () => {
        stubPortClosed();
        stubExecSync(true);
        makeSpawnMock();

        const mgr = new ServerManager(makeContext());
        await mgr.start('7b');
        await mgr.stop();
        expect(mgr.isRunning()).toBe(false);
    });
});

// ── _findSquishBin() ──────────────────────────────────────────────────────────

type WorkspaceMock = {
    _setConfig: (k: string, v: unknown) => void;
    _setWorkspaceFolders: (f: Array<{ uri: { fsPath: string } }>) => void;
};

describe('_findSquishBin() — tier 1: squish.venvPath setting', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        stubFsDefault();
        mockCp.execSync.mockImplementation(() => { throw new Error('not found'); });
        (vscode.workspace as unknown as WorkspaceMock)._setWorkspaceFolders([]);
    });

    test('uses path directly when venvPath is an executable file', async () => {
        stubPortClosed();
        makeSpawnMock();
        const binPath = '/custom/bin/squish';
        (vscode.workspace as unknown as WorkspaceMock)._setConfig('venvPath', binPath);
        mockFs.existsSync.mockImplementation((p) => p === binPath);
        mockFs.statSync.mockImplementation(() => ({ isDirectory: () => false } as fs.Stats));

        const mgr = new ServerManager(makeContext());
        await mgr.start('7b');

        expect(mockCp.spawn).toHaveBeenCalledWith(
            binPath,
            expect.arrayContaining(['run', '7b']),
            expect.anything(),
        );
    });

    test('resolves bin/squish inside venvPath directory', async () => {
        stubPortClosed();
        makeSpawnMock();
        const venvDir = '/Users/wscholl/squish/.venv';
        const expectedBin = '/Users/wscholl/squish/.venv/bin/squish';
        (vscode.workspace as unknown as WorkspaceMock)._setConfig('venvPath', venvDir);
        mockFs.existsSync.mockImplementation((p) => p === venvDir || p === expectedBin);
        mockFs.statSync.mockImplementation((p) => ({
            isDirectory: () => p === venvDir,
        } as fs.Stats));

        const mgr = new ServerManager(makeContext());
        await mgr.start('7b');

        expect(mockCp.spawn).toHaveBeenCalledWith(
            expectedBin,
            expect.arrayContaining(['run', '7b']),
            expect.anything(),
        );
    });

    test('falls through to PATH when venvPath file does not exist', async () => {
        stubPortClosed();
        stubExecSync(true);
        makeSpawnMock();
        (vscode.workspace as unknown as WorkspaceMock)._setConfig('venvPath', '/nonexistent/squish');
        mockFs.existsSync.mockReturnValue(false);

        const mgr = new ServerManager(makeContext());
        await mgr.start('7b');

        // Should fall through to which/squish found via execSync
        expect(mockCp.spawn).toHaveBeenCalled();
        expect(vscode.window.showErrorMessage).not.toHaveBeenCalled();
    });
});

describe('_findSquishBin() — tier 3: workspace venv paths', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        stubFsDefault();
        mockCp.execSync.mockImplementation(() => { throw new Error('not found'); });
        (vscode.workspace as unknown as WorkspaceMock)._setConfig('venvPath', '');
    });

    test('finds .venv/bin/squish in workspace folder', async () => {
        stubPortClosed();
        makeSpawnMock();
        const wsRoot = '/Users/wscholl/squish';
        const expectedBin = `${wsRoot}/.venv/bin/squish`;
        (vscode.workspace as unknown as WorkspaceMock)._setWorkspaceFolders([
            { uri: { fsPath: wsRoot } },
        ]);
        mockFs.existsSync.mockImplementation((p) => p === expectedBin);

        const mgr = new ServerManager(makeContext());
        await mgr.start('7b');

        expect(mockCp.spawn).toHaveBeenCalledWith(
            expectedBin,
            expect.arrayContaining(['run', '7b']),
            expect.anything(),
        );
    });

    test('finds venv/bin/squish when .venv absent', async () => {
        stubPortClosed();
        makeSpawnMock();
        const wsRoot = '/project';
        const expectedBin = `${wsRoot}/venv/bin/squish`;
        (vscode.workspace as unknown as WorkspaceMock)._setWorkspaceFolders([
            { uri: { fsPath: wsRoot } },
        ]);
        mockFs.existsSync.mockImplementation((p) => p === expectedBin);

        const mgr = new ServerManager(makeContext());
        await mgr.start('7b');

        expect(mockCp.spawn).toHaveBeenCalledWith(
            expectedBin,
            expect.arrayContaining(['run', '7b']),
            expect.anything(),
        );
    });

    test('skips workspace tier when no workspace folders open', async () => {
        stubPortClosed();
        (vscode.workspace as unknown as WorkspaceMock)._setWorkspaceFolders([]);
        mockFs.existsSync.mockReturnValue(false);

        const mgr = new ServerManager(makeContext());
        await mgr.start('7b');

        expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
            expect.stringContaining('not found'),
            expect.anything(),
            expect.anything(),
        );
    });
});

describe('_findSquishBin() — tier 4: global install paths', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        stubFsDefault();
        mockCp.execSync.mockImplementation(() => { throw new Error('not found'); });
        (vscode.workspace as unknown as WorkspaceMock)._setConfig('venvPath', '');
        (vscode.workspace as unknown as WorkspaceMock)._setWorkspaceFolders([]);
        mockOs.homedir.mockReturnValue('/home/user');
    });

    test('finds pip --user install at ~/.local/bin/squish', async () => {
        stubPortClosed();
        makeSpawnMock();
        const expectedBin = '/home/user/.local/bin/squish';
        mockFs.existsSync.mockImplementation((p) => p === expectedBin);

        const mgr = new ServerManager(makeContext());
        await mgr.start('7b');

        expect(mockCp.spawn).toHaveBeenCalledWith(
            expectedBin,
            expect.arrayContaining(['run', '7b']),
            expect.anything(),
        );
    });

    test('finds pipx install at ~/.local/pipx/venvs/squish/bin/squish', async () => {
        stubPortClosed();
        makeSpawnMock();
        const expectedBin = '/home/user/.local/pipx/venvs/squish/bin/squish';
        mockFs.existsSync.mockImplementation((p) => p === expectedBin);

        const mgr = new ServerManager(makeContext());
        await mgr.start('7b');

        expect(mockCp.spawn).toHaveBeenCalledWith(
            expectedBin,
            expect.arrayContaining(['run', '7b']),
            expect.anything(),
        );
    });

    test('finds homebrew install at /opt/homebrew/bin/squish', async () => {
        stubPortClosed();
        makeSpawnMock();
        const expectedBin = '/opt/homebrew/bin/squish';
        mockFs.existsSync.mockImplementation((p) => p === expectedBin);

        const mgr = new ServerManager(makeContext());
        await mgr.start('7b');

        expect(mockCp.spawn).toHaveBeenCalledWith(
            expectedBin,
            expect.arrayContaining(['run', '7b']),
            expect.anything(),
        );
    });

    test('shows error with settings action when all tiers exhausted', async () => {
        stubPortClosed();
        mockFs.existsSync.mockReturnValue(false);

        const mgr = new ServerManager(makeContext());
        await mgr.start('7b');

        expect(vscode.window.showErrorMessage).toHaveBeenCalledWith(
            expect.stringContaining('not found'),
            'Open Settings',
            'Copy pip command',
        );
        expect(mockCp.spawn).not.toHaveBeenCalled();
    });
});
