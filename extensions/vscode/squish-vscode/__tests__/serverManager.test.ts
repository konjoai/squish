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

// ── Windows cross-platform tests ──────────────────────────────────────────────
// Note: path.join() uses POSIX separators on macOS/Linux regardless of platform
// mock, so these tests use forward-slash paths that are consistent with what
// path.join() produces at runtime on this machine.

describe('_findSquishBin() — Windows tier 1: venvPath Scripts layout', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        stubFsDefault();
        mockCp.execSync.mockImplementation(() => { throw new Error('not found'); });
        mockOs.platform.mockReturnValue('win32');
        (vscode.workspace as unknown as WorkspaceMock)._setWorkspaceFolders([]);
    });

    test('resolves Scripts/squish.exe inside venvPath on Windows', async () => {
        stubPortClosed();
        makeSpawnMock();
        const venvDir = '/fake/win/venv';
        // path.join(venvDir, 'Scripts', 'squish.exe') on POSIX → /fake/win/venv/Scripts/squish.exe
        const expectedBin = `${venvDir}/Scripts/squish.exe`;
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

    test('resolves Scripts/squish.cmd inside venvPath on Windows (.exe absent)', async () => {
        stubPortClosed();
        makeSpawnMock();
        const venvDir = '/fake/win/venv';
        const exeBin  = `${venvDir}/Scripts/squish.exe`;
        const cmdBin  = `${venvDir}/Scripts/squish.cmd`;
        (vscode.workspace as unknown as WorkspaceMock)._setConfig('venvPath', venvDir);
        // .exe does not exist, .cmd does
        mockFs.existsSync.mockImplementation((p) => p === venvDir || p === cmdBin);
        mockFs.statSync.mockImplementation((p) => ({
            isDirectory: () => p === venvDir,
        } as fs.Stats));

        const mgr = new ServerManager(makeContext());
        await mgr.start('7b');

        expect(mockCp.spawn).toHaveBeenCalledWith(
            cmdBin,
            expect.arrayContaining(['run', '7b']),
            expect.anything(),
        );
        // .exe was checked first, then .cmd
        expect(mockFs.existsSync).toHaveBeenCalledWith(exeBin);
        expect(mockFs.existsSync).toHaveBeenCalledWith(cmdBin);
    });
});

describe('_findSquishBin() — Windows tier 2: PATH via where', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        stubFsDefault();
        mockOs.platform.mockReturnValue('win32');
        (vscode.workspace as unknown as WorkspaceMock)._setConfig('venvPath', '');
        (vscode.workspace as unknown as WorkspaceMock)._setWorkspaceFolders([]);
    });

    test('uses "where squish.exe" instead of "which" on Windows', async () => {
        stubPortClosed();
        makeSpawnMock();
        mockCp.execSync.mockImplementation((cmd: string) => {
            if (cmd === 'where squish.exe') { return Buffer.from('C:\\Python312\\Scripts\\squish.exe\n'); }
            throw new Error('not found');
        });

        const mgr = new ServerManager(makeContext());
        await mgr.start('7b');

        expect(mockCp.execSync).toHaveBeenCalledWith('where squish.exe', expect.anything());
        expect(mockCp.spawn).toHaveBeenCalledWith(
            'squish.exe',
            expect.arrayContaining(['run', '7b']),
            expect.anything(),
        );
    });

    test('falls back to squish.cmd when squish.exe not on PATH', async () => {
        stubPortClosed();
        makeSpawnMock();
        mockCp.execSync.mockImplementation((cmd: string) => {
            if (cmd === 'where squish.cmd') { return Buffer.from('C:\\Scripts\\squish.cmd\n'); }
            throw new Error('not found');
        });

        const mgr = new ServerManager(makeContext());
        await mgr.start('7b');

        expect(mockCp.execSync).toHaveBeenCalledWith('where squish.exe', expect.anything());
        expect(mockCp.execSync).toHaveBeenCalledWith('where squish.cmd', expect.anything());
        expect(mockCp.spawn).toHaveBeenCalledWith(
            'squish.cmd',
            expect.arrayContaining(['run', '7b']),
            expect.anything(),
        );
    });
});

describe('_findSquishBin() — Windows tier 3: workspace venv Scripts layout', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        stubFsDefault();
        mockCp.execSync.mockImplementation(() => { throw new Error('not found'); });
        mockOs.platform.mockReturnValue('win32');
        (vscode.workspace as unknown as WorkspaceMock)._setConfig('venvPath', '');
    });

    test('finds .venv/Scripts/squish.exe in workspace on Windows', async () => {
        stubPortClosed();
        makeSpawnMock();
        const wsRoot = '/ws/myproject';
        const expectedBin = `${wsRoot}/.venv/Scripts/squish.exe`;
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

    test('finds venv/Scripts/squish.cmd in workspace (.exe absent)', async () => {
        stubPortClosed();
        makeSpawnMock();
        const wsRoot = '/ws/myproject';
        const expectedBin = `${wsRoot}/venv/Scripts/squish.cmd`;
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
});

describe('_findSquishBin() — Windows tier 4: global install paths', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        stubFsDefault();
        mockCp.execSync.mockImplementation(() => { throw new Error('not found'); });
        mockOs.platform.mockReturnValue('win32');
        mockOs.homedir.mockReturnValue('/home/winuser');
        (vscode.workspace as unknown as WorkspaceMock)._setConfig('venvPath', '');
        (vscode.workspace as unknown as WorkspaceMock)._setWorkspaceFolders([]);
    });

    test('finds pip --user install at AppData/Roaming/Python/Scripts/squish.exe', async () => {
        stubPortClosed();
        makeSpawnMock();
        const expectedBin = '/home/winuser/AppData/Roaming/Python/Scripts/squish.exe';
        mockFs.existsSync.mockImplementation((p) => p === expectedBin);

        const mgr = new ServerManager(makeContext());
        await mgr.start('7b');

        expect(mockCp.spawn).toHaveBeenCalledWith(
            expectedBin,
            expect.arrayContaining(['run', '7b']),
            expect.anything(),
        );
    });

    test('finds pipx install at AppData/Local/pipx/venvs/squish/Scripts/squish.exe', async () => {
        stubPortClosed();
        makeSpawnMock();
        const expectedBin = '/home/winuser/AppData/Local/pipx/venvs/squish/Scripts/squish.exe';
        mockFs.existsSync.mockImplementation((p) => p === expectedBin);

        const mgr = new ServerManager(makeContext());
        await mgr.start('7b');

        expect(mockCp.spawn).toHaveBeenCalledWith(
            expectedBin,
            expect.arrayContaining(['run', '7b']),
            expect.anything(),
        );
    });
});

describe('ServerManager.stop() — Windows', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        stubFsDefault();
        mockOs.platform.mockReturnValue('win32');
        (vscode.workspace as unknown as WorkspaceMock)._setConfig('venvPath', '');
        (vscode.workspace as unknown as WorkspaceMock)._setWorkspaceFolders([]);
    });

    test('calls kill() without signal on Windows', async () => {
        stubPortClosed();
        mockCp.execSync.mockImplementation((cmd: string) => {
            if (cmd === 'where squish.exe') { return Buffer.from('squish.exe\n'); }
            throw new Error('not found');
        });
        const { proc } = makeSpawnMock();

        const mgr = new ServerManager(makeContext());
        await mgr.start('7b');
        await mgr.stop();

        expect(proc.kill).toHaveBeenCalledWith();
        expect(proc.kill).not.toHaveBeenCalledWith('SIGTERM');
    });

    test('spawn includes shell: true on Windows', async () => {
        stubPortClosed();
        mockCp.execSync.mockImplementation((cmd: string) => {
            if (cmd === 'where squish.exe') { return Buffer.from('squish.exe\n'); }
            throw new Error('not found');
        });
        makeSpawnMock();

        const mgr = new ServerManager(makeContext());
        await mgr.start('7b');

        expect(mockCp.spawn).toHaveBeenCalledWith(
            expect.anything(),
            expect.anything(),
            expect.objectContaining({ shell: true }),
        );
    });
});
