/**
 * __tests__/serverManager.test.ts
 *
 * Unit tests for ServerManager.
 * Mocks vscode (via __mocks__/vscode.ts), child_process, and net.
 */
import * as child_process from 'child_process';
import * as net from 'net';
import * as vscode from 'vscode';

jest.mock('child_process');
jest.mock('net');

const mockCp = child_process as jest.Mocked<typeof child_process>;
const mockNet = net as jest.Mocked<typeof net>;

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

describe('ServerManager.start()', () => {
    beforeEach(() => {
        jest.clearAllMocks();
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
    beforeEach(() => { jest.clearAllMocks(); });

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
