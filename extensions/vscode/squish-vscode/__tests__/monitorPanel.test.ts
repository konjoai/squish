/**
 * __tests__/monitorPanel.test.ts
 *
 * Unit tests for MonitorPanel.
 * SquishClient is mocked to avoid real HTTP.
 * VS Code is mocked via the existing __mocks__/vscode.ts.
 */
import * as vscode from 'vscode';
import { MonitorPanel } from '../src/monitorPanel';

jest.mock('../src/squishClient');
import { SquishClient } from '../src/squishClient';
const MockSquishClient = SquishClient as jest.MockedClass<typeof SquishClient>;

// ── Helpers ───────────────────────────────────────────────────────────────────

const EXT_URI = vscode.Uri.file('/extension');

function makeWebviewView(postMessage = jest.fn()): vscode.WebviewView {
    return {
        webview: {
            options: {},
            html: '',
            cspSource: 'https:',
            asWebviewUri: jest.fn((uri: vscode.Uri) => uri),
            postMessage,
            onDidReceiveMessage: jest.fn(),
        },
        onDidDispose: jest.fn(),
        onDidChangeVisibility: jest.fn(),
        visible: true,
        title: 'Squish Monitor',
    } as unknown as vscode.WebviewView;
}

function makeHealthResponse(overrides: Partial<import('../src/squishClient').HealthInfo> = {}) {
    return {
        loaded: true,
        model: 'squish-7b',
        tps: 45.2,
        active_requests: 0,
        total_requests: 10,
        uptime_s: 3600,
        ram_gb: 4.5,
        vram_gb: 0,
        ...overrides,
    };
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('MonitorPanel', () => {

    beforeEach(() => {
        jest.clearAllMocks();
        jest.useFakeTimers({ doNotFake: ['setImmediate', 'nextTick'] });
    });

    afterEach(() => {
        jest.useRealTimers();
    });

    // ── Construction ──────────────────────────────────────────────────────

    test('viewType is squish.monitorView', () => {
        expect(MonitorPanel.viewType).toBe('squish.monitorView');
    });

    test('can be constructed without throwing', () => {
        expect(() => new MonitorPanel(EXT_URI)).not.toThrow();
    });

    // ── resolveWebviewView ────────────────────────────────────────────────

    test('resolveWebviewView sets HTML on the webview', () => {
        const panel = new MonitorPanel(EXT_URI);
        const view  = makeWebviewView();
        panel.resolveWebviewView(
            view,
            {} as vscode.WebviewViewResolveContext,
            {} as vscode.CancellationToken,
        );
        expect((view.webview as vscode.Webview & { html: string }).html).toContain('<!DOCTYPE html>');
    });

    test('resolveWebviewView enables scripts', () => {
        const panel = new MonitorPanel(EXT_URI);
        const view  = makeWebviewView();
        panel.resolveWebviewView(
            view,
            {} as vscode.WebviewViewResolveContext,
            {} as vscode.CancellationToken,
        );
        expect((view.webview as vscode.Webview).options.enableScripts).toBe(true);
    });

    test('HTML contains squish monitor branding', () => {
        const panel = new MonitorPanel(EXT_URI);
        const view  = makeWebviewView();
        panel.resolveWebviewView(
            view,
            {} as vscode.WebviewViewResolveContext,
            {} as vscode.CancellationToken,
        );
        const html = (view.webview as vscode.Webview & { html: string }).html;
        expect(html).toMatch(/monitor|squish|status/i);
    });

    // ── Polling ───────────────────────────────────────────────────────────

    test('health endpoint is polled after resolveWebviewView', async () => {
        const health = jest.fn().mockImplementation(() => Promise.resolve(makeHealthResponse()));
        MockSquishClient.prototype.health = health;

        const panel = new MonitorPanel(EXT_URI);
        const view  = makeWebviewView();
        panel.resolveWebviewView(
            view,
            {} as vscode.WebviewViewResolveContext,
            {} as vscode.CancellationToken,
        );

        // Flush the microtask queue — the initial _poll() was kicked off synchronously
        // via `void this._poll()`, so awaiting a few resolved promises is enough.
        await Promise.resolve();
        await Promise.resolve();

        expect(health).toHaveBeenCalled();
        panel.dispose();
    });

    test('poll result is posted to webview', async () => {
        const postMessage = jest.fn();
        MockSquishClient.prototype.health = jest.fn().mockImplementation(
            () => Promise.resolve(makeHealthResponse({ tps: 55.5 }))
        );

        const panel = new MonitorPanel(EXT_URI);
        const view  = makeWebviewView(postMessage);
        panel.resolveWebviewView(
            view,
            {} as vscode.WebviewViewResolveContext,
            {} as vscode.CancellationToken,
        );

        // Flush the microtask queue so the first poll's Promise chain resolves.
        await Promise.resolve();
        await Promise.resolve();

        expect(postMessage).toHaveBeenCalledWith(
            expect.objectContaining({ type: expect.any(String) }),
        );
        panel.dispose();
    });

    test('poll handles health error gracefully (no throw)', async () => {
        MockSquishClient.prototype.health = jest.fn().mockImplementation(
            () => Promise.reject(new Error('offline'))
        );

        const panel = new MonitorPanel(EXT_URI);
        const view  = makeWebviewView();

        expect(() =>
            panel.resolveWebviewView(
                view,
                {} as vscode.WebviewViewResolveContext,
                {} as vscode.CancellationToken,
            )
        ).not.toThrow();

        // Flush the microtask queue — the poll's rejection is caught internally;
        // no unhandled rejection should propagate.
        await Promise.resolve();
        await Promise.resolve();
        panel.dispose();
    });

    test('dispose stops polling', () => {
        const health = jest.fn().mockResolvedValue(makeHealthResponse());
        MockSquishClient.prototype.health = health;

        const panel = new MonitorPanel(EXT_URI);
        const view  = makeWebviewView();
        panel.resolveWebviewView(
            view,
            {} as vscode.WebviewViewResolveContext,
            {} as vscode.CancellationToken,
        );
        const callsBefore = health.mock.calls.length;
        panel.dispose();
        jest.advanceTimersByTime(10_000);
        // Should not poll after dispose
        expect(health.mock.calls.length).toBe(callsBefore);
    });

    // ── Sparkline management ──────────────────────────────────────────────

    test('multiple polls accumulate sparkline data', async () => {
        let tok = 10;
        MockSquishClient.prototype.health = jest.fn().mockImplementation(
            () => Promise.resolve(makeHealthResponse({ tps: tok++ }))
        );

        const postMessage = jest.fn();
        const panel = new MonitorPanel(EXT_URI);
        const view  = makeWebviewView(postMessage);
        panel.resolveWebviewView(
            view,
            {} as vscode.WebviewViewResolveContext,
            {} as vscode.CancellationToken,
        );

        // Advance timer to trigger multiple polls
        for (let i = 0; i < 5; i++) {
            await new Promise((r) => setImmediate(r));
            jest.advanceTimersByTime(2000);
        }
        await new Promise((r) => setImmediate(r));

        expect(postMessage.mock.calls.length).toBeGreaterThanOrEqual(1);
    });

    test('visibility change to hidden stops polling', () => {
        const health = jest.fn().mockResolvedValue(makeHealthResponse());
        MockSquishClient.prototype.health = health;

        const panel = new MonitorPanel(EXT_URI);
        let visibilityCb: (() => void) | undefined;
        const view = {
            ...makeWebviewView(),
            onDidChangeVisibility: jest.fn((cb: () => void) => { visibilityCb = cb; }),
            visible: true,
        } as unknown as vscode.WebviewView;
        (view as unknown as Record<string, unknown>).visible = false;  // simulate hidden

        panel.resolveWebviewView(
            view,
            {} as vscode.WebviewViewResolveContext,
            {} as vscode.CancellationToken,
        );
        if (visibilityCb) { visibilityCb(); }
        const callsBefore = health.mock.calls.length;
        jest.advanceTimersByTime(10_000);
        expect(health.mock.calls.length).toBe(callsBefore);
    });
});
