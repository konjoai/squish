/**
 * __tests__/chatPanel.test.ts
 *
 * Unit tests for ChatPanel.
 * The WebviewView is fully mocked — no VS Code process needed.
 */
import * as vscode from 'vscode';
import { ChatPanel } from '../src/chatPanel';
import { HistoryManager } from '../src/historyManager';

// Mock squishClient so we can control streaming
jest.mock('../src/squishClient');
import { SquishClient } from '../src/squishClient';
const MockSquishClient = SquishClient as jest.MockedClass<typeof SquishClient>;

// Mock HistoryManager so tests don't touch the filesystem
jest.mock('../src/historyManager');
const MockHistoryManager = HistoryManager as jest.MockedClass<typeof HistoryManager>;

// ── Helpers ───────────────────────────────────────────────────────────────────

function makeHistoryManager(): HistoryManager {
    const mgr = new MockHistoryManager();
    (mgr.createSession as jest.Mock).mockReturnValue({
        id: 'test-session-id',
        title: 'New conversation',
        createdAt: Date.now(),
        updatedAt: Date.now(),
        messages: [],
    });
    (mgr.save as jest.Mock).mockImplementation(() => undefined);
    (mgr.list as jest.Mock).mockReturnValue([]);
    (mgr.load as jest.Mock).mockReturnValue(null);
    (mgr.delete as jest.Mock).mockImplementation(() => undefined);
    (mgr.rename as jest.Mock).mockImplementation(() => undefined);
    return mgr;
}

function makePanel(extUri = vscode.Uri.file('/extension')): ChatPanel {
    return new ChatPanel(extUri, makeHistoryManager());
}

function makeWebviewView(postMessage = jest.fn()): vscode.WebviewView {
    const webviewView = {
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
        title: 'Squish Chat',
        visible: true,
    };
    return webviewView as unknown as vscode.WebviewView;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('ChatPanel', () => {
    const extUri = vscode.Uri.file('/extension');

    beforeEach(() => {
        jest.clearAllMocks();
    });

    test('resolveWebviewView sets HTML and wires message handler', () => {
        const panel = makePanel(extUri);
        const view = makeWebviewView();

        panel.resolveWebviewView(
            view,
            {} as vscode.WebviewViewResolveContext,
            {} as vscode.CancellationToken,
        );

        expect(view.webview.html).toContain('<!DOCTYPE html>');
        expect(view.webview.html).toContain('Squish');
        expect(view.webview.onDidReceiveMessage).toHaveBeenCalled();
    });

    test('clearHistory posts clearHistory message and resets history', () => {
        const postMessage = jest.fn();
        const panel = makePanel(extUri);
        const view = makeWebviewView(postMessage);

        panel.resolveWebviewView(view, {} as vscode.WebviewViewResolveContext, {} as vscode.CancellationToken);
        panel.clearHistory();

        expect(postMessage).toHaveBeenCalledWith({ type: 'clearHistory' });
    });

    test('userMessage triggers streamChat with configured model', () => {
        const postMessage = jest.fn();
        const panel = makePanel(extUri);
        const view = makeWebviewView(postMessage);

        // Capture the message handler
        let messageHandler: ((msg: unknown) => void) | undefined;
        (view.webview.onDidReceiveMessage as jest.Mock).mockImplementation(
            (cb: (msg: unknown) => void) => { messageHandler = cb; }
        );

        // Mock streamChat to immediately call onChunk with done
        MockSquishClient.prototype.streamChat = jest.fn(
            (_msgs, _max, _temp, _model, onChunk, _onError) => {
                onChunk({ delta: 'Hello', done: false, finishReason: null });
                onChunk({ delta: '', done: true, finishReason: 'stop' });
            }
        );

        panel.resolveWebviewView(view, {} as vscode.WebviewViewResolveContext, {} as vscode.CancellationToken);
        messageHandler?.({ type: 'userMessage', text: 'test' });

        // Verify streamChat was called
        expect(MockSquishClient.prototype.streamChat).toHaveBeenCalled();
        const call = (MockSquishClient.prototype.streamChat as jest.Mock).mock.calls[0];
        // messages arg: index 0
        expect(call[0]).toEqual(expect.arrayContaining([
            expect.objectContaining({ role: 'user', content: 'test' }),
        ]));
        // model arg: index 3
        expect(call[3]).toBe('7b');
    });

    test('streamStart/streamChunk/streamEnd posts are relayed to webview', async () => {
        const postMessage = jest.fn();
        const panel = makePanel(extUri);
        const view = makeWebviewView(postMessage);

        let messageHandler: ((msg: unknown) => void) | undefined;
        (view.webview.onDidReceiveMessage as jest.Mock).mockImplementation(
            (cb: (msg: unknown) => void) => { messageHandler = cb; }
        );

        MockSquishClient.prototype.streamChat = jest.fn(
            (_msgs, _max, _temp, _model, onChunk, _onError) => {
                onChunk({ delta: 'Token', done: false, finishReason: null });
                onChunk({ delta: '', done: true, finishReason: 'stop' });
            }
        );

        panel.resolveWebviewView(view, {} as vscode.WebviewViewResolveContext, {} as vscode.CancellationToken);
        messageHandler?.({ type: 'userMessage', text: 'hi' });

        // Wait for async handler
        await new Promise(r => setTimeout(r, 10));

        const types = postMessage.mock.calls.map(([m]: [{ type: string }]) => m.type);
        expect(types).toContain('streamStart');
        expect(types).toContain('streamChunk');
        expect(types).toContain('streamEnd');
    });

    test('streamError is relayed to webview', async () => {
        const postMessage = jest.fn();
        const panel = makePanel(extUri);
        const view = makeWebviewView(postMessage);

        let messageHandler: ((msg: unknown) => void) | undefined;
        (view.webview.onDidReceiveMessage as jest.Mock).mockImplementation(
            (cb: (msg: unknown) => void) => { messageHandler = cb; }
        );

        MockSquishClient.prototype.streamChat = jest.fn(
            (_msgs, _max, _temp, _model, _onChunk, onError) => {
                onError(new Error('GPU on fire'));
            }
        );

        panel.resolveWebviewView(view, {} as vscode.WebviewViewResolveContext, {} as vscode.CancellationToken);
        messageHandler?.({ type: 'userMessage', text: 'hi' });

        await new Promise(r => setTimeout(r, 10));

        const errorMsg = postMessage.mock.calls.find(
            ([m]: [{ type: string }]) => m.type === 'streamError'
        );
        expect(errorMsg).toBeDefined();
        expect(errorMsg[0].message).toContain('GPU on fire');
    });

    test('clearHistory message from webview triggers clearHistory()', () => {
        const postMessage = jest.fn();
        const panel = makePanel(extUri);
        const view = makeWebviewView(postMessage);

        let messageHandler: ((msg: unknown) => void) | undefined;
        (view.webview.onDidReceiveMessage as jest.Mock).mockImplementation(
            (cb: (msg: unknown) => void) => { messageHandler = cb; }
        );

        panel.resolveWebviewView(view, {} as vscode.WebviewViewResolveContext, {} as vscode.CancellationToken);
        postMessage.mockClear();
        messageHandler?.({ type: 'clearHistory' });

        expect(postMessage).toHaveBeenCalledWith({ type: 'clearHistory' });
    });

    test('clearHistory does not crash when view is not resolved', () => {
        const panel = makePanel(extUri);
        expect(() => panel.clearHistory()).not.toThrow();
    });

    test('think blocks are filtered from streamed response', async () => {
        const postMessage = jest.fn();
        const panel = makePanel(extUri);
        const view = makeWebviewView(postMessage);

        let messageHandler: ((msg: unknown) => void) | undefined;
        (view.webview.onDidReceiveMessage as jest.Mock).mockImplementation(
            (cb: (msg: unknown) => void) => { messageHandler = cb; }
        );

        MockSquishClient.prototype.streamChat = jest.fn(
            (_msgs, _max, _temp, _model, onChunk, _onError) => {
                // Simulate: think block first, then real content
                onChunk({ delta: '<think>some reasoning</think>', done: false, finishReason: null });
                onChunk({ delta: 'Actual answer', done: false, finishReason: null });
                onChunk({ delta: '', done: true, finishReason: 'stop' });
            }
        );

        panel.resolveWebviewView(view, {} as vscode.WebviewViewResolveContext, {} as vscode.CancellationToken);
        messageHandler?.({ type: 'userMessage', text: 'hi' });
        await new Promise(r => setTimeout(r, 10));

        const chunks = postMessage.mock.calls
            .filter(([m]: [{ type: string }]) => m.type === 'streamChunk')
            .map(([m]: [{ type: string; delta: string }]) => m.delta);
        const combined = chunks.join('');
        expect(combined).not.toContain('<think>');
        expect(combined).not.toContain('some reasoning');
        expect(combined).toContain('Actual answer');
    });

    test('split think tag across chunk boundary is handled', async () => {
        const postMessage = jest.fn();
        const panel = makePanel(extUri);
        const view = makeWebviewView(postMessage);

        let messageHandler: ((msg: unknown) => void) | undefined;
        (view.webview.onDidReceiveMessage as jest.Mock).mockImplementation(
            (cb: (msg: unknown) => void) => { messageHandler = cb; }
        );

        MockSquishClient.prototype.streamChat = jest.fn(
            (_msgs, _max, _temp, _model, onChunk, _onError) => {
                // Split '<think>' across two chunks
                onChunk({ delta: '<thi', done: false, finishReason: null });
                onChunk({ delta: 'nk>hidden</think>visible', done: false, finishReason: null });
                onChunk({ delta: '', done: true, finishReason: 'stop' });
            }
        );

        panel.resolveWebviewView(view, {} as vscode.WebviewViewResolveContext, {} as vscode.CancellationToken);
        messageHandler?.({ type: 'userMessage', text: 'hi' });
        await new Promise(r => setTimeout(r, 10));

        const chunks = postMessage.mock.calls
            .filter(([m]: [{ type: string }]) => m.type === 'streamChunk')
            .map(([m]: [{ type: string; delta: string }]) => m.delta);
        const combined = chunks.join('');
        expect(combined).not.toContain('hidden');
        expect(combined).toContain('visible');
    });

    // ── Tool calling loop ──────────────────────────────────────────────────

    test('tool call loop: toolCallStart/End messages posted on tool invocation', async () => {
        const postMessage = jest.fn();
        const panel = makePanel(extUri);
        const view = makeWebviewView(postMessage);

        let messageHandler: ((msg: unknown) => void) | undefined;
        (view.webview.onDidReceiveMessage as jest.Mock).mockImplementation(
            (cb: (msg: unknown) => void) => { messageHandler = cb; }
        );

        // First streamChat call returns tool_calls, second returns final answer
        let callCount = 0;
        MockSquishClient.prototype.streamChat = jest.fn(
            (_msgs, _max, _temp, _model, onChunk, _onError) => {
                callCount++;
                if (callCount === 1) {
                    // Model wants to call read_file
                    onChunk({
                        delta: '',
                        done: true,
                        finishReason: 'tool_calls',
                        toolCalls: [{
                            id: 'call_abc',
                            type: 'function',
                            function: { name: 'read_file', arguments: '{"path":"README.md"}' },
                        }],
                    });
                } else {
                    // Final answer after tool result
                    onChunk({ delta: 'File read done.', done: false, finishReason: null });
                    onChunk({ delta: '', done: true, finishReason: 'stop' });
                }
            }
        );

        panel.resolveWebviewView(view, {} as vscode.WebviewViewResolveContext, {} as vscode.CancellationToken);
        messageHandler?.({ type: 'userMessage', text: 'read README.md' });

        // Wait for async tool execution and second streamChat call
        await new Promise(r => setTimeout(r, 50));

        const types = postMessage.mock.calls.map(([m]: [{ type: string }]) => m.type);
        expect(types).toContain('toolCallStart');
        expect(types).toContain('toolCallEnd');
        expect(types).toContain('streamEnd');

        const startMsg = postMessage.mock.calls.find(([m]: [{ type: string }]) => m.type === 'toolCallStart');
        expect(startMsg![0]).toMatchObject({ type: 'toolCallStart', id: 'call_abc', name: 'read_file' });
    });

    test('tool call loop: passes tools array to streamChat', async () => {
        const postMessage = jest.fn();
        const panel = makePanel(extUri);
        const view = makeWebviewView(postMessage);

        let messageHandler: ((msg: unknown) => void) | undefined;
        (view.webview.onDidReceiveMessage as jest.Mock).mockImplementation(
            (cb: (msg: unknown) => void) => { messageHandler = cb; }
        );

        MockSquishClient.prototype.streamChat = jest.fn(
            (_msgs, _max, _temp, _model, onChunk, _onError) => {
                onChunk({ delta: 'ok', done: false, finishReason: null });
                onChunk({ delta: '', done: true, finishReason: 'stop' });
            }
        );

        panel.resolveWebviewView(view, {} as vscode.WebviewViewResolveContext, {} as vscode.CancellationToken);
        messageHandler?.({ type: 'userMessage', text: 'hi' });
        await new Promise(r => setTimeout(r, 20));

        const call = (MockSquishClient.prototype.streamChat as jest.Mock).mock.calls[0];
        // 7th argument (index 6) should be the tools array
        const tools = call[6];
        expect(Array.isArray(tools)).toBe(true);
        expect(tools.length).toBeGreaterThan(0);
        expect(tools[0]).toMatchObject({ type: 'function' });
    });

    test('tool call loop: tool result appended to messages on second call', async () => {
        const postMessage = jest.fn();
        const panel = makePanel(extUri);
        const view = makeWebviewView(postMessage);

        let messageHandler: ((msg: unknown) => void) | undefined;
        (view.webview.onDidReceiveMessage as jest.Mock).mockImplementation(
            (cb: (msg: unknown) => void) => { messageHandler = cb; }
        );

        let callCount = 0;
        let secondCallMessages: unknown[] = [];
        MockSquishClient.prototype.streamChat = jest.fn(
            (msgs, _max, _temp, _model, onChunk, _onError) => {
                callCount++;
                if (callCount === 1) {
                    onChunk({
                        delta: '',
                        done: true,
                        finishReason: 'tool_calls',
                        toolCalls: [{
                            id: 'call_xyz',
                            type: 'function',
                            function: { name: 'get_selection', arguments: '{}' },
                        }],
                    });
                } else {
                    secondCallMessages = msgs as unknown[];
                    onChunk({ delta: 'done', done: false, finishReason: null });
                    onChunk({ delta: '', done: true, finishReason: 'stop' });
                }
            }
        );

        panel.resolveWebviewView(view, {} as vscode.WebviewViewResolveContext, {} as vscode.CancellationToken);
        messageHandler?.({ type: 'userMessage', text: 'get selection' });
        await new Promise(r => setTimeout(r, 50));

        // Second call should include the assistant tool_calls message + tool result
        const assistantMsg = secondCallMessages.find(
            (m: unknown) => (m as { role: string }).role === 'assistant'
        ) as { role: string; tool_calls: unknown[] } | undefined;
        expect(assistantMsg).toBeDefined();
        expect(assistantMsg!.tool_calls).toBeDefined();

        const toolMsg = secondCallMessages.find(
            (m: unknown) => (m as { role: string }).role === 'tool'
        ) as { role: string; tool_call_id: string } | undefined;
        expect(toolMsg).toBeDefined();
        expect(toolMsg!.tool_call_id).toBe('call_xyz');
    });

    test('tool call loop: stops after max rounds to prevent infinite loop', async () => {
        const postMessage = jest.fn();
        const panel = makePanel(extUri);
        const view = makeWebviewView(postMessage);

        let messageHandler: ((msg: unknown) => void) | undefined;
        (view.webview.onDidReceiveMessage as jest.Mock).mockImplementation(
            (cb: (msg: unknown) => void) => { messageHandler = cb; }
        );

        // Always return tool_calls — should stop at MAX_TOOL_ROUNDS (10)
        let callCount = 0;
        MockSquishClient.prototype.streamChat = jest.fn(
            (_msgs, _max, _temp, _model, onChunk, _onError) => {
                callCount++;
                onChunk({
                    delta: '',
                    done: true,
                    finishReason: 'tool_calls',
                    toolCalls: [{
                        id: `call_${callCount}`,
                        type: 'function',
                        function: { name: 'get_selection', arguments: '{}' },
                    }],
                });
            }
        );

        panel.resolveWebviewView(view, {} as vscode.WebviewViewResolveContext, {} as vscode.CancellationToken);
        messageHandler?.({ type: 'userMessage', text: 'loop test' });

        // Give generous time for all rounds
        await new Promise(r => setTimeout(r, 200));

        // Should have stopped — MAX_TOOL_ROUNDS is 10, so 11 calls total (1 initial + 10 rounds)
        expect(callCount).toBeLessThanOrEqual(11);

        // streamEnd should have been posted (loop terminated)
        const types = postMessage.mock.calls.map(([m]: [{ type: string }]) => m.type);
        expect(types).toContain('streamEnd');
    });

    // ── Stop generation ───────────────────────────────────────────────────

    test('stopGeneration message triggers abort and posts streamEnd', () => {
        const postMessage = jest.fn();
        const panel = makePanel(extUri);
        const view = makeWebviewView(postMessage);

        let messageHandler: ((msg: unknown) => void) | undefined;
        (view.webview.onDidReceiveMessage as jest.Mock).mockImplementation(
            (cb: (msg: unknown) => void) => { messageHandler = cb; }
        );

        // streamChat that never calls back (simulates long in-flight request)
        MockSquishClient.prototype.streamChat = jest.fn();
        MockSquishClient.prototype.abort = jest.fn();

        panel.resolveWebviewView(view, {} as vscode.WebviewViewResolveContext, {} as vscode.CancellationToken);

        // Start a generation
        messageHandler?.({ type: 'userMessage', text: 'slow query' });
        postMessage.mockClear();

        // User clicks Stop
        messageHandler?.({ type: 'stopGeneration' });

        expect(MockSquishClient.prototype.abort).toHaveBeenCalled();
        const types = postMessage.mock.calls.map(([m]: [{ type: string }]) => m.type);
        expect(types).toContain('streamEnd');
    });

    // ── New tool implementations ──────────────────────────────────────────

    test('write_file tool creates a new file without confirmation', async () => {
        const postMessage = jest.fn();
        const panel = makePanel(extUri);
        const view = makeWebviewView(postMessage);

        let messageHandler: ((msg: unknown) => void) | undefined;
        (view.webview.onDidReceiveMessage as jest.Mock).mockImplementation(
            (cb: (msg: unknown) => void) => { messageHandler = cb; }
        );

        const { workspace } = require('vscode') as typeof import('vscode') & {
            workspace: { fs: { stat: jest.Mock; writeFile: jest.Mock }; _setWorkspaceFolders: (f: unknown[]) => void };
        };
        // stat throws → file does not exist (new file)
        workspace.fs.stat.mockRejectedValueOnce(new Error('not found'));
        workspace.fs.writeFile.mockResolvedValueOnce(undefined);
        workspace._setWorkspaceFolders([{ uri: { fsPath: '/ws', toString: () => '/ws' } }]);

        let callCount = 0;
        MockSquishClient.prototype.streamChat = jest.fn(
            (_msgs, _max, _temp, _model, onChunk, _onError) => {
                callCount++;
                if (callCount === 1) {
                    onChunk({
                        delta: '',
                        done: true,
                        finishReason: 'tool_calls',
                        toolCalls: [{
                            id: 'tc1',
                            type: 'function',
                            function: { name: 'write_file', arguments: JSON.stringify({ path: 'hello.txt', content: 'hi' }) },
                        }],
                    });
                } else {
                    onChunk({ delta: 'Done', done: false, finishReason: null });
                    onChunk({ delta: '', done: true, finishReason: 'stop' });
                }
            }
        );

        panel.resolveWebviewView(view, {} as vscode.WebviewViewResolveContext, {} as vscode.CancellationToken);
        messageHandler?.({ type: 'userMessage', text: 'write file' });
        await new Promise(r => setTimeout(r, 50));

        expect(workspace.fs.writeFile).toHaveBeenCalled();
        const toolEndMsg = postMessage.mock.calls.find(
            ([m]: [{ type: string }]) => m.type === 'toolCallEnd'
        );
        expect(toolEndMsg).toBeDefined();
        expect(toolEndMsg![0].result).toContain('Created');
    });

    test('write_file tool shows confirmation before overwriting and cancels', async () => {
        const postMessage = jest.fn();
        const panel = makePanel(extUri);
        const view = makeWebviewView(postMessage);

        let messageHandler: ((msg: unknown) => void) | undefined;
        (view.webview.onDidReceiveMessage as jest.Mock).mockImplementation(
            (cb: (msg: unknown) => void) => { messageHandler = cb; }
        );

        const { workspace, window: vsWindow } = require('vscode') as typeof import('vscode') & {
            workspace: { fs: { stat: jest.Mock; writeFile: jest.Mock } };
            window: { showWarningMessage: jest.Mock };
        };
        // stat resolves → file exists
        workspace.fs.stat.mockResolvedValueOnce({});
        // User clicks Cancel
        vsWindow.showWarningMessage.mockResolvedValueOnce('Cancel');

        let callCount = 0;
        MockSquishClient.prototype.streamChat = jest.fn(
            (_msgs, _max, _temp, _model, onChunk, _onError) => {
                callCount++;
                if (callCount === 1) {
                    onChunk({
                        delta: '',
                        done: true,
                        finishReason: 'tool_calls',
                        toolCalls: [{
                            id: 'tc2',
                            type: 'function',
                            function: { name: 'write_file', arguments: JSON.stringify({ path: 'existing.txt', content: 'new' }) },
                        }],
                    });
                } else {
                    onChunk({ delta: 'ok', done: false, finishReason: null });
                    onChunk({ delta: '', done: true, finishReason: 'stop' });
                }
            }
        );

        panel.resolveWebviewView(view, {} as vscode.WebviewViewResolveContext, {} as vscode.CancellationToken);
        messageHandler?.({ type: 'userMessage', text: 'overwrite file' });
        await new Promise(r => setTimeout(r, 50));

        expect(workspace.fs.writeFile).not.toHaveBeenCalled();
        const toolEndMsg = postMessage.mock.calls.find(
            ([m]: [{ type: string }]) => m.type === 'toolCallEnd'
        );
        expect(toolEndMsg![0].result).toBe('Cancelled.');
    });

    test('list_directory tool returns entry names', async () => {
        const postMessage = jest.fn();
        const panel = makePanel(extUri);
        const view = makeWebviewView(postMessage);

        let messageHandler: ((msg: unknown) => void) | undefined;
        (view.webview.onDidReceiveMessage as jest.Mock).mockImplementation(
            (cb: (msg: unknown) => void) => { messageHandler = cb; }
        );

        const { workspace } = require('vscode') as typeof import('vscode') & {
            workspace: { fs: { readDirectory: jest.Mock }; _setWorkspaceFolders: (f: unknown[]) => void };
        };
        workspace.fs.readDirectory.mockResolvedValueOnce([
            ['src', 2 /* Directory */],
            ['README.md', 1 /* File */],
        ]);
        workspace._setWorkspaceFolders([{ uri: { fsPath: '/ws', toString: () => '/ws' } }]);

        let callCount = 0;
        MockSquishClient.prototype.streamChat = jest.fn(
            (_msgs, _max, _temp, _model, onChunk, _onError) => {
                callCount++;
                if (callCount === 1) {
                    onChunk({
                        delta: '',
                        done: true,
                        finishReason: 'tool_calls',
                        toolCalls: [{
                            id: 'tc3',
                            type: 'function',
                            function: { name: 'list_directory', arguments: '{}' },
                        }],
                    });
                } else {
                    onChunk({ delta: 'listed', done: false, finishReason: null });
                    onChunk({ delta: '', done: true, finishReason: 'stop' });
                }
            }
        );

        panel.resolveWebviewView(view, {} as vscode.WebviewViewResolveContext, {} as vscode.CancellationToken);
        messageHandler?.({ type: 'userMessage', text: 'list root' });
        await new Promise(r => setTimeout(r, 50));

        const toolEndMsg = postMessage.mock.calls.find(
            ([m]: [{ type: string }]) => m.type === 'toolCallEnd'
        );
        expect(toolEndMsg).toBeDefined();
        expect(toolEndMsg![0].result).toContain('src/');
        expect(toolEndMsg![0].result).toContain('README.md');
    });

    test('get_diagnostics tool returns formatted error list', async () => {
        const postMessage = jest.fn();
        const panel = makePanel(extUri);
        const view = makeWebviewView(postMessage);

        let messageHandler: ((msg: unknown) => void) | undefined;
        (view.webview.onDidReceiveMessage as jest.Mock).mockImplementation(
            (cb: (msg: unknown) => void) => { messageHandler = cb; }
        );

        const { languages, workspace: ws } = require('vscode') as typeof import('vscode') & {
            languages: { getDiagnostics: jest.Mock };
            workspace: { _setWorkspaceFolders: (f: unknown[]) => void };
        };
        languages.getDiagnostics.mockReturnValueOnce([
            [
                { fsPath: '/ws/src/index.ts' },
                [{ severity: 0 /* Error */, range: { start: { line: 4 } }, message: "Cannot find name 'x'" }],
            ],
        ]);
        ws._setWorkspaceFolders([{ uri: { fsPath: '/ws', toString: () => '/ws' } }]);

        let callCount = 0;
        MockSquishClient.prototype.streamChat = jest.fn(
            (_msgs, _max, _temp, _model, onChunk, _onError) => {
                callCount++;
                if (callCount === 1) {
                    onChunk({
                        delta: '',
                        done: true,
                        finishReason: 'tool_calls',
                        toolCalls: [{
                            id: 'tc4',
                            type: 'function',
                            function: { name: 'get_diagnostics', arguments: '{}' },
                        }],
                    });
                } else {
                    onChunk({ delta: 'done', done: false, finishReason: null });
                    onChunk({ delta: '', done: true, finishReason: 'stop' });
                }
            }
        );

        panel.resolveWebviewView(view, {} as vscode.WebviewViewResolveContext, {} as vscode.CancellationToken);
        messageHandler?.({ type: 'userMessage', text: 'show errors' });
        await new Promise(r => setTimeout(r, 50));

        const toolEndMsg = postMessage.mock.calls.find(
            ([m]: [{ type: string }]) => m.type === 'toolCallEnd'
        );
        expect(toolEndMsg).toBeDefined();
        expect(toolEndMsg![0].result).toContain('[error]');
        expect(toolEndMsg![0].result).toContain("Cannot find name 'x'");
        expect(toolEndMsg![0].result).toContain(':5 ');
    });

    test('get_diagnostics returns friendly message when no issues', async () => {
        const postMessage = jest.fn();
        const panel = makePanel(extUri);
        const view = makeWebviewView(postMessage);

        let messageHandler: ((msg: unknown) => void) | undefined;
        (view.webview.onDidReceiveMessage as jest.Mock).mockImplementation(
            (cb: (msg: unknown) => void) => { messageHandler = cb; }
        );

        const { languages } = require('vscode') as typeof import('vscode') & {
            languages: { getDiagnostics: jest.Mock };
        };
        languages.getDiagnostics.mockReturnValueOnce([]);

        MockSquishClient.prototype.streamChat = jest.fn(
            (_msgs, _max, _temp, _model, onChunk, _onError) => {
                onChunk({
                    delta: '', done: true, finishReason: 'tool_calls',
                    toolCalls: [{ id: 'tc5', type: 'function', function: { name: 'get_diagnostics', arguments: '{}' } }],
                });
            }
        );

        panel.resolveWebviewView(view, {} as vscode.WebviewViewResolveContext, {} as vscode.CancellationToken);
        messageHandler?.({ type: 'userMessage', text: 'any errors?' });
        await new Promise(r => setTimeout(r, 30));

        const toolEndMsg = postMessage.mock.calls.find(
            ([m]: [{ type: string }]) => m.type === 'toolCallEnd'
        );
        expect(toolEndMsg![0].result).toBe('No diagnostics.');
    });

    test('HTML template contains stop button', () => {
        const panel = makePanel(extUri);
        const view = makeWebviewView();
        panel.resolveWebviewView(view, {} as vscode.WebviewViewResolveContext, {} as vscode.CancellationToken);
        expect(view.webview.html).toContain('btn-stop');
    });

    test('tools array includes new tool definitions', () => {
        const panel = makePanel(extUri);
        const view = makeWebviewView();
        panel.resolveWebviewView(view, {} as vscode.WebviewViewResolveContext, {} as vscode.CancellationToken);

        let messageHandler2: ((msg: unknown) => void) | undefined;
        const view2 = makeWebviewView();
        (view2.webview.onDidReceiveMessage as jest.Mock).mockImplementation(
            (cb) => { messageHandler2 = cb; }
        );
        MockSquishClient.prototype.streamChat = jest.fn(
            (_msgs, _max, _temp, _model, onChunk, _onError) => {
                onChunk({ delta: 'ok', done: false, finishReason: null });
                onChunk({ delta: '', done: true, finishReason: 'stop' });
            }
        );
        panel.resolveWebviewView(view2, {} as vscode.WebviewViewResolveContext, {} as vscode.CancellationToken);
        messageHandler2?.({ type: 'userMessage', text: 'hi' });

        const call = (MockSquishClient.prototype.streamChat as jest.Mock).mock.calls[0];
        const tools = call[6] as Array<{ function: { name: string } }>;
        const names = tools.map(t => t.function.name);
        expect(names).toContain('write_file');
        expect(names).toContain('list_directory');
        expect(names).toContain('get_diagnostics');
    });
});
