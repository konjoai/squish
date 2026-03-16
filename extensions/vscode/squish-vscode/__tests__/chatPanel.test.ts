/**
 * __tests__/chatPanel.test.ts
 *
 * Unit tests for ChatPanel.
 * The WebviewView is fully mocked — no VS Code process needed.
 */
import * as vscode from 'vscode';
import { ChatPanel } from '../src/chatPanel';

// Mock squishClient so we can control streaming
jest.mock('../src/squishClient');
import { SquishClient } from '../src/squishClient';
const MockSquishClient = SquishClient as jest.MockedClass<typeof SquishClient>;

// ── Helpers ───────────────────────────────────────────────────────────────────

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
        const panel = new ChatPanel(extUri);
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
        const panel = new ChatPanel(extUri);
        const view = makeWebviewView(postMessage);

        panel.resolveWebviewView(view, {} as vscode.WebviewViewResolveContext, {} as vscode.CancellationToken);
        panel.clearHistory();

        expect(postMessage).toHaveBeenCalledWith({ type: 'clearHistory' });
    });

    test('userMessage triggers streamChat with configured model', () => {
        const postMessage = jest.fn();
        const panel = new ChatPanel(extUri);
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
        const panel = new ChatPanel(extUri);
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
        const panel = new ChatPanel(extUri);
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
        const panel = new ChatPanel(extUri);
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
        const panel = new ChatPanel(extUri);
        expect(() => panel.clearHistory()).not.toThrow();
    });
});
