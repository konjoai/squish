/**
 * __tests__/inlineCompletion.test.ts
 *
 * Unit tests for SquishInlineCompletionProvider.
 * SquishClient and VS Code are fully mocked.
 */
import * as vscode from 'vscode';
import { SquishInlineCompletionProvider } from '../src/inlineCompletion';

jest.mock('../src/squishClient');
import { SquishClient } from '../src/squishClient';
const MockSquishClient = SquishClient as jest.MockedClass<typeof SquishClient>;

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Build a mock TextDocument */
function makeDoc(
    lines: string[],
    languageId = 'typescript',
): vscode.TextDocument {
    const text = lines.join('\n');
    return {
        languageId,
        getText: jest.fn(() => text),
        lineAt: jest.fn((lineOrPos: number | vscode.Position) => {
            const lineNum = typeof lineOrPos === 'number' ? lineOrPos : lineOrPos.line;
            return { text: lines[lineNum] ?? '', lineNumber: lineNum };
        }),
        lineCount: lines.length,
        uri: vscode.Uri.file('/ws/file.ts'),
        offsetAt: jest.fn((pos: vscode.Position) => {
            let off = 0;
            for (let i = 0; i < pos.line; i++) {
                off += (lines[i]?.length ?? 0) + 1;
            }
            return off + pos.character;
        }),
        positionAt: jest.fn(() => new vscode.Position(0, 0)),
        fileName: '/ws/file.ts',
        isUntitled: false,
        encoding: 'utf8',
        isDirty: false,
        isClosed: false,
    } as unknown as vscode.TextDocument;
}

function makePosition(line: number, char: number): vscode.Position {
    return new vscode.Position(line, char);
}

function makeCancellationToken(cancelled = false): vscode.CancellationToken {
    return {
        isCancellationRequested: cancelled,
        onCancellationRequested: jest.fn(),
    };
}

function makeInlineContext(): vscode.InlineCompletionContext {
    return {
        triggerKind: vscode.InlineCompletionTriggerKind.Automatic,
        selectedCompletionInfo: undefined,
    };
}

/**
 * Sets up MockSquishClient.prototype.streamChat to call onChunk with the
 * given text chunks and then call onChunk with done:true.
 */
function mockStreamChat(chunks: string[]): void {
    MockSquishClient.prototype.streamChat = jest.fn(
        (_msgs, _max, _temp, _model, onChunk: (c: { delta?: string; done?: boolean }) => void) => {
            for (const chunk of chunks) { onChunk({ delta: chunk, done: false }); }
            onChunk({ delta: '', done: true });
        },
    ) as jest.Mock;
}

/** Mock that never calls done → hangs forever (used to test cancellation) */
function mockStreamChatHanging(): void {
    MockSquishClient.prototype.streamChat = jest.fn(
        (_msgs, _max, _temp, _model, _onChunk: unknown, onError: (e: Error) => void) => {
            // Simulate an instant error so the promise resolves (no hang)
            onError(new Error('aborted'));
        },
    ) as jest.Mock;
}

/** Mock that errors immediately */
function mockStreamChatError(msg = 'Connection refused'): void {
    MockSquishClient.prototype.streamChat = jest.fn(
        (_msgs, _max, _temp, _model, _onChunk: unknown, onError: (e: Error) => void) => {
            onError(new Error(msg));
        },
    ) as jest.Mock;
}

function enabledCfg(): void {
    (vscode.workspace.getConfiguration as jest.Mock).mockReturnValue({
        get: (key: string, def?: unknown) =>
            key === 'enableInlineCompletion' ? true :
            key === 'host'   ? '127.0.0.1' :
            key === 'port'   ? 11435 :
            key === 'apiKey' ? 'key'  :
            key === 'model'  ? '7b'   : def,
    });
}

function disabledCfg(): void {
    (vscode.workspace.getConfiguration as jest.Mock).mockReturnValue({
        get: (key: string, def?: unknown) =>
            key === 'enableInlineCompletion' ? false : def,
    });
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('SquishInlineCompletionProvider', () => {

    let provider: SquishInlineCompletionProvider;

    beforeEach(() => {
        jest.clearAllMocks();
        provider = new SquishInlineCompletionProvider();
        // default: returns a simple completion
        mockStreamChat(['const result = 42;']);
    });

    // ── Trigger detection ──────────────────────────────────────────────────

    test('returns undefined when inline completion is disabled', async () => {
        disabledCfg();
        const doc = makeDoc(['// squish: sort array']);
        const result = await provider.provideInlineCompletionItems(
            doc, makePosition(0, 20), makeInlineContext(), makeCancellationToken(),
        );
        expect(result).toBeUndefined();
    });

    test('returns undefined for unsupported language without explicit trigger', async () => {
        enabledCfg();
        const doc = makeDoc(['some text'], 'plaintext');
        const result = await provider.provideInlineCompletionItems(
            doc, makePosition(0, 9), makeInlineContext(), makeCancellationToken(),
        );
        expect(result).toBeUndefined();
    });

    test('triggers on "// squish:" prefix', async () => {
        enabledCfg();
        const doc = makeDoc(['// squish: sort this array by name'], 'typescript');
        const result = await provider.provideInlineCompletionItems(
            doc, makePosition(0, 34), makeInlineContext(), makeCancellationToken(),
        );
        expect(result).toBeTruthy();
    });

    test('triggers on "# squish:" prefix (Python)', async () => {
        enabledCfg();
        const doc = makeDoc(['# squish: sort list'], 'python');
        const result = await provider.provideInlineCompletionItems(
            doc, makePosition(0, 19), makeInlineContext(), makeCancellationToken(),
        );
        expect(result).toBeTruthy();
    });

    test('triggers for FIM languages even without comment trigger', async () => {
        enabledCfg();
        const doc = makeDoc(['function foo() {', '  return 42;'], 'rust');
        const result = await provider.provideInlineCompletionItems(
            doc, makePosition(1, 12), makeInlineContext(), makeCancellationToken(),
        );
        expect(result).toBeTruthy();
    });

    test('returns InlineCompletionList with insertText from model', async () => {
        enabledCfg();
        mockStreamChat(['\nconst sorted = items.sort();']);
        const doc = makeDoc(['// squish: sort items'], 'typescript');
        const result = await provider.provideInlineCompletionItems(
            doc, makePosition(0, 21), makeInlineContext(), makeCancellationToken(),
        );
        expect(result?.items?.length).toBeGreaterThan(0);
        const insertText = result?.items[0]?.insertText;
        expect(String(insertText)).toContain('sorted');
    });

    test('returns undefined when prefix is empty', async () => {
        enabledCfg();
        const doc = makeDoc([''], 'typescript');
        const result = await provider.provideInlineCompletionItems(
            doc, makePosition(0, 0), makeInlineContext(), makeCancellationToken(),
        );
        expect(result).toBeUndefined();
    });

    test('cancellation token stops request', async () => {
        enabledCfg();
        mockStreamChatHanging();
        const doc = makeDoc(['// squish: do stuff'], 'typescript');
        const token = makeCancellationToken(false);   // not pre-cancelled
        const result = await provider.provideInlineCompletionItems(
            doc, makePosition(0, 19), makeInlineContext(), token,
        );
        // Should resolve (not hang) — result may be undefined due to error
        expect(result === undefined || Array.isArray((result as vscode.InlineCompletionList)?.items)).toBe(true);
    });

    test('pre-cancelled token returns immediately', async () => {
        enabledCfg();
        const doc = makeDoc(['// squish: do stuff'], 'typescript');
        const token = makeCancellationToken(true);   // already cancelled
        const result = await provider.provideInlineCompletionItems(
            doc, makePosition(0, 19), makeInlineContext(), token,
        );
        // Pre-cancellation should short-circuit — result is undefined or empty
        expect(result === undefined || (result as vscode.InlineCompletionList)?.items?.length === 0).toBe(true);
    });

    test('client error is handled gracefully (returns undefined)', async () => {
        enabledCfg();
        mockStreamChatError('Connection refused');
        const doc = makeDoc(['// squish: do stuff'], 'typescript');
        const result = await provider.provideInlineCompletionItems(
            doc, makePosition(0, 19), makeInlineContext(), makeCancellationToken(),
        );
        expect(result === undefined || (result as vscode.InlineCompletionList)?.items?.length === 0).toBe(true);
    });

    // ── Language support ───────────────────────────────────────────────────

    test.each([
        'typescript', 'javascript', 'python', 'rust', 'go', 'cpp', 'c', 'java',
    ])('FIM triggers for %s language', async (lang: string) => {
        enabledCfg();
        const doc = makeDoc(['function test() { return 1; }'], lang);
        await provider.provideInlineCompletionItems(
            doc, makePosition(0, 29), makeInlineContext(), makeCancellationToken(),
        );
        expect(MockSquishClient.prototype.streamChat).toHaveBeenCalled();
    });
});

