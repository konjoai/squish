/**
 * __tests__/codeLens.test.ts
 *
 * Unit tests for SquishCodeLensProvider.
 * VS Code is mocked via __mocks__/vscode.ts.
 */
import * as vscode from 'vscode';
import { SquishCodeLensProvider } from '../src/codeLens';

// ── Helpers ───────────────────────────────────────────────────────────────────

function makeDoc(lines: string[], languageId: string): vscode.TextDocument {
    return {
        languageId,
        lineCount: lines.length,
        lineAt: jest.fn((lineOrPos: number | vscode.Position) => {
            const n = typeof lineOrPos === 'number' ? lineOrPos : lineOrPos.line;
            return { text: lines[n] ?? '', lineNumber: n };
        }),
        uri: vscode.Uri.file('/ws/test.' + languageId),
    } as unknown as vscode.TextDocument;
}

function makeCancelToken(): vscode.CancellationToken {
    return { isCancellationRequested: false, onCancellationRequested: jest.fn() };
}

function makeCfg(enabled: boolean) {
    return {
        get: (key: string, def?: unknown) =>
            key === 'enableCodeLens' ? enabled : def,
    };
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('SquishCodeLensProvider', () => {

    let provider: SquishCodeLensProvider;

    beforeEach(() => {
        jest.clearAllMocks();
        provider = new SquishCodeLensProvider();
    });

    // ── Disabled ───────────────────────────────────────────────────────────

    test('returns empty array when enableCodeLens is false', () => {
        (vscode.workspace.getConfiguration as jest.Mock).mockReturnValue(makeCfg(false));
        const doc = makeDoc(['function foo() {}'], 'typescript');
        expect(provider.provideCodeLenses(doc, makeCancelToken())).toEqual([]);
    });

    // ── Unsupported languages ──────────────────────────────────────────────

    test('returns empty array for unsupported language', () => {
        (vscode.workspace.getConfiguration as jest.Mock).mockReturnValue(makeCfg(true));
        const doc = makeDoc(['some plain text'], 'plaintext');
        expect(provider.provideCodeLenses(doc, makeCancelToken())).toEqual([]);
    });

    test('returns empty array for json language', () => {
        (vscode.workspace.getConfiguration as jest.Mock).mockReturnValue(makeCfg(true));
        const doc = makeDoc(['{"key": "value"}'], 'json');
        expect(provider.provideCodeLenses(doc, makeCancelToken())).toEqual([]);
    });

    // ── TypeScript ─────────────────────────────────────────────────────────

    test('detects TypeScript function and returns 4 lenses', () => {
        (vscode.workspace.getConfiguration as jest.Mock).mockReturnValue(makeCfg(true));
        const doc = makeDoc(['function greet(name: string): string {'], 'typescript');
        const lenses = provider.provideCodeLenses(doc, makeCancelToken());
        expect(lenses.length).toBe(4);
    });

    test('TypeScript lenses have correct commands', () => {
        (vscode.workspace.getConfiguration as jest.Mock).mockReturnValue(makeCfg(true));
        const doc = makeDoc(['export function compute() {}'], 'typescript');
        const lenses = provider.provideCodeLenses(doc, makeCancelToken());
        const cmds = lenses.map((l) => l.command?.command);
        expect(cmds).toContain('squish.explainSelection');
        expect(cmds).toContain('squish.documentFunction');
        expect(cmds).toContain('squish.refactorSelection');
        expect(cmds).toContain('squish.generateTests');
    });

    test('detects TypeScript class', () => {
        (vscode.workspace.getConfiguration as jest.Mock).mockReturnValue(makeCfg(true));
        const doc = makeDoc(['export class MyService {'], 'typescript');
        const lenses = provider.provideCodeLenses(doc, makeCancelToken());
        expect(lenses.length).toBe(4);
    });

    test('detects async TypeScript function', () => {
        (vscode.workspace.getConfiguration as jest.Mock).mockReturnValue(makeCfg(true));
        const doc = makeDoc(['export async function fetchData() {'], 'typescript');
        const lenses = provider.provideCodeLenses(doc, makeCancelToken());
        expect(lenses.length).toBe(4);
    });

    test('two TypeScript functions produce 8 lenses', () => {
        (vscode.workspace.getConfiguration as jest.Mock).mockReturnValue(makeCfg(true));
        const doc = makeDoc([
            'function alpha() {}',
            '',
            'function beta() {}',
        ], 'typescript');
        const lenses = provider.provideCodeLenses(doc, makeCancelToken());
        expect(lenses.length).toBe(8);
    });

    // ── Python ─────────────────────────────────────────────────────────────

    test('detects Python def', () => {
        (vscode.workspace.getConfiguration as jest.Mock).mockReturnValue(makeCfg(true));
        const doc = makeDoc(['def my_function(x, y):'], 'python');
        const lenses = provider.provideCodeLenses(doc, makeCancelToken());
        expect(lenses.length).toBe(4);
    });

    test('detects Python class', () => {
        (vscode.workspace.getConfiguration as jest.Mock).mockReturnValue(makeCfg(true));
        const doc = makeDoc(['class MyClass:'], 'python');
        const lenses = provider.provideCodeLenses(doc, makeCancelToken());
        expect(lenses.length).toBe(4);
    });

    test('detects async Python def', () => {
        (vscode.workspace.getConfiguration as jest.Mock).mockReturnValue(makeCfg(true));
        const doc = makeDoc(['async def handler():'], 'python');
        const lenses = provider.provideCodeLenses(doc, makeCancelToken());
        expect(lenses.length).toBe(4);
    });

    // ── Rust ───────────────────────────────────────────────────────────────

    test('detects Rust fn', () => {
        (vscode.workspace.getConfiguration as jest.Mock).mockReturnValue(makeCfg(true));
        const doc = makeDoc(['pub fn compute(x: i32) -> i32 {'], 'rust');
        const lenses = provider.provideCodeLenses(doc, makeCancelToken());
        expect(lenses.length).toBe(4);
    });

    test('detects Rust struct', () => {
        (vscode.workspace.getConfiguration as jest.Mock).mockReturnValue(makeCfg(true));
        const doc = makeDoc(['pub struct Config {'], 'rust');
        const lenses = provider.provideCodeLenses(doc, makeCancelToken());
        expect(lenses.length).toBe(4);
    });

    // ── Go ─────────────────────────────────────────────────────────────────

    test('detects Go func', () => {
        (vscode.workspace.getConfiguration as jest.Mock).mockReturnValue(makeCfg(true));
        const doc = makeDoc(['func ProcessData(input string) error {'], 'go');
        const lenses = provider.provideCodeLenses(doc, makeCancelToken());
        expect(lenses.length).toBe(4);
    });

    test('detects Go method with receiver', () => {
        (vscode.workspace.getConfiguration as jest.Mock).mockReturnValue(makeCfg(true));
        const doc = makeDoc(['func (s *Server) Start() error {'], 'go');
        const lenses = provider.provideCodeLenses(doc, makeCancelToken());
        expect(lenses.length).toBe(4);
    });

    // ── JavaScript ─────────────────────────────────────────────────────────

    test('detects JavaScript function', () => {
        (vscode.workspace.getConfiguration as jest.Mock).mockReturnValue(makeCfg(true));
        const doc = makeDoc(['function validate(input) {'], 'javascript');
        const lenses = provider.provideCodeLenses(doc, makeCancelToken());
        expect(lenses.length).toBe(4);
    });

    test('lines with no match produce no lenses', () => {
        (vscode.workspace.getConfiguration as jest.Mock).mockReturnValue(makeCfg(true));
        const doc = makeDoc([
            '// just a comment',
            'const x = 1;',
            'return x;',
        ], 'typescript');
        expect(provider.provideCodeLenses(doc, makeCancelToken())).toEqual([]);
    });
});
