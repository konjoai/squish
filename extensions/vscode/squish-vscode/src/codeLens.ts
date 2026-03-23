/**
 * squish-vscode/src/codeLens.ts
 *
 * CodeLensProvider that adds "Squish" action links above each function /
 * class / method definition in supported languages.  Clicking a lens sends
 * the code block to the Squish Agent chat panel with a pre-built prompt.
 *
 * Supported lenses (per definition):
 *   Squish: Explain | Document | Refactor | Test
 */
import * as vscode from 'vscode';

// ── Per-language function/class detection regexes ─────────────────────────

interface LangPattern {
    /** Pattern must capture the symbol name in group 1. */
    re: RegExp;
    kind: 'function' | 'class' | 'method';
}

const LANG_PATTERNS: Record<string, LangPattern[]> = {
    typescript: [
        { re: /^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(\w+)/,        kind: 'function' },
        { re: /^\s*(?:export\s+)?(?:abstract\s+)?class\s+(\w+)/,                        kind: 'class'    },
        { re: /^\s*(?:public|private|protected|static|async|override)[\w\s]*?\s(\w+)\s*\(/, kind: 'method'   },
    ],
    javascript: [
        { re: /^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(\w+)/,        kind: 'function' },
        { re: /^\s*(?:export\s+)?class\s+(\w+)/,                                        kind: 'class'    },
    ],
    python: [
        { re: /^\s*def\s+(\w+)\s*\(/,   kind: 'function' },
        { re: /^\s*class\s+(\w+)[\s:(]/, kind: 'class'   },
        { re: /^\s*async\s+def\s+(\w+)\s*\(/, kind: 'function' },
    ],
    rust: [
        { re: /^\s*(?:pub(?:\(.*?\))?\s+)?(?:async\s+)?fn\s+(\w+)/, kind: 'function' },
        { re: /^\s*(?:pub\s+)?(?:struct|enum|trait|impl)\s+(\w+)/, kind: 'class' },
    ],
    go: [
        { re: /^\s*func\s+(?:\(.*?\)\s+)?(\w+)\s*\(/,   kind: 'function' },
        { re: /^\s*type\s+(\w+)\s+(?:struct|interface)/, kind: 'class' },
    ],
    cpp: [
        { re: /^\s*(?:[\w:*&<>\s]+)\s+(\w+)\s*\((?!.*\).*=\s*0)/, kind: 'function' },
        { re: /^\s*class\s+(\w+)/,                                   kind: 'class' },
    ],
    c: [
        { re: /^\s*(?:[\w\s*]+)\s+(\w+)\s*\([^;]*$/, kind: 'function' },
    ],
    java: [
        { re: /^\s*(?:public|private|protected|static|final|synchronized|\s)+[\w<>\[\]]+\s+(\w+)\s*\(/, kind: 'method' },
        { re: /^\s*(?:public\s+)?(?:abstract\s+)?class\s+(\w+)/,   kind: 'class' },
    ],
};

// ── Lens commands ─────────────────────────────────────────────────────────

interface LensAction {
    label: string;
    command: string;
}

const LENS_ACTIONS: LensAction[] = [
    { label: 'Explain',  command: 'squish.explainSelection' },
    { label: 'Document', command: 'squish.documentFunction'  },
    { label: 'Refactor', command: 'squish.refactorSelection' },
    { label: 'Test',     command: 'squish.generateTests'     },
];

// ── Provider ──────────────────────────────────────────────────────────────

export class SquishCodeLensProvider implements vscode.CodeLensProvider {
    private readonly _onDidChange = new vscode.EventEmitter<void>();
    readonly onDidChangeCodeLenses = this._onDidChange.event;

    provideCodeLenses(
        document: vscode.TextDocument,
        _token: vscode.CancellationToken,
    ): vscode.CodeLens[] {
        const cfg = vscode.workspace.getConfiguration('squish');
        if (!cfg.get<boolean>('enableCodeLens', true)) {
            return [];
        }

        const patterns = LANG_PATTERNS[document.languageId];
        if (!patterns) {
            return [];
        }

        const lenses: vscode.CodeLens[] = [];

        for (let lineNum = 0; lineNum < document.lineCount; lineNum++) {
            const lineText = document.lineAt(lineNum).text;
            for (const { re, kind } of patterns) {
                const m = re.exec(lineText);
                if (!m) continue;

                const symbolName = m[1] ?? lineText.trim().slice(0, 30);
                const range      = new vscode.Range(lineNum, 0, lineNum, lineText.length);

                for (const action of LENS_ACTIONS) {
                    lenses.push(
                        new vscode.CodeLens(range, {
                            title: action.label,
                            command: action.command,
                            arguments: [{ range, symbolName, kind, lineText }],
                            tooltip: `Squish: ${action.label} "${symbolName}"`,
                        }),
                    );
                }
                break; // only one pattern match per line
            }
        }

        return lenses;
    }

    refresh(): void {
        this._onDidChange.fire();
    }

    dispose(): void {
        this._onDidChange.dispose();
    }
}

export const CODE_LENS_LANGUAGES = Object.keys(LANG_PATTERNS);
