/**
 * __mocks__/vscode.ts
 *
 * Minimal VS Code API mock for Jest unit tests.
 * Only surfaces the APIs exercised by squish-vscode's source modules.
 */

export const StatusBarAlignment = { Left: 1, Right: 2 };
export const ConfigurationTarget = { Global: 1, Workspace: 2, WorkspaceFolder: 3 };
export const FileType = { Unknown: 0, File: 1, Directory: 2, SymbolicLink: 64 };
export const DiagnosticSeverity = { Error: 0, Warning: 1, Information: 2, Hint: 3 };
export class ThemeColor { constructor(public id: string) {} }

const _config: Record<string, unknown> = {
    'squish.host':            '127.0.0.1',
    'squish.port':            11435,
    'squish.apiKey':          'squish',
    'squish.model':           '7b',
    'squish.autoStart':       false,
    'squish.maxTokens':       2048,
    'squish.temperature':     0.7,
    'squish.systemPrompt':    '',
    'squish.thinkingBudget':  0,
    'squish.venvPath':        '',
};

let _workspaceFolders: Array<{ uri: { fsPath: string } }> = [];

const _mockConfig = {
    get: jest.fn((key: string, defaultVal?: unknown) => {
        const full = _config[`squish.${key}`] ?? _config[key];
        return full !== undefined ? full : defaultVal;
    }),
    update: jest.fn().mockResolvedValue(undefined),
};

export const workspace = {
    getConfiguration: jest.fn().mockReturnValue(_mockConfig),
    _setConfig: (key: string, val: unknown) => { _config[`squish.${key}`] = val; },
    get workspaceFolders() { return _workspaceFolders; },
    _setWorkspaceFolders: (folders: Array<{ uri: { fsPath: string } }>) => {
        _workspaceFolders = folders;
    },
    _mockConfig,
    fs: {
        stat:          jest.fn().mockRejectedValue(new Error('FileNotFound')),
        writeFile:     jest.fn().mockResolvedValue(undefined),
        readDirectory: jest.fn().mockResolvedValue([] as [string, number][]),
        readFile:      jest.fn().mockResolvedValue(Buffer.from('')),
    },
    textDocuments: [] as Array<{ isUntitled: boolean; uri: { scheme: string; fsPath: string } }>,
};

export const window = {
    createStatusBarItem: jest.fn().mockReturnValue({
        command: '',
        text: '',
        tooltip: '',
        backgroundColor: undefined,
        show: jest.fn(),
        dispose: jest.fn(),
    }),
    registerWebviewViewProvider: jest.fn().mockReturnValue({ dispose: jest.fn() }),
    showInformationMessage: jest.fn().mockResolvedValue(undefined),
    showErrorMessage: jest.fn().mockResolvedValue(undefined),
    showWarningMessage: jest.fn().mockResolvedValue(undefined),
    showQuickPick: jest.fn().mockResolvedValue(undefined),
};

export const commands = {
    registerCommand: jest.fn().mockReturnValue({ dispose: jest.fn() }),
    executeCommand: jest.fn().mockResolvedValue(undefined),
};

export const env = {
    clipboard: {
        writeText: jest.fn().mockResolvedValue(undefined),
    },
};

export const languages = {
    getDiagnostics: jest.fn().mockReturnValue([] as [{ fsPath: string }, { severity: number; range: { start: { line: number } }; message: string }[]][]),
};

export const Uri = {
    joinPath: jest.fn((_base: unknown, ...parts: string[]) => ({
        toString: () => parts.join('/'),
        fsPath: parts.join('/'),
    })),
    file: jest.fn((p: string) => ({ fsPath: p })),
};
