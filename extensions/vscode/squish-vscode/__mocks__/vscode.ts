/**
 * __mocks__/vscode.ts
 *
 * Minimal VS Code API mock for Jest unit tests.
 * Only surfaces the APIs exercised by squish-vscode's source modules.
 */

export const StatusBarAlignment = { Left: 1, Right: 2 };
export const ConfigurationTarget = { Global: 1, Workspace: 2, WorkspaceFolder: 3 };
export class ThemeColor { constructor(public id: string) {} }

const _config: Record<string, unknown> = {
    'squish.host':            '127.0.0.1',
    'squish.port':            11435,
    'squish.apiKey':          'squish',
    'squish.model':           '7b',
    'squish.autoStart':       false,
    'squish.maxTokens':       1024,
    'squish.temperature':     0.7,
    'squish.systemPrompt':    '',
    'squish.thinkingBudget':  0,
};

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
    _mockConfig,
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

export const Uri = {
    joinPath: jest.fn((_base: unknown, ...parts: string[]) => ({
        toString: () => parts.join('/'),
        fsPath: parts.join('/'),
    })),
    file: jest.fn((p: string) => ({ fsPath: p })),
};
