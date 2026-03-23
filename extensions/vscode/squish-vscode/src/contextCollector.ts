/**
 * squish-vscode/src/contextCollector.ts
 *
 * Builds a rich workspace-context string that is prepended to the system
 * prompt on every agentic request so the model has full situational
 * awareness without needing to ask for it explicitly.
 *
 * Collected context:
 *  - Active editor file name + language + selected text (if any)
 *  - All open editor file paths (relative)
 *  - First-category diagnostics for the active file (errors + warnings)
 *  - Git branch + last commit message (if `git` is available)
 */
import * as vscode from 'vscode';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export interface WorkspaceContext {
    /** Relative path of the active file or undefined. */
    activeFile?: string;
    /** Language identifier of the active file (e.g. 'typescript'). */
    language?: string;
    /** Currently selected text (empty string if no selection). */
    selection: string;
    /** Relative paths of all open editors. */
    openFiles: string[];
    /** Formatted diagnostics string (empty if none). */
    diagnostics: string;
    /** Git branch + last commit (empty if not in a git repo). */
    gitInfo: string;
    /** Full formatted context block for use in a system prompt. */
    formatted: string;
}

export class ContextCollector {
    /**
     * Collect workspace context from the current VS Code state.
     * All VS Code API calls are synchronous or minimal-latency.
     * The git call is async but has a 2-second timeout.
     */
    async collect(): Promise<WorkspaceContext> {
        const editor = vscode.window.activeTextEditor;
        const root = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? '';

        // Active file
        const activeFile = editor
            ? _relativePath(editor.document.uri.fsPath, root)
            : undefined;
        const language = editor?.document.languageId;

        // Selection
        const selection =
            editor && !editor.selection.isEmpty
                ? editor.document.getText(editor.selection)
                : '';

        // Open files
        const openFiles = vscode.window.tabGroups.all
            .flatMap((g) => g.tabs)
            .map((t) => {
                const input = t.input as { uri?: vscode.Uri } | null;
                return input?.uri ? _relativePath(input.uri.fsPath, root) : null;
            })
            .filter((p): p is string => p !== null && p !== '');
        const unique = [...new Set(openFiles)].slice(0, 20);

        // Diagnostics for active file
        const diagnostics = editor
            ? _formatDiagnostics(
                  vscode.languages
                      .getDiagnostics(editor.document.uri)
                      .filter((d) => d.severity <= vscode.DiagnosticSeverity.Warning)
                      .slice(0, 10),
              )
            : '';

        // Git info
        const gitInfo = await _gitInfo(root);

        // Build formatted block
        const parts: string[] = [];
        if (activeFile) {
            parts.push(`Active file: ${activeFile}${language ? ` (${language})` : ''}`);
        }
        if (selection) {
            parts.push(`Selected text:\n\`\`\`\n${selection.slice(0, 2000)}\n\`\`\``);
        }
        if (unique.length) {
            parts.push(`Open files: ${unique.join(', ')}`);
        }
        if (diagnostics) {
            parts.push(`Diagnostics:\n${diagnostics}`);
        }
        if (gitInfo) {
            parts.push(`Git: ${gitInfo}`);
        }

        const formatted =
            parts.length
                ? `--- Workspace Context ---\n${parts.join('\n')}\n------------------------`
                : '';

        return { activeFile, language, selection, openFiles: unique, diagnostics, gitInfo, formatted };
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

function _relativePath(absolute: string, root: string): string {
    if (root && absolute.startsWith(root)) {
        return absolute.slice(root.length).replace(/^[\\/]/, '');
    }
    return absolute;
}

function _formatDiagnostics(diags: vscode.Diagnostic[]): string {
    if (!diags.length) {
        return '';
    }
    return diags
        .map((d) => {
            const sev = d.severity === vscode.DiagnosticSeverity.Error ? 'ERROR' : 'WARN';
            const ln = d.range.start.line + 1;
            return `  [${sev} L${ln}] ${d.message}`;
        })
        .join('\n');
}

async function _gitInfo(root: string): Promise<string> {
    if (!root) {
        return '';
    }
    try {
        const opts = { cwd: root, timeout: 2000 };
        const [branchResult, logResult] = await Promise.all([
            execAsync('git rev-parse --abbrev-ref HEAD', opts).catch(() => ({ stdout: '' })),
            execAsync('git log -1 --pretty=%s', opts).catch(() => ({ stdout: '' })),
        ]);
        const branch = branchResult.stdout.trim();
        const commit = logResult.stdout.trim().slice(0, 80);
        if (!branch) {
            return '';
        }
        return commit ? `${branch} — "${commit}"` : branch;
    } catch {
        return '';
    }
}
