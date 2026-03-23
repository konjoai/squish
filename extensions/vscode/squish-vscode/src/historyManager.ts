/**
 * squish-vscode/src/historyManager.ts
 *
 * Persists conversation sessions to disk as JSON files in
 * ~/.squish/history/ (configurable via squish.historyPath).
 *
 * Each session is stored as:
 *   <historyPath>/<session-id>.json
 *
 * Session JSON schema:
 * {
 *   id: string,          // UUID-style unique identifier
 *   title: string,       // First user message, truncated
 *   createdAt: number,   // Unix ms
 *   updatedAt: number,   // Unix ms
 *   messages: ChatMessage[]
 * }
 */
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { ChatMessage } from './squishClient';

export interface Session {
    id: string;
    title: string;
    createdAt: number;
    updatedAt: number;
    messages: ChatMessage[];
}

export class HistoryManager {
    private readonly _dir: string;

    constructor(customPath?: string) {
        this._dir = customPath && customPath.trim()
            ? customPath.trim()
            : path.join(os.homedir(), '.squish', 'history');
        this._ensureDir();
    }

    // ── Public API ────────────────────────────────────────────────────────

    /** Create a brand-new session in memory (does not write to disk yet). */
    createSession(): Session {
        return {
            id: _uuid(),
            title: 'New conversation',
            createdAt: Date.now(),
            updatedAt: Date.now(),
            messages: [],
        };
    }

    /** Persist (or overwrite) a session to disk. */
    save(session: Session): void {
        session.updatedAt = Date.now();
        const file = this._filePath(session.id);
        fs.writeFileSync(file, JSON.stringify(session, null, 2), 'utf8');
    }

    /** Load a session by id.  Returns null if not found. */
    load(id: string): Session | null {
        const file = this._filePath(id);
        if (!fs.existsSync(file)) {
            return null;
        }
        try {
            return JSON.parse(fs.readFileSync(file, 'utf8')) as Session;
        } catch {
            return null;
        }
    }

    /** List all sessions sorted by updatedAt descending. */
    list(): Session[] {
        let files: string[];
        try {
            files = fs.readdirSync(this._dir).filter((f) => f.endsWith('.json'));
        } catch {
            return [];
        }
        const sessions: Session[] = [];
        for (const f of files) {
            try {
                const raw = fs.readFileSync(path.join(this._dir, f), 'utf8');
                sessions.push(JSON.parse(raw) as Session);
            } catch {
                // skip corrupt files
            }
        }
        sessions.sort((a, b) => b.updatedAt - a.updatedAt);
        return sessions;
    }

    /** Delete a session from disk.  Silently succeeds if already absent. */
    delete(id: string): void {
        const file = this._filePath(id);
        if (fs.existsSync(file)) {
            fs.unlinkSync(file);
        }
    }

    /** Rename a session's title and save. */
    rename(id: string, newTitle: string): void {
        const session = this.load(id);
        if (!session) {
            return;
        }
        session.title = newTitle.slice(0, 80);
        this.save(session);
    }

    /** Derive a title from the first user message in the list. */
    static titleFromMessages(messages: ChatMessage[]): string {
        const first = messages.find((m) => m.role === 'user');
        if (!first || !first.content) {
            return 'New conversation';
        }
        const text = typeof first.content === 'string' ? first.content : '';
        return text.length > 60 ? text.slice(0, 57) + '…' : text || 'New conversation';
    }

    // ── Private ───────────────────────────────────────────────────────────

    private _filePath(id: string): string {
        // Sanitise id to avoid path traversal
        const safe = id.replace(/[^a-zA-Z0-9_-]/g, '_');
        return path.join(this._dir, `${safe}.json`);
    }

    private _ensureDir(): void {
        if (!fs.existsSync(this._dir)) {
            fs.mkdirSync(this._dir, { recursive: true });
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

/** Simple collision-resistant ID (no crypto dependency). */
function _uuid(): string {
    const hex = (n: number) =>
        Math.floor(Math.random() * (2 ** (n * 4))).toString(16).padStart(n, '0');
    return `${hex(8)}-${hex(4)}-4${hex(3)}-${(8 + Math.floor(Math.random() * 4)).toString(16)}${hex(3)}-${hex(12)}`;
}
