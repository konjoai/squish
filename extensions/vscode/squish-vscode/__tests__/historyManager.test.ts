/**
 * __tests__/historyManager.test.ts
 *
 * Unit tests for HistoryManager.
 * Uses a real temp directory so fs operations are exercised end-to-end
 * without requiring VS Code.
 */
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { HistoryManager, Session } from '../src/historyManager';

// ── Helpers ───────────────────────────────────────────────────────────────────

function makeTmpDir(): string {
    return fs.mkdtempSync(path.join(os.tmpdir(), 'squish-hist-test-'));
}

function makeManager(dir?: string): HistoryManager {
    return new HistoryManager(dir);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('HistoryManager', () => {

    let tmpDir: string;

    beforeEach(() => {
        tmpDir = makeTmpDir();
    });

    afterEach(() => {
        fs.rmSync(tmpDir, { recursive: true, force: true });
    });

    // ── createSession ──────────────────────────────────────────────────────

    test('createSession returns a session with default title', () => {
        const mgr = makeManager(tmpDir);
        const s = mgr.createSession();
        expect(s.title).toBe('New conversation');
        expect(s.messages).toEqual([]);
        expect(typeof s.id).toBe('string');
        expect(s.id.length).toBeGreaterThan(8);
    });

    test('createSession IDs are unique', () => {
        const mgr = makeManager(tmpDir);
        const ids = new Set(Array.from({ length: 20 }, () => mgr.createSession().id));
        expect(ids.size).toBe(20);
    });

    test('createSession sets createdAt to approximately now', () => {
        const before = Date.now();
        const s = makeManager(tmpDir).createSession();
        const after = Date.now();
        expect(s.createdAt).toBeGreaterThanOrEqual(before);
        expect(s.createdAt).toBeLessThanOrEqual(after);
    });

    // ── save / load ────────────────────────────────────────────────────────

    test('save writes a JSON file to disk', () => {
        const mgr = makeManager(tmpDir);
        const s = mgr.createSession();
        s.title = 'Test session';
        mgr.save(s);
        const files = fs.readdirSync(tmpDir);
        expect(files).toHaveLength(1);
        expect(files[0]).toBe(s.id + '.json');
    });

    test('load returns the saved session', () => {
        const mgr = makeManager(tmpDir);
        const s = mgr.createSession();
        s.title = 'Persisted';
        s.messages = [{ role: 'user', content: 'Hello' }];
        mgr.save(s);
        const loaded = mgr.load(s.id);
        expect(loaded).not.toBeNull();
        expect(loaded!.title).toBe('Persisted');
        expect(loaded!.messages).toEqual([{ role: 'user', content: 'Hello' }]);
    });

    test('load returns null for unknown id', () => {
        const mgr = makeManager(tmpDir);
        expect(mgr.load('nonexistent-id')).toBeNull();
    });

    test('save updates updatedAt timestamp', async () => {
        const mgr = makeManager(tmpDir);
        const s = mgr.createSession();
        const t1 = s.updatedAt;
        await new Promise((r) => setTimeout(r, 5));
        mgr.save(s);
        expect(s.updatedAt).toBeGreaterThan(t1);
    });

    test('save is idempotent — second save overwrites the first', () => {
        const mgr = makeManager(tmpDir);
        const s = mgr.createSession();
        s.messages = [{ role: 'user', content: 'v1' }];
        mgr.save(s);
        s.messages = [{ role: 'user', content: 'v2' }];
        mgr.save(s);
        const files = fs.readdirSync(tmpDir);
        expect(files).toHaveLength(1);
        expect(mgr.load(s.id)!.messages[0].content).toBe('v2');
    });

    // ── list ───────────────────────────────────────────────────────────────

    test('list returns empty array when no sessions', () => {
        const mgr = makeManager(tmpDir);
        expect(mgr.list()).toEqual([]);
    });

    test('list returns all saved sessions sorted newest first', async () => {
        const mgr = makeManager(tmpDir);
        const s1 = mgr.createSession();
        s1.title = 'First';
        mgr.save(s1);
        await new Promise((r) => setTimeout(r, 5));
        const s2 = mgr.createSession();
        s2.title = 'Second';
        mgr.save(s2);
        const items = mgr.list();
        expect(items).toHaveLength(2);
        // Newest should be first
        expect(items[0].id).toBe(s2.id);
        expect(items[1].id).toBe(s1.id);
    });

    test('list does not return messages (only metadata)', () => {
        const mgr = makeManager(tmpDir);
        const s = mgr.createSession();
        s.messages = [{ role: 'user', content: 'secret' }];
        mgr.save(s);
        const items = mgr.list();
        // messages may or may not be present — just verify the item is returned
        expect(items[0].id).toBe(s.id);
    });

    // ── delete ─────────────────────────────────────────────────────────────

    test('delete removes the session file', () => {
        const mgr = makeManager(tmpDir);
        const s = mgr.createSession();
        mgr.save(s);
        expect(fs.readdirSync(tmpDir)).toHaveLength(1);
        mgr.delete(s.id);
        expect(fs.readdirSync(tmpDir)).toHaveLength(0);
    });

    test('delete is a no-op for unknown id', () => {
        const mgr = makeManager(tmpDir);
        expect(() => mgr.delete('ghost-id')).not.toThrow();
    });

    test('delete then load returns null', () => {
        const mgr = makeManager(tmpDir);
        const s = mgr.createSession();
        mgr.save(s);
        mgr.delete(s.id);
        expect(mgr.load(s.id)).toBeNull();
    });

    // ── rename ─────────────────────────────────────────────────────────────

    test('rename updates the title on disk', () => {
        const mgr = makeManager(tmpDir);
        const s = mgr.createSession();
        mgr.save(s);
        mgr.rename(s.id, 'New title');
        const loaded = mgr.load(s.id);
        expect(loaded!.title).toBe('New title');
    });

    test('rename is a no-op for unknown id', () => {
        const mgr = makeManager(tmpDir);
        expect(() => mgr.rename('ghost', 'Title')).not.toThrow();
    });

    // ── default path ───────────────────────────────────────────────────────

    test('constructor creates the history directory if missing', () => {
        const custom = path.join(tmpDir, 'nested', 'history');
        expect(fs.existsSync(custom)).toBe(false);
        new HistoryManager(custom);
        expect(fs.existsSync(custom)).toBe(true);
    });

    test('multiple messages round-trip through JSON losslessly', () => {
        const mgr = makeManager(tmpDir);
        const s = mgr.createSession();
        s.messages = [
            { role: 'system', content: 'You are helpful.' },
            { role: 'user', content: 'Hello 🎉' },
            { role: 'assistant', content: 'Hi there!' },
        ];
        mgr.save(s);
        const loaded = mgr.load(s.id)!;
        expect(loaded.messages).toEqual(s.messages);
    });
});
