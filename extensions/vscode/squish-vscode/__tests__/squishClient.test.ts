/**
 * __tests__/squishClient.test.ts
 *
 * Unit tests for SquishClient.
 * Mocks Node http so we never open a real socket.
 */
import * as http from 'http';
import { SquishClient, ChatChunk } from '../src/squishClient';

jest.mock('http');

const mockedHttp = http as jest.Mocked<typeof http>;

// ── Helpers ───────────────────────────────────────────────────────────────────

function mockGetResponse(body: string, statusCode = 200): void {
    const mockRes = Object.assign(
        Object.create(require('events').EventEmitter.prototype),
        { statusCode, setEncoding: jest.fn() },
    );
    const mockReq = Object.assign(
        Object.create(require('events').EventEmitter.prototype),
        { end: jest.fn(), write: jest.fn(), setTimeout: jest.fn() },
    );
    mockedHttp.request.mockImplementation((_opts: unknown, cb?: unknown) => {
        if (typeof cb === 'function') {
            setTimeout(() => {
                cb(mockRes as http.IncomingMessage);
                mockRes.emit('data', body);
                mockRes.emit('end');
            }, 0);
        }
        return mockReq as unknown as http.ClientRequest;
    });
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('SquishClient.health()', () => {
    const client = new SquishClient('127.0.0.1', 11435, 'squish');

    test('loaded=true when server is ready', async () => {
        mockGetResponse(JSON.stringify({
            loaded: true, model: 'qwen3:8b', avg_tps: 12.5, uptime_s: 300, requests: 7,
        }));
        const h = await client.health();
        expect(h.loaded).toBe(true);
        expect(h.model).toBe('qwen3:8b');
        expect(h.tps).toBeCloseTo(12.5);
        expect(h.uptime).toBe(300);
        expect(h.requests).toBe(7);
    });

    test('loaded=false when server reports not loaded', async () => {
        mockGetResponse(JSON.stringify({ loaded: false, status: 'no_model' }));
        const h = await client.health();
        expect(h.loaded).toBe(false);
    });

    test('rejects on network error', async () => {
        const mockReq = Object.assign(
            Object.create(require('events').EventEmitter.prototype),
            { end: jest.fn(), write: jest.fn(), setTimeout: jest.fn() },
        );
        mockedHttp.request.mockImplementation(() => {
            setTimeout(() => mockReq.emit('error', new Error('ECONNREFUSED')), 0);
            return mockReq as unknown as http.ClientRequest;
        });
        await expect(client.health()).rejects.toThrow('ECONNREFUSED');
    });

    test('sets Authorization header', async () => {
        mockGetResponse(JSON.stringify({ loaded: false }));
        await client.health();
        const callArgs = mockedHttp.request.mock.calls.at(-1)![0] as http.RequestOptions;
        expect((callArgs.headers as Record<string, string>)?.['Authorization']).toBe('Bearer squish');
    });
});

describe('SquishClient.models()', () => {
    const client = new SquishClient('127.0.0.1', 11435, 'squish');

    test('returns model id list', async () => {
        mockGetResponse(JSON.stringify({
            data: [{ id: 'qwen3:8b' }, { id: 'gemma3:4b' }],
        }));
        const models = await client.models();
        expect(models).toEqual(['qwen3:8b', 'gemma3:4b']);
    });

    test('returns empty list when data is missing', async () => {
        mockGetResponse(JSON.stringify({}));
        const models = await client.models();
        expect(models).toEqual([]);
    });
});

describe('SquishClient.streamChat()', () => {
    const client = new SquishClient('127.0.0.1', 11435, 'squish');

    function sseLines(chunks: object[]): string {
        return chunks.map(c => `data: ${JSON.stringify(c)}\n`).join('\n') + '\ndata: [DONE]\n';
    }

    function mockStreamResponse(body: string): void {
        const mockRes = Object.assign(
            Object.create(require('events').EventEmitter.prototype),
            { setEncoding: jest.fn() },
        );
        const mockReq = Object.assign(
            Object.create(require('events').EventEmitter.prototype),
            { end: jest.fn(), write: jest.fn() },
        );
        mockedHttp.request.mockImplementation((_opts: unknown, cb?: unknown) => {
            if (typeof cb === 'function') {
                setTimeout(() => {
                    cb(mockRes as http.IncomingMessage);
                    mockRes.emit('data', body);
                    mockRes.emit('end');
                }, 0);
            }
            return mockReq as unknown as http.ClientRequest;
        });
    }

    test('calls onChunk with delta tokens', (done) => {
        const sse = sseLines([
            { choices: [{ delta: { content: 'Hello' }, finish_reason: null }] },
            { choices: [{ delta: { content: ' world' }, finish_reason: 'stop' }] },
        ]);
        mockStreamResponse(sse);

        const chunks: ChatChunk[] = [];
        client.streamChat(
            [{ role: 'user', content: 'hi' }],
            128,
            0.7,
            'qwen3:8b',
            (c) => { chunks.push(c); if (c.done) {
                const deltas = chunks.filter(c => c.delta).map(c => c.delta);
                expect(deltas).toContain('Hello');
                expect(deltas).toContain(' world');
                done();
            }},
            (err) => done(err),
        );
    });

    test('includes model in POST body', (done) => {
        const sse = 'data: [DONE]\n';
        const mockRes = Object.assign(
            Object.create(require('events').EventEmitter.prototype),
            { setEncoding: jest.fn() },
        );
        let written = '';
        const mockReq = Object.assign(
            Object.create(require('events').EventEmitter.prototype),
            { end: jest.fn(), write: jest.fn((d: string) => { written += d; }) },
        );
        mockedHttp.request.mockImplementation((_opts: unknown, cb?: unknown) => {
            if (typeof cb === 'function') {
                setTimeout(() => {
                    cb(mockRes as http.IncomingMessage);
                    mockRes.emit('data', sse);
                    mockRes.emit('end');
                }, 0);
            }
            return mockReq as unknown as http.ClientRequest;
        });

        client.streamChat(
            [{ role: 'user', content: 'test' }],
            256, 0.5, 'gemma3:4b',
            (c) => { if (c.done) {
                const body = JSON.parse(written);
                expect(body.model).toBe('gemma3:4b');
                expect(body.stream).toBe(true);
                done();
            }},
            (err) => done(err),
        );
    });

    test('calls onError on request error', (done) => {
        const mockReq = Object.assign(
            Object.create(require('events').EventEmitter.prototype),
            { end: jest.fn(), write: jest.fn() },
        );
        mockedHttp.request.mockImplementation(() => {
            setTimeout(() => mockReq.emit('error', new Error('network down')), 0);
            return mockReq as unknown as http.ClientRequest;
        });
        client.streamChat(
            [{ role: 'user', content: 'hi' }],
            128, 0.7, '7b',
            () => {},
            (err) => { expect(err.message).toBe('network down'); done(); },
        );
    });

    test('handles res end with done callback', (done) => {
        const mockRes = Object.assign(
            Object.create(require('events').EventEmitter.prototype),
            { setEncoding: jest.fn() },
        );
        const mockReq = Object.assign(
            Object.create(require('events').EventEmitter.prototype),
            { end: jest.fn(), write: jest.fn() },
        );
        mockedHttp.request.mockImplementation((_opts: unknown, cb?: unknown) => {
            if (typeof cb === 'function') {
                setTimeout(() => {
                    cb(mockRes as http.IncomingMessage);
                    // Emit end without data to trigger res.on('end') path
                    mockRes.emit('end');
                }, 0);
            }
            return mockReq as unknown as http.ClientRequest;
        });

        client.streamChat(
            [{ role: 'user', content: 'test' }],
            128, 0.7, '7b',
            (c) => { if (c.done) { done(); }},
            (err) => done(err),
        );
    });
});
