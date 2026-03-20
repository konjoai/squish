/**
 * __tests__/squishClient.test.ts
 *
 * Unit tests for SquishClient.
 * Mocks Node http so we never open a real socket.
 */
import * as http from 'http';
import { SquishClient, ChatChunk, ToolDefinition } from '../src/squishClient';

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

    // ── Tool calling ───────────────────────────────────────────────────────

    test('includes tools array in POST body when provided', (done) => {
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

        const tools: ToolDefinition[] = [{
            type: 'function',
            function: {
                name: 'read_file',
                description: 'Read a file',
                parameters: { type: 'object', properties: { path: { type: 'string' } }, required: ['path'] },
            },
        }];

        client.streamChat(
            [{ role: 'user', content: 'read src/index.ts' }],
            256, 0.5, 'qwen3:8b',
            (c) => { if (c.done) {
                const body = JSON.parse(written);
                expect(body.tools).toBeDefined();
                expect(body.tools).toHaveLength(1);
                expect(body.tools[0].function.name).toBe('read_file');
                expect(body.tool_choice).toBe('auto');
                done();
            }},
            (err) => done(err),
            tools,
        );
    });

    test('omits tools from body when empty array provided', (done) => {
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
            [{ role: 'user', content: 'hi' }],
            128, 0.7, '7b',
            (c) => { if (c.done) {
                const body = JSON.parse(written);
                expect(body.tools).toBeUndefined();
                expect(body.tool_choice).toBeUndefined();
                done();
            }},
            (err) => done(err),
            [],
        );
    });

    test('accumulates streamed tool_calls across SSE chunks and emits on finish', (done) => {
        // Simulate a model that emits tool_calls spread across multiple SSE events
        const sse = [
            `data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'read_file', arguments: '' } }] }, finish_reason: null }] })}\n`,
            `data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: '{"path":' } }] }, finish_reason: null }] })}\n`,
            `data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: '"src/index.ts"}' } }] }, finish_reason: 'tool_calls' }] })}\n`,
            'data: [DONE]\n',
        ].join('\n');
        mockStreamResponse(sse);

        client.streamChat(
            [{ role: 'user', content: 'read the file' }],
            128, 0.7, '7b',
            (c) => { if (c.done) {
                expect(c.finishReason).toBe('tool_calls');
                expect(c.toolCalls).toBeDefined();
                expect(c.toolCalls).toHaveLength(1);
                const tc = c.toolCalls![0];
                expect(tc.id).toBe('call_1');
                expect(tc.function.name).toBe('read_file');
                expect(tc.function.arguments).toBe('{"path":"src/index.ts"}');
                done();
            }},
            (err) => done(err),
        );
    });

    test('tool_calls is undefined when model produces only text', (done) => {
        const sse = sseLines([
            { choices: [{ delta: { content: 'Just text' }, finish_reason: 'stop' }] },
        ]);
        mockStreamResponse(sse);

        client.streamChat(
            [{ role: 'user', content: 'hi' }],
            128, 0.7, '7b',
            (c) => { if (c.done) {
                expect(c.toolCalls).toBeUndefined();
                done();
            }},
            (err) => done(err),
        );
    });

    test('tool message role is accepted in messages array', (done) => {
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
            [
                { role: 'user', content: 'read src/index.ts' },
                { role: 'assistant', content: null, tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'read_file', arguments: '{"path":"src/index.ts"}' } }] },
                { role: 'tool', content: 'file contents here', tool_call_id: 'call_1', name: 'read_file' },
            ],
            128, 0.7, '7b',
            (c) => { if (c.done) {
                const body = JSON.parse(written);
                const msgs = body.messages;
                expect(msgs[1].role).toBe('assistant');
                expect(msgs[1].tool_calls).toBeDefined();
                expect(msgs[2].role).toBe('tool');
                expect(msgs[2].tool_call_id).toBe('call_1');
                done();
            }},
            (err) => done(err),
        );
    });

    test('falls back to JSON parsing when server returns non-SSE completion', (done) => {
        // Squish server forces stream=false when tools are provided and model
        // produces text — returns plain JSON instead of SSE
        const jsonBody = JSON.stringify({
            id: 'chatcmpl-test',
            object: 'chat.completion',
            choices: [{
                index: 0,
                message: { role: 'assistant', content: 'Here is your answer.' },
                finish_reason: 'stop',
            }],
        });
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
                    mockRes.emit('data', jsonBody);
                    mockRes.emit('end');
                }, 0);
            }
            return mockReq as unknown as http.ClientRequest;
        });

        const chunks: ChatChunk[] = [];
        client.streamChat(
            [{ role: 'user', content: 'hi' }],
            128, 0.7, '7b',
            (c) => {
                chunks.push(c);
                if (c.done) {
                    const content = chunks.filter(c => c.delta).map(c => c.delta).join('');
                    expect(content).toBe('Here is your answer.');
                    expect(c.finishReason).toBeNull(); // emitDone() called without reason
                    done();
                }
            },
            (err) => done(err),
        );
    });

    test('JSON fallback handles non-streaming tool_calls response', (done) => {
        const jsonBody = JSON.stringify({
            id: 'chatcmpl-tools',
            object: 'chat.completion',
            choices: [{
                index: 0,
                message: {
                    role: 'assistant',
                    content: null,
                    tool_calls: [{
                        id: 'call_abc',
                        type: 'function',
                        function: { name: 'read_file', arguments: '{"path":"README.md"}' },
                    }],
                },
                finish_reason: 'tool_calls',
            }],
        });
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
                    mockRes.emit('data', jsonBody);
                    mockRes.emit('end');
                }, 0);
            }
            return mockReq as unknown as http.ClientRequest;
        });

        client.streamChat(
            [{ role: 'user', content: 'read file' }],
            128, 0.7, '7b',
            (c) => { if (c.done) {
                expect(c.finishReason).toBe('tool_calls');
                expect(c.toolCalls).toBeDefined();
                expect(c.toolCalls).toHaveLength(1);
                expect(c.toolCalls![0].id).toBe('call_abc');
                expect(c.toolCalls![0].function.name).toBe('read_file');
                done();
            }},
            (err) => done(err),
        );
    });
});

// ── SquishClient.abort() ──────────────────────────────────────────────────────

describe('SquishClient.abort()', () => {
    test('calls destroy on the active request', () => {
        const mockRes = Object.assign(
            Object.create(require('events').EventEmitter.prototype),
            { setEncoding: jest.fn() },
        );
        const mockReq = Object.assign(
            Object.create(require('events').EventEmitter.prototype),
            { end: jest.fn(), write: jest.fn(), destroy: jest.fn() },
        );
        mockedHttp.request.mockImplementation((_opts: unknown, cb?: unknown) => {
            if (typeof cb === 'function') {
                // Register the callback but do not emit anything — simulates an
                // in-flight request that has not produced any data yet.
                setTimeout(() => { cb(mockRes as http.IncomingMessage); }, 100);
            }
            return mockReq as unknown as http.ClientRequest;
        });

        const client = new SquishClient('127.0.0.1', 11435, 'squish');
        client.streamChat(
            [{ role: 'user', content: 'hi' }],
            128, 0.7, '7b',
            () => {},
            () => {},
        );
        client.abort();
        expect(mockReq.destroy).toHaveBeenCalled();
    });

    test('is safe to call when no active request', () => {
        const client = new SquishClient('127.0.0.1', 11435, 'squish');
        expect(() => client.abort()).not.toThrow();
    });
});

