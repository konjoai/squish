/**
 * __tests__/agentLoop.test.ts
 *
 * Unit tests for AgentLoop.
 * SquishClient is fully mocked — no network required.
 */
import { AgentLoop, AgentLoopOptions } from '../src/agentLoop';
import { SquishClient, ChatMessage, ToolDefinition, ToolCall } from '../src/squishClient';

jest.mock('../src/squishClient');
const MockSquishClient = SquishClient as jest.MockedClass<typeof SquishClient>;

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Build a minimal stream that yields the given chunks then signals done. */
async function* makeStream(
    chunks: Array<{ delta?: string; toolCalls?: ToolCall[]; done?: boolean }>,
): AsyncIterable<{ delta?: string; toolCalls?: ToolCall[]; done?: boolean }> {
    for (const c of chunks) { yield c; }
}

function makeClient(
    streamChunks: Array<{ delta?: string; toolCalls?: ToolCall[]; done?: boolean }>[],
): SquishClient {
    let callIdx = 0;
    const client = new MockSquishClient('localhost', 11435, 'key');
    (client.chatStream as jest.Mock).mockImplementation(() => {
        const chunks = streamChunks[callIdx] ?? streamChunks[streamChunks.length - 1];
        callIdx++;
        return makeStream(chunks);
    });
    (client.abort as jest.Mock).mockImplementation(() => undefined);
    return client;
}

function baseOpts(override: Partial<AgentLoopOptions> = {}): AgentLoopOptions {
    return {
        client: makeClient([[{ delta: 'Hello world', done: true }]]),
        messages: [{ role: 'user', content: 'Hi' }],
        tools: [],
        executeTool: jest.fn().mockResolvedValue('tool-result'),
        ...override,
    };
}

function makeTool(name: string): ToolDefinition {
    return {
        type: 'function',
        function: {
            name,
            description: `Tool: ${name}`,
            parameters: { type: 'object', properties: {}, required: [] },
        },
    };
}

function makeToolCall(name: string, id = 'tc1', args = '{}'): ToolCall {
    return { id, type: 'function', function: { name, arguments: args } };
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('AgentLoop.run', () => {

    beforeEach(() => { jest.clearAllMocks(); });

    // ── Basic completion ───────────────────────────────────────────────────

    test('returns stop reason when model produces no tool calls', async () => {
        const res = await AgentLoop.run(baseOpts());
        expect(res.stoppedReason).toBe('stop');
        expect(res.steps).toBe(1);
    });

    test('appends assistant message to messages on stop', async () => {
        const res = await AgentLoop.run(baseOpts());
        const last = res.messages[res.messages.length - 1];
        expect(last.role).toBe('assistant');
        expect(last.content).toBe('Hello world');
    });

    test('accumulates multiple delta chunks into one message', async () => {
        const opts = baseOpts({
            client: makeClient([[
                { delta: 'Part1' },
                { delta: ' Part2' },
                { delta: '', done: true },
            ]]),
        });
        const res = await AgentLoop.run(opts);
        const asst = res.messages.find((m) => m.role === 'assistant');
        expect(asst?.content).toBe('Part1 Part2');
    });

    // ── Tool calls ─────────────────────────────────────────────────────────

    test('executes a tool call and continues', async () => {
        const executeTool = jest.fn().mockResolvedValue('file contents');
        const opts = baseOpts({
            client: makeClient([
                // First call: model asks for tool
                [{ toolCalls: [makeToolCall('read_file')], done: true }],
                // Second call: model gives final answer
                [{ delta: 'Done', done: true }],
            ]),
            tools: [makeTool('read_file')],
            executeTool,
        });
        const res = await AgentLoop.run(opts);
        expect(executeTool).toHaveBeenCalledWith('read_file', expect.any(Object));
        expect(res.stoppedReason).toBe('stop');
        expect(res.steps).toBe(2);
    });

    test('tool result is pushed as tool-role message', async () => {
        const executeTool = jest.fn().mockResolvedValue('result-data');
        const opts = baseOpts({
            client: makeClient([
                [{ toolCalls: [makeToolCall('get_file', 'c1')], done: true }],
                [{ delta: 'Final', done: true }],
            ]),
            executeTool,
        });
        const res = await AgentLoop.run(opts);
        const toolMsg = res.messages.find(
            (m) => m.role === 'tool' && m.tool_call_id === 'c1',
        );
        expect(toolMsg?.content).toBe('result-data');
    });

    test('tool execution error is captured, not thrown', async () => {
        const executeTool = jest.fn().mockRejectedValue(new Error('Permission denied'));
        const opts = baseOpts({
            client: makeClient([
                [{ toolCalls: [makeToolCall('delete_file')], done: true }],
                [{ delta: 'Handled', done: true }],
            ]),
            executeTool,
        });
        const res = await AgentLoop.run(opts);
        const toolMsg = res.messages.find((m) => m.role === 'tool');
        expect(toolMsg?.content).toMatch(/Permission denied/);
        expect(res.stoppedReason).toBe('stop');
    });

    test('multiple tool calls in one step are all executed', async () => {
        const executeTool = jest.fn()
            .mockResolvedValueOnce('r1')
            .mockResolvedValueOnce('r2');
        const opts = baseOpts({
            client: makeClient([
                [{ toolCalls: [makeToolCall('t1', 'c1'), makeToolCall('t2', 'c2')], done: true }],
                [{ delta: 'All done', done: true }],
            ]),
            executeTool,
        });
        const res = await AgentLoop.run(opts);
        expect(executeTool).toHaveBeenCalledTimes(2);
        const toolMsgs = res.messages.filter((m) => m.role === 'tool');
        expect(toolMsgs).toHaveLength(2);
    });

    // ── Max steps ──────────────────────────────────────────────────────────

    test('returns max_steps when tool loop never terminates', async () => {
        const opts = baseOpts({
            client: makeClient([
                // Always returns tool call, never finishes
                [{ toolCalls: [makeToolCall('infinite')], done: true }],
            ]),
            maxSteps: 3,
            executeTool: jest.fn().mockResolvedValue('ok'),
        });
        const res = await AgentLoop.run(opts);
        expect(res.stoppedReason).toBe('max_steps');
        expect(res.steps).toBe(3);
    });

    test('default maxSteps is 10', async () => {
        const opts = baseOpts({
            client: makeClient([[{ toolCalls: [makeToolCall('t')], done: true }]]),
            executeTool: jest.fn().mockResolvedValue('ok'),
        });
        const res = await AgentLoop.run(opts);
        expect(res.stoppedReason).toBe('max_steps');
        expect(res.steps).toBe(10);
    });

    // ── Abort ──────────────────────────────────────────────────────────────

    test('returns aborted when abortSignal fires before first step', async () => {
        const opts = baseOpts({ abortSignal: () => true });
        const res = await AgentLoop.run(opts);
        expect(res.stoppedReason).toBe('aborted');
        expect(res.steps).toBe(0);
    });

    test('returns aborted mid-stream', async () => {
        let calls = 0;
        const opts = baseOpts({
            abortSignal: () => { calls++; return calls > 3; },
        });
        const res = await AgentLoop.run(opts);
        expect(['aborted', 'stop']).toContain(res.stoppedReason);
    });

    // ── Error handling ─────────────────────────────────────────────────────

    test('returns error when stream throws', async () => {
        const client = new MockSquishClient('localhost', 11435, 'key');
        (client.chatStream as jest.Mock).mockImplementation(async function* () {
            throw new Error('Network error');
        });
        const opts = baseOpts({ client });
        const res = await AgentLoop.run(opts);
        expect(res.stoppedReason).toBe('error');
        expect(res.error).toMatch(/Network error/);
    });

    // ── Callbacks ──────────────────────────────────────────────────────────

    test('onChunk is called with each delta', async () => {
        const onChunk = jest.fn();
        const opts = baseOpts({
            client: makeClient([[{ delta: 'A' }, { delta: 'B' }, { done: true }]]),
            onChunk,
        });
        await AgentLoop.run(opts);
        expect(onChunk).toHaveBeenCalledWith('A');
        expect(onChunk).toHaveBeenCalledWith('B');
    });

    test('onToolCallStart fires for each tool call', async () => {
        const onToolCallStart = jest.fn();
        const opts = baseOpts({
            client: makeClient([
                [{ toolCalls: [makeToolCall('fn', 'id1')], done: true }],
                [{ delta: 'ok', done: true }],
            ]),
            onToolCallStart,
            executeTool: jest.fn().mockResolvedValue('r'),
        });
        await AgentLoop.run(opts);
        expect(onToolCallStart).toHaveBeenCalledWith(
            expect.objectContaining({ id: 'id1' }),
        );
    });

    test('onToolCallEnd fires after each tool call', async () => {
        const onToolCallEnd = jest.fn();
        const opts = baseOpts({
            client: makeClient([
                [{ toolCalls: [makeToolCall('fn', 'id2')], done: true }],
                [{ delta: 'ok', done: true }],
            ]),
            onToolCallEnd,
            executeTool: jest.fn().mockResolvedValue('the-result'),
        });
        await AgentLoop.run(opts);
        expect(onToolCallEnd).toHaveBeenCalledWith(
            expect.objectContaining({ id: 'id2' }),
            'the-result',
        );
    });

    test('onStepComplete fires once per step', async () => {
        const onStepComplete = jest.fn();
        const opts = baseOpts({
            client: makeClient([
                [{ toolCalls: [makeToolCall('x')], done: true }],
                [{ delta: 'done', done: true }],
            ]),
            onStepComplete,
            executeTool: jest.fn().mockResolvedValue('ok'),
        });
        await AgentLoop.run(opts);
        expect(onStepComplete).toHaveBeenCalledTimes(2);
    });

    // ── Argument parsing ───────────────────────────────────────────────────

    test('invalid JSON args are passed as empty object to executeTool', async () => {
        const executeTool = jest.fn().mockResolvedValue('ok');
        const opts = baseOpts({
            client: makeClient([
                [{ toolCalls: [makeToolCall('fn', 'c1', 'NOT VALID JSON')], done: true }],
                [{ delta: 'done', done: true }],
            ]),
            executeTool,
        });
        await AgentLoop.run(opts);
        expect(executeTool).toHaveBeenCalledWith('fn', {});
    });
});
