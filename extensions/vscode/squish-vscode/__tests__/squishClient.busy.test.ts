/**
 * __tests__/squishClient.busy.test.ts
 *
 * SquishClient.busy must reflect in-flight streamChat generations. The health
 * poll relies on it to avoid flashing "offline" while the model is busy (a
 * blocking prefill can stall the port probe / health response past their
 * timeouts — that is NOT the server being down).
 *
 * Own file so the process-wide _inFlight counter starts clean (jest isolates
 * module state per test file). Node http is mocked — no real socket.
 */
import * as http from 'http';
import { EventEmitter } from 'events';

jest.mock('http');
const mockedHttp = http as jest.Mocked<typeof http>;

import { SquishClient } from '../src/squishClient';

function mockReqEmitter(extra: Record<string, unknown> = {}): EventEmitter {
    const req = Object.assign(new EventEmitter(), { end: jest.fn(), write: jest.fn() }, extra);
    mockedHttp.request.mockImplementation(() => req as unknown as http.ClientRequest);
    return req as unknown as EventEmitter;
}

function startStream(client: SquishClient): void {
    client.streamChat(
        [{ role: 'user', content: 'hi' }],
        8, 0.7, 'model',
        () => { /* onChunk */ },
        () => { /* onError */ },
    );
}

describe('SquishClient.busy — in-flight tracking', () => {
    test('false when idle', () => {
        expect(SquishClient.busy).toBe(false);
    });

    test('true during a streamChat, false after the request closes', () => {
        const req = mockReqEmitter();
        const client = new SquishClient('127.0.0.1', 11435, 'squish');

        startStream(client);
        // The increment is synchronous (before req.write/end), so the poll sees
        // "busy" immediately — even before the first token streams back.
        expect(SquishClient.busy).toBe(true);

        req.emit('close');   // single decrement point (success / error / abort)
        expect(SquishClient.busy).toBe(false);
    });

    test('decrements on abort too — destroy() emits close', () => {
        const req = mockReqEmitter({
            destroy: jest.fn(function (this: EventEmitter) { this.emit('close'); }),
        });
        const client = new SquishClient('127.0.0.1', 11435, 'squish');

        startStream(client);
        expect(SquishClient.busy).toBe(true);

        client.abort();
        expect(SquishClient.busy).toBe(false);
    });
});
