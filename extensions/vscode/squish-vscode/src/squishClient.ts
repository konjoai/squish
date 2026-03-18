/**
 * squish-vscode/src/squishClient.ts
 *
 * HTTP client wrapping all squish server API calls.
 * Uses the Node.js built-in `http` module so there are no extra dependencies.
 */
import * as http from 'http';

export interface HealthInfo {
    loaded: boolean;
    model?: string;
    tps?: number;
    requests?: number;
    uptime?: number;
}

export interface ChatMessage {
    role: 'system' | 'user' | 'assistant';
    content: string;
}

export interface ChatChunk {
    delta: string;
    done: boolean;
    finishReason?: string | null;
}

export class SquishClient {
    constructor(
        private readonly host: string,
        private readonly port: number,
        private readonly apiKey: string,
    ) {}

    // ── Health check ──────────────────────────────────────────────────────

    async health(): Promise<HealthInfo> {
        const body = await this._get('/health');
        const parsed = JSON.parse(body);
        return {
            loaded: parsed.loaded === true,
            model: parsed.model ?? undefined,
            tps: parsed.avg_tps ?? undefined,
            requests: parsed.requests ?? undefined,
            uptime: parsed.uptime_s ?? undefined,
        };
    }

    // ── Model list ────────────────────────────────────────────────────────

    async models(): Promise<string[]> {
        const body = await this._get('/v1/models');
        const parsed = JSON.parse(body);
        return (parsed.data ?? []).map((m: { id: string }) => m.id);
    }

    // ── Streaming chat completions ────────────────────────────────────────
    /**
     * Stream a chat completion.
     * Calls `onChunk` for each streamed delta, then once more with `done: true`.
     */
    streamChat(
        messages: ChatMessage[],
        maxTokens: number,
        temperature: number,
        model: string,
        onChunk: (chunk: ChatChunk) => void,
        onError: (err: Error) => void,
    ): void {
        const payload = JSON.stringify({
            model,
            messages,
            stream: true,
            max_tokens: maxTokens,
            temperature,
        });

        const options: http.RequestOptions = {
            hostname: this.host,
            port: this.port,
            path: '/v1/chat/completions',
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.apiKey}`,
                'Content-Length': Buffer.byteLength(payload),
            },
        };

        const req = http.request(options, (res) => {
            let buffer = '';
            let finished = false;
            const emitDone = () => {
                if (finished) { return; }
                finished = true;
                onChunk({ delta: '', done: true });
            };
            res.setEncoding('utf8');

            res.on('data', (chunk: string) => {
                buffer += chunk;
                const lines = buffer.split('\n');
                buffer = lines.pop() ?? '';

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed || !trimmed.startsWith('data:')) {
                        continue;
                    }
                    const data = trimmed.slice(5).trim();
                    if (data === '[DONE]') {
                        emitDone();
                        return;
                    }
                    try {
                        const parsed = JSON.parse(data);
                        const delta: string =
                            parsed.choices?.[0]?.delta?.content ?? '';
                        const finishReason: string | null =
                            parsed.choices?.[0]?.finish_reason ?? null;
                        const done = finishReason != null;
                        onChunk({ delta, done, finishReason });
                        if (done) { finished = true; }
                    } catch {
                        // malformed SSE line — ignore
                    }
                }
            });

            res.on('end', () => {
                emitDone();
            });

            res.on('error', onError);
        });

        req.on('error', onError);
        req.write(payload);
        req.end();
    }

    // ── Internal ──────────────────────────────────────────────────────────

    private _get(path: string, timeoutMs = 3000): Promise<string> {
        return new Promise((resolve, reject) => {
            const options: http.RequestOptions = {
                hostname: this.host,
                port: this.port,
                path,
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${this.apiKey}`,
                },
            };
            const req = http.request(options, (res) => {
                let body = '';
                res.setEncoding('utf8');
                res.on('data', (chunk: string) => { body += chunk; });
                res.on('end', () => resolve(body));
                res.on('error', reject);
            });
            req.setTimeout(timeoutMs, () => {
                req.destroy(new Error(`Health check timed out after ${timeoutMs}ms`));
            });
            req.on('error', reject);
            req.end();
        });
    }
}
