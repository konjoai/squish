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

// ── Tool calling types (OpenAI-compatible) ─────────────────────────────────

export interface ToolParameterSchema {
    type: string;
    properties?: Record<string, { type: string; description?: string; enum?: string[] }>;
    required?: string[];
}

export interface ToolDefinition {
    type: 'function';
    function: {
        name: string;
        description: string;
        parameters: ToolParameterSchema;
    };
}

export interface ToolCall {
    id: string;
    type: 'function';
    function: {
        name: string;
        arguments: string;   // JSON string
    };
}

export interface ChatMessage {
    role: 'system' | 'user' | 'assistant' | 'tool';
    content: string | null;
    tool_calls?: ToolCall[];
    tool_call_id?: string;   // required when role === 'tool'
    name?: string;
}

export interface ChatChunk {
    delta: string;
    done: boolean;
    finishReason?: string | null;
    // Set when the model wants to call tools instead of producing text
    toolCalls?: ToolCall[];
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
     * When the model invokes tools, `chunk.toolCalls` is populated and
     * `chunk.done` is `true` with `finishReason === 'tool_calls'`.
     */
    streamChat(
        messages: ChatMessage[],
        maxTokens: number,
        temperature: number,
        model: string,
        onChunk: (chunk: ChatChunk) => void,
        onError: (err: Error) => void,
        tools?: ToolDefinition[],
    ): void {
        const payload = JSON.stringify({
            model,
            messages,
            stream: true,
            max_tokens: maxTokens,
            temperature,
            ...(tools && tools.length > 0 ? { tools, tool_choice: 'auto' } : {}),
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

            // Accumulate streaming tool call fragments across SSE chunks.
            // tool_calls arrive as delta patches: [{index, id?, type?, function:{name?,arguments?}}]
            const pendingToolCalls: Record<number, {
                id: string;
                type: 'function';
                function: { name: string; arguments: string };
            }> = {};

            const emitDone = (finishReason?: string | null) => {
                if (finished) { return; }
                finished = true;
                const toolCallList = Object.values(pendingToolCalls);
                onChunk({
                    delta: '',
                    done: true,
                    finishReason: finishReason ?? null,
                    toolCalls: toolCallList.length > 0 ? toolCallList as ToolCall[] : undefined,
                });
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
                        const delta   = parsed.choices?.[0]?.delta ?? {};
                        const finish  = parsed.choices?.[0]?.finish_reason ?? null;

                        // Accumulate streamed tool call fragments
                        if (delta.tool_calls) {
                            for (const tc of delta.tool_calls as Array<{
                                index: number;
                                id?: string;
                                type?: string;
                                function?: { name?: string; arguments?: string };
                            }>) {
                                const idx = tc.index ?? 0;
                                if (!pendingToolCalls[idx]) {
                                    pendingToolCalls[idx] = {
                                        id: tc.id ?? `call_${idx}`,
                                        type: 'function',
                                        function: { name: '', arguments: '' },
                                    };
                                }
                                const p = pendingToolCalls[idx];
                                if (tc.id)                        { p.id = tc.id; }
                                if (tc.function?.name)            { p.function.name += tc.function.name; }
                                if (tc.function?.arguments)       { p.function.arguments += tc.function.arguments; }
                            }
                        }

                        const content: string = delta.content ?? '';
                        if (finish != null) {
                            if (content) {
                                onChunk({ delta: content, done: false });
                            }
                            emitDone(finish);
                        } else if (content) {
                            onChunk({ delta: content, done: false });
                        }
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
