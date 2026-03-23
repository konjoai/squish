/**
 * squish-vscode/src/agentLoop.ts
 *
 * Multi-step agentic execution engine.
 *
 * The agent loop calls the model, collects any tool calls, executes them
 * (with optional approval for destructive operations), injects the tool
 * results back into the conversation, and repeats until:
 *  - The model stops calling tools (finish_reason === 'stop')
 *  - The maximum step count is reached
 *  - The operation is aborted
 *
 * Tool execution is delegated to the caller via the `executeTool` callback
 * so that VS Code APIs remain in the extension host process and this class
 * stays testable without a live VS Code instance.
 */
import { SquishClient, ChatMessage, ToolDefinition, ToolCall } from './squishClient';

export type ToolExecutor = (name: string, args: Record<string, unknown>) => Promise<string>;

export interface AgentLoopOptions {
    client: SquishClient;
    messages: ChatMessage[];
    tools: ToolDefinition[];
    executeTool: ToolExecutor;
    maxSteps?: number;
    onChunk?: (delta: string) => void;
    onToolCallStart?: (call: ToolCall) => void;
    onToolCallEnd?: (call: ToolCall, result: string) => void;
    onStepComplete?: (step: number, messages: ChatMessage[]) => void;
    abortSignal?: () => boolean;   // returns true if should abort
}

export interface AgentLoopResult {
    messages: ChatMessage[];        // full updated conversation
    steps: number;                  // number of steps executed
    stoppedReason: 'stop' | 'max_steps' | 'aborted' | 'error';
    error?: string;
}

export class AgentLoop {
    static async run(opts: AgentLoopOptions): Promise<AgentLoopResult> {
        const {
            client,
            tools,
            executeTool,
            maxSteps = 10,
            onChunk,
            onToolCallStart,
            onToolCallEnd,
            onStepComplete,
            abortSignal,
        } = opts;

        const messages: ChatMessage[] = [...opts.messages];
        let steps = 0;

        while (steps < maxSteps) {
            if (abortSignal?.()) {
                return { messages, steps, stoppedReason: 'aborted' };
            }

            steps++;
            let fullText = '';
            let pendingToolCalls: ToolCall[] = [];

            // ── Stream one model turn ──────────────────────────────────────
            try {
                for await (const chunk of client.chatStream(messages, tools)) {
                    if (abortSignal?.()) {
                        client.abort();
                        return { messages, steps, stoppedReason: 'aborted' };
                    }
                    if (chunk.toolCalls?.length) {
                        pendingToolCalls = chunk.toolCalls;
                    }
                    if (chunk.delta) {
                        fullText += chunk.delta;
                        onChunk?.(chunk.delta);
                    }
                    if (chunk.done) {
                        break;
                    }
                }
            } catch (err) {
                const msg = err instanceof Error ? err.message : String(err);
                return { messages, steps, stoppedReason: 'error', error: msg };
            }

            // ── No tool calls → model is done ──────────────────────────────
            if (!pendingToolCalls.length) {
                messages.push({ role: 'assistant', content: fullText });
                onStepComplete?.(steps, messages);
                return { messages, steps, stoppedReason: 'stop' };
            }

            // ── Execute tool calls ─────────────────────────────────────────
            messages.push({
                role: 'assistant',
                content: fullText || null,
                tool_calls: pendingToolCalls,
            });

            for (const call of pendingToolCalls) {
                onToolCallStart?.(call);
                let result: string;
                try {
                    const args = _parseArgs(call.function.arguments);
                    result = await executeTool(call.function.name, args);
                } catch (err) {
                    result = `Error: ${err instanceof Error ? err.message : String(err)}`;
                }
                onToolCallEnd?.(call, result);
                messages.push({
                    role: 'tool',
                    tool_call_id: call.id,
                    name: call.function.name,
                    content: result,
                });
            }

            onStepComplete?.(steps, messages);
        }

        return { messages, steps, stoppedReason: 'max_steps' };
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

function _parseArgs(raw: string): Record<string, unknown> {
    try {
        return JSON.parse(raw) as Record<string, unknown>;
    } catch {
        return {};
    }
}
