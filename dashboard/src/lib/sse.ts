/**
 * SSE / NDJSON parser for squish.
 *
 * /v1/chat/completions emits SSE: `data: {json}\n\n`, terminator `data: [DONE]\n\n`.
 * /api/chat emits NDJSON: `{json}\n` per line, no terminator.
 *
 * Auto-detects: if the buffer contains `\n\n`, treat as SSE; otherwise NDJSON.
 */

export interface ParsedFrame {
  json: string;
  done: boolean;
}

export function parseStreamChunk(buffer: string): { frames: ParsedFrame[]; rest: string } {
  const frames: ParsedFrame[] = [];

  if (buffer.includes("\n\n")) {
    const parts = buffer.split("\n\n");
    const rest = parts.pop() ?? "";
    for (const block of parts) {
      for (const line of block.split("\n")) {
        if (!line || line.startsWith(":")) continue;
        if (line.startsWith("data:")) {
          const payload = line.slice(5).trim();
          if (payload === "[DONE]") frames.push({ json: "", done: true });
          else if (payload) frames.push({ json: payload, done: false });
        }
      }
    }
    return { frames, rest };
  }

  const lines = buffer.split("\n");
  const rest = lines.pop() ?? "";
  for (const raw of lines) {
    const line = raw.trim();
    if (!line || line.startsWith(":")) continue;
    const payload = line.startsWith("data:") ? line.slice(5).trim() : line;
    if (payload === "[DONE]") frames.push({ json: "", done: true });
    else if (payload) frames.push({ json: payload, done: false });
  }
  return { frames, rest };
}
