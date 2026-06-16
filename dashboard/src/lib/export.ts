/**
 * Conversation exporters — pure, testable serializers used by the "copy as
 * Markdown" / "download JSON" command-palette actions. No DOM, no clipboard.
 */
import type { ChatTurn } from "./types";

/** Render the conversation as readable Markdown. */
export function conversationToMarkdown(turns: ChatTurn[], exportedAtMs = Date.now()): string {
  const header = [
    "# squish conversation",
    `> exported ${new Date(exportedAtMs).toISOString()} · ${turns.length} turn${turns.length === 1 ? "" : "s"}`,
    "",
  ];
  const body = turns.map((t) => {
    const meta: string[] = [];
    if (t.ttftS != null) meta.push(`ttft ${t.ttftS.toFixed(2)}s`);
    if (t.totalS != null) meta.push(`${t.totalS.toFixed(2)}s total`);
    if (t.tokens && t.tokens.length > 0) meta.push(`${t.tokens.length} tok`);
    const suffix = meta.length > 0 ? ` _(${meta.join(" · ")})_` : "";
    return `**${t.role}**${suffix}\n\n${t.content}\n`;
  });
  return [...header, ...body].join("\n").trimEnd() + "\n";
}

/** Render the conversation as a structured, pretty-printed JSON document. */
export function conversationToJSON(turns: ChatTurn[], exportedAtMs = Date.now()): string {
  return JSON.stringify(
    {
      exported_at: new Date(exportedAtMs).toISOString(),
      turn_count: turns.length,
      turns: turns.map((t) => ({
        role: t.role,
        content: t.content,
        ...(t.ttftS != null ? { ttft_s: Number(t.ttftS.toFixed(4)) } : {}),
        ...(t.totalS != null ? { total_s: Number(t.totalS.toFixed(4)) } : {}),
        ...(t.tokens && t.tokens.length > 0 ? { token_count: t.tokens.length } : {}),
      })),
    },
    null,
    2,
  );
}
