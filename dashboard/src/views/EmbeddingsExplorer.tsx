import { useState } from "react";
import { motion } from "motion/react";
import { ease } from "@konjoai/ui";
import { embedText } from "../lib/api";
import { similarityMatrix } from "../lib/vector";
import type { EmbeddingResponse } from "../lib/types";

export interface EmbeddingsExplorerProps {
  onFromMockChange?: (fromMock: boolean) => void;
}

const DEFAULT_TEXTS = [
  "The cat sat on the warm windowsill.",
  "A feline rested in the sunny window.",
  "Quantize the KV cache to INT4 for speed.",
];

/**
 * Embeddings Explorer — drives POST /v1/embeddings, then renders the live
 * cosine-similarity heatmap across the input texts. Demonstrates squish's
 * OpenAI-compatible embeddings endpoint: semantically close sentences light up,
 * unrelated ones stay dark. Mean-pooled vectors are computed server-side; the
 * similarity math runs client-side (lib/vector.ts). Mock fallback offline.
 */
export function EmbeddingsExplorer({ onFromMockChange }: EmbeddingsExplorerProps) {
  const [texts, setTexts] = useState<string[]>(DEFAULT_TEXTS);
  const [resp, setResp] = useState<EmbeddingResponse | null>(null);
  const [fromMock, setFromMock] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(false);

  const run = () => {
    const inputs = texts.map((t) => t.trim()).filter((t) => t.length > 0);
    if (inputs.length < 2) return;
    setLoading(true);
    void embedText(inputs).then(({ data, fromMock }) => {
      setResp(data);
      setFromMock(fromMock);
      onFromMockChange?.(fromMock);
      setLoading(false);
    });
  };

  const setText = (i: number, v: string) =>
    setTexts((arr) => arr.map((t, j) => (j === i ? v : t)));
  const addText = () => setTexts((arr) => (arr.length >= 6 ? arr : [...arr, ""]));
  const removeText = (i: number) =>
    setTexts((arr) => (arr.length <= 2 ? arr : arr.filter((_, j) => j !== i)));

  const vectors = resp?.data.map((d) => d.embedding) ?? [];
  const matrix = vectors.length >= 2 ? similarityMatrix(vectors) : [];
  const dim = vectors[0]?.length ?? 0;

  return (
    <section className="space-y-3" id="embeddings">
      <header className="flex items-baseline justify-between flex-wrap gap-2">
        <div>
          <h2 className="text-konjo-display text-konjo-fg" style={{ fontSize: 20, fontWeight: 600 }}>
            Embeddings explorer
          </h2>
          <p className="text-konjo-fg-muted text-[13px] mt-1">
            Cosine similarity via <span className="text-konjo-mono">/v1/embeddings</span>
            {resp && <> · <span className="text-konjo-fg">{resp.model}</span> · {dim}-dim · {fromMock ? "mock" : "live"}</>}
          </p>
        </div>
        <button
          type="button"
          onClick={run}
          disabled={loading}
          className="px-5 py-2 rounded-konjo text-konjo-mono uppercase tracking-[0.18em] text-[11px] bg-konjo-accent text-konjo-bg hover:brightness-110 cursor-pointer shadow-konjo-glow disabled:opacity-50 transition"
        >
          {loading ? "embedding…" : "embed"}
        </button>
      </header>

      <div className="glass-konjo rounded-konjo-lg p-5 grid lg:grid-cols-[1fr_auto] gap-6 items-start">
        <div className="space-y-2 min-w-0">
          {texts.map((t, i) => (
            <div key={i} className="flex items-center gap-2">
              <span className="text-konjo-mono text-[10px] tabular-nums w-4 text-right" style={{ color: HUE(i) }}>
                {i + 1}
              </span>
              <input
                value={t}
                onChange={(e) => setText(i, e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") run(); }}
                placeholder={`text ${i + 1}`}
                className="flex-1 bg-konjo-surface/50 border border-konjo-line rounded-konjo px-2.5 py-1.5 outline-none text-konjo-fg text-[13px] placeholder:text-konjo-fg-faint focus:shadow-konjo-glow transition-shadow"
              />
              <button
                type="button"
                onClick={() => removeText(i)}
                disabled={texts.length <= 2}
                className="text-konjo-fg-faint hover:text-konjo-hot disabled:opacity-30 text-konjo-mono text-[14px] px-1"
                title="remove"
                aria-label={`remove text ${i + 1}`}
              >
                ×
              </button>
            </div>
          ))}
          <button
            type="button"
            onClick={addText}
            disabled={texts.length >= 6}
            className="text-konjo-mono text-[10px] px-2 py-1 rounded-konjo-sm border border-konjo-line/60 text-konjo-fg-muted hover:text-konjo-fg hover:border-konjo-violet/60 disabled:opacity-30 transition-colors"
          >
            + add text
          </button>
        </div>

        {matrix.length >= 2 ? (
          <Heatmap matrix={matrix} />
        ) : (
          <div className="flex items-center justify-center text-konjo-fg-muted text-konjo-mono text-[12px]" style={{ minWidth: 200, minHeight: 160 }}>
            press embed to compare
          </div>
        )}
      </div>
    </section>
  );
}

function Heatmap({ matrix }: { matrix: number[][] }) {
  const n = matrix.length;
  const cell = 40;
  const gap = 3;
  const size = n * (cell + gap) - gap;
  return (
    <div className="flex flex-col items-center gap-2 shrink-0">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-label="cosine similarity matrix">
        {matrix.flatMap((row, r) =>
          row.map((v, c) => {
            // Map similarity [-1,1] → [0,1] for intensity; diagonal is always 1.
            const t = Math.max(0, Math.min(1, (v + 1) / 2));
            return (
              <motion.g key={`${r}-${c}`}>
                <motion.rect
                  x={c * (cell + gap)}
                  y={r * (cell + gap)}
                  width={cell}
                  height={cell}
                  rx={4}
                  initial={{ opacity: 0, scale: 0.6 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.3, ease: ease.kanjo, delay: (r * n + c) * 0.02 }}
                  style={{ transformBox: "fill-box", transformOrigin: "center" }}
                  fill={`color-mix(in oklch, var(--color-konjo-accent) ${Math.round(t * 100)}%, var(--color-konjo-surface))`}
                />
                <text
                  x={c * (cell + gap) + cell / 2}
                  y={r * (cell + gap) + cell / 2}
                  textAnchor="middle"
                  dominantBaseline="central"
                  style={{ fontSize: 10, fontFamily: "var(--font-konjo-mono, monospace)", fill: t > 0.55 ? "var(--color-konjo-bg)" : "var(--color-konjo-fg-muted)" }}
                >
                  {v.toFixed(2)}
                </text>
              </motion.g>
            );
          }),
        )}
      </svg>
      <div className="text-konjo-mono uppercase tracking-[0.18em] text-[9px] text-konjo-fg-muted">
        cosine similarity · bright = close
      </div>
    </div>
  );
}

const HUES = [
  "var(--color-konjo-accent)",
  "var(--color-konjo-violet)",
  "var(--color-konjo-cool)",
  "var(--color-konjo-warm)",
  "var(--color-konjo-good)",
  "var(--color-konjo-hot)",
];
function HUE(i: number): string { return HUES[i % HUES.length]; }
