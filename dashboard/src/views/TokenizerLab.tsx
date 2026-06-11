import { useEffect, useRef, useState } from "react";
import { motion } from "motion/react";
import { ease } from "@konjoai/ui";
import { tokenizeText } from "../lib/api";
import { AnimatedNumber } from "../components/AnimatedNumber";
import type { TokenizeResult } from "../lib/types";

export interface TokenizerLabProps {
  onFromMockChange?: (fromMock: boolean) => void;
}

const TOKEN_HUES = [
  "var(--color-konjo-accent)",
  "var(--color-konjo-violet)",
  "var(--color-konjo-cool)",
  "var(--color-konjo-warm)",
  "var(--color-konjo-good)",
];

const SAMPLE = "Squish squeezes a 7B model onto your laptop — quantized to INT4, served at 50 tok/s.";

/**
 * Tokenizer Lab — drives POST /v1/tokenize. As you type (debounced 250 ms), the
 * text is segmented into colored token chips backed by the live token IDs the
 * model's tokenizer returns. Shows count, byte-per-token density, and the raw
 * ID stream. Falls back to a deterministic mock BPE when offline.
 */
export function TokenizerLab({ onFromMockChange }: TokenizerLabProps) {
  const [text, setText] = useState<string>(SAMPLE);
  const [result, setResult] = useState<TokenizeResult | null>(null);
  const [fromMock, setFromMock] = useState<boolean>(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      void tokenizeText(text).then(({ data, fromMock }) => {
        setResult(data);
        setFromMock(fromMock);
        onFromMockChange?.(fromMock);
      });
    }, 250);
    return () => { if (timer.current) clearTimeout(timer.current); };
  }, [text, onFromMockChange]);

  const ids = result?.token_ids ?? [];
  const count = result?.token_count ?? 0;
  const chars = text.length;
  const density = count > 0 ? chars / count : 0;

  // Render chips by re-segmenting the text the same way the mock does; for the
  // live path we render per-ID chips since the server returns IDs only.
  const chips = segment(text);
  const showChips = chips.length === count || fromMock;

  return (
    <section className="space-y-3" id="tokenizer">
      <header className="flex items-baseline justify-between flex-wrap gap-2">
        <div>
          <h2 className="text-konjo-display text-konjo-fg" style={{ fontSize: 20, fontWeight: 600 }}>
            Tokenizer lab
          </h2>
          <p className="text-konjo-fg-muted text-[13px] mt-1">
            Live from <span className="text-konjo-mono">/v1/tokenize</span> ·{" "}
            <span className="text-konjo-fg">{result?.model ?? "—"}</span>
          </p>
        </div>
        <div className="flex gap-4">
          <Metric label="tokens" value={count} />
          <Metric label="chars" value={chars} />
          <Metric label="chars / tok" value={density} format={(v) => v.toFixed(2)} accent="var(--color-konjo-violet)" />
        </div>
      </header>

      <div className="glass-konjo rounded-konjo-lg p-5 space-y-4">
        <textarea
          rows={3}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="type to tokenize…"
          className="w-full bg-konjo-surface/50 border border-konjo-line rounded-konjo px-3 py-2 outline-none text-konjo-fg placeholder:text-konjo-fg-faint resize-none focus:shadow-konjo-glow transition-shadow"
          style={{ fontSize: 14, lineHeight: 1.55 }}
        />

        <div>
          <div className="text-konjo-mono uppercase tracking-[0.18em] text-[10px] text-konjo-fg-muted mb-2">
            tokens {showChips ? "" : "· id stream"}
          </div>
          <div className="flex flex-wrap gap-1">
            {showChips
              ? chips.map((tok, i) => {
                  const c = TOKEN_HUES[i % TOKEN_HUES.length];
                  return (
                    <motion.span
                      key={i}
                      initial={{ opacity: 0, scale: 0.85 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ duration: 0.2, ease: ease.kanjo, delay: Math.min(i * 0.006, 0.4) }}
                      title={ids[i] != null ? `id ${ids[i]}` : undefined}
                      className="text-konjo-mono text-[12px] rounded-konjo-sm px-1.5 py-0.5"
                      style={{
                        background: `color-mix(in oklch, ${c} 16%, transparent)`,
                        border: `1px solid color-mix(in oklch, ${c} 35%, transparent)`,
                        color: "var(--color-konjo-fg)",
                        whiteSpace: "pre",
                      }}
                    >
                      {tok.replace(/ /g, "·")}
                    </motion.span>
                  );
                })
              : ids.map((id, i) => {
                  const c = TOKEN_HUES[i % TOKEN_HUES.length];
                  return (
                    <motion.span
                      key={i}
                      initial={{ opacity: 0, scale: 0.85 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ duration: 0.2, ease: ease.kanjo, delay: Math.min(i * 0.006, 0.4) }}
                      className="text-konjo-mono text-[11px] tabular-nums rounded-konjo-sm px-1.5 py-0.5"
                      style={{ background: `color-mix(in oklch, ${c} 14%, transparent)`, color: c }}
                    >
                      {id}
                    </motion.span>
                  );
                })}
            {count === 0 && (
              <span className="text-konjo-fg-faint text-konjo-mono text-[12px]">empty</span>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

/** Mirror the mock segmentation so live chips line up with the visual model. */
function segment(text: string): string[] {
  return text.match(/\s*[\w']+|\s*[^\s\w]/g) ?? [];
}

function Metric({
  label, value, format, accent,
}: { label: string; value: number; format?: (v: number) => string; accent?: string }) {
  return (
    <div className="text-right">
      <div className="text-konjo-mono uppercase tracking-[0.16em] text-[9px] text-konjo-fg-muted">{label}</div>
      <div className="text-konjo-mono tabular-nums" style={{ fontSize: 18, fontWeight: 600, color: accent ?? "var(--color-konjo-fg)" }}>
        <AnimatedNumber value={value} format={format} />
      </div>
    </div>
  );
}
