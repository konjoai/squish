import { useEffect, useRef, useState } from "react";

export interface AnimatedNumberProps {
  value: number;
  /** Formatter for the displayed number. Default: rounded integer. */
  format?: (v: number) => string;
  /** Tween duration in ms. */
  durationMs?: number;
  className?: string;
}

/**
 * Count-up number that tweens from its previous value to the next whenever
 * `value` changes — the small motion that makes live telemetry feel alive.
 *
 * The display is seeded with the target value on mount so the correct number
 * is present immediately (important for SSR and test environments where
 * requestAnimationFrame may not tick), then animates on subsequent updates.
 */
export function AnimatedNumber({
  value,
  format = (v) => `${Math.round(v)}`,
  durationMs = 600,
  className,
}: AnimatedNumberProps) {
  const [display, setDisplay] = useState(value);
  const fromRef = useRef(value);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    const from = fromRef.current;
    const to = value;
    if (from === to) return;

    if (typeof requestAnimationFrame !== "function") {
      // No animation frames (e.g. test/SSR): settle on the next tick rather
      // than synchronously inside the effect body.
      const id = setTimeout(() => { setDisplay(to); fromRef.current = to; }, 0);
      return () => clearTimeout(id);
    }

    const start = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / durationMs);
      // easeOutCubic — settles gently, like the rest of the cockpit.
      const eased = 1 - Math.pow(1 - t, 3);
      const cur = from + (to - from) * eased;
      setDisplay(cur);
      if (t < 1) {
        rafRef.current = requestAnimationFrame(tick);
      } else {
        fromRef.current = to;
      }
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      fromRef.current = to;
    };
  }, [value, durationMs]);

  return <span className={className}>{format(display)}</span>;
}
