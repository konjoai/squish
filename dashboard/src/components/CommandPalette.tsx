import { useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { ease } from "@konjoai/ui";
import { fuzzyRank } from "../lib/fuzzy";

export interface Command {
  id: string;
  title: string;
  /** Short hint shown on the right (e.g. "section", "action"). */
  group: string;
  /** Extra terms to match against beyond the title. */
  keywords?: string;
  run: () => void;
}

export interface CommandPaletteProps {
  commands: Command[];
}

interface Ranked extends Command {
  haystack: string;
}

/**
 * ⌘K / Ctrl-K command palette — fuzzy-filter every section and action, fly
 * anywhere with the keyboard. ↑/↓ to move, Enter to run, Esc to close. The
 * spine of "make it Konjo": the whole cockpit is one keystroke away.
 */
export function CommandPalette({ commands }: CommandPaletteProps) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [cursor, setCursor] = useState(0);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null);

  // Global ⌘K / Ctrl-K toggle (and Esc to close). Also opens on a custom
  // "konjo:cmdk" event so any affordance (e.g. a hint pill) can trigger it.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((o) => !o);
      } else if (e.key === "Escape") {
        setOpen(false);
      }
    };
    const onOpen = () => setOpen(true);
    window.addEventListener("keydown", onKey);
    window.addEventListener("konjo:cmdk", onOpen);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("konjo:cmdk", onOpen);
    };
  }, []);

  // Focus on open; reset query/cursor on close (deferred so we never call
  // setState synchronously inside the effect body) for a clean next open.
  useEffect(() => {
    if (open) {
      const id = setTimeout(() => inputRef.current?.focus(), 20);
      return () => clearTimeout(id);
    }
    const id = setTimeout(() => { setQuery(""); setCursor(0); }, 0);
    return () => clearTimeout(id);
  }, [open]);

  const ranked = useMemo<Ranked[]>(() => {
    const withHay: Ranked[] = commands.map((c) => ({
      ...c,
      haystack: `${c.title} ${c.group} ${c.keywords ?? ""}`,
    }));
    return fuzzyRank(query, withHay);
  }, [commands, query]);

  // Effective cursor, clamped to the (possibly shrunk) result set at render.
  const activeIdx = Math.min(cursor, Math.max(0, ranked.length - 1));

  const choose = (cmd: Command | undefined) => {
    if (!cmd) return;
    setOpen(false);
    cmd.run();
  };

  const onListKey = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setCursor((c) => Math.min(ranked.length - 1, c + 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setCursor((c) => Math.max(0, c - 1));
    } else if (e.key === "Enter") {
      e.preventDefault();
      choose(ranked[activeIdx]);
    }
  };

  // Scroll the active row into view.
  useEffect(() => {
    const el = listRef.current?.querySelector<HTMLElement>(`[data-idx="${activeIdx}"]`);
    el?.scrollIntoView({ block: "nearest" });
  }, [activeIdx]);

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-50 flex items-start justify-center"
          style={{ paddingTop: "14vh" }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.18, ease: ease.kanjo }}
          onMouseDown={() => setOpen(false)}
        >
          <div className="absolute inset-0 bg-konjo-bg/70" style={{ backdropFilter: "blur(4px)" }} />
          <motion.div
            role="dialog"
            aria-label="command palette"
            className="relative glass-konjo rounded-konjo-lg overflow-hidden w-full"
            style={{ maxWidth: 560, boxShadow: "0 24px 80px rgba(0,0,0,0.5)" }}
            initial={{ opacity: 0, y: -12, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.99 }}
            transition={{ duration: 0.22, ease: ease.kanjo }}
            onMouseDown={(e) => e.stopPropagation()}
            onKeyDown={onListKey}
          >
            <div className="flex items-center gap-2 px-4 py-3 border-b border-konjo-line/60">
              <span className="text-konjo-violet text-konjo-mono text-[13px]">⌘</span>
              <input
                ref={inputRef}
                value={query}
                onChange={(e) => { setQuery(e.target.value); setCursor(0); }}
                placeholder="jump to a panel or run a command…"
                className="flex-1 bg-transparent border-0 outline-none text-konjo-fg placeholder:text-konjo-fg-faint text-[14px]"
                aria-label="command query"
              />
              <kbd className="text-konjo-mono text-[9px] text-konjo-fg-faint border border-konjo-line/60 rounded px-1.5 py-0.5">esc</kbd>
            </div>

            <div ref={listRef} className="max-h-[50vh] overflow-y-auto py-1">
              {ranked.length === 0 ? (
                <div className="px-4 py-6 text-center text-konjo-fg-muted text-konjo-mono text-[12px]">
                  no matches
                </div>
              ) : (
                ranked.map((cmd, i) => (
                  <button
                    key={cmd.id}
                    type="button"
                    data-idx={i}
                    onMouseEnter={() => setCursor(i)}
                    onClick={() => choose(cmd)}
                    aria-selected={i === activeIdx}
                    className={[
                      "w-full flex items-center gap-3 px-4 py-2 text-left transition-colors",
                      i === activeIdx ? "bg-konjo-accent/12" : "hover:bg-konjo-surface/50",
                    ].join(" ")}
                  >
                    <span
                      className="inline-block rounded-full shrink-0"
                      style={{
                        width: 5, height: 5,
                        background: i === activeIdx ? "var(--color-konjo-accent)" : "var(--color-konjo-line)",
                        boxShadow: i === activeIdx ? "0 0 8px var(--color-konjo-accent)" : "none",
                      }}
                    />
                    <span className="flex-1 text-konjo-fg text-[13px] truncate">{cmd.title}</span>
                    <span className="text-konjo-mono uppercase tracking-[0.16em] text-[9px] text-konjo-fg-faint shrink-0">
                      {cmd.group}
                    </span>
                  </button>
                ))
              )}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
