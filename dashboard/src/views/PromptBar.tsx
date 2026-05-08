import { useState } from "react";
import { motion } from "motion/react";
import { ease } from "@konjoai/ui";

export interface PromptBarProps {
  prompt: string;
  onPromptChange: (s: string) => void;
  onSubmit: () => void;
  onClear: () => void;
  disabled: boolean;
  submitLabel?: string;
}

/**
 * Chat input with submit (⌘/ctrl-Enter) + clear-conversation buttons.
 */
export function PromptBar({
  prompt, onPromptChange, onSubmit, onClear, disabled, submitLabel = "send",
}: PromptBarProps) {
  const [focused, setFocused] = useState(false);
  const empty = prompt.trim().length === 0;
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: ease.kanjo }}
      className={[
        "flex items-end gap-2 glass-konjo rounded-konjo p-3",
        focused ? "shadow-konjo-glow" : "",
      ].join(" ")}
    >
      <textarea
        rows={2}
        value={prompt}
        onChange={(e) => onPromptChange(e.target.value)}
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && !disabled && !empty) onSubmit();
        }}
        placeholder="ask the model · ⌘/ctrl-enter to send"
        className="flex-1 bg-transparent border-0 outline-none text-konjo-fg placeholder:text-konjo-fg-faint resize-none"
        style={{ fontSize: 14, lineHeight: 1.55 }}
      />
      <button
        type="button"
        onClick={onClear}
        disabled={disabled}
        className="px-3 py-2 rounded-konjo border border-konjo-line bg-konjo-surface text-konjo-fg-muted text-konjo-mono uppercase tracking-[0.18em] text-[10px] hover:bg-konjo-surface-2 transition-colors disabled:opacity-40"
        title="clear conversation"
      >
        clear
      </button>
      <button
        type="button"
        onClick={onSubmit}
        disabled={disabled || empty}
        className={[
          "px-5 py-2 rounded-konjo text-konjo-mono uppercase tracking-[0.18em] text-[11px] transition-colors",
          disabled || empty
            ? "bg-konjo-surface text-konjo-fg-faint cursor-not-allowed"
            : "bg-konjo-accent text-konjo-bg hover:brightness-110 cursor-pointer shadow-konjo-glow",
        ].join(" ")}
      >
        {submitLabel}
      </button>
    </motion.div>
  );
}
