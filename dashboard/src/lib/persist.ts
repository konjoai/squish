/**
 * Conversation persistence — keeps the chat across reloads via localStorage.
 *
 * Pure-ish helpers (the only side effect is localStorage, guarded for SSR /
 * private-mode / quota failures). Load validates every turn's shape so a
 * corrupt or stale payload can never crash the cockpit.
 */
import type { ChatTurn } from "./types";

const KEY = "squish.cockpit.conversation.v1";
const MAX_TURNS = 200;

function storage(): Storage | null {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    // Accessing localStorage can throw in sandboxed iframes — treat as absent.
    return null;
  }
}

function isValidTurn(t: unknown): t is ChatTurn {
  if (!t || typeof t !== "object") return false;
  const o = t as Record<string, unknown>;
  return (
    typeof o.id === "string" &&
    typeof o.content === "string" &&
    (o.role === "user" || o.role === "assistant" || o.role === "system")
  );
}

/** Load persisted turns, or [] when absent/corrupt. Never throws. */
export function loadTurns(): ChatTurn[] {
  const s = storage();
  if (!s) return [];
  try {
    const raw = s.getItem(KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isValidTurn).slice(-MAX_TURNS);
  } catch {
    // Malformed JSON — drop it rather than crash.
    return [];
  }
}

/** Persist the (trimmed) conversation. Non-fatal on quota / unavailability. */
export function saveTurns(turns: ChatTurn[]): void {
  const s = storage();
  if (!s) return;
  try {
    s.setItem(KEY, JSON.stringify(turns.slice(-MAX_TURNS)));
  } catch {
    // Quota exceeded or storage disabled — losing persistence is acceptable.
  }
}

/** Remove the persisted conversation. */
export function clearTurns(): void {
  const s = storage();
  if (!s) return;
  try {
    s.removeItem(KEY);
  } catch {
    // Nothing actionable if removal fails.
  }
}
