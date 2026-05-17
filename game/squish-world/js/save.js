/** save.js — localStorage persistence with graceful fallback. */

const KEY = 'squishworld_save';
const AVATAR_KEY = 'squishworld_avatar';

const defaultSave = () => ({
  stage_unlocked: 0,
  activities_completed: {},
  time_in_stage0: 0,
  avatar_config: null,
  journal: [],
  first_launch: Date.now(),
});

let _save = null;

export function loadSave() {
  try {
    const raw = localStorage.getItem(KEY);
    _save = raw ? { ...defaultSave(), ...JSON.parse(raw) } : defaultSave();
  } catch (_) {
    _save = defaultSave();
  }
  return _save;
}

export function getSave() {
  if (!_save) return loadSave();
  return _save;
}

export function writeSave(partial) {
  _save = { ..._save, ...partial };
  try { localStorage.setItem(KEY, JSON.stringify(_save)); } catch (_) {}
}

export function addJournalEntry(event) {
  const s = getSave();
  const entry = { date: new Date().toISOString(), event };
  s.journal = [entry, ...(s.journal || [])].slice(0, 100);
  writeSave({ journal: s.journal });
}

export function markActivityComplete(id) {
  const s = getSave();
  if (!s.activities_completed[id]) {
    s.activities_completed[id] = Date.now();
    addJournalEntry(`Completed ${id}`);
    writeSave({ activities_completed: s.activities_completed });
  }
}

export function addStage0Time(seconds) {
  const s = getSave();
  writeSave({ time_in_stage0: (s.time_in_stage0 || 0) + seconds });
}

export function unlockStage1() {
  const s = getSave();
  if (s.stage_unlocked < 1) {
    addJournalEntry('Unlocked Stage 1 — Intention!');
    writeSave({ stage_unlocked: 1 });
  }
}

// Avatar
export function loadAvatar() {
  try {
    const raw = localStorage.getItem(AVATAR_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (_) { return null; }
}

export function saveAvatar(config) {
  try { localStorage.setItem(AVATAR_KEY, JSON.stringify(config)); } catch (_) {}
}
