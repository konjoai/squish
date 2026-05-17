/** audio.js — Web Audio API engine. All tones are procedural; no audio files. */

let _ctx = null;
let _masterGain = null;

/** Must be called on first user gesture to comply with browser autoplay policy. */
export function initAudio() {
  if (_ctx) return;
  _ctx = new (window.AudioContext || window.webkitAudioContext)();
  _masterGain = _ctx.createGain();
  _masterGain.gain.value = 0.35;
  _masterGain.connect(_ctx.destination);
}

export function resumeAudio() {
  if (_ctx && _ctx.state === 'suspended') _ctx.resume();
}

function note(freq, type, duration, attack, decay, volume = 1) {
  if (!_ctx) return;
  const t = _ctx.currentTime;
  const osc = _ctx.createOscillator();
  const gain = _ctx.createGain();
  osc.type = type;
  osc.frequency.value = freq;
  gain.gain.setValueAtTime(0, t);
  gain.gain.linearRampToValueAtTime(volume, t + attack);
  gain.gain.exponentialRampToValueAtTime(0.001, t + attack + decay);
  osc.connect(gain);
  gain.connect(_masterGain);
  osc.start(t);
  osc.stop(t + attack + decay + 0.05);
}

// Pentatonic scale C4 D4 E4 G4 A4
export const PENTA = [261.63, 293.66, 329.63, 392.00, 440.00];

export function playNote(freq) { note(freq, 'sine', 0.4, 0.01, 0.4); }

export function playPop(pitch = 1) {
  if (!_ctx) return;
  const t = _ctx.currentTime;
  const osc = _ctx.createOscillator();
  const gain = _ctx.createGain();
  osc.type = 'sine';
  osc.frequency.setValueAtTime(800 * pitch, t);
  osc.frequency.exponentialRampToValueAtTime(200 * pitch, t + 0.12);
  gain.gain.setValueAtTime(0.5, t);
  gain.gain.exponentialRampToValueAtTime(0.001, t + 0.15);
  osc.connect(gain);
  gain.connect(_masterGain);
  osc.start(t);
  osc.stop(t + 0.2);
}

export function playSuccess() {
  if (!_ctx) return;
  [261.63, 329.63, 392.00].forEach((f, i) => {
    const t = _ctx.currentTime + i * 0.12;
    const osc = _ctx.createOscillator();
    const gain = _ctx.createGain();
    osc.type = 'sine';
    osc.frequency.value = f;
    gain.gain.setValueAtTime(0, t);
    gain.gain.linearRampToValueAtTime(0.4, t + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.4);
    osc.connect(gain);
    gain.connect(_masterGain);
    osc.start(t);
    osc.stop(t + 0.5);
  });
}

export function playWobble() {
  if (!_ctx) return;
  const t = _ctx.currentTime;
  const osc = _ctx.createOscillator();
  const gain = _ctx.createGain();
  osc.type = 'sine';
  osc.frequency.setValueAtTime(300, t);
  osc.frequency.linearRampToValueAtTime(250, t + 0.15);
  osc.frequency.linearRampToValueAtTime(280, t + 0.3);
  gain.gain.setValueAtTime(0.25, t);
  gain.gain.exponentialRampToValueAtTime(0.001, t + 0.35);
  osc.connect(gain);
  gain.connect(_masterGain);
  osc.start(t);
  osc.stop(t + 0.4);
}

export function playChime(noteIndex = 0) {
  playNote(PENTA[noteIndex % PENTA.length]);
}

export function playSplash() {
  if (!_ctx) return;
  const t = _ctx.currentTime;
  for (let i = 0; i < 4; i++) {
    const osc = _ctx.createOscillator();
    const gain = _ctx.createGain();
    osc.type = 'triangle';
    osc.frequency.value = 200 + Math.random() * 600;
    gain.gain.setValueAtTime(0.15, t + i * 0.03);
    gain.gain.exponentialRampToValueAtTime(0.001, t + i * 0.03 + 0.25);
    osc.connect(gain);
    gain.connect(_masterGain);
    osc.start(t + i * 0.03);
    osc.stop(t + i * 0.03 + 0.3);
  }
}

export function playBounce() {
  if (!_ctx) return;
  const t = _ctx.currentTime;
  const osc = _ctx.createOscillator();
  const gain = _ctx.createGain();
  osc.type = 'sine';
  osc.frequency.setValueAtTime(400, t);
  osc.frequency.exponentialRampToValueAtTime(150, t + 0.25);
  gain.gain.setValueAtTime(0.3, t);
  gain.gain.exponentialRampToValueAtTime(0.001, t + 0.3);
  osc.connect(gain);
  gain.connect(_masterGain);
  osc.start(t);
  osc.stop(t + 0.35);
}

export function playGiggle() {
  if (!_ctx) return;
  const freqs = [500, 600, 550, 650, 600];
  freqs.forEach((f, i) => {
    const t = _ctx.currentTime + i * 0.07;
    const osc = _ctx.createOscillator();
    const gain = _ctx.createGain();
    osc.type = 'sine';
    osc.frequency.value = f;
    gain.gain.setValueAtTime(0.2, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.08);
    osc.connect(gain);
    gain.connect(_masterGain);
    osc.start(t);
    osc.stop(t + 0.1);
  });
}

export function playFirefly() {
  if (!_ctx) return;
  const t = _ctx.currentTime;
  const osc = _ctx.createOscillator();
  const gain = _ctx.createGain();
  osc.type = 'sine';
  osc.frequency.setValueAtTime(880, t);
  osc.frequency.linearRampToValueAtTime(1100, t + 0.2);
  gain.gain.setValueAtTime(0, t);
  gain.gain.linearRampToValueAtTime(0.18, t + 0.05);
  gain.gain.exponentialRampToValueAtTime(0.001, t + 0.4);
  osc.connect(gain);
  gain.connect(_masterGain);
  osc.start(t);
  osc.stop(t + 0.45);
}

export function playMagic() {
  if (!_ctx) return;
  const t = _ctx.currentTime;
  [700, 900, 1100].forEach((f, i) => {
    const osc = _ctx.createOscillator();
    const gain = _ctx.createGain();
    osc.type = 'sine';
    osc.frequency.value = f + Math.random() * 100;
    gain.gain.setValueAtTime(0.12, t + i * 0.04);
    gain.gain.exponentialRampToValueAtTime(0.001, t + i * 0.04 + 0.2);
    osc.connect(gain);
    gain.connect(_masterGain);
    osc.start(t + i * 0.04);
    osc.stop(t + i * 0.04 + 0.25);
  });
}
