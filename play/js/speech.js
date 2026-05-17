/** speech.js — Web Speech API wrapper. Gracefully silent if unavailable. */

let _voice = null;
let _available = false;

export function initSpeech() {
  if (!window.speechSynthesis) return;
  _available = true;

  function pickVoice() {
    const voices = speechSynthesis.getVoices();
    // Prefer warm female English voices
    const preferred = voices.find(v =>
      v.lang.startsWith('en') && (
        v.name.includes('Samantha') || v.name.includes('Karen') ||
        v.name.includes('Moira') || v.name.includes('Tessa') ||
        v.name.includes('Vicki') || (v.gender && v.gender === 'female')
      )
    );
    _voice = preferred || voices.find(v => v.lang.startsWith('en')) || null;
  }

  pickVoice();
  if (speechSynthesis.onvoiceschanged !== undefined) {
    speechSynthesis.onvoiceschanged = pickVoice;
  }
}

export function say(text, { rate = 0.88, pitch = 1.15, volume = 1 } = {}) {
  if (!_available || !window.speechSynthesis) return;
  try {
    speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(text);
    if (_voice) utt.voice = _voice;
    utt.rate = rate;
    utt.pitch = pitch;
    utt.volume = volume;
    utt.lang = 'en-US';
    speechSynthesis.speak(utt);
  } catch (_) {
    // silent fallback
  }
}

// Squish companion phrases
export const SQUISH_IDLE = [
  "Let's play!",
  "Hi Lily!",
  "You're doing great!",
  "I love you!",
  "What shall we do?",
  "You're so clever!",
];

export const SQUISH_SUCCESS = [
  "Yay! You did it!",
  "Amazing!",
  "Wonderful!",
  "So good!",
  "You're a star!",
  "Hooray!",
];

export const SQUISH_EXPLORE = [
  "Want to try something new?",
  "Let's explore more!",
  "Come see what's inside!",
];

export const ANIMAL_NAMES = {
  duck: "Quack quack! Duck!",
  bunny: "Hop hop! Bunny!",
  cat: "Meow! Kitty cat!",
  dog: "Woof! Puppy dog!",
  frog: "Ribbit! Froggy!",
  bear: "Rawr! Bear!",
};

export function sayRandom(arr) {
  say(arr[Math.floor(Math.random() * arr.length)]);
}
