/**
 * stage1/activities.js — Stage 1 "Intention" activities.
 * Each class: constructor(w,h), update(dt), draw(ctx), handleTap(x,y),
 * handleDrag(x,y,id), handleDragEnd(x,y,id), resize(w,h), destroy()
 * All return {done: true} from handleTap when an activity is fully complete.
 */
import { burst, confetti, sparkle, updateParticles, drawParticles } from '../particles.js';
import { playPop, playSuccess, playWobble, playChime } from '../audio.js';
import { say } from '../speech.js';
import { markActivityComplete } from '../save.js';

const RAINBOW = ['#f43f5e','#f59e0b','#22c55e','#3b82f6','#a855f7','#14b8a6','#f97316','#ec4899'];

// ─────────────────────────────────────────────────────────────────────────────
// A. COLOR MATCH — drag colored shapes into matching colored buckets
// ─────────────────────────────────────────────────────────────────────────────
const MATCH_COLORS = ['#f43f5e','#f59e0b','#22c55e','#3b82f6','#a855f7','#14b8a6'];

export class ColorMatch {
  constructor(w, h) {
    this.W = w; this.H = h;
    this.done = false;
    this.matched = 0;
    this.drag = null; // {shapeI, ox, oy}
    this._build();
  }

  _build() {
    const W = this.W, H = this.H;
    const n = 6;
    const shapeW = Math.min(W / (n + 1), 64);
    const bucketW = Math.min(W / (n + 1), 70);

    this.shapes = MATCH_COLORS.map((c, i) => ({
      x: (i + 1) * W / (n + 1),
      y: H * 0.28 + (i % 2) * 30,
      baseX: (i + 1) * W / (n + 1),
      baseY: H * 0.28 + (i % 2) * 30,
      r: shapeW / 2,
      color: c,
      matched: false,
      wobble: 0,
      wobbleT: 0,
    }));

    this.buckets = MATCH_COLORS.map((c, i) => ({
      x: (i + 1) * W / (n + 1),
      y: H * 0.74,
      r: bucketW / 2,
      color: c,
      filled: false,
    }));
  }

  resize(w, h) { this.W = w; this.H = h; this._build(); }

  update(dt) {
    for (const s of this.shapes) {
      if (s.wobbleT > 0) {
        s.wobbleT -= dt * 3;
        s.wobble = Math.sin(s.wobbleT * 20) * 8;
      } else {
        s.wobble *= 0.8;
      }
    }
  }

  draw(ctx) {
    const W = this.W, H = this.H;
    // Title
    ctx.fillStyle = '#fff';
    ctx.font = `bold ${Math.min(W, H) * 0.05}px system-ui`;
    ctx.textAlign = 'center';
    ctx.fillText('Match the colors! 🎨', W / 2, H * 0.1);

    // Buckets
    for (const b of this.buckets) {
      ctx.save();
      ctx.beginPath();
      ctx.roundRect(b.x - b.r, b.y - b.r * 0.4, b.r * 2, b.r * 1.6, 8);
      ctx.fillStyle = b.filled ? b.color : b.color + '44';
      ctx.strokeStyle = b.color;
      ctx.lineWidth = 3;
      ctx.fill(); ctx.stroke();
      // Bucket label
      ctx.fillStyle = b.filled ? '#fff' : b.color;
      ctx.font = `${b.r * 0.7}px system-ui`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(b.filled ? '✓' : '?', b.x, b.y + b.r * 0.3);
      ctx.textBaseline = 'alphabetic';
      ctx.restore();
    }

    // Shapes (unmatched only)
    for (const s of this.shapes) {
      if (s.matched) continue;
      ctx.save();
      ctx.translate(s.x + s.wobble, s.y);
      ctx.beginPath();
      ctx.arc(0, 0, s.r, 0, Math.PI * 2);
      ctx.fillStyle = s.color;
      ctx.shadowColor = s.color;
      ctx.shadowBlur = 14;
      ctx.fill();
      ctx.restore();
    }

    // Dragging shape follows cursor
    if (this.drag !== null) {
      const s = this.shapes[this.drag.shapeI];
      ctx.save();
      ctx.shadowColor = s.color;
      ctx.shadowBlur = 20;
      ctx.beginPath();
      ctx.arc(this.drag.x, this.drag.y, s.r * 1.1, 0, Math.PI * 2);
      ctx.fillStyle = s.color;
      ctx.fill();
      ctx.restore();
    }

    // Progress
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = `${Math.min(W, H) * 0.038}px system-ui`;
    ctx.textAlign = 'center';
    ctx.fillText(`${this.matched} / 6 matched`, W / 2, H * 0.92);
  }

  handleTap(x, y) { return this._startDrag(x, y, false); }

  _startDrag(x, y) {
    if (this.done) return false;
    for (let i = 0; i < this.shapes.length; i++) {
      const s = this.shapes[i];
      if (!s.matched && Math.hypot(x - s.x, y - s.y) < s.r + 16) {
        this.drag = { shapeI: i, x, y };
        playPop(1.2);
        return true;
      }
    }
    return false;
  }

  handleDrag(x, y) {
    if (this.drag !== null) { this.drag.x = x; this.drag.y = y; }
  }

  handleDragEnd(x, y) {
    if (this.drag === null) return false;
    const s = this.shapes[this.drag.shapeI];
    let matched = false;
    for (const b of this.buckets) {
      if (!b.filled && b.color === s.color && Math.hypot(x - b.x, y - b.y) < b.r + 20) {
        s.matched = true;
        b.filled = true;
        matched = true;
        this.matched++;
        burst(b.x, b.y, 12, [s.color]);
        playChime(this.matched);
        if (this.matched >= 6) {
          this.done = true;
          confetti(this.W / 2, this.H * 0.5, 50);
          playSuccess();
          say("You did it! All colors matched!");
          markActivityComplete('color_match');
          return { done: true };
        }
        break;
      }
    }
    if (!matched) {
      // Bounce back
      s.x = s.baseX;
      s.y = s.baseY;
      s.wobbleT = 1;
      playWobble();
    }
    this.drag = null;
    return false;
  }

  destroy() {}
}

// ─────────────────────────────────────────────────────────────────────────────
// B. SORT THE SHAPES — tap shape, then tap the right bin
// ─────────────────────────────────────────────────────────────────────────────
const SHAPE_TYPES = ['circle','square','triangle','star'];
const SHAPE_COLORS = ['#f43f5e','#f59e0b','#22c55e','#3b82f6','#a855f7','#14b8a6','#f97316','#ec4899'];

export class SortShapes {
  constructor(w, h) {
    this.W = w; this.H = h;
    this.done = false;
    this.sorted = 0;
    this.selected = null; // index into this.shapes
    this.shapes = [];
    this.bins = [];
    this._build();
  }

  _build() {
    const W = this.W, H = this.H;
    const sz = Math.min(W, H) * 0.09;
    // 8 shapes scattered
    this.shapes = Array.from({ length: 8 }, (_, i) => ({
      type: SHAPE_TYPES[i % 4],
      color: SHAPE_COLORS[i],
      x: 60 + Math.random() * (W - 120),
      y: H * 0.15 + Math.random() * (H * 0.5),
      size: sz,
      jiggle: Math.random() * Math.PI * 2,
      sorted: false,
    }));
    // 3 bins (circle/square/triangle — star goes to star bin or any)
    const bW = Math.min((W - 40) / 4, 90);
    this.bins = SHAPE_TYPES.map((type, i) => ({
      type,
      x: (i + 0.5) * W / 4 + W * 0.0,
      y: H * 0.82,
      w: bW, h: bW * 0.8,
    }));
  }

  resize(w, h) { this.W = w; this.H = h; this._build(); }

  update(dt) {
    this.t = (this.t || 0) + dt;
    for (const s of this.shapes) s.jiggleV = Math.sin(this.t * 4 + s.jiggle) * 3;
  }

  draw(ctx) {
    const W = this.W, H = this.H;
    ctx.fillStyle = '#fff';
    ctx.font = `bold ${Math.min(W, H) * 0.05}px system-ui`;
    ctx.textAlign = 'center';
    ctx.fillText('Sort the shapes! ⬡', W / 2, H * 0.1);

    // Bins
    for (const b of this.bins) {
      const active = this.selected !== null && this.shapes[this.selected].type === b.type;
      ctx.save();
      ctx.beginPath();
      ctx.roundRect(b.x - b.w / 2, b.y - b.h / 2, b.w, b.h, 10);
      ctx.fillStyle = active ? 'rgba(124,58,237,0.35)' : 'rgba(255,255,255,0.08)';
      ctx.strokeStyle = active ? '#7c3aed' : 'rgba(255,255,255,0.25)';
      ctx.lineWidth = 2;
      ctx.fill(); ctx.stroke();
      // Shape icon
      this._drawShape(ctx, b.x, b.y - 8, b.w * 0.3, b.type, '#fff');
      ctx.fillStyle = '#fff';
      ctx.font = `${Math.min(W, H) * 0.025}px system-ui`;
      ctx.textAlign = 'center';
      ctx.fillText(b.type, b.x, b.y + b.h / 2 - 4);
      ctx.restore();
    }

    // Shapes
    for (let i = 0; i < this.shapes.length; i++) {
      const s = this.shapes[i];
      if (s.sorted) continue;
      const jigY = s.jiggleV || 0;
      const selected = this.selected === i;
      ctx.save();
      if (selected) {
        ctx.shadowColor = s.color;
        ctx.shadowBlur = 20;
        ctx.scale(1.12, 1.12);
        ctx.translate(s.x * (1 - 1 / 1.12), (s.y + jigY) * (1 - 1 / 1.12));
      }
      this._drawShape(ctx, s.x, s.y + jigY, s.size, s.type, s.color);
      ctx.restore();
    }

    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = `${Math.min(W, H) * 0.035}px system-ui`;
    ctx.textAlign = 'center';
    ctx.fillText(`${this.sorted} / 8 sorted`, W / 2, H * 0.95);
  }

  _drawShape(ctx, x, y, sz, type, color) {
    ctx.fillStyle = color;
    ctx.beginPath();
    if (type === 'circle') {
      ctx.arc(x, y, sz * 0.5, 0, Math.PI * 2);
    } else if (type === 'square') {
      ctx.roundRect(x - sz * 0.45, y - sz * 0.45, sz * 0.9, sz * 0.9, sz * 0.12);
    } else if (type === 'triangle') {
      ctx.moveTo(x, y - sz * 0.5);
      ctx.lineTo(x + sz * 0.5, y + sz * 0.4);
      ctx.lineTo(x - sz * 0.5, y + sz * 0.4);
      ctx.closePath();
    } else if (type === 'star') {
      for (let i = 0; i < 10; i++) {
        const a = (Math.PI / 5) * i - Math.PI / 2;
        const r = i % 2 === 0 ? sz * 0.5 : sz * 0.22;
        if (i === 0) ctx.moveTo(x + r * Math.cos(a), y + r * Math.sin(a));
        else ctx.lineTo(x + r * Math.cos(a), y + r * Math.sin(a));
      }
      ctx.closePath();
    }
    ctx.shadowColor = color; ctx.shadowBlur = 10;
    ctx.fill();
    ctx.shadowBlur = 0;
  }

  handleTap(x, y) {
    if (this.done) return false;
    // Check if tapping a bin (while shape selected)
    if (this.selected !== null) {
      for (const b of this.bins) {
        if (Math.abs(x - b.x) < b.w / 2 + 16 && Math.abs(y - b.y) < b.h / 2 + 16) {
          const s = this.shapes[this.selected];
          if (s.type === b.type) {
            s.sorted = true;
            this.sorted++;
            burst(b.x, b.y, 10, [s.color]);
            playChime(this.sorted);
            say(s.type + '!');
            if (this.sorted >= 8) {
              this.done = true;
              confetti(this.W / 2, this.H / 2, 50);
              playSuccess();
              say("Amazing! You sorted all the shapes!");
              markActivityComplete('sort_shapes');
              this.selected = null;
              return { done: true };
            }
            this.selected = null;
            return true;
          } else {
            // Wrong bin — gently bounce back
            this.shapes[this.selected].jiggle = this.shapes[this.selected].jiggle;
            playWobble();
            this.selected = null;
            return true;
          }
        }
      }
    }
    // Select a shape
    for (let i = 0; i < this.shapes.length; i++) {
      const s = this.shapes[i];
      if (!s.sorted && Math.hypot(x - s.x, y - s.y) < s.size * 0.6) {
        this.selected = i;
        playPop(1.1);
        say('Where does the ' + s.type + ' go?');
        return true;
      }
    }
    this.selected = null;
    return false;
  }

  handleDrag() {} handleDragEnd() {} destroy() {}
}

// ─────────────────────────────────────────────────────────────────────────────
// C. PATTERN PARADE — AB/ABB pattern completion
// ─────────────────────────────────────────────────────────────────────────────
const PARADE_ANIMALS = [
  { name:'duck',  color:'#f59e0b', emoji:'🦆' },
  { name:'bunny', color:'#f9a8d4', emoji:'🐰' },
  { name:'frog',  color:'#22c55e', emoji:'🐸' },
  { name:'bear',  color:'#92400e', emoji:'🐻' },
];

export class PatternParade {
  constructor(w, h) {
    this.W = w; this.H = h;
    this.done = false;
    this.level = 0;
    this.streak = 0;
    this.choices = [];
    this.pattern = [];
    this.answer = null;
    this.feedback = null; // {correct, t}
    this._nextRound();
  }

  resize(w, h) { this.W = w; this.H = h; }

  _nextRound() {
    this.feedback = null;
    const W = this.W, H = this.H;
    // Level 0-2: AB, level 3+: ABB
    const patLen = this.level < 3 ? 2 : 3;
    const useAnimals = PARADE_ANIMALS.slice(0, patLen === 2 ? 2 : 3);
    const pattern = Array.from({ length: 3 }, (_, i) => useAnimals[i % patLen]);
    const answerAnimal = useAnimals[3 % patLen]; // what comes next
    this.pattern = pattern;
    this.answer = answerAnimal;
    // 3 choices: correct + 2 wrong
    const wrong = PARADE_ANIMALS.filter(a => a.name !== answerAnimal.name).slice(0, 2);
    this.choices = [answerAnimal, ...wrong].sort(() => Math.random() - 0.5);

    // Narrate pattern
    setTimeout(() => {
      const names = pattern.map(a => a.name).join('... ');
      say(`What comes next? ${names}... ?`);
    }, 300);
  }

  update(dt) {
    if (this.feedback) {
      this.feedback.t -= dt;
      if (this.feedback.t <= 0) {
        this.feedback = null;
        if (this.streak >= 5) {
          this.done = true;
          confetti(this.W / 2, this.H / 2, 50);
          playSuccess();
          say("Wonderful! You're a pattern expert!");
          markActivityComplete('pattern_parade');
        } else {
          this.level = Math.min(this.level + 1, 4);
          this._nextRound();
        }
      }
    }
  }

  draw(ctx) {
    const W = this.W, H = this.H;
    ctx.fillStyle = '#fff';
    ctx.font = `bold ${Math.min(W, H) * 0.05}px system-ui`;
    ctx.textAlign = 'center';
    ctx.fillText("What comes next? 🔮", W / 2, H * 0.1);

    // Pattern row
    const cellW = Math.min(W * 0.18, 90);
    const n = this.pattern.length;
    const startX = W / 2 - (n * cellW * 1.2) / 2;
    for (let i = 0; i < n; i++) {
      const a = this.pattern[i];
      const cx = startX + i * cellW * 1.2 + cellW / 2;
      const cy = H * 0.32;
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, cellW * 0.45, 0, Math.PI * 2);
      ctx.fillStyle = a.color;
      ctx.shadowColor = a.color; ctx.shadowBlur = 12;
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.font = `${cellW * 0.55}px system-ui`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(a.emoji, cx, cy);
      ctx.textBaseline = 'alphabetic';
      ctx.restore();
    }
    // Arrow + question mark
    const qx = startX + n * cellW * 1.2 + cellW / 2;
    const qy = H * 0.32;
    ctx.fillStyle = '#f59e0b';
    ctx.font = `${cellW * 0.6}px system-ui`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('→ ?', qx, qy);
    ctx.textBaseline = 'alphabetic';

    // Divider
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(W * 0.1, H * 0.5);
    ctx.lineTo(W * 0.9, H * 0.5);
    ctx.stroke();

    // "Choose:" label
    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    ctx.font = `${Math.min(W, H) * 0.04}px system-ui`;
    ctx.textAlign = 'center';
    ctx.fillText('Tap the right one!', W / 2, H * 0.57);

    // Choices
    const choiceW = cellW * 1.1;
    const cStartX = W / 2 - (this.choices.length * choiceW * 1.3) / 2;
    for (let i = 0; i < this.choices.length; i++) {
      const c = this.choices[i];
      const cx = cStartX + i * choiceW * 1.3 + choiceW / 2;
      const cy = H * 0.7;
      const correct = this.feedback && this.feedback.correct && c.name === this.answer.name;
      const wrong = this.feedback && !this.feedback.correct && c.name === this.feedback.chosen;
      ctx.save();
      ctx.scale(correct ? 1.15 : 1, correct ? 1.15 : 1);
      ctx.translate(correct ? cx * (1 - 1 / 1.15) : 0, correct ? cy * (1 - 1 / 1.15) : 0);
      ctx.beginPath();
      ctx.arc(cx, cy, choiceW * 0.48, 0, Math.PI * 2);
      ctx.fillStyle = wrong ? '#f43f5e44' : c.color + (correct ? '' : '99');
      ctx.shadowColor = c.color; ctx.shadowBlur = correct ? 20 : 10;
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.font = `${choiceW * 0.58}px system-ui`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(c.emoji, cx, cy);
      ctx.textBaseline = 'alphabetic';
      if (correct) {
        ctx.fillStyle = '#22c55e';
        ctx.font = `bold ${choiceW * 0.3}px system-ui`;
        ctx.fillText('✓', cx + choiceW * 0.4, cy - choiceW * 0.4);
      }
      ctx.restore();
    }

    // Progress stars
    const starSize = 18;
    for (let i = 0; i < 5; i++) {
      ctx.fillStyle = i < this.streak ? '#f59e0b' : 'rgba(255,255,255,0.2)';
      ctx.font = `${starSize}px system-ui`;
      ctx.textAlign = 'center';
      ctx.fillText('★', W / 2 + (i - 2) * (starSize + 8), H * 0.92);
    }
  }

  handleTap(x, y) {
    if (this.done || this.feedback) return false;
    const W = this.W, H = this.H;
    const choiceW = Math.min(this.W * 0.18, 90) * 1.1;
    const cStartX = W / 2 - (this.choices.length * choiceW * 1.3) / 2;

    for (let i = 0; i < this.choices.length; i++) {
      const c = this.choices[i];
      const cx = cStartX + i * choiceW * 1.3 + choiceW / 2;
      const cy = H * 0.7;
      if (Math.hypot(x - cx, y - cy) < choiceW * 0.5 + 14) {
        const correct = c.name === this.answer.name;
        this.feedback = { correct, t: 1.2, chosen: c.name };
        if (correct) {
          this.streak++;
          burst(cx, cy, 12, [c.color]);
          playChime(this.streak);
          say(`Yes! ${c.name}!`);
        } else {
          playWobble();
          say(`Try again! Look at the pattern!`);
          this.streak = Math.max(0, this.streak - 1);
        }
        return true;
      }
    }
    return false;
  }

  handleDrag() {} handleDragEnd() {} destroy() {}
}
