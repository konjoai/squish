/**
 * stage0/activities.js — All 8 Stage 0 delight activities.
 * Each activity exports a class with: constructor(w,h), update(dt), draw(ctx),
 * handleTap(x,y), handleDrag(x,y), handleDragEnd(x,y), resize(w,h), destroy()
 */
import { burst, sparkle, confetti, updateParticles, drawParticles } from '../particles.js';
import {
  playPop, playSuccess, playSplash, playBounce, playGiggle,
  playFirefly, playMagic, playChime, playNote, PENTA
} from '../audio.js';
import { say, ANIMAL_NAMES } from '../speech.js';

const RAINBOW = ['#f43f5e','#f59e0b','#22c55e','#3b82f6','#a855f7','#14b8a6','#f97316','#ec4899'];

// ─────────────────────────────────────────────────────────────────────────────
// 1. BUBBLE POP
// ─────────────────────────────────────────────────────────────────────────────
export class BubblePop {
  constructor(w, h) { this.resize(w, h); this.t = 0; this.bubbles = []; }

  resize(w, h) {
    this.W = w; this.H = h;
    this.maxBubbles = Math.floor(w * h / 18000) + 8;
  }

  _spawn() {
    if (this.bubbles.length >= this.maxBubbles) return;
    const r = 24 + Math.random() * 36;
    this.bubbles.push({
      x: r + Math.random() * (this.W - r * 2),
      y: this.H + r,
      r,
      vx: (Math.random() - 0.5) * 0.8,
      vy: -(0.6 + Math.random() * 1.2),
      color: RAINBOW[Math.floor(Math.random() * RAINBOW.length)],
      wobble: Math.random() * Math.PI * 2,
      alpha: 0.85,
      popping: false,
      popT: 0,
    });
  }

  update(dt) {
    this.t += dt;
    if (Math.random() < dt * 3) this._spawn();
    for (let i = this.bubbles.length - 1; i >= 0; i--) {
      const b = this.bubbles[i];
      if (b.popping) {
        b.popT += dt * 5;
        b.r += 4;
        b.alpha -= 0.15;
        if (b.alpha <= 0) this.bubbles.splice(i, 1);
      } else {
        b.wobble += dt * 2;
        b.x += b.vx + Math.sin(b.wobble) * 0.4;
        b.y += b.vy;
        if (b.y < -b.r * 2) this.bubbles.splice(i, 1);
      }
    }
  }

  draw(ctx) {
    for (const b of this.bubbles) {
      ctx.save();
      ctx.globalAlpha = Math.max(0, b.alpha);
      ctx.beginPath();
      ctx.arc(b.x, b.y, b.r, 0, Math.PI * 2);
      ctx.strokeStyle = b.color;
      ctx.lineWidth = 2.5;
      ctx.stroke();
      // Inner shimmer
      const g = ctx.createRadialGradient(b.x - b.r * 0.3, b.y - b.r * 0.3, 1, b.x, b.y, b.r);
      g.addColorStop(0, b.color + '55');
      g.addColorStop(0.7, b.color + '18');
      g.addColorStop(1, b.color + '00');
      ctx.fillStyle = g;
      ctx.fill();
      // Shine
      ctx.beginPath();
      ctx.arc(b.x - b.r * 0.32, b.y - b.r * 0.32, b.r * 0.2, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,0.55)';
      ctx.fill();
      ctx.restore();
    }
  }

  handleTap(x, y) {
    for (let i = this.bubbles.length - 1; i >= 0; i--) {
      const b = this.bubbles[i];
      if (!b.popping && Math.hypot(x - b.x, y - b.y) < b.r + 12) {
        b.popping = true;
        burst(b.x, b.y, 10, [b.color]);
        playPop(1 + (b.r - 24) / 36 * 0.5);
        return true;
      }
    }
    return false;
  }

  handleDrag(x, y) { this.handleTap(x, y); }
  handleDragEnd() {}
  destroy() {}
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. ANIMAL PARADE
// ─────────────────────────────────────────────────────────────────────────────
const ANIMALS = [
  { name:'duck',   color:'#f59e0b', emoji:'🦆', sound:'Quack quack!' },
  { name:'bunny',  color:'#f9a8d4', emoji:'🐰', sound:'Hop hop!' },
  { name:'cat',    color:'#f97316', emoji:'🐱', sound:'Meow!' },
  { name:'dog',    color:'#a78bfa', emoji:'🐶', sound:'Woof woof!' },
  { name:'frog',   color:'#22c55e', emoji:'🐸', sound:'Ribbit!' },
  { name:'bear',   color:'#92400e', emoji:'🐻', sound:'Rawr!' },
];

export class AnimalParade {
  constructor(w, h) { this.resize(w, h); this.t = 0; this.critters = []; this._init(); }

  resize(w, h) { this.W = w; this.H = h; }

  _init() {
    this.critters = ANIMALS.map((a, i) => ({
      ...a, x: (i + 1) * this.W / (ANIMALS.length + 1),
      y: this.H * 0.55 + Math.sin(i) * 30,
      baseY: this.H * 0.55 + Math.sin(i) * 30,
      phase: i * 0.8,
      dancing: false, danceT: 0, scale: 1, rot: 0,
      size: Math.min(this.W, this.H) * 0.1,
    }));
  }

  update(dt) {
    this.t += dt;
    for (const c of this.critters) {
      c.y = c.baseY + Math.sin(this.t * 1.8 + c.phase) * 8;
      if (c.dancing) {
        c.danceT += dt;
        c.scale = 1 + Math.sin(c.danceT * 12) * 0.15;
        c.rot = Math.sin(c.danceT * 8) * 0.3;
        if (c.danceT > 1.5) { c.dancing = false; c.scale = 1; c.rot = 0; }
      }
    }
  }

  draw(ctx) {
    for (const c of this.critters) {
      ctx.save();
      ctx.translate(c.x, c.y);
      ctx.rotate(c.rot);
      ctx.scale(c.scale, c.scale);
      // Body circle
      ctx.beginPath();
      ctx.arc(0, 0, c.size * 0.5, 0, Math.PI * 2);
      ctx.fillStyle = c.color;
      ctx.shadowColor = c.color;
      ctx.shadowBlur = 14;
      ctx.fill();
      ctx.shadowBlur = 0;
      // Emoji face
      ctx.font = `${c.size * 0.72}px system-ui`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(c.emoji, 0, 0);
      // Name tag
      ctx.fillStyle = '#fff';
      ctx.font = `bold ${c.size * 0.28}px system-ui`;
      ctx.textBaseline = 'alphabetic';
      ctx.fillText(c.name, 0, c.size * 0.72);
      ctx.restore();
    }
  }

  handleTap(x, y) {
    for (const c of this.critters) {
      if (Math.hypot(x - c.x, y - c.y) < c.size * 0.6) {
        c.dancing = true; c.danceT = 0;
        burst(c.x, c.y - c.size * 0.3, 10, [c.color]);
        playChime(Math.floor(Math.random() * 5));
        say(`${c.sound} ${c.name}!`);
        return true;
      }
    }
    return false;
  }

  handleDrag() {} handleDragEnd() {} destroy() {}
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. MAGIC PAINT
// ─────────────────────────────────────────────────────────────────────────────
export class MagicPaint {
  constructor(w, h) {
    this.resize(w, h);
    this.trail = [];
    this.painting = false;
    this.colorI = 0;
    this.colors = ['#7c3aed','#f43f5e','#f59e0b','#14b8a6','#22c55e','#3b82f6'];
    this.lastX = 0; this.lastY = 0;
    this.sparkTimer = 0;
  }

  resize(w, h) { this.W = w; this.H = h; }

  update(dt) {
    this.sparkTimer += dt;
    for (let i = this.trail.length - 1; i >= 0; i--) {
      this.trail[i].life -= dt * 0.25;
      if (this.trail[i].life <= 0) this.trail.splice(i, 1);
    }
  }

  draw(ctx) {
    for (const p of this.trail) {
      ctx.save();
      ctx.globalAlpha = Math.max(0, p.life * 0.8);
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = p.color;
      ctx.shadowColor = p.color;
      ctx.shadowBlur = 12;
      ctx.fill();
      ctx.restore();
    }
  }

  handleTap(x, y) {
    this.painting = true;
    this.lastX = x; this.lastY = y;
    this._addDot(x, y);
    playMagic();
  }

  handleDrag(x, y) {
    if (!this.painting) return;
    this._addDot(x, y);
    if (this.sparkTimer > 0.08) {
      sparkle(x, y, this.colors[this.colorI]);
      this.sparkTimer = 0;
    }
    const dist = Math.hypot(x - this.lastX, y - this.lastY);
    if (dist > 10) {
      this.colorI = (this.colorI + 1) % this.colors.length;
    }
    this.lastX = x; this.lastY = y;
  }

  _addDot(x, y) {
    this.trail.push({ x, y, r: 12 + Math.random() * 8, color: this.colors[this.colorI], life: 1 });
  }

  handleDragEnd() { this.painting = false; }
  destroy() {}
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. MUSIC STARS
// ─────────────────────────────────────────────────────────────────────────────
export class MusicStars {
  constructor(w, h) { this.resize(w, h); this.t = 0; this.stars = []; this._spawn(12); }

  resize(w, h) { this.W = w; this.H = h; }

  _spawn(n = 1) {
    for (let i = 0; i < n; i++) {
      const noteI = Math.floor(Math.random() * PENTA.length);
      this.stars.push({
        x: 60 + Math.random() * (this.W - 120),
        y: this.H * 0.15 + Math.random() * (this.H * 0.65),
        r: 20 + Math.random() * 20,
        color: RAINBOW[Math.floor(Math.random() * RAINBOW.length)],
        freq: PENTA[noteI],
        noteI,
        phase: Math.random() * Math.PI * 2,
        hit: false, hitT: 0,
        vx: (Math.random() - 0.5) * 0.4,
        vy: (Math.random() - 0.5) * 0.3,
        rot: 0,
        rotV: (Math.random() - 0.5) * 0.02,
      });
    }
  }

  update(dt) {
    this.t += dt;
    for (let i = this.stars.length - 1; i >= 0; i--) {
      const s = this.stars[i];
      if (s.hit) {
        s.hitT += dt * 4;
        s.r += 3;
        if (s.hitT > 1) {
          this.stars.splice(i, 1);
          this._spawn(1);
        }
      } else {
        s.x += s.vx;
        s.y += s.vy;
        s.rot += s.rotV;
        if (s.x < s.r || s.x > this.W - s.r) s.vx *= -1;
        if (s.y < s.r || s.y > this.H - s.r) s.vy *= -1;
        s.pulse = 1 + Math.sin(this.t * 3 + s.phase) * 0.08;
      }
    }
  }

  draw(ctx) {
    for (const s of this.stars) {
      ctx.save();
      const a = s.hit ? Math.max(0, 1 - s.hitT) : s.pulse || 1;
      ctx.globalAlpha = a;
      ctx.translate(s.x, s.y);
      ctx.rotate(s.rot);
      ctx.scale(s.pulse || 1, s.pulse || 1);
      drawStar5(ctx, 0, 0, s.r * 0.42, s.r, s.color);
      // Musical note dots
      ctx.fillStyle = 'rgba(255,255,255,0.7)';
      ctx.font = `${s.r * 0.65}px system-ui`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('♪', 0, 0);
      ctx.restore();
    }
  }

  handleTap(x, y) {
    for (const s of this.stars) {
      if (!s.hit && Math.hypot(x - s.x, y - s.y) < s.r + 14) {
        s.hit = true;
        burst(s.x, s.y, 12, [s.color]);
        playNote(s.freq);
        return true;
      }
    }
    return false;
  }

  handleDrag() {} handleDragEnd() {} destroy() {}
}

function drawStar5(ctx, cx, cy, r1, r2, color) {
  ctx.beginPath();
  for (let i = 0; i < 10; i++) {
    const angle = (Math.PI / 5) * i - Math.PI / 2;
    const r = i % 2 === 0 ? r2 : r1;
    if (i === 0) ctx.moveTo(cx + r * Math.cos(angle), cy + r * Math.sin(angle));
    else ctx.lineTo(cx + r * Math.cos(angle), cy + r * Math.sin(angle));
  }
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.shadowColor = color;
  ctx.shadowBlur = 18;
  ctx.fill();
  ctx.shadowBlur = 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. COLOR RAIN
// ─────────────────────────────────────────────────────────────────────────────
export class ColorRain {
  constructor(w, h) { this.resize(w, h); this.t = 0; this.drops = []; this.pools = []; }

  resize(w, h) { this.W = w; this.H = h; }

  _spawn() {
    this.drops.push({
      x: 20 + Math.random() * (this.W - 40),
      y: -20,
      r: 10 + Math.random() * 16,
      vy: 3 + Math.random() * 4,
      color: RAINBOW[Math.floor(Math.random() * RAINBOW.length)],
      hit: false,
    });
  }

  update(dt) {
    this.t += dt;
    if (Math.random() < dt * 3.5) this._spawn();
    for (let i = this.drops.length - 1; i >= 0; i--) {
      const d = this.drops[i];
      if (!d.hit) { d.y += d.vy; if (d.y > this.H + 30) this.drops.splice(i, 1); }
    }
    for (const p of this.pools) { p.life -= dt * 0.08; p.r = Math.min(p.r + dt * 20, p.maxR); }
    this.pools.filter(p => p.life > 0);
  }

  draw(ctx) {
    // Pools on ground
    for (const p of this.pools) {
      if (p.life <= 0) continue;
      ctx.save();
      ctx.globalAlpha = Math.min(0.7, p.life * 0.7);
      ctx.beginPath();
      ctx.ellipse(p.x, p.y, p.r, p.r * 0.3, 0, 0, Math.PI * 2);
      ctx.fillStyle = p.color;
      ctx.fill();
      ctx.restore();
    }
    // Drops
    for (const d of this.drops) {
      ctx.save();
      ctx.beginPath();
      ctx.arc(d.x, d.y, d.r, 0, Math.PI * 2);
      ctx.fillStyle = d.color;
      ctx.shadowColor = d.color;
      ctx.shadowBlur = 10;
      ctx.fill();
      ctx.restore();
    }
  }

  handleTap(x, y) {
    for (let i = this.drops.length - 1; i >= 0; i--) {
      const d = this.drops[i];
      if (!d.hit && Math.hypot(x - d.x, y - d.y) < d.r + 14) {
        d.hit = true;
        burst(d.x, d.y, 8, [d.color]);
        playSplash();
        this.pools.push({ x: d.x, y: this.H * 0.9, r: d.r, maxR: d.r * 4, color: d.color, life: 1 });
        this.drops.splice(i, 1);
        return true;
      }
    }
    return false;
  }

  handleDrag() {} handleDragEnd() {} destroy() {}
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. SQUISH BOUNCE (companion interaction)
// ─────────────────────────────────────────────────────────────────────────────
export class SquishBounce {
  constructor(w, h, squish) {
    this.W = w; this.H = h;
    this.squish = squish;
    this.prompt = 'Tap Squish!';
    this.t = 0;
  }
  resize(w, h) { this.W = w; this.H = h; }
  update(dt) { this.t += dt; }
  draw(ctx) {
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.font = `bold ${Math.min(this.W, this.H) * 0.055}px system-ui`;
    ctx.textAlign = 'center';
    ctx.fillText(this.prompt, this.W / 2, this.H * 0.15);
    // Arrow pointing to Squish
    const sx = this.squish.x + this.squish.size / 2;
    const sy = this.squish.y - 20 + Math.sin(this.t * 3) * 8;
    ctx.fillStyle = '#f59e0b';
    ctx.font = `${Math.min(this.W, this.H) * 0.07}px system-ui`;
    ctx.fillText('↓', sx, sy);
  }
  handleTap() { return false; } // Squish handles its own tap
  handleDrag() {} handleDragEnd() {} destroy() {}
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. FIREFLY GARDEN
// ─────────────────────────────────────────────────────────────────────────────
export class FireflyGarden {
  constructor(w, h) { this.resize(w, h); this.t = 0; this.flies = []; this._init(); }

  resize(w, h) { this.W = w; this.H = h; }

  _init() {
    const count = Math.floor((this.W * this.H) / 14000) + 12;
    this.flies = Array.from({ length: count }, (_, i) => ({
      x: 40 + Math.random() * (this.W - 80),
      y: this.H * 0.2 + Math.random() * (this.H * 0.65),
      r: 5 + Math.random() * 5,
      phase: Math.random() * Math.PI * 2,
      vx: (Math.random() - 0.5) * 0.6,
      vy: (Math.random() - 0.5) * 0.5,
      lit: false, litT: 0, ring: 0,
    }));
  }

  update(dt) {
    this.t += dt;
    for (const f of this.flies) {
      f.x += f.vx + Math.sin(this.t * 1.5 + f.phase) * 0.3;
      f.y += f.vy + Math.cos(this.t * 1.2 + f.phase) * 0.2;
      if (f.x < 20 || f.x > this.W - 20) f.vx *= -1;
      if (f.y < 50 || f.y > this.H - 50) f.vy *= -1;
      f.pulse = 0.5 + 0.5 * Math.sin(this.t * 3 + f.phase);
      if (f.lit) {
        f.litT += dt;
        f.ring += dt * 80;
        if (f.litT > 2) f.lit = false;
      }
    }
  }

  draw(ctx) {
    for (const f of this.flies) {
      ctx.save();
      const glow = f.lit ? 1 : f.pulse * 0.7;
      ctx.globalAlpha = 0.3 + glow * 0.7;
      // Glow halo
      const gr = ctx.createRadialGradient(f.x, f.y, 0, f.x, f.y, f.r * (f.lit ? 6 : 3));
      gr.addColorStop(0, '#f59e0b');
      gr.addColorStop(1, 'rgba(245,158,11,0)');
      ctx.fillStyle = gr;
      ctx.beginPath();
      ctx.arc(f.x, f.y, f.r * (f.lit ? 6 : 3), 0, Math.PI * 2);
      ctx.fill();
      // Core
      ctx.globalAlpha = 1;
      ctx.beginPath();
      ctx.arc(f.x, f.y, f.r, 0, Math.PI * 2);
      ctx.fillStyle = f.lit ? '#fef3c7' : '#fbbf24';
      ctx.fill();
      // Ring effect
      if (f.lit && f.ring < f.r * 8) {
        ctx.beginPath();
        ctx.arc(f.x, f.y, f.ring, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(245,158,11,${Math.max(0, 0.7 - f.ring / (f.r * 8))})`;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
      ctx.restore();
    }
  }

  handleTap(x, y) {
    for (const f of this.flies) {
      if (Math.hypot(x - f.x, y - f.y) < f.r + 16) {
        f.lit = true; f.litT = 0; f.ring = 0;
        playFirefly();
        // Ring of light sparkles
        for (let i = 0; i < 8; i++) {
          const a = (Math.PI * 2 * i) / 8;
          sparkle(f.x + Math.cos(a) * 30, f.y + Math.sin(a) * 30, '#f59e0b');
        }
        return true;
      }
    }
    return false;
  }

  handleDrag() {} handleDragEnd() {} destroy() {}
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. WEATHER WAND
// ─────────────────────────────────────────────────────────────────────────────
const WEATHERS = ['sunny', 'cloudy', 'rain', 'rainbow', 'night'];
const WEATHER_LABELS = ['☀️ Sunny', '☁️ Cloudy', '🌧 Rain', '🌈 Rainbow', '🌟 Night'];

export class WeatherWand {
  constructor(w, h) {
    this.resize(w, h);
    this.t = 0;
    this.wI = 0;
    this.target = 0;
    this.swipeStart = null;
    this.raindrops = [];
    this.stars = [];
    this.swipeX = null;
    this._spawnNight();
  }

  resize(w, h) { this.W = w; this.H = h; }

  _spawnNight() {
    this.stars = Array.from({ length: 60 }, () => ({
      x: Math.random() * this.W,
      y: Math.random() * this.H * 0.6,
      r: 1 + Math.random() * 2,
      phase: Math.random() * Math.PI * 2,
    }));
  }

  _spawnRain() {
    while (this.raindrops.length < 40) {
      this.raindrops.push({
        x: Math.random() * this.W,
        y: Math.random() * this.H,
        vy: 5 + Math.random() * 4,
        color: RAINBOW[Math.floor(Math.random() * RAINBOW.length)],
        len: 10 + Math.random() * 15,
      });
    }
  }

  update(dt) {
    this.t += dt;
    if (this.wI === 2) { // rain
      this._spawnRain();
      for (const d of this.raindrops) {
        d.y += d.vy;
        if (d.y > this.H) d.y = -d.len;
      }
    } else {
      this.raindrops = [];
    }
  }

  draw(ctx) {
    const weather = WEATHERS[this.wI];

    // Sky gradient per weather
    const skyColors = {
      sunny:   ['#0ea5e9','#7dd3fc'],
      cloudy:  ['#6b7280','#9ca3af'],
      rain:    ['#374151','#6b7280'],
      rainbow: ['#818cf8','#c4b5fd'],
      night:   ['#0a0a1a','#1e1b4b'],
    };
    const [c1, c2] = skyColors[weather];
    const bg = ctx.createLinearGradient(0, 0, 0, this.H);
    bg.addColorStop(0, c1); bg.addColorStop(1, c2);
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, this.W, this.H);

    // Ground
    ctx.fillStyle = weather === 'night' ? '#1a0a2e' : '#22c55e';
    ctx.beginPath();
    ctx.roundRect(0, this.H * 0.78, this.W, this.H * 0.22, [30, 30, 0, 0]);
    ctx.fill();

    // Weather elements
    if (weather === 'sunny') {
      const sunPulse = 1 + Math.sin(this.t * 2) * 0.04;
      ctx.save();
      ctx.translate(this.W * 0.75, this.H * 0.22);
      ctx.scale(sunPulse, sunPulse);
      ctx.beginPath();
      ctx.arc(0, 0, 55, 0, Math.PI * 2);
      ctx.fillStyle = '#fbbf24';
      ctx.shadowColor = '#f59e0b';
      ctx.shadowBlur = 30;
      ctx.fill();
      ctx.restore();
      // Rays
      for (let i = 0; i < 8; i++) {
        const a = (Math.PI * 2 * i / 8) + this.t * 0.3;
        ctx.beginPath();
        ctx.moveTo(this.W * 0.75 + Math.cos(a) * 65, this.H * 0.22 + Math.sin(a) * 65);
        ctx.lineTo(this.W * 0.75 + Math.cos(a) * 85, this.H * 0.22 + Math.sin(a) * 85);
        ctx.strokeStyle = '#fbbf24'; ctx.lineWidth = 5; ctx.lineCap = 'round'; ctx.stroke();
      }
    }

    if (weather === 'cloudy' || weather === 'rain') {
      this._drawCloud(ctx, this.W * 0.25, this.H * 0.2, 70, '#e5e7eb');
      this._drawCloud(ctx, this.W * 0.65, this.H * 0.15, 55, '#d1d5db');
    }

    if (weather === 'rain') {
      for (const d of this.raindrops) {
        ctx.beginPath();
        ctx.moveTo(d.x, d.y);
        ctx.lineTo(d.x - 2, d.y + d.len);
        ctx.strokeStyle = d.color + 'bb';
        ctx.lineWidth = 2; ctx.stroke();
      }
    }

    if (weather === 'rainbow') {
      const arcColors = ['#f43f5e','#f97316','#f59e0b','#22c55e','#3b82f6','#a855f7'];
      for (let i = 0; i < arcColors.length; i++) {
        ctx.beginPath();
        ctx.arc(this.W / 2, this.H * 0.8, this.W * 0.35 + i * 20, Math.PI, 0);
        ctx.strokeStyle = arcColors[i] + 'cc';
        ctx.lineWidth = 16; ctx.stroke();
      }
      this._drawCloud(ctx, this.W * 0.1, this.H * 0.45, 50, '#fff');
      this._drawCloud(ctx, this.W * 0.85, this.H * 0.45, 50, '#fff');
    }

    if (weather === 'night') {
      for (const s of this.stars) {
        const a = 0.5 + 0.5 * Math.sin(this.t * 2.5 + s.phase);
        ctx.save();
        ctx.globalAlpha = a;
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fillStyle = '#fff';
        ctx.fill();
        ctx.restore();
      }
      // Moon
      ctx.beginPath();
      ctx.arc(this.W * 0.75, this.H * 0.2, 40, 0, Math.PI * 2);
      ctx.fillStyle = '#fef3c7';
      ctx.shadowColor = '#fbbf24'; ctx.shadowBlur = 25;
      ctx.fill();
      ctx.shadowBlur = 0;
      // Moon crater (cutout effect)
      ctx.beginPath();
      ctx.arc(this.W * 0.75 + 22, this.H * 0.2 - 12, 30, 0, Math.PI * 2);
      ctx.fillStyle = weather === 'night' ? '#1e1b4b' : '#818cf8';
      ctx.fill();
    }

    // Label
    ctx.fillStyle = 'rgba(255,255,255,0.85)';
    ctx.font = `bold ${Math.min(this.W, this.H) * 0.048}px system-ui`;
    ctx.textAlign = 'center';
    ctx.fillText(WEATHER_LABELS[this.wI], this.W / 2, this.H * 0.88);
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.font = `${Math.min(this.W, this.H) * 0.033}px system-ui`;
    ctx.fillText('← swipe to change weather →', this.W / 2, this.H * 0.93);
  }

  _drawCloud(ctx, x, y, r, color) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.arc(x + r * 0.8, y - r * 0.3, r * 0.75, 0, Math.PI * 2);
    ctx.arc(x - r * 0.7, y + r * 0.1, r * 0.6, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }

  handleTap() { return false; }

  handleDrag(x, y) {
    if (this.swipeStart === null) { this.swipeStart = x; return; }
    this.swipeX = x;
  }

  handleDragEnd() {
    if (this.swipeStart !== null && this.swipeX !== null) {
      const dx = this.swipeX - this.swipeStart;
      if (Math.abs(dx) > 50) {
        this.wI = (this.wI + (dx > 0 ? 1 : -1) + WEATHERS.length) % WEATHERS.length;
        burst(this.W / 2, this.H * 0.5, 14, RAINBOW);
        playMagic();
        say(WEATHER_LABELS[this.wI].replace(/[^\w ]/g, ''));
      }
    }
    this.swipeStart = null;
    this.swipeX = null;
  }

  destroy() {}
}
