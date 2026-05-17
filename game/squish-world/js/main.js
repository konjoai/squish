/**
 * main.js — Squish World game engine.
 * State machine: AVATAR_CREATE → STAGE0 → STAGE1_HUB → STAGE1_ACTIVITY
 */
import { initAudio, resumeAudio } from './audio.js';
import { initSpeech, say } from './speech.js';
import { loadSave, getSave, writeSave, addStage0Time, unlockStage1, loadAvatar } from './save.js';
import { updateParticles, drawParticles } from './particles.js';
import { Squish } from './squish.js';
import { AvatarCreatorScene, drawAvatar } from './avatar.js';
import {
  BubblePop, AnimalParade, MagicPaint, MusicStars,
  ColorRain, SquishBounce, FireflyGarden, WeatherWand
} from './stage0/activities.js';
import { ColorMatch, SortShapes, PatternParade } from './stage1/activities.js';

// ── Canvas setup ─────────────────────────────────────────────────────────────
const canvas = document.getElementById('game');
const ctx = canvas.getContext('2d');

function resize() {
  canvas.width  = window.innerWidth;
  canvas.height = window.innerHeight;
  if (game) game.onResize(canvas.width, canvas.height);
}
window.addEventListener('resize', resize);

// ── Input ─────────────────────────────────────────────────────────────────────
function pos(e, touch = false) {
  const src = touch ? e.changedTouches[0] : e;
  const r = canvas.getBoundingClientRect();
  return { x: src.clientX - r.left, y: src.clientY - r.top };
}

canvas.addEventListener('mousedown',  e => { resumeAudio(); game && game.onTap(pos(e), 'start'); });
canvas.addEventListener('mousemove',  e => { game && game.onDrag(pos(e)); });
canvas.addEventListener('mouseup',    e => { game && game.onTapEnd(pos(e)); });
canvas.addEventListener('touchstart', e => { e.preventDefault(); resumeAudio(); game && game.onTap(pos(e, true), 'start'); }, { passive: false });
canvas.addEventListener('touchmove',  e => { e.preventDefault(); game && game.onDrag(pos(e, true)); }, { passive: false });
canvas.addEventListener('touchend',   e => { e.preventDefault(); game && game.onTapEnd(pos(e, true)); }, { passive: false });

// ── Stage 0 activity list ─────────────────────────────────────────────────────
const STAGE0_ACTIVITIES = [
  { id:'bubbles',  label:'Bubble Pop',     emoji:'🫧', desc:'Pop the rainbow bubbles!' },
  { id:'animals',  label:'Animal Parade',  emoji:'🦆', desc:'Tap the animals!' },
  { id:'paint',    label:'Magic Paint',    emoji:'🎨', desc:'Draw sparkly trails!' },
  { id:'stars',    label:'Music Stars',    emoji:'⭐', desc:'Tap stars for music!' },
  { id:'rain',     label:'Color Rain',     emoji:'🌈', desc:'Splash the colorful drops!' },
  { id:'squish',   label:'Squish Bounce',  emoji:'💜', desc:'Tap Squish!' },
  { id:'firefly',  label:'Firefly Garden', emoji:'✨', desc:'Light up the fireflies!' },
  { id:'weather',  label:'Weather Wand',   emoji:'🌦', desc:'Swipe to change weather!' },
];

const STAGE1_ACTIVITIES = [
  { id:'colormatch', label:'Color Match',    emoji:'🎨', desc:'Match colors to buckets!', Cls: ColorMatch },
  { id:'sortshapes', label:'Sort Shapes',    emoji:'⬡',  desc:'Sort shapes by type!',    Cls: SortShapes },
  { id:'patterns',   label:'Pattern Parade', emoji:'🔮', desc:'What comes next?',        Cls: PatternParade },
];

// ── Main game object ──────────────────────────────────────────────────────────
let game = null;

class Game {
  constructor(w, h) {
    this.W = w; this.H = h;
    this.t = 0;
    this.lastTime = null;
    this.squish = new Squish(w, h);
    this.state = 'loading';  // loading | avatar | stage0 | stage0_activity | stage1_hub | stage1_activity
    this.scene = null;       // current activity/scene object
    this.activeActivityId = null;
    this.avatarConfig = null;
    this.stage0Time = 0;
    this.stage0Prompted = false;
    this.hubButtons = [];
    this.stage0Buttons = [];
    this.backBtn = null;
    this._init();
  }

  _init() {
    loadSave();
    this.avatarConfig = loadAvatar();
    if (!this.avatarConfig) {
      this._goto('avatar');
    } else {
      this._goto('stage0');
    }
    this._buildStage0Buttons();
  }

  _goto(state, extra = {}) {
    if (this.scene && this.scene.destroy) this.scene.destroy();
    this.scene = null;
    this.activeActivityId = null;
    this.state = state;

    if (state === 'avatar') {
      this.scene = new AvatarCreatorScene(canvas, cfg => {
        this.avatarConfig = cfg;
        this._goto('stage0');
      });
    } else if (state === 'stage0_activity') {
      this._startStage0Activity(extra.id);
    } else if (state === 'stage1_activity') {
      this._startStage1Activity(extra.id);
    } else if (state === 'stage0') {
      this._buildStage0Buttons();
      say("Let's play! Tap something!");
    } else if (state === 'stage1_hub') {
      this._buildHubButtons();
      say("Welcome! Which door will you choose?");
    }

    this.backBtn = (state === 'stage0_activity' || state === 'stage1_activity' || state === 'stage1_hub') ? {
      x: 46, y: 46, r: 32, label: '←',
    } : null;
  }

  _buildStage0Buttons() {
    const W = this.W, H = this.H;
    const cols = W > 500 ? 4 : 2;
    const rows = Math.ceil(STAGE0_ACTIVITIES.length / cols);
    const bw = (W - 32) / cols - 12;
    const bh = Math.min(bw, (H * 0.72 - 32) / rows - 12);
    this.stage0Buttons = STAGE0_ACTIVITIES.map((a, i) => ({
      ...a,
      x: 16 + (i % cols) * (bw + 12) + bw / 2,
      y: H * 0.16 + Math.floor(i / cols) * (bh + 12) + bh / 2,
      w: bw, h: bh,
    }));
  }

  _buildHubButtons() {
    const W = this.W, H = this.H;
    const bw = Math.min((W - 48) / 3 - 12, 160);
    this.hubButtons = STAGE1_ACTIVITIES.map((a, i) => ({
      ...a,
      x: 20 + i * (bw + 12) + bw / 2,
      y: H * 0.52,
      w: bw, h: bw * 1.1,
    }));
  }

  _startStage0Activity(id) {
    const W = this.W, H = this.H;
    this.activeActivityId = id;
    switch (id) {
      case 'bubbles':  this.scene = new BubblePop(W, H); break;
      case 'animals':  this.scene = new AnimalParade(W, H); break;
      case 'paint':    this.scene = new MagicPaint(W, H); break;
      case 'stars':    this.scene = new MusicStars(W, H); break;
      case 'rain':     this.scene = new ColorRain(W, H); break;
      case 'squish':   this.scene = new SquishBounce(W, H, this.squish); break;
      case 'firefly':  this.scene = new FireflyGarden(W, H); break;
      case 'weather':  this.scene = new WeatherWand(W, H); break;
    }
  }

  _startStage1Activity(id) {
    const W = this.W, H = this.H;
    this.activeActivityId = id;
    const def = STAGE1_ACTIVITIES.find(a => a.id === id);
    if (def) this.scene = new def.Cls(W, H);
  }

  onResize(w, h) {
    this.W = w; this.H = h;
    this.squish.resize(w, h);
    if (this.scene && this.scene.resize) this.scene.resize(w, h);
    this._buildStage0Buttons();
    this._buildHubButtons();
    if (this.backBtn) { this.backBtn.x = 46; this.backBtn.y = 46; }
  }

  onTap(p, phase) {
    const { x, y } = p;
    initAudio();

    // Back button
    if (this.backBtn && Math.hypot(x - this.backBtn.x, y - this.backBtn.y) < this.backBtn.r + 10) {
      if (this.state === 'stage0_activity') this._goto('stage0');
      else if (this.state === 'stage1_activity' || this.state === 'stage1_hub') this._goto('stage0');
      return;
    }

    // Squish companion tap (always active)
    if (this.squish.hit(x, y)) {
      this.squish.onTap();
      if (this.state === 'stage0') {
        // Hint toward Stage 1 unlock
        const s = getSave();
        if (s.time_in_stage0 > 55) {
          say("Tap 'Explore More' to try new adventures!");
        }
      }
      return;
    }

    switch (this.state) {
      case 'avatar':
        if (this.scene) this.scene.handleTap(x, y);
        break;

      case 'stage0':
        for (const btn of this.stage0Buttons) {
          if (Math.abs(x - btn.x) < btn.w / 2 + 10 && Math.abs(y - btn.y) < btn.h / 2 + 10) {
            this._goto('stage0_activity', { id: btn.id });
            return;
          }
        }
        // "Explore more" button
        if (this.stage0Prompted || getSave().time_in_stage0 >= 60) {
          const ebx = this.W / 2, eby = this.H * 0.94;
          if (Math.abs(x - ebx) < 120 && Math.abs(y - eby) < 30) {
            unlockStage1();
            this._goto('stage1_hub');
            return;
          }
        }
        break;

      case 'stage0_activity':
        if (this.scene) {
          const result = this.scene.handleTap(x, y);
          if (result && result.done) {
            this.squish.celebrate();
            setTimeout(() => this._goto('stage0'), 2000);
          }
        }
        break;

      case 'stage1_hub':
        for (const btn of this.hubButtons) {
          if (Math.abs(x - btn.x) < btn.w / 2 + 14 && Math.abs(y - btn.y) < btn.h / 2 + 14) {
            this._goto('stage1_activity', { id: btn.id });
            return;
          }
        }
        break;

      case 'stage1_activity':
        if (this.scene) {
          const result = this.scene.handleTap(x, y);
          if (result && result.done) {
            this.squish.celebrate();
            setTimeout(() => this._goto('stage1_hub'), 2500);
          }
        }
        break;
    }
  }

  onDrag(p) {
    const { x, y } = p;
    if (this.scene && this.scene.handleDrag) this.scene.handleDrag(x, y);
  }

  onTapEnd(p) {
    const { x, y } = p;
    if (this.scene && this.scene.handleDragEnd) {
      const result = this.scene.handleDragEnd(x, y);
      if (result && result.done) {
        this.squish.celebrate();
        setTimeout(() => {
          if (this.state === 'stage1_activity') this._goto('stage1_hub');
          else this._goto('stage0');
        }, 2500);
      }
    }
  }

  update(dt) {
    this.t += dt;
    this.squish.update(dt);
    if (this.scene && this.scene.update) this.scene.update(dt);
    updateParticles(dt);

    // Track stage0 time
    if (this.state === 'stage0' || this.state === 'stage0_activity') {
      this.stage0Time += dt;
      // Auto-save every 10s
      if (Math.floor(this.stage0Time) % 10 === 0 && this.stage0Time > 1) {
        addStage0Time(dt);
      }
      // Prompt after 60s
      const total = getSave().time_in_stage0 + this.stage0Time;
      if (total >= 60 && !this.stage0Prompted) {
        this.stage0Prompted = true;
        this.squish.shimmer();
        say("Want to explore more? Tap me to find out!");
      }
    }
  }

  draw() {
    const W = this.W, H = this.H;

    // Base background
    ctx.fillStyle = '#0a0a1a';
    ctx.fillRect(0, 0, W, H);

    // Ambient stars (always visible)
    if (this.state !== 'stage0_activity' || this.activeActivityId !== 'weather') {
      drawAmbientStars(ctx, W, H, this.t);
    }

    // Draw scene
    if (this.scene && this.scene.draw) this.scene.draw(ctx);

    // Stage 0 grid
    if (this.state === 'stage0') this._drawStage0(ctx);

    // Stage 1 hub
    if (this.state === 'stage1_hub') this._drawHub(ctx);

    // Particles
    drawParticles(ctx);

    // Squish (on top)
    this.squish.draw(ctx);

    // Avatar in top-left (outside avatar creator)
    if (this.avatarConfig && this.state !== 'avatar') {
      const avSize = Math.min(W, H) * 0.095;
      drawAvatar(ctx, avSize * 0.55, H - avSize * 0.65, avSize, this.avatarConfig);
    }

    // Back button
    if (this.backBtn) this._drawBack(ctx);

    // "Explore More" button
    if (this.state === 'stage0' && (this.stage0Prompted || getSave().time_in_stage0 >= 60)) {
      const pulse = 1 + Math.sin(this.t * 3) * 0.04;
      const bx = W / 2, by = H * 0.94;
      ctx.save();
      ctx.translate(bx, by);
      ctx.scale(pulse, pulse);
      ctx.beginPath();
      ctx.roundRect(-110, -24, 220, 48, 24);
      ctx.fillStyle = '#7c3aed';
      ctx.shadowColor = '#7c3aed'; ctx.shadowBlur = 20;
      ctx.fill();
      ctx.fillStyle = '#fff';
      ctx.font = `bold ${Math.min(W, H) * 0.038}px system-ui`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('✨ Explore More!', 0, 0);
      ctx.restore();
      ctx.textBaseline = 'alphabetic';
    }
  }

  _drawStage0(ctx) {
    const W = this.W, H = this.H;
    // Title
    const ts = Math.min(W, H) * 0.058;
    ctx.fillStyle = '#f59e0b';
    ctx.font = `bold ${ts}px system-ui`;
    ctx.textAlign = 'center';
    ctx.shadowColor = '#f59e0b'; ctx.shadowBlur = 18;
    ctx.fillText("✨ Squish World ✨", W / 2, H * 0.1);
    ctx.shadowBlur = 0;

    // Activity tiles
    for (const btn of this.stage0Buttons) {
      const hovered = false; // track mouse hover if needed
      ctx.save();
      ctx.beginPath();
      ctx.roundRect(btn.x - btn.w / 2, btn.y - btn.h / 2, btn.w, btn.h, 16);
      const g = ctx.createLinearGradient(btn.x - btn.w / 2, btn.y - btn.h / 2, btn.x + btn.w / 2, btn.y + btn.h / 2);
      g.addColorStop(0, 'rgba(124,58,237,0.25)');
      g.addColorStop(1, 'rgba(20,184,166,0.15)');
      ctx.fillStyle = g;
      ctx.strokeStyle = 'rgba(255,255,255,0.18)';
      ctx.lineWidth = 1.5;
      ctx.fill();
      ctx.stroke();
      // Emoji
      const es = Math.min(btn.w, btn.h) * 0.38;
      ctx.font = `${es}px system-ui`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(btn.emoji, btn.x, btn.y - btn.h * 0.08);
      // Label
      ctx.fillStyle = '#fff';
      ctx.font = `bold ${Math.min(btn.w, btn.h) * 0.17}px system-ui`;
      ctx.textBaseline = 'middle';
      ctx.fillText(btn.label, btn.x, btn.y + btn.h * 0.3);
      ctx.restore();
    }
    ctx.textBaseline = 'alphabetic';
  }

  _drawHub(ctx) {
    const W = this.W, H = this.H;
    // Title
    ctx.fillStyle = '#f59e0b';
    ctx.font = `bold ${Math.min(W, H) * 0.055}px system-ui`;
    ctx.textAlign = 'center';
    ctx.shadowColor = '#f59e0b'; ctx.shadowBlur = 14;
    ctx.fillText("🏠 Squish World", W / 2, H * 0.1);
    ctx.shadowBlur = 0;

    // Cozy house background element
    ctx.fillStyle = 'rgba(124,58,237,0.12)';
    ctx.beginPath();
    ctx.roundRect(W * 0.12, H * 0.15, W * 0.76, H * 0.78, 24);
    ctx.fill();

    // Ground
    ctx.fillStyle = 'rgba(34,197,94,0.2)';
    ctx.beginPath();
    ctx.roundRect(0, H * 0.68, W, H * 0.32, [20, 20, 0, 0]);
    ctx.fill();

    // Hub doors/buttons
    for (const btn of this.hubButtons) {
      ctx.save();
      // Door frame
      ctx.beginPath();
      ctx.roundRect(btn.x - btn.w / 2, btn.y - btn.h / 2, btn.w, btn.h, [20, 20, 4, 4]);
      const g = ctx.createLinearGradient(btn.x, btn.y - btn.h / 2, btn.x, btn.y + btn.h / 2);
      g.addColorStop(0, 'rgba(124,58,237,0.55)');
      g.addColorStop(1, 'rgba(20,184,166,0.35)');
      ctx.fillStyle = g;
      ctx.strokeStyle = '#7c3aed';
      ctx.lineWidth = 2.5;
      ctx.shadowColor = '#7c3aed'; ctx.shadowBlur = 16;
      ctx.fill();
      ctx.stroke();
      ctx.shadowBlur = 0;
      // Emoji
      ctx.font = `${btn.w * 0.45}px system-ui`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(btn.emoji, btn.x, btn.y - btn.h * 0.1);
      // Label
      ctx.fillStyle = '#fff';
      ctx.font = `bold ${btn.w * 0.17}px system-ui`;
      ctx.textBaseline = 'middle';
      ctx.fillText(btn.label, btn.x, btn.y + btn.h * 0.3);
      ctx.restore();
    }
    ctx.textBaseline = 'alphabetic';

    // Subtitle
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.font = `${Math.min(W, H) * 0.035}px system-ui`;
    ctx.textAlign = 'center';
    ctx.fillText('Choose a door to enter!', W / 2, H * 0.9);
  }

  _drawBack(ctx) {
    const b = this.backBtn;
    ctx.save();
    ctx.beginPath();
    ctx.arc(b.x, b.y, b.r, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(124,58,237,0.75)';
    ctx.shadowColor = '#7c3aed'; ctx.shadowBlur = 12;
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.fillStyle = '#fff';
    ctx.font = `bold ${b.r}px system-ui`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(b.label, b.x, b.y);
    ctx.restore();
    ctx.textBaseline = 'alphabetic';
  }
}

// ── Ambient background stars ──────────────────────────────────────────────────
function drawAmbientStars(ctx, W, H, t) {
  for (let i = 0; i < 35; i++) {
    const sx = ((i * 137.508) % W);
    const sy = ((i * 97.3)    % (H * 0.92));
    const sr = 0.8 + (i % 3) * 0.5;
    const alpha = 0.2 + 0.25 * Math.sin(t * 1.5 + i * 0.7);
    ctx.fillStyle = `rgba(255,255,255,${alpha})`;
    ctx.beginPath();
    ctx.arc(sx, sy, sr, 0, Math.PI * 2);
    ctx.fill();
  }
}

// ── Game loop ─────────────────────────────────────────────────────────────────
function loop(ts) {
  const dt = game.lastTime === null ? 0.016 : Math.min((ts - game.lastTime) / 1000, 0.05);
  game.lastTime = ts;
  game.update(dt);
  game.draw();
  requestAnimationFrame(loop);
}

// ── Boot ──────────────────────────────────────────────────────────────────────
function boot() {
  initSpeech();
  resize();
  game = new Game(canvas.width, canvas.height);
  requestAnimationFrame(loop);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
