/** avatar.js — Avatar creator scene + avatar renderer. */
import { saveAvatar } from './save.js';
import { playPop, playSuccess } from './audio.js';
import { say } from './speech.js';
import { burst } from './particles.js';

const HAIR_COLORS = ['#f59e0b','#a16207','#dc2626','#1d4ed8','#7c3aed'];
const SKIN_TONES  = ['#fde68a','#fbbf24','#d97706','#92400e','#7c3cd3','#f87171'];
const OUTFIT_COLORS = ['#7c3aed','#f43f5e','#14b8a6','#f59e0b','#22c55e'];

const HAIR_ICONS = ['curly','wavy','straight','braids','buns'];
const OUTFIT_ICONS = ['stars','hearts','stripes','dots','rainbows'];

export class AvatarCreatorScene {
  constructor(canvas, onDone) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.onDone = onDone;
    this.config = {
      hairColor: HAIR_COLORS[0],
      skinTone: SKIN_TONES[2],
      outfitColor: OUTFIT_COLORS[0],
      hairStyle: 0,
      outfitStyle: 0,
    };
    this.t = 0;
    this.doneBtn = null;
    this.swatches = [];
    this._buildUI();
    say("Hi! Let's make your character! Pick your favorite colors!");
  }

  _buildUI() {
    const { width: W, height: H } = this.canvas;
    const sw = 44, gap = 10;
    this.swatches = [];

    const row = (colors, y, type) =>
      colors.map((c, i) => ({
        x: W / 2 - (colors.length * (sw + gap)) / 2 + i * (sw + gap) + sw / 2,
        y, r: sw / 2, color: c, type,
      }));

    const sectionY = H * 0.55;
    this.swatches = [
      ...row(HAIR_COLORS, sectionY, 'hair'),
      ...row(SKIN_TONES, sectionY + sw + gap * 2, 'skin'),
      ...row(OUTFIT_COLORS, sectionY + (sw + gap * 2) * 2, 'outfit'),
    ];

    this.doneBtn = {
      x: W / 2, y: H * 0.92,
      w: 180, h: 56,
      label: "Let's go! ✨",
    };
  }

  resize() {
    this._buildUI();
  }

  update(dt) { this.t += dt; }

  draw() {
    const { ctx, canvas } = this;
    const { width: W, height: H } = canvas;
    ctx.clearRect(0, 0, W, H);

    // Background gradient
    const bg = ctx.createLinearGradient(0, 0, 0, H);
    bg.addColorStop(0, '#0a0a2e');
    bg.addColorStop(1, '#1a0a3e');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, W, H);

    // Stars
    for (let i = 0; i < 40; i++) {
      const sx = ((i * 137.5) % W);
      const sy = ((i * 73.1) % (H * 0.45));
      const sr = 1 + (i % 3);
      const alpha = 0.4 + 0.4 * Math.sin(this.t * 2 + i);
      ctx.fillStyle = `rgba(255,255,255,${alpha})`;
      ctx.beginPath();
      ctx.arc(sx, sy, sr, 0, Math.PI * 2);
      ctx.fill();
    }

    // Title
    ctx.fillStyle = '#f59e0b';
    ctx.font = `bold ${Math.min(W, H) * 0.07}px system-ui, sans-serif`;
    ctx.textAlign = 'center';
    ctx.fillText("Make your character!", W / 2, H * 0.09);

    // Avatar preview
    drawAvatar(ctx, W / 2, H * 0.32, Math.min(W, H) * 0.22, this.config);

    // Section labels
    const labelFont = `bold ${Math.min(W, H) * 0.038}px system-ui`;
    ctx.font = labelFont;
    ctx.fillStyle = '#fff';
    const sw = 44, gap = 10;
    const sectionY = H * 0.55;
    ctx.textAlign = 'center';
    ctx.fillText("Hair", W / 2, sectionY - 8);
    ctx.fillText("Skin", W / 2, sectionY + sw + gap * 2 - 8);
    ctx.fillText("Outfit", W / 2, sectionY + (sw + gap * 2) * 2 - 8);

    // Swatches
    for (const s of this.swatches) {
      const active = this._isActive(s);
      ctx.save();
      if (active) {
        ctx.shadowColor = s.color;
        ctx.shadowBlur = 14;
        ctx.scale(1.15, 1.15);
        ctx.translate(s.x * (1 - 1 / 1.15), s.y * (1 - 1 / 1.15));
      }
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fillStyle = s.color;
      ctx.fill();
      if (active) {
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 3;
        ctx.stroke();
      }
      ctx.restore();
    }

    // Done button
    const btn = this.doneBtn;
    ctx.save();
    const pulse = 1 + Math.sin(this.t * 3) * 0.03;
    ctx.translate(btn.x, btn.y);
    ctx.scale(pulse, pulse);
    ctx.beginPath();
    ctx.roundRect(-btn.w / 2, -btn.h / 2, btn.w, btn.h, 28);
    const btnGrad = ctx.createLinearGradient(-btn.w / 2, 0, btn.w / 2, 0);
    btnGrad.addColorStop(0, '#7c3aed');
    btnGrad.addColorStop(1, '#a855f7');
    ctx.fillStyle = btnGrad;
    ctx.shadowColor = '#7c3aed';
    ctx.shadowBlur = 20;
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.font = `bold ${Math.min(W, H) * 0.046}px system-ui`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(btn.label, 0, 0);
    ctx.restore();
    ctx.textBaseline = 'alphabetic';
  }

  _isActive(s) {
    if (s.type === 'hair')   return this.config.hairColor === s.color;
    if (s.type === 'skin')   return this.config.skinTone === s.color;
    if (s.type === 'outfit') return this.config.outfitColor === s.color;
    return false;
  }

  handleTap(x, y) {
    // Swatches
    for (const s of this.swatches) {
      if (Math.hypot(x - s.x, y - s.y) < s.r + 12) {
        playPop(1.2);
        if (s.type === 'hair')   this.config.hairColor = s.color;
        if (s.type === 'skin')   this.config.skinTone = s.color;
        if (s.type === 'outfit') this.config.outfitColor = s.color;
        burst(x, y, 8, [s.color]);
        return;
      }
    }

    // Done button
    const btn = this.doneBtn;
    if (Math.abs(x - btn.x) < btn.w / 2 + 10 && Math.abs(y - btn.y) < btn.h / 2 + 10) {
      playSuccess();
      say("Let's go! Time to play!");
      burst(x, y, 20);
      saveAvatar(this.config);
      setTimeout(() => this.onDone(this.config), 400);
    }
  }
}

export function drawAvatar(ctx, cx, cy, size, cfg = {}) {
  const skin = cfg.skinTone || '#d97706';
  const hair = cfg.hairColor || '#f59e0b';
  const outfit = cfg.outfitColor || '#7c3aed';
  const s = size;

  ctx.save();
  ctx.translate(cx, cy);

  // Body (outfit)
  ctx.beginPath();
  ctx.roundRect(-s * 0.28, -s * 0.08, s * 0.56, s * 0.62, s * 0.12);
  ctx.fillStyle = outfit;
  ctx.fill();

  // Head
  ctx.beginPath();
  ctx.arc(0, -s * 0.22, s * 0.28, 0, Math.PI * 2);
  ctx.fillStyle = skin;
  ctx.fill();

  // Hair
  ctx.beginPath();
  ctx.arc(0, -s * 0.38, s * 0.3, Math.PI, 0);
  ctx.closePath();
  ctx.fillStyle = hair;
  ctx.fill();
  // Hair bun
  ctx.beginPath();
  ctx.arc(0, -s * 0.58, s * 0.12, 0, Math.PI * 2);
  ctx.fillStyle = hair;
  ctx.fill();

  // Eyes
  ctx.fillStyle = '#1a0a3e';
  ctx.beginPath(); ctx.arc(-s * 0.09, -s * 0.2, s * 0.045, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.arc( s * 0.09, -s * 0.2, s * 0.045, 0, Math.PI * 2); ctx.fill();

  // Eye shine
  ctx.fillStyle = '#fff';
  ctx.beginPath(); ctx.arc(-s * 0.075, -s * 0.215, s * 0.018, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.arc( s * 0.105, -s * 0.215, s * 0.018, 0, Math.PI * 2); ctx.fill();

  // Smile
  ctx.beginPath();
  ctx.arc(0, -s * 0.1, s * 0.1, 0.2, Math.PI - 0.2);
  ctx.strokeStyle = '#7c3c28';
  ctx.lineWidth = s * 0.03;
  ctx.lineCap = 'round';
  ctx.stroke();

  // Arms
  ctx.beginPath();
  ctx.roundRect(-s * 0.46, 0, s * 0.18, s * 0.35, s * 0.09);
  ctx.fillStyle = skin; ctx.fill();
  ctx.beginPath();
  ctx.roundRect( s * 0.28, 0, s * 0.18, s * 0.35, s * 0.09);
  ctx.fillStyle = skin; ctx.fill();

  // Legs
  ctx.beginPath();
  ctx.roundRect(-s * 0.22, s * 0.5, s * 0.18, s * 0.32, s * 0.09);
  ctx.fillStyle = skin; ctx.fill();
  ctx.beginPath();
  ctx.roundRect( s * 0.04, s * 0.5, s * 0.18, s * 0.32, s * 0.09);
  ctx.fillStyle = skin; ctx.fill();

  ctx.restore();
}
