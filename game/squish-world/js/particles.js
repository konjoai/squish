/** particles.js — Shared particle system for tap bursts, confetti, etc. */

const particles = [];

const RAINBOW = ['#f43f5e','#f59e0b','#22c55e','#3b82f6','#a855f7','#14b8a6','#f97316','#ec4899'];

export function burst(x, y, count = 12, colors = RAINBOW, speed = 5) {
  for (let i = 0; i < count; i++) {
    const angle = (Math.PI * 2 * i) / count + Math.random() * 0.4;
    const s = speed * (0.6 + Math.random() * 0.8);
    particles.push({
      x, y,
      vx: Math.cos(angle) * s,
      vy: Math.sin(angle) * s - 2,
      r: 4 + Math.random() * 6,
      color: colors[i % colors.length],
      alpha: 1,
      gravity: 0.18,
      life: 1,
      decay: 0.025 + Math.random() * 0.02,
    });
  }
}

export function sparkle(x, y, color = '#f59e0b', count = 8) {
  for (let i = 0; i < count; i++) {
    const angle = Math.random() * Math.PI * 2;
    const s = 2 + Math.random() * 4;
    particles.push({
      x: x + Math.random() * 20 - 10,
      y: y + Math.random() * 20 - 10,
      vx: Math.cos(angle) * s,
      vy: Math.sin(angle) * s,
      r: 2 + Math.random() * 3,
      color,
      alpha: 1,
      gravity: 0.05,
      life: 1,
      decay: 0.03 + Math.random() * 0.03,
      star: true,
    });
  }
}

export function confetti(cx, cy, count = 40) {
  const colors = RAINBOW;
  for (let i = 0; i < count; i++) {
    particles.push({
      x: cx + Math.random() * 200 - 100,
      y: cy,
      vx: (Math.random() - 0.5) * 8,
      vy: -8 - Math.random() * 8,
      r: 5 + Math.random() * 5,
      ry: 1,
      rvy: 0.2 + Math.random() * 0.3,
      color: colors[i % colors.length],
      alpha: 1,
      gravity: 0.3,
      life: 1,
      decay: 0.012,
      rect: true,
      w: 6 + Math.random() * 8,
      h: 4 + Math.random() * 4,
      rot: Math.random() * Math.PI * 2,
    });
  }
}

export function updateParticles(dt) {
  for (let i = particles.length - 1; i >= 0; i--) {
    const p = particles[i];
    p.x += p.vx;
    p.y += p.vy;
    p.vy += p.gravity;
    p.vx *= 0.97;
    p.life -= p.decay;
    p.alpha = Math.max(0, p.life);
    if (p.rot !== undefined) p.rot += p.rvy || 0.1;
    if (p.life <= 0) particles.splice(i, 1);
  }
}

export function drawParticles(ctx) {
  for (const p of particles) {
    ctx.save();
    ctx.globalAlpha = p.alpha;
    ctx.fillStyle = p.color;
    if (p.rect) {
      ctx.translate(p.x, p.y);
      ctx.rotate(p.rot || 0);
      ctx.fillRect(-p.w / 2, -p.h / 2, p.w, p.h);
    } else if (p.star) {
      ctx.translate(p.x, p.y);
      drawStar(ctx, 0, 0, p.r * 0.5, p.r, 4);
    } else {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }
}

function drawStar(ctx, cx, cy, r1, r2, n) {
  ctx.beginPath();
  for (let i = 0; i < n * 2; i++) {
    const angle = (Math.PI / n) * i - Math.PI / 2;
    const r = i % 2 === 0 ? r2 : r1;
    if (i === 0) ctx.moveTo(cx + r * Math.cos(angle), cy + r * Math.sin(angle));
    else ctx.lineTo(cx + r * Math.cos(angle), cy + r * Math.sin(angle));
  }
  ctx.closePath();
  ctx.fill();
}
