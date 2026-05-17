/** squish.js — Squish companion character. Always present. */
import { sparkle, burst } from './particles.js';
import { playBounce, playGiggle } from './audio.js';
import { say, SQUISH_IDLE, SQUISH_SUCCESS, sayRandom } from './speech.js';

export class Squish {
  constructor(canvasW, canvasH) {
    this.W = canvasW;
    this.H = canvasH;
    this.size = Math.min(canvasW, canvasH) * 0.12;
    // Floating in bottom-right corner
    this.baseX = canvasW - this.size - 20;
    this.baseY = canvasH - this.size - 20;
    this.x = this.baseX;
    this.y = this.baseY;
    this.t = 0;
    this.scale = 1;
    this.rotation = 0;
    this.color = '#7c3aed';
    this.colors = ['#7c3aed','#f43f5e','#f59e0b','#14b8a6','#22c55e','#3b82f6'];
    this.colorIndex = 0;
    this.bounceVy = 0;
    this.bounceY = 0;
    this.state = 'idle'; // idle | bounce | spin | shimmer
    this.stateTimer = 0;
    this.idleTimer = 0;
    this.idleInterval = 8 + Math.random() * 7; // seconds between idle phrases
    this.tapCount = 0;
  }

  resize(w, h) {
    this.W = w; this.H = h;
    this.size = Math.min(w, h) * 0.12;
    this.baseX = w - this.size - 20;
    this.baseY = h - this.size - 20;
    this.x = this.baseX;
    this.y = this.baseY;
  }

  update(dt) {
    this.t += dt;
    this.idleTimer += dt;

    // Idle bob
    const bob = Math.sin(this.t * 2.2) * 6;

    // State machine
    switch (this.state) {
      case 'idle':
        this.bounceY += (bob - this.bounceY) * 0.1;
        this.scale += (1 - this.scale) * 0.1;
        this.rotation += (0 - this.rotation) * 0.1;
        if (this.idleTimer > this.idleInterval) {
          this.idleTimer = 0;
          this.idleInterval = 8 + Math.random() * 7;
          sayRandom(SQUISH_IDLE);
        }
        break;
      case 'bounce':
        this.bounceVy += 0.8;
        this.bounceY += this.bounceVy;
        this.scale = 1 + Math.sin(this.stateTimer * 15) * 0.12;
        if (this.bounceY > 0) {
          this.bounceY = 0;
          this.bounceVy = -this.bounceVy * 0.55;
          if (Math.abs(this.bounceVy) < 1) this.state = 'idle';
        }
        this.stateTimer += dt;
        break;
      case 'spin':
        this.rotation += 6 * dt;
        this.stateTimer += dt;
        this.scale = 1 + Math.sin(this.stateTimer * 8) * 0.08;
        if (this.stateTimer > 1.2) {
          this.state = 'idle';
          this.rotation = 0;
        }
        break;
      case 'shimmer':
        this.scale = 1 + Math.sin(this.stateTimer * 12) * 0.06;
        this.stateTimer += dt;
        if (this.stateTimer > 0.8) this.state = 'idle';
        break;
    }

    this.y = this.baseY + this.bounceY;
    this.x = this.baseX + Math.sin(this.t * 1.1) * 4;
  }

  draw(ctx) {
    ctx.save();
    ctx.translate(this.x + this.size / 2, this.y + this.size / 2);
    ctx.rotate(this.rotation);
    ctx.scale(this.scale, this.scale);

    const s = this.size;
    const r = s * 0.22;

    // Glow
    const glow = ctx.createRadialGradient(0, 0, s * 0.2, 0, 0, s * 0.7);
    glow.addColorStop(0, this.color + '60');
    glow.addColorStop(1, this.color + '00');
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(0, 0, s * 0.7, 0, Math.PI * 2);
    ctx.fill();

    // Body (rounded cube face)
    ctx.beginPath();
    const hs = s / 2;
    ctx.roundRect(-hs, -hs, s, s, r);
    const bodyGrad = ctx.createLinearGradient(-hs, -hs, hs, hs);
    bodyGrad.addColorStop(0, lighten(this.color, 30));
    bodyGrad.addColorStop(1, this.color);
    ctx.fillStyle = bodyGrad;
    ctx.shadowColor = this.color;
    ctx.shadowBlur = 18;
    ctx.fill();
    ctx.shadowBlur = 0;

    // Eyes
    const eyeY = -s * 0.08;
    const eyeR = s * 0.1;
    const eyeX = s * 0.16;
    ctx.fillStyle = '#fff';
    ctx.beginPath(); ctx.arc(-eyeX, eyeY, eyeR, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(eyeX,  eyeY, eyeR, 0, Math.PI * 2); ctx.fill();

    // Pupils with happy squint during bounce
    const pupilR = eyeR * 0.55;
    const squint = this.state === 'bounce' ? 0.3 : 0;
    ctx.fillStyle = '#1a0a3e';
    if (squint > 0) {
      // Happy squint eyes
      ctx.beginPath();
      ctx.arc(-eyeX, eyeY + squint * eyeR, pupilR, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath();
      ctx.arc(eyeX, eyeY + squint * eyeR, pupilR, 0, Math.PI * 2); ctx.fill();
    } else {
      ctx.beginPath(); ctx.arc(-eyeX, eyeY, pupilR, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(eyeX, eyeY, pupilR, 0, Math.PI * 2); ctx.fill();
    }

    // Smile
    const smileY = s * 0.15;
    ctx.beginPath();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = s * 0.05;
    ctx.lineCap = 'round';
    ctx.arc(0, smileY, s * 0.2, 0.2, Math.PI - 0.2);
    ctx.stroke();

    // Shimmer stars during shimmer state
    if (this.state === 'shimmer') {
      ctx.fillStyle = '#f59e0b';
      for (let i = 0; i < 4; i++) {
        const a = (i / 4) * Math.PI * 2 + this.t * 3;
        const sr = s * 0.55;
        ctx.beginPath();
        ctx.arc(Math.cos(a) * sr, Math.sin(a) * sr, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    ctx.restore();
  }

  hit(px, py) {
    const cx = this.x + this.size / 2;
    const cy = this.y + this.size / 2;
    const r = this.size * 0.6;
    return Math.hypot(px - cx, py - cy) < r;
  }

  onTap() {
    this.tapCount++;
    playBounce();
    if (this.tapCount % 3 === 0) playGiggle();
    this.state = this.tapCount % 4 === 0 ? 'spin' : 'bounce';
    this.stateTimer = 0;
    this.bounceVy = -12;
    // Cycle color every 3 taps
    if (this.tapCount % 3 === 0) {
      this.colorIndex = (this.colorIndex + 1) % this.colors.length;
      this.color = this.colors[this.colorIndex];
    }
    sparkle(this.x + this.size / 2, this.y + this.size / 2, this.color);
    burst(this.x + this.size / 2, this.y, 8, [this.color]);
    const phrases = ['Hee hee!', 'Wheee!', 'Again again!', 'Woo hoo!', ''];
    say(phrases[this.tapCount % phrases.length]);
  }

  celebrate() {
    this.state = 'spin';
    this.stateTimer = 0;
    burst(this.x + this.size / 2, this.y, 16);
    sayRandom(SQUISH_SUCCESS);
  }

  shimmer() {
    this.state = 'shimmer';
    this.stateTimer = 0;
  }
}

function lighten(hex, amount) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  const lr = Math.min(255, r + amount);
  const lg = Math.min(255, g + amount);
  const lb = Math.min(255, b + amount);
  return `#${lr.toString(16).padStart(2,'0')}${lg.toString(16).padStart(2,'0')}${lb.toString(16).padStart(2,'0')}`;
}
