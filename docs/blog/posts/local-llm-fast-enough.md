---
date:
  created: 2026-06-26
  updated: 2026-07-06
slug: local-llm-fast-enough
title: "I Couldn't Find a Local LLM Tool Fast Enough, So I Built My Own Called Squish"
subtitle: "Fast local LLM inference for Apple Silicon (M-series)."
description: "A local LLM inference server for Apple Silicon, 1.15 to 14.7× faster than Ollama depending on how much your prompts repeat, with the honest benchmarks."
authors:
  - wesley
og_author: Wesley Scholl
categories:
  - Benchmarks
tags:
  - local-llm
  - apple-silicon
  - mlx
  - rust
  - quantization
  - benchmarks
image: assets/blog/chart-speed.png
---

# I Couldn't Find a Local LLM Tool Fast Enough, So I Built My Own Called Squish

<div style="text-align:center; margin: 32px 0;">
<style>
  @keyframes squishFloat {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
  }
  .squish-mascot {
    animation: squishFloat 3.5s ease-in-out infinite;
    transition: transform 0.25s ease;
    max-width: 220px;
    width: 100%;
    height: auto;
  }
  .squish-mascot:hover {
    transform: scale(1.06);
  }
</style>
<img class="squish-mascot" src="/assets/blog/squish-hero.png" alt="Squish, the mascot">
</div>

*This requires Apple Silicon (M-series) and macOS 13 (Ventura) or later. If you're on Intel, Linux, or Windows, the numbers in this article will not apply to your hardware.*

**TL;DR:** Squish is a local LLM inference server for Apple Silicon, 1.15 to 14.7× faster than Ollama depending on how much your prompts repeat. [Skip the story and install it →](#how-you-use-it)

Since February, many of my commits have been written by a local AI in under two seconds. No API keys, no rate limits, no internet connection, and no data leaving my machine. Getting it working took considerable effort, but once it did, it's been very reliable across dozens of repositories. The local AI software I'm describing didn't exist, not without heavy modifications to source code you don't control, or building your own from scratch. So I built it, and it solved my problem. It may not solve yours, but you can clone it, fork it, take it apart, and have fun with it.

It started with Gemini. I wired the API to a script I wrote that automates git commits and pushes. It was fast and intelligent, until I hit the hard free tier rate limits. In my case, I hit them every day. Sometimes on the second commit, sometimes on the first. Gemini returned a response message saying I was rate limited. I used it once or used it twice and was cut off. This was for simple stuff, maybe 500 to 1,000 tokens for a commit, a little more for a large diff. I put up with it for a while, but the annoying limits drove me to drop it. I searched around but couldn't find anything that could solve my problem.

So I went local, off the cloud entirely, with no rate-limiting. I pointed the script at Ollama running Mistral and configured and tested it for weeks: built a custom Modelfile, iterated prompts, fine-tuned output until the commit messages described the diff accurately. The descriptions came out great. They also took too long. End to end, a response landed anywhere from seven to ten seconds on a normal commit, and north of a minute for a large diff. So I pulled every lever and turned every knob, but the adjustments failed to reduce the response time, and my problem was still unsolved. The slow and unpredictable responses were the hard wall. When the software is written by someone else, their speed limit is your speed limit. I wanted a coherent commit message in under five seconds, ideally under three. Ollama and the models it serves could not deliver. I thought about it for over a month and came to the conclusion that I had to build something myself. Something lean and elegant that doesn't just work, it beats the baseline outright. That something is called Squish, a local AI inference server.

## What Squish Is

**To be clear:** I did not build a model. I built the server that runs models, a framework that quantizes and compresses them, and a local format that reduces how much memory they need to run. Squish is an MLX-based local inference server, and five architectural components set it apart from existing tools:

<style>
  .squish-diagram-zoom { position: relative; cursor: zoom-in; }
  .squish-diagram-zoom::after {
    content: "\2922";
    position: absolute;
    top: 12px; right: 16px;
    color: rgba(255,255,255,0.45);
    font-size: 20px;
    line-height: 1;
    pointer-events: none;
  }
  .squish-diagram-zoom.squish-zoomed {
    position: fixed !important;
    top: 0 !important; left: 0 !important; right: 0 !important; bottom: 0 !important;
    width: 100vw !important; height: 100vh !important;
    margin: 0 !important;
    padding: 24px !important;
    border-radius: 0 !important;
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: zoom-out;
    background: rgba(0,0,0,0.97) !important;
    overflow: auto;
  }
  .squish-diagram-zoom.squish-zoomed::after { content: "\2715"; top: 20px; right: 24px; font-size: 22px; }
  .squish-diagram-zoom.squish-zoomed > div {
    width: 92vw !important;
    max-width: 1400px !important;
    max-height: 92vh;
    flex-shrink: 0;
  }
  .squish-diagram-zoom.squish-zoomed svg { display: block; width: 100% !important; height: auto !important; }
  body.squish-zoom-lock { overflow: hidden; }
  body.squish-zoom-lock .md-top,
  body.squish-zoom-lock .md-header,
  body.squish-zoom-lock .md-tabs { display: none !important; }
</style>
<script>
  function squishToggleZoom(el) {
    el.classList.toggle('squish-zoomed');
    document.body.classList.toggle('squish-zoom-lock', el.classList.contains('squish-zoomed'));
  }
  document.addEventListener('keydown', function (e) {
    if (e.key !== 'Escape') return;
    document.querySelectorAll('.squish-diagram-zoom.squish-zoomed').forEach(function (el) {
      el.classList.remove('squish-zoomed');
    });
    document.body.classList.remove('squish-zoom-lock');
  });
</script>

<div class="squish-diagram-zoom" onclick="squishToggleZoom(this)" style="background:#000000; padding:24px 0; border-radius:10px; margin: 24px 0;">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
  .mono{font-family:"IBM Plex Mono",ui-monospace,monospace}
  .serif{font-family:"Fraunces",Georgia,serif}
</style>
<div style="max-width:940px; margin:0 auto;">
<svg viewBox="0 0 940 600" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <pattern id="grid" width="26" height="26" patternUnits="userSpaceOnUse">
      <path d="M26 0 L0 0 0 26" fill="none" stroke="rgba(168,85,247,0.07)" stroke-width="1"/>
    </pattern>
    <marker id="aGreen" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#a855f7"/></marker>
    <marker id="aAmber" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#f59e0b"/></marker>
    <marker id="aLine" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#4c3575"/></marker>
  </defs>

  <rect x="0" y="0" width="940" height="600" fill="#000000"/>
  <rect x="0" y="0" width="940" height="600" fill="url(#grid)"/>

  <!-- header -->
  <text x="40" y="30" class="mono" font-size="10.5" letter-spacing="3" fill="#9385b0">KONJO AI  /  SQUISH INFERENCE SERVER</text>
  <text x="40" y="60" class="serif" font-size="27" font-weight="600" fill="#f8f6fc">How Squish runs on Apple Silicon</text>
  <text x="40" y="82" class="mono" font-size="12.5" fill="#9385b0">One shared pool of memory: the weights, the cache, and the GPU all live in it.</text>

  <!-- prompt entry (once per request, not animated: this is not a per-token event) -->
  <rect x="406" y="98" width="150" height="38" rx="5" fill="rgba(10,5,20,0.5)" stroke="#2d1b4e"/>
  <rect x="406" y="98" width="4" height="38" rx="2" fill="#9385b0"/>
  <text x="481" y="122" class="mono" font-size="13.5" font-weight="600" fill="#f8f6fc" text-anchor="middle">Your prompt</text>
  <path d="M481 136 L481 162" stroke="#4c3575" stroke-width="2" marker-end="url(#aLine)"/>

  <!-- hardware frame -->
  <rect x="40" y="166" width="624" height="312" rx="10" fill="rgba(10,5,20,0.3)" stroke="#2d1b4e" stroke-dasharray="4 3"/>
  <rect x="52" y="158" width="250" height="16" fill="#000000"/>
  <text x="56" y="170" class="mono" font-size="9.5" letter-spacing="2.5" fill="#9385b0">APPLE SILICON &#183; UNIFIED MEMORY</text>

  <!-- GPU box, pulses briefly when the weight read arrives (compute happening) -->
  <rect x="74" y="250" width="200" height="120" rx="6" fill="rgba(168,85,247,0.06)" stroke="rgba(168,85,247,0.5)">
    <animate attributeName="stroke-width" dur="3s" repeatCount="indefinite" values="1;1;2.6;1;1" keyTimes="0;0.33;0.42;0.5;1"/>
    <animate attributeName="fill" dur="3s" repeatCount="indefinite" values="rgba(168,85,247,0.06);rgba(168,85,247,0.06);rgba(168,85,247,0.16);rgba(168,85,247,0.06);rgba(168,85,247,0.06)" keyTimes="0;0.33;0.42;0.5;1"/>
  </rect>
  <text x="92" y="278" class="mono" font-size="11" font-weight="600" fill="#a855f7">GPU</text>
  <text x="92" y="302" class="mono" font-size="15" font-weight="600" fill="#f8f6fc">The engine</text>
  <text x="92" y="324" class="mono" font-size="11" fill="#9385b0">math on the</text>
  <text x="92" y="340" class="mono" font-size="11" fill="#9385b0">compressed weights</text>

  <!-- RAM box -->
  <rect x="360" y="190" width="270" height="252" rx="6" fill="rgba(34,211,238,0.05)" stroke="rgba(34,211,238,0.45)"/>
  <text x="376" y="214" class="mono" font-size="11" font-weight="600" fill="#22d3ee">Unified memory (RAM)</text>

  <rect x="376" y="228" width="238" height="52" rx="4" fill="rgba(10,5,20,0.55)" stroke="#2d1b4e"/>
  <text x="392" y="250" class="mono" font-size="12" fill="#d2c4ea">Quantized weights</text>
  <text x="392" y="268" class="mono" font-size="10" fill="#9385b0">small (INT3), kept warm</text>

  <!-- KV cache, flashes amber on eviction -->
  <rect x="376" y="292" width="238" height="52" rx="4" fill="rgba(10,5,20,0.55)" stroke="#2d1b4e">
    <animate attributeName="stroke" dur="9s" repeatCount="indefinite" values="#2d1b4e;#2d1b4e;#f59e0b;#2d1b4e;#2d1b4e" keyTimes="0;0.1;0.15;0.22;1"/>
  </rect>
  <text x="392" y="314" class="mono" font-size="12" fill="#d2c4ea">KV cache</text>
  <text x="392" y="332" class="mono" font-size="10" fill="#9385b0">reuses seen work</text>

  <!-- Scratch, flashes amber on eviction -->
  <rect x="376" y="356" width="238" height="52" rx="4" fill="rgba(10,5,20,0.55)" stroke="#2d1b4e">
    <animate attributeName="stroke" dur="9s" repeatCount="indefinite" values="#2d1b4e;#2d1b4e;#f59e0b;#2d1b4e;#2d1b4e" keyTimes="0;0.1;0.15;0.22;1"/>
  </rect>
  <text x="392" y="378" class="mono" font-size="12" fill="#d2c4ea">Scratch</text>
  <text x="392" y="396" class="mono" font-size="10" fill="#9385b0">working buffers</text>

  <!-- weights stream RAM -> GPU (static arrow + traveling pulse) -->
  <path id="weightPath" d="M360 254 C322 254 318 300 276 300" fill="none" stroke="#a855f7" stroke-width="2" marker-end="url(#aGreen)"/>
  <text x="174" y="212" class="mono" font-size="10.5" fill="#a855f7" text-anchor="middle">the full weight set,</text>
  <text x="174" y="227" class="mono" font-size="10.5" fill="#a855f7" text-anchor="middle">moved every token</text>
  <circle r="4.5" fill="#a855f7">
    <animateMotion dur="3s" repeatCount="indefinite" keyPoints="0;0;1;1;1" keyTimes="0;0.02;0.33;0.35;1" path="M360 254 C322 254 318 300 276 300"/>
    <animate attributeName="opacity" dur="3s" repeatCount="indefinite" values="0;1;1;0;0" keyTimes="0;0.02;0.33;0.35;1"/>
  </circle>

  <!-- prompt -> RAM entry -->
  <path d="M481 162 L481 190" stroke="#4c3575" stroke-width="2" marker-end="url(#aLine)"/>

  <!-- engine -> tokens (static arrow + traveling pulse) -->
  <path d="M174 370 L174 506" stroke="#4c3575" stroke-width="2" marker-end="url(#aLine)"/>
  <circle r="4.5" fill="#a855f7">
    <animateMotion dur="3s" repeatCount="indefinite" keyPoints="0;0;1;1;1" keyTimes="0;0.46;0.78;0.8;1" path="M174 370 L174 506"/>
    <animate attributeName="opacity" dur="3s" repeatCount="indefinite" values="0;1;1;0;0" keyTimes="0;0.46;0.78;0.8;1"/>
  </circle>
  <rect x="86" y="508" width="176" height="38" rx="5" fill="rgba(10,5,20,0.5)" stroke="#2d1b4e">
    <animate attributeName="stroke" dur="3s" repeatCount="indefinite" values="#2d1b4e;#2d1b4e;#a855f7;#2d1b4e;#2d1b4e" keyTimes="0;0.77;0.8;0.86;1"/>
  </rect>
  <rect x="86" y="508" width="4" height="38" rx="2" fill="#9385b0"/>
  <text x="174" y="532" class="mono" font-size="13.5" font-weight="600" fill="#f8f6fc" text-anchor="middle">Tokens stream back</text>

  <!-- governor side panel -->
  <rect x="700" y="190" width="200" height="252" rx="6" fill="rgba(245,158,11,0.06)" stroke="rgba(245,158,11,0.5)" stroke-dasharray="4 3"/>
  <text x="716" y="214" class="mono" font-size="9.5" letter-spacing="2" fill="#f59e0b">RUNS IN THE BACKGROUND</text>
  <text x="716" y="238" class="mono" font-size="14" font-weight="600" fill="#f8f6fc">Memory governor</text>
  <text x="716" y="268" class="mono" font-size="11" fill="#9385b0">Watches memory in tiers:</text>
  <text x="716" y="286" class="mono" font-size="11" fill="#9385b0">shrinks caches gently, then</text>
  <text x="716" y="302" class="mono" font-size="11" fill="#d2c4ea">more aggressively under</text>
  <text x="716" y="318" class="mono" font-size="11" fill="#9385b0">pressure. Rejects new work</text>
  <text x="716" y="334" class="mono" font-size="11" fill="#9385b0">only if it gets severe.</text>
  <text x="716" y="356" class="mono" font-size="11" fill="#9385b0">The weights stay resident.</text>

  <!-- watches: slow continuous breathing pulse, this is always happening -->
  <path d="M700 212 C672 212 660 205 634 205" fill="none" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="5 4" marker-end="url(#aAmber)">
    <animate attributeName="opacity" dur="2.2s" repeatCount="indefinite" values="0.35;0.9;0.35"/>
  </path>
  <text x="666" y="199" class="mono" font-size="9" fill="#f59e0b" text-anchor="middle">watches</text>

  <!-- evicts: rare, fires once every 9s, with a traveling pulse -->
  <path id="evictKV" d="M700 318 C672 318 656 318 618 318" fill="none" stroke="#f59e0b" stroke-width="1.5" marker-end="url(#aAmber)" opacity="0.85"/>
  <text x="660" y="311" class="mono" font-size="9" fill="#f59e0b" text-anchor="middle">evicts</text>
  <circle r="4" fill="#f59e0b">
    <animateMotion dur="9s" repeatCount="indefinite" keyPoints="0;0;1;1;1" keyTimes="0;0.1;0.15;0.17;1" path="M700 318 C672 318 656 318 618 318"/>
    <animate attributeName="opacity" dur="9s" repeatCount="indefinite" values="0;1;1;0;0" keyTimes="0;0.1;0.15;0.17;1"/>
  </circle>

  <path id="evictScratch" d="M700 382 C672 382 656 382 618 382" fill="none" stroke="#f59e0b" stroke-width="1.5" marker-end="url(#aAmber)" opacity="0.85"/>
  <text x="660" y="375" class="mono" font-size="9" fill="#f59e0b" text-anchor="middle">evicts</text>
  <circle r="4" fill="#f59e0b">
    <animateMotion dur="9s" repeatCount="indefinite" keyPoints="0;0;1;1;1" keyTimes="0;0.1;0.15;0.17;1" path="M700 382 C672 382 656 382 618 382"/>
    <animate attributeName="opacity" dur="9s" repeatCount="indefinite" values="0;1;1;0;0" keyTimes="0;0.1;0.15;0.17;1"/>
  </circle>

  <!-- footer -->
  <line x1="40" y1="564" x2="900" y2="564" stroke="#2d1b4e"/>
  <text x="40" y="583" class="mono" font-size="10" fill="#9385b0">PLATFORM <tspan fill="#d2c4ea">Apple Silicon</tspan>   &#183;   MEMORY <tspan fill="#d2c4ea">unified</tspan>   &#183;   DEFAULT <tspan fill="#d2c4ea">INT3, KV fp16</tspan></text>
</svg>
</div>
</div>

**1. A persistent daemon that keeps the model awake.** Ollama loads its model lazily, on the first request, and that first request pays for it: twenty to thirty seconds of cold start while the weights load, before a single token comes back. Worse, the model gets unloaded once it sits idle, so an intermittent workflow pays that toll over and over. Squish loads the model once, when the daemon starts, and keeps it resident for as long as it's running. The cost is paid a single time; every request after is warm. Load once, never load again.

**2. A two-tier KV cache that remembers.** Before a model can answer a prompt, it runs prefill. It reads the entire prompt and builds the internal attention state required in memory before producing a single token. That memory state is called the KV cache. Normally the KV cache is discarded once a response is completed, and on the next request it gets rebuilt from scratch. Squish retains the KV cache, in two layers.

- **The prompt cache** handles exact repeats, an identical prompt sent again. If that prompt is resent, there's nothing to rebuild. Time-to-first-token (TTFT) drops to roughly 4 to 11 ms, versus about 800 ms to prefill from scratch.
- **The block cache** handles prompts that partially overlap previous prompts. The cache stores KV state in fixed-size blocks on disk. A block is computed only once. Any future prompt that contains overlapping blocks reuses the stored copy. This ensures the model only computes tokens it hasn't seen before. Examples include agent loops that resend a long system prompt each turn, and multi-turn conversations.

Exact or partial prompts are processed once, never repeated.

**3. INT3 quantization that genuinely works, on some models.** A model's knowledge lives in its parameters, also called weights, the values it learned during training. Quantization stores each parameter at lower precision, like rounding a long decimal to a couple of places. This saves memory, and on Apple Silicon it also makes the model faster. The reason is simple: the slow part of producing each token is not the math, it is moving the model's weights out of memory. Smaller weights mean less to move, and less to move means more tokens per second. The tradeoff is accuracy. Three-bit quantization (INT3) is aggressive enough that it usually breaks a model outright. However, for some model families, INT3 is stable, more on that below.

<div class="squish-diagram-zoom" onclick="squishToggleZoom(this)" style="background:#000000; padding:24px 0; border-radius:10px; margin: 24px 0;">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
  .mono{font-family:"IBM Plex Mono",ui-monospace,monospace}
  .serif{font-family:"Fraunces",Georgia,serif}
</style>
<div style="max-width:900px; margin:0 auto;">
<svg viewBox="0 0 900 380" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <pattern id="grid" width="26" height="26" patternUnits="userSpaceOnUse">
      <path d="M26 0 L0 0 0 26" fill="none" stroke="rgba(168,85,247,0.07)" stroke-width="1"/>
    </pattern>
  </defs>
  <rect x="0" y="0" width="900" height="380" fill="#000000"/>
  <rect x="0" y="0" width="900" height="380" fill="url(#grid)"/>

  <text x="40" y="32" class="mono" font-size="10.5" letter-spacing="2" fill="#9385b0">SQUISH &#183; HOW INT3 QUANTIZATION WORKS</text>
  <text x="40" y="62" class="serif" font-size="21" font-weight="600" fill="#f8f6fc">Each weight is rounded to the nearest of 8 stored levels.</text>
  <text x="40" y="84" class="mono" font-size="11" fill="#9385b0">3 bits = 2&#179; = 8 possible values per group. The gap to the nearest level is the rounding error.</text>

  <!-- number line -->
  <line x1="120" y1="220" x2="780" y2="220" stroke="#4c3575" stroke-width="2"/>
  <!-- 8 ticks, evenly spaced -->
  <g id="ticks">
    <!-- x positions for 8 ticks from 130 to 770 -->
  </g>
  <g stroke="#22d3ee" stroke-width="2">
    <line x1="130" y1="212" x2="130" y2="228"/>
    <line x1="221" y1="212" x2="221" y2="228"/>
    <line x1="312" y1="212" x2="312" y2="228"/>
    <line x1="403" y1="212" x2="403" y2="228"/>
    <line x1="494" y1="212" x2="494" y2="228"/>
    <line x1="585" y1="212" x2="585" y2="228"/>
    <line x1="676" y1="212" x2="676" y2="228"/>
    <line x1="770" y1="212" x2="770" y2="228"/>
  </g>
  <text x="130" y="246" class="mono" font-size="8.5" fill="#9385b0" text-anchor="middle">-1.0</text>
  <text x="770" y="246" class="mono" font-size="8.5" fill="#9385b0" text-anchor="middle">1.0</text>
  <text x="450" y="266" class="mono" font-size="9.5" fill="#6b5b8a" text-anchor="middle">8 storable levels, evenly spaced across the group's range</text>

  <!-- four example values: original position (top, faint) animates snapping down to nearest tick -->
  <!-- value 1: 0.83 -> snaps to tick at 0.714 (x=676) -->
  <g>
    <circle cx="712" cy="150" r="5" fill="none" stroke="#f59e0b" stroke-width="1.6">
      <animate attributeName="cy" dur="4s" repeatCount="indefinite" values="150;150;220;220;150" keyTimes="0;0.15;0.4;0.85;1"/>
      <animate attributeName="cx" dur="4s" repeatCount="indefinite" values="712;712;676;676;712" keyTimes="0;0.15;0.4;0.85;1"/>
    </circle>
    <text x="712" y="138" class="mono" font-size="10" fill="#f59e0b" text-anchor="middle">0.831406
      <animate attributeName="opacity" dur="4s" repeatCount="indefinite" values="1;1;0;0;1" keyTimes="0;0.15;0.4;0.85;1"/>
    </text>
    <text x="676" y="200" class="mono" font-size="10" fill="#a855f7" text-anchor="middle">0.71
      <animate attributeName="opacity" dur="4s" repeatCount="indefinite" values="0;0;1;1;0" keyTimes="0;0.15;0.4;0.85;1"/>
    </text>
  </g>

  <!-- value 2: -0.21 -> snaps to tick at -0.143 (x=403) -->
  <g>
    <circle cx="380" cy="150" r="5" fill="none" stroke="#f59e0b" stroke-width="1.6">
      <animate attributeName="cy" dur="4s" repeatCount="indefinite" values="150;150;220;220;150" keyTimes="0;0.2;0.45;0.85;1"/>
      <animate attributeName="cx" dur="4s" repeatCount="indefinite" values="380;380;403;403;380" keyTimes="0;0.2;0.45;0.85;1"/>
    </circle>
    <text x="380" y="138" class="mono" font-size="10" fill="#f59e0b" text-anchor="middle">-0.209831
      <animate attributeName="opacity" dur="4s" repeatCount="indefinite" values="1;1;0;0;1" keyTimes="0;0.2;0.45;0.85;1"/>
    </text>
    <text x="403" y="200" class="mono" font-size="10" fill="#a855f7" text-anchor="middle">-0.14
      <animate attributeName="opacity" dur="4s" repeatCount="indefinite" values="0;0;1;1;0" keyTimes="0;0.2;0.45;0.85;1"/>
    </text>
  </g>

  <!-- value 3: 0.47 -> snaps to tick at 0.429 (x=585) -->
  <g>
    <circle cx="605" cy="150" r="5" fill="none" stroke="#f59e0b" stroke-width="1.6">
      <animate attributeName="cy" dur="4s" repeatCount="indefinite" values="150;150;220;220;150" keyTimes="0;0.25;0.5;0.85;1"/>
      <animate attributeName="cx" dur="4s" repeatCount="indefinite" values="605;605;585;585;605" keyTimes="0;0.25;0.5;0.85;1"/>
    </circle>
    <text x="605" y="138" class="mono" font-size="10" fill="#f59e0b" text-anchor="middle">0.467213
      <animate attributeName="opacity" dur="4s" repeatCount="indefinite" values="1;1;0;0;1" keyTimes="0;0.25;0.5;0.85;1"/>
    </text>
    <text x="585" y="200" class="mono" font-size="10" fill="#a855f7" text-anchor="middle">0.43
      <animate attributeName="opacity" dur="4s" repeatCount="indefinite" values="0;0;1;1;0" keyTimes="0;0.25;0.5;0.85;1"/>
    </text>
  </g>

  <!-- value 4: -0.68 -> snaps to tick at -0.714 (x=221) -->
  <g>
    <circle cx="250" cy="150" r="5" fill="none" stroke="#f59e0b" stroke-width="1.6">
      <animate attributeName="cy" dur="4s" repeatCount="indefinite" values="150;150;220;220;150" keyTimes="0;0.3;0.55;0.85;1"/>
      <animate attributeName="cx" dur="4s" repeatCount="indefinite" values="250;250;221;221;250" keyTimes="0;0.3;0.55;0.85;1"/>
    </circle>
    <text x="250" y="138" class="mono" font-size="10" fill="#f59e0b" text-anchor="middle">-0.681904
      <animate attributeName="opacity" dur="4s" repeatCount="indefinite" values="1;1;0;0;1" keyTimes="0;0.3;0.55;0.85;1"/>
    </text>
    <text x="221" y="200" class="mono" font-size="10" fill="#a855f7" text-anchor="middle">-0.71
      <animate attributeName="opacity" dur="4s" repeatCount="indefinite" values="0;0;1;1;0" keyTimes="0;0.3;0.55;0.85;1"/>
    </text>
  </g>

  <!-- before / after block size comparison -->
  <text x="150" y="308" class="mono" font-size="10" fill="#9385b0" text-anchor="middle">BEFORE &#183; full precision</text>
  <rect x="70" y="316" width="160" height="34" rx="4" fill="rgba(245,158,11,0.08)" stroke="#f59e0b" stroke-width="1.4"/>
  <text x="150" y="337" class="mono" font-size="9" fill="#f59e0b" text-anchor="middle">16 bits per weight</text>

  <text x="750" y="308" class="mono" font-size="10" fill="#9385b0" text-anchor="middle">AFTER &#183; INT3 stored</text>
  <rect x="705" y="316" width="90" height="34" rx="4" fill="rgba(168,85,247,0.1)" stroke="#a855f7" stroke-width="1.4"/>
  <text x="750" y="337" class="mono" font-size="9" fill="#a855f7" text-anchor="middle">3 bits per weight</text>

  <path d="M234 333 L701 333" stroke="#6b5b8a" stroke-width="1.2" stroke-dasharray="3 3" fill="none" marker-end="url(#am)"/>
  <defs>
    <marker id="am" markerWidth="8" markerHeight="8" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#6b5b8a"/></marker>
  </defs>
</svg>
</div>
</div>

**4. The optimization engine: where the speed and memory savings come from.** To generate each new token, the GPU works through every weight in the model. Most tools expand the rounded-off weights back to full precision in memory before the GPU can use them. Squish never expands the full model to full-size in memory, saving a significant amount of memory. To achieve this, the quantized model weights are streamed from memory in small blocks by the GPU. During this process, each quantized block is expanded to full-size, then processed, and the next block overwrites the previous block sequentially. Only a few blocks are expanded to full-size inside the GPU at a time. Squish runs the same computation as full precision, without expanding the model weights to full-size in memory. With this memory savings, fewer bytes move through memory, improving speed at the same time.

<div class="squish-diagram-zoom" onclick="squishToggleZoom(this)" style="background:#000000; padding:24px 0; border-radius:10px; margin: 24px 0;">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
  .mono{font-family:"IBM Plex Mono",ui-monospace,monospace}
  .serif{font-family:"Fraunces",Georgia,serif}
</style>
<div style="max-width:900px; margin:0 auto;">
<svg viewBox="0 0 900 360" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <pattern id="grid" width="26" height="26" patternUnits="userSpaceOnUse">
      <path d="M26 0 L0 0 0 26" fill="none" stroke="rgba(168,85,247,0.07)" stroke-width="1"/>
    </pattern>
    <marker id="aGreen" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#a855f7"/></marker>
    <marker id="aBlue" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#22d3ee"/></marker>
  </defs>

  <rect x="0" y="0" width="900" height="360" fill="#000000"/>
  <rect x="0" y="0" width="900" height="360" fill="url(#grid)"/>

  <!-- header -->
  <text x="40" y="32" class="mono" font-size="10.5" letter-spacing="2" fill="#9385b0">SQUISH &#183; BLOCK-NATIVE FUSED MULTIPLY ENGINE</text>
  <text x="40" y="62" class="serif" font-size="21" font-weight="600" fill="#f8f6fc">Each quantized block is expanded and processed by the GPU.</text>
  <text x="40" y="84" class="mono" font-size="11" fill="#9385b0">The next block overwrites the current block in place, so no additional memory is required.</text>

  <!-- MEMORY box -->
  <rect x="40" y="150" width="220" height="170" rx="8" fill="rgba(34,211,238,0.05)" stroke="rgba(34,211,238,0.45)"/>
  <text x="150" y="138" class="mono" font-size="10" letter-spacing="1" fill="#22d3ee" text-anchor="middle">MEMORY &#183; quantized model weights</text>
  <g fill="rgba(34,211,238,0.18)" stroke="#4c3575" stroke-width="1.4">
    <rect x="60"  y="170" width="26" height="22" rx="3"/><rect x="98"  y="170" width="26" height="22" rx="3"/><rect x="136" y="170" width="26" height="22" rx="3"/><rect x="174" y="170" width="26" height="22" rx="3"/><rect x="212" y="170" width="26" height="22" rx="3"/>
    <rect x="60"  y="204" width="26" height="22" rx="3"/><rect x="98"  y="204" width="26" height="22" rx="3"/><rect x="136" y="204" width="26" height="22" rx="3"/><rect x="174" y="204" width="26" height="22" rx="3"/><rect x="212" y="204" width="26" height="22" rx="3"/>
    <rect x="60"  y="238" width="26" height="22" rx="3"/><rect x="98"  y="238" width="26" height="22" rx="3"/><rect x="136" y="238" width="26" height="22" rx="3"/><rect x="174" y="238" width="26" height="22" rx="3"/><rect x="212" y="238" width="26" height="22" rx="3"/>
    <rect x="60"  y="272" width="26" height="22" rx="3"/><rect x="98"  y="272" width="26" height="22" rx="3"/><rect x="136" y="272" width="26" height="22" rx="3"/><rect x="174" y="272" width="26" height="22" rx="3"/><rect x="212" y="272" width="26" height="22" rx="3"/>
  </g>

  <!-- fetch / return arrows (static, general mechanism) -->
  <text x="410" y="186" class="mono" font-size="10" fill="#a855f7" text-anchor="middle">GPU fetches each block</text>
  <path d="M560 200 L260 200" fill="none" stroke="#a855f7" stroke-width="2" marker-end="url(#aGreen)"/>
  <path d="M260 224 L560 224" fill="none" stroke="#22d3ee" stroke-width="2" marker-end="url(#aBlue)"/>
  <text x="410" y="246" class="mono" font-size="10" fill="#22d3ee" text-anchor="middle">blocks expand for processing</text>

  <!-- GPU panel -->
  <rect x="560" y="150" width="300" height="200" rx="8" fill="rgba(168,85,247,0.05)" stroke="rgba(168,85,247,0.5)"/>
  <text x="710" y="130" class="mono" font-size="10" letter-spacing="1.5" fill="#a855f7" text-anchor="middle">GPU</text>
  <text x="710" y="144" class="mono" font-size="9.5" fill="#9385b0" text-anchor="middle">processes each expanded block sequentially</text>

  <!-- fixed slot outline: the one physical location every block lands in -->
  <rect x="662" y="208" width="96" height="96" rx="6" fill="none" stroke="#5b3a8a" stroke-width="1.4" stroke-dasharray="4 4"/>

  <!-- three blocks, cycling: green(1) -> yellow(2) -> magenta(3), each held ~2.4s -->
  <!-- shared path: memory edge -> GPU edge (riding the return arrow) -> slot center -->
  <!-- loop = 10.5s total, repeatCount indefinite, shared across all animated elements -->

  <!-- BLOCK 1 . green . window 0.000 - 0.3333 -->
  <g>
    <animateMotion dur="10.5s" repeatCount="indefinite"
      path="M260,224 L560,224 L710,256"
      keyPoints="0;0;1;1" keyTimes="0;0;0.0571;1"/>
    <animateTransform attributeName="transform" type="scale" additive="sum" dur="10.5s" repeatCount="indefinite"
      values="0.55;0.55;3.2;3.2" keyTimes="0;0;0.0571;1"/>
    <animate attributeName="opacity" dur="10.5s" repeatCount="indefinite"
      values="1;1;0;0" keyTimes="0;0.3904;0.4047;1"/>
    <rect x="-15" y="-15" width="30" height="30" rx="4" fill="rgba(168,85,247,0.85)" stroke="#a855f7" stroke-width="2"/>
    <text x="0" y="5" class="mono" font-size="10" font-weight="700" fill="#000000" text-anchor="middle">1</text>
  </g>

  <!-- BLOCK 2 . yellow . window 0.3333 - 0.6667 -->
  <g>
    <animateMotion dur="10.5s" repeatCount="indefinite"
      path="M260,224 L560,224 L710,256"
      keyPoints="0;0;0;1;1" keyTimes="0;0;0.3333;0.3904;1"/>
    <animateTransform attributeName="transform" type="scale" additive="sum" dur="10.5s" repeatCount="indefinite"
      values="0.55;0.55;0.55;3.2;3.2" keyTimes="0;0;0.3333;0.3904;1"/>
    <animate attributeName="opacity" dur="10.5s" repeatCount="indefinite"
      values="0;0;1;1;0;0" keyTimes="0;0.3333;0.3333;0.7238;0.7381;1"/>
    <rect x="-15" y="-15" width="30" height="30" rx="4" fill="rgba(251,191,36,0.85)" stroke="#fbbf24" stroke-width="2"/>
    <text x="0" y="5" class="mono" font-size="10" font-weight="700" fill="#000000" text-anchor="middle">2</text>
  </g>

  <!-- BLOCK 3 . magenta . window 0.6667 - 1.0 -->
  <g>
    <animateMotion dur="10.5s" repeatCount="indefinite"
      path="M260,224 L560,224 L710,256"
      keyPoints="1;1;0;0;1;1" keyTimes="0;0.0714;0.0714;0.6667;0.7238;1"/>
    <animateTransform attributeName="transform" type="scale" additive="sum" dur="10.5s" repeatCount="indefinite"
      values="3.2;3.2;0.55;0.55;3.2;3.2" keyTimes="0;0.0714;0.0714;0.6667;0.7238;1"/>
    <animate attributeName="opacity" dur="10.5s" repeatCount="indefinite"
      values="1;1;0;0;1;1" keyTimes="0;0.0571;0.0714;0.6667;0.6667;1"/>
    <rect x="-15" y="-15" width="30" height="30" rx="4" fill="rgba(236,72,153,0.85)" stroke="#ec4899" stroke-width="2"/>
    <text x="0" y="5" class="mono" font-size="10" font-weight="700" fill="#000000" text-anchor="middle">3</text>
  </g>

  <!-- static label: what the slot is doing -->
  <text x="710" y="322" class="mono" font-size="10" fill="#d2c4ea" text-anchor="middle">processing current block</text>

  <!-- flashing label: fires at each swap, held longer, no longer just a blip -->
  <text x="710" y="340" class="mono" font-size="10.5" fill="#f59e0b" text-anchor="middle">overwritten
    <animate attributeName="opacity" dur="10.5s" repeatCount="indefinite"
      values="1;1;0;0;1;1;0;0;1;1;0;0;1;1"
      keyTimes="0;0.0952;0.0952;0.3523;0.3523;0.4285;0.4285;0.6857;0.6857;0.7619;0.7619;0.9619;0.9619;1"/>
  </text>
</svg>
</div>
</div>

**5. A memory governor that reacts to real pressure.** On a 16 GB Mac, everything shares the same memory, including the model, long conversation contexts, macOS, and every other running application. If all of these exceed 16 GB, the memory spills into disk space, which is slow and can cause thrashing, or macOS may kill the process outright. Squish watches memory-pressure signals from macOS. As memory pressure rises, the governor reduces the KV cache's size, first gently, then more aggressively if pressure keeps climbing. Under memory pressure, new requests are also given a smaller context window. This prevents allocating space the machine does not have. If memory pressure is critical, Squish rejects new requests with a clear response instead of crashing. Any requests already generating remain to finish processing. Once memory pressure reduces to a normal level, the original KV cache's size and prompt context window limits are restored.

## The Benchmarking Methodology

Before I present any numbers, I'll share my benchmarking protocol. Plenty of "Ollama vs X" articles contain at least one apples-to-oranges measurement that favors the tool the author is selling. The most overlooked one is thermal. Run the favored tool first on a cool machine, then run the competing tool second once the machine is hot. This manufactures a win out of nothing but execution order. So I controlled for it. Each inference server was measured from the same 50°C baseline. A consistency check confirmed the first and last runs were taken at the same chip temperature and held to within 1.7%, so performance didn't degrade as things heated up. The temperature of the chip's silicon (its die) was logged live throughout the benchmark tests. These numbers reflect each inference server, not the benchmark order.

**Hardware:** Apple M3 MacBook Pro, 16 GB unified memory, running macOS 26 Tahoe, the current OS version. Normal hardware, controlled procedure: no external memory, SSD, or compute, and no fresh reboot to game the result.

**Models:** Qwen2.5-7B-Instruct for both, Q4_K_M on Ollama and INT4/INT3 on Squish. The two models are comparable in parameter count and quantization level. Ollama shipped a major version jump partway through this project, so I ran the full suite against both 0.18.2 and 0.30.7. They came out identical, matching to a tenth of a token per second at short and medium context, so the comparison below isn't pinned to a single convenient version. The numbers that follow use 0.30.7, the current release.

**Protocol:** I ran five runs per metric, reported the median result value, and used identical prompts and lengths for both Ollama and Squish. The benchmarks included prompt sizes of 75, 2000, and 4000 tokens. 75-token prompts are what most benchmarks publish. Coding agents and document Q&A workloads are typically around 4000 tokens. The raw per-run JSON results are committed in the repo, and every number can be reproduced with an M3.

## The Honest Benchmark Comparison

With that protocol settled, here is how the two inference servers compare (apples-to-apples). Every number is reproducible from the repo via `benchmarks/ollama_vs_squish/bench_thermal_h2h.py`.

| Metric | Ollama 0.30.7 | Squish |
|---|---:|---:|
| Cold start: load + first token (1.5B) | 20–30 s | ≈0.5 s (54×) |
| Full response @ 4000-token prompt (repeated exactly)* | 37.5 s | 3.8 s (9.8×) |
| Decode throughput @ 75 tokens | 20.3 tok/s | 24.0 tok/s (INT3) |
| Decode throughput @ 2000 tokens | 19.7 tok/s | 22.6 tok/s (INT3) |
| Inter-token tail p95 @ 75 tokens | 52.4 ms | 42.7 ms (INT3) |
| Inter-token tail p95 @ 2000 tokens | 52.9 ms | 45.4 ms (INT3) |
| Repeat-prompt TTFT (KV cache hit) | ~160 ms | 4–11 ms |
| Peak RAM during inference | 5.14 GB | 3.50 GB |
| Disk: 7B INT4 / INT3 | 4.36 GB / n/a | 4.00 / 3.56 GB |

*See "How Prompt Reuse Scales" below, this is the ceiling, not the typical case.*

*Ollama 0.18.2 was tested identically. Its decode throughput and latency matched 0.30.7 to within noise.*

Squish leads on every metric in this table: decode throughput, tail latency, peak memory, and full end-to-end response time. An earlier version of this comparison also reported an Ollama win on short-prompt TTFT, 167 ms vs 192 ms, using the same fixed prompt sent five times. That number didn't hold up under scrutiny: it was measuring Ollama's own cache hit on the repeat sends, not a genuinely cold comparison. Under real cold, unique-prompt conditions, covered in the next section, Squish leads TTFT too, at every length tested, down to the shortest prompt measured.

Let's say you have an agent resending the system prompt every turn. For a 4,000-token prompt, Ollama takes around 37.5 seconds to return a full response. Squish's full response returns in 3.8 seconds flat. It's 9.8× faster, and it's the difference between a tool that responds in a timely manner and one where users stare at the screen wondering why it hasn't responded yet.

## How Prompt Reuse Scales

The 9.8× number above assumes the prompt repeats exactly. What if the prompt is completely unique with nothing to reuse at all? I benchmarked Qwen2.5-7B-Instruct using Squish INT4 vs Ollama Q4_K_M. Tested five context lengths and verified 0% reuse for every single request.

| Context | Ollama TTFT | Squish TTFT | Ollama decode | Squish decode | Ollama E2E | Squish E2E | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| 75 | 812 ms | 800 ms | 16.4 tok/s | 17.5 tok/s | 14.12 s | 12.30 s | 1.15× |
| 512 | 3.90 s | 3.33 s | 16.7 tok/s | 19.1 tok/s | 16.27 s | 13.74 s | 1.18× |
| 1024 | 10.01 s | 8.59 s | 10.8 tok/s | 14.8 tok/s | 28.93 s | 21.98 s | 1.32× |
| 2048 | 19.79 s | 17.31 s | 10.5 tok/s | 14.2 tok/s | 38.84 s | 32.24 s | 1.20× |
| 4096 | 41.50 s | 36.14 s | 9.4 tok/s | 11.9 tok/s | 62.80 s | 52.93 s | 1.19× |

*Squish's 4096-token TTFT ranged from 25.4 to 43.6 seconds across its five measured runs, wider than the other rows, though each run independently confirmed 0% cache hit. This row should be treated as the least precise of the five. The 75-token result is the closest race of the set, Squish's TTFT edge there is just 12 ms, about 1.5%, still a real win but a thin one.*

Even with 0% prompt reuse, Squish wins TTFT, decode, and end-to-end response time at every context size tested, down to the shortest prompt measured. The floor is 1.15 to 1.32× faster than Ollama.

The 9.8× ceiling resulted from the same 200-token, thermally-controlled benchmark as the floor above, so the numbers are directly comparable. The floor tests completely unique prompts; the ceiling resends an identical prompt every time: **1.15 to 1.32× on unique prompts, up to 9.8× on exact repeats.** The benchmark below measures the prompt reuse effect in isolation and reaches 14.7× with 95% overlapping prompts.

I also wanted to see how speedup scales as prompt reuse increases from 5% to 95%. The second benchmark isolates and tests the reuse effect at four context lengths and five overlap percentage levels. Each result was identically validated byte-for-byte against a cold run to confirm reuse output is the exact same.

| Overlap | 512 tok | 1024 tok | 2048 tok | 4096 tok |
|---|---:|---:|---:|---:|
| 5% | 1.15× | 1.07× | 1.06× | 1.07× |
| 25% | 1.32× | 1.30× | 1.34× | 1.30× |
| 50% | 1.55× | 1.84× | 1.97× | 1.91× |
| 75% | 2.75× | 3.23× | 3.51× | 3.64× |
| 95% | 5.86× | 6.95× | 11.0× | 14.7× |

*The results above reflect three runs, lighter than the 5-run protocol above. These five overlap levels are quartiles pulled from a finer sweep measured at every 5% interval between 5% and 95%. The full 19-point sweep isn't committed to the repo, reproduce it with `python -m benchmarks.prefix_reuse_curve --contexts 512,1024,2048,4096`.*

This table's results are higher than the floor-to-ceiling range above. An 8-token generation is almost entirely prefill and the ratio isolates the prompt reuse effect at full strength. The 9.8× ceiling includes a full 200-token response including decode time. Reuse can't speed up decode, reducing the ratio. Both measure the same mechanism, one in isolation, and the other inside a real response. Speedup is modest at low overlap percentages and improves significantly at 50% and above. At 95% overlap, a 512-token prompt sees 5.9× while a 4096-token prompt sees 14.7×. At 75% and 95% overlap, the longer the context the faster the response, though the pattern is less clean right at 50%.

Agent loops and multi-turn conversations benefit heavily from increasing context with overlapping prompts. However, reuse only helps when there's previously processed content to reuse. The other lever is making the weights themselves smaller, and that's where things get interesting.

*Ollama version 0.31.1 was released five days after the benchmark above was measured. A spot-check using 0.31.1 on the cold/unique floor reproduced the exact same 1.32× result at 1024 tokens. The reuse-ceiling spot-check initially failed its own drift check due to unrelated machine load; after clearing it, a clean re-run landed at 8.8×, within noise of the published 9.8×.*

## The Most Interesting Thing I Found

I wanted local models to run faster and use less memory. I started with INT4 quantization, which is well studied and a stable standard. But I wanted to know exactly how small the models would compress, so I tried quantizing using INT3. Most models broke. Qwen2.5-7B didn't. The INT3 quantized model remained stable, and tied INT4 on the arc_easy reasoning benchmark. The scores were 0.551 to 0.541, INT3 slightly ahead but within the margin of error. The INT3 model decodes roughly 18% faster and compresses the model from 4.00 GB to 3.56 GB on disk. The other Qwen models I tested remained stable using INT3, within roughly 1% of the original model's performance. I also quantized Gemma-3 using INT3, and it collapsed, a fifteen-point drop in accuracy.

<div class="squish-diagram-zoom" onclick="squishToggleZoom(this)" style="background:#000000; padding:24px 0; border-radius:10px; margin: 24px 0;">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
  .mono{font-family:"IBM Plex Mono",ui-monospace,monospace}
  .serif{font-family:"Fraunces",Georgia,serif}
</style>
<div style="max-width:900px; margin:0 auto;">
<svg viewBox="0 0 900 420" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <pattern id="grid" width="26" height="26" patternUnits="userSpaceOnUse">
      <path d="M26 0 L0 0 0 26" fill="none" stroke="rgba(168,85,247,0.07)" stroke-width="1"/>
    </pattern>
  </defs>
  <rect x="0" y="0" width="900" height="420" fill="#000000"/>
  <rect x="0" y="0" width="900" height="420" fill="url(#grid)"/>

  <text x="40" y="32" class="mono" font-size="10.5" letter-spacing="2" fill="#9385b0">SQUISH &#183; THE ACCURACY TRADEOFF</text>
  <text x="40" y="62" class="serif" font-size="21" font-weight="600" fill="#f8f6fc">Smaller does not mean worse, where INT3 ships at all.</text>
  <text x="40" y="84" class="mono" font-size="11" fill="#9385b0">arc_easy accuracy, Qwen2.5-1.5B. INT3 and INT4 are statistically tied.</text>

  <!-- bar chart, axis 50% to 58% (tight range so the real ~1pp gap is visible and honest) -->
  <line x1="120" y1="300" x2="500" y2="300" stroke="#4c3575" stroke-width="1.6"/>
  <text x="110" y="304" class="mono" font-size="8.5" fill="#6b5b8a" text-anchor="end">50%</text>
  <line x1="120" y1="140" x2="500" y2="140" stroke="#4c3575" stroke-width="1.6"/>
  <text x="110" y="144" class="mono" font-size="8.5" fill="#6b5b8a" text-anchor="end">58%</text>
  <line x1="120" y1="220" x2="500" y2="220" stroke="#2d1b4e" stroke-width="1" stroke-dasharray="2 3"/>
  <text x="110" y="224" class="mono" font-size="8.5" fill="#6b5b8a" text-anchor="end">54%</text>

  <!-- INT4 bar: 54.1%, scale 20px per point from the 50% baseline at y=300 -->
  <text x="200" y="330" class="mono" font-size="10.5" fill="#22d3ee" text-anchor="middle">INT4</text>
  <rect x="160" y="300" width="80" height="0" fill="rgba(34,211,238,0.25)" stroke="#22d3ee" stroke-width="1.8">
    <animate attributeName="height" dur="3.5s" repeatCount="indefinite" values="0;82;82;0" keyTimes="0;0.3;0.85;1"/>
    <animate attributeName="y" dur="3.5s" repeatCount="indefinite" values="300;218;218;300" keyTimes="0;0.3;0.85;1"/>
  </rect>
  <text x="200" y="206" class="mono" font-size="11" fill="#22d3ee" text-anchor="middle">
    <animate attributeName="opacity" dur="3.5s" repeatCount="indefinite" values="0;1;1;0" keyTimes="0;0.32;0.85;1"/>
    54.1%
  </text>

  <!-- INT3 bar: 55.1%, one percentage point higher -->
  <text x="380" y="330" class="mono" font-size="10.5" fill="#a855f7" text-anchor="middle">INT3</text>
  <rect x="340" y="300" width="80" height="0" fill="rgba(168,85,247,0.25)" stroke="#a855f7" stroke-width="1.8">
    <animate attributeName="height" dur="3.5s" repeatCount="indefinite" values="0;102;102;0" keyTimes="0;0.3;0.85;1"/>
    <animate attributeName="y" dur="3.5s" repeatCount="indefinite" values="300;198;198;300" keyTimes="0;0.3;0.85;1"/>
  </rect>
  <text x="380" y="186" class="mono" font-size="11" fill="#a855f7" text-anchor="middle">
    <animate attributeName="opacity" dur="3.5s" repeatCount="indefinite" values="0;1;1;0" keyTimes="0;0.32;0.85;1"/>
    55.1%
  </text>

  <!-- tied bracket -->
  <path d="M200 172 C200 160 380 160 380 174" fill="none" stroke="#9385b0" stroke-width="1.2"/>
  <text x="290" y="152" class="mono" font-size="9.5" fill="#9385b0" text-anchor="middle">tied, n=1000</text>

  <!-- right side: the gate -->
  <rect x="580" y="140" width="280" height="170" rx="8" fill="rgba(245,158,11,0.06)" stroke="rgba(245,158,11,0.4)"/>
  <text x="720" y="164" class="mono" font-size="10" letter-spacing="1" fill="#f59e0b" text-anchor="middle">THE CI GATE</text>
  <text x="600" y="190" class="mono" font-size="10.5" fill="#d2c4ea">If INT3 costs a model family too</text>
  <text x="600" y="206" class="mono" font-size="10.5" fill="#d2c4ea">much accuracy, it is not shipped</text>
  <text x="600" y="222" class="mono" font-size="10.5" fill="#d2c4ea">for that family. No exceptions.</text>
  <line x1="600" y1="240" x2="840" y2="240" stroke="#4c3575" stroke-dasharray="2 3"/>
  <text x="600" y="262" class="mono" font-size="10" fill="#9385b0">example: gemma-3, &#8804;4B</text>
  <text x="600" y="280" class="mono" font-size="10" fill="#fb923c">-15pp accuracy &#8594; blocked</text>
  <text x="600" y="298" class="mono" font-size="9" fill="#6b5b8a">never shipped, not degraded</text>

  <line x1="40" y1="352" x2="860" y2="352" stroke="#2d1b4e"/>
  <text x="40" y="374" class="mono" font-size="9" fill="#6b5b8a">SOURCE  CI accuracy gates, lm_eval arc_easy. Qwen2.5-1.5B INT4 54.1% / INT3 55.1% (n=1000), shipped. gemma-3 family INT3 -15pp, blocked at CI, never shipped.</text>
</svg>
</div>
</div>

The Qwen models I tested could handle INT3, in some cases marginally beating INT4 within the margin of error. With that finding in hand, I wanted to find the actual floor, so I moved on to INT2. Qwen broke, and for good reason. The model responded to my prompt with gibberish: `IFYINGIFYIN`, completely incoherent, basically random. When a 16-bit model is crushed down to 2 bits, its intelligence is lost. Aggressive quantization has a hard floor, and eventually every model breaks. That breaking point is different for every model and every model family. That's why Squish has a quantization safety mechanism that blocks INT3 for the families it fails on and INT2 altogether. The result is a tool that compresses every model as far as it can safely go, and no further.

## How You Use It

<div style="text-align:center; margin: 32px 0;">
<style>
  @keyframes squishFloat {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
  }
  .squish-mascot {
    animation: squishFloat 3.5s ease-in-out infinite;
    transition: transform 0.25s ease;
    max-width: 320px;
    width: 100%;
    height: auto;
  }
  .squish-mascot:hover {
    transform: scale(1.06);
  }
</style>
<img class="squish-mascot" src="/assets/blog/squish-fistpump.png" alt="Squish, celebrating">
</div>

There are many ways to use Squish, but at its most basic level it serves an OpenAI API on a local port (11435). I built it as a drop-in replacement for Ollama and for any application that speaks the OpenAI API endpoint spec.

Installing Squish via Homebrew is recommended. Nothing compiles, and every dependency comes bundled:

```bash
brew tap konjoai/squish
brew install squish
```

Or through Python:

```bash
pip install squish-ai    # Python 3.11–3.14 required - OR
pipx install squish-ai --python python3.13    # Isolated in its own environment
```

With Squish installed, start a model with `squish run`. Specifying a model is optional. With no argument, Squish pulls a default sized to your machine's RAM, so you get the largest model that comfortably fits.

```bash
squish run    # or specify a model, e.g. squish run qwen2.5:7b
```

Behind the scenes on that first run, Squish pulls a pre-squished model from the squishai Hugging Face community, the conversion and accuracy-gating already done, then loads and serves it. Once the model is loaded and ready, the web UI opens automatically.

A caveat before you go looking for models: Squish doesn't run every model, only those in MLX format (fp16 or bf16 weights). More on that in the next section.

With the server running, point any OpenAI client at it. Set your base URL to the local endpoint and use `squish` as the API key:

```bash
export OPENAI_BASE_URL=http://localhost:11435/v1
export OPENAI_API_KEY=squish
```

Ollama clients work too: point `OLLAMA_HOST` at the same port and they won't know the difference.

From there, there are several ways to work with the running model:

- **The web UI**, a chat client backed by a live instrument panel: the KV cache filling and reusing in real time, a quantization comparator, a latency waterfall splitting prefill from decode, and Apple Silicon power telemetry.
- **The VS Code extension**, the same chat plus agentic features and repo access, right in your editor.
- **The macOS app**, in two parts: a menu bar showing the Squish logo and live tokens per second, and a chat window for talking to the loaded model.
- **The command line**, POSTing directly to the OpenAI API endpoint.

Here are the basic Squish CLI commands:

```bash
squish run        # start the server (auto-picks a model for your RAM)
squish pull       # download and compress a model for your machine
squish doctor     # check your environment is set up correctly
squish catalog    # browse available models
```

To keep the server always available, `squish daemon install` registers it to start at login. For the full command list, see the repo.

## What Squish Doesn't Do

Squish only runs on Apple Silicon (M-series). There is no support for CUDA, Intel, Windows, or Linux. These platforms, chips, and kernels are not on the roadmap. Squish's speed comes from the Mac's Metal and unified memory. If you're not on an M-series Mac, Squish will not work for you.

Squish doesn't run every model on Hugging Face. It works with models already in MLX format, the fp16 or bf16 weights the `mlx-community` organization publishes, which is the layout Apple's MLX framework expects. If a model is already in that format, you can point Squish's convert command at it, and it quantizes and packs the weights into Squish's own format. What you can't do is hand it an arbitrary checkpoint in some other layout and expect it to work.

## What I Use It For Now

When I first started building Squish, my goal was to get my commit messages written quickly and without rate limits. I accomplished that goal and then some. My commit and pull request scripts still run against Squish. However, my agents now write many of those commits and PRs themselves. Now my local Squish workflow has grown to include:

- Private and local chat, every prompt and response stays on the machine.
- Local code review and codebase Q&A, the Squish agent reads my repo without a third party involved.
- A drop-in OpenAI endpoint for testing other applications against a real model, no API bill.
- Model benchmarking and inference metrics: TTFT, end-to-end latency, RAM/GPU utilization live in the web dashboard and VS Code extension.

Squish solves my problems, and it may or may not solve yours. But if you have more than 16 GB RAM, Squish can run larger models than I've been able to while using significantly less RAM and hard disk space. If you want to modify it, the source is available and every benchmark number is reproducible from the repo.

I built Squish because I was tired of waiting. It keeps the model loaded, the cache warm, and answers before I've moved my hands. What I haven't seen is what it can do beyond my one machine.

## Join the Squish Community

<div style="text-align:center; margin: 32px 0;">
<style>
  @keyframes squishFloat {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
  }
  .squish-mascot {
    animation: squishFloat 3.5s ease-in-out infinite;
    transition: transform 0.25s ease;
    max-width: 320px;
    width: 100%;
    height: auto;
  }
  .squish-mascot:hover {
    transform: scale(1.06);
  }
</style>
<img class="squish-mascot" src="/assets/blog/squish-flying.png" alt="Squish, taking off">
</div>

I've run and tested Squish on one machine, an M3 MacBook Pro with 16 GB of RAM. Every number in this article reflects that same laptop. I have no idea what Squish looks like on a Mac Studio with 128 GB, or what happens when you point it at a 70B model instead of a 7B one, and I'd like to find out.

If you have access to a Mac with 32, 64, or 128+ GB of unified memory, running Squish against larger models and publishing what you find would answer questions I can't answer myself:

- How does the memory governor behave when there's real headroom to work with?
- Does the reuse curve hold the same shape for larger models?
- Where is the actual ceiling beyond the 16 GB RAM constraint?

A few other ways to help, beyond high RAM benchmarking:

- **Try to break the numbers in this article.** Every benchmark here is reproducible from the repo. If you can find a configuration where the claims don't hold, that's a more valuable contribution than a benchmark that confirms what's already published.
- **Test model families I haven't.** The INT3 accuracy gate currently covers a handful of Qwen models and Gemma-3. There are a lot of other model families out there I simply haven't had time to run through it.
- **File real bugs from actual use.** Edge cases from actual daily workflows surface things a benchmark suite never will.
- **Contribute to the model catalog.** If there's an MLX-format model on Hugging Face you think belongs in Squish's catalog, open an issue or a PR.
- **Improve the docs.** If something in this article or the repo's docs confused you, it'll confuse someone else too.

Open an issue or a PR on GitHub, whichever of these you want to take on. I built Squish alone, but I'd rather not be the only one who knows what it can do. Break something and have fun.

<div style="text-align:center; margin: 32px 0;">
<style>
  @keyframes squishFloat {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
  }
  .squish-mascot {
    animation: squishFloat 3.5s ease-in-out infinite;
    transition: transform 0.25s ease;
    max-width: 320px;
    width: 100%;
    height: auto;
  }
  .squish-mascot:hover {
    transform: scale(1.06);
  }
</style>
<img class="squish-mascot" src="/assets/blog/squish-building.png" alt="Squish, building Squish">
</div>

---

Requests for models to add to the squishai Hugging Face community, or trouble installing or running Squish? Open an issue on GitHub.
