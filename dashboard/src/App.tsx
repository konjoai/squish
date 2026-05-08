import { useEffect, useRef, useState } from "react";
import { KonjoApp } from "@konjoai/ui";
import { ChatPanel } from "./views/ChatPanel";
import { PromptBar } from "./views/PromptBar";
import { ThroughputCard } from "./views/ThroughputCard";
import { KVCacheView } from "./views/KVCacheView";
import { QuantComparator } from "./views/QuantComparator";
import { LatencyWaterfall } from "./views/LatencyWaterfall";
import { ThermalDial } from "./views/ThermalDial";
import { ModelInfo } from "./views/ModelInfo";
import { MetaInspector } from "./views/MetaInspector";
import {
  chatStream, fetchHealth, fetchMetrics, summarizeMetricsText,
} from "./lib/api";
import { MOCK_HEALTH, MOCK_PROM_TEXT } from "./lib/mock";
import type {
  ChatTurn, HealthResponse, CockpitMetrics, KVMode,
} from "./lib/types";

const EMPTY_METRICS: CockpitMetrics = summarizeMetricsText(MOCK_PROM_TEXT);

function inferKVMode(loader: string): KVMode {
  const l = loader.toLowerCase();
  if (l.includes("int2")) return "int2";
  if (l.includes("int3")) return "int3";
  if (l.includes("int4")) return "int4";
  if (l.includes("int8") || l.includes("kivi") || l.includes("snap")) return "int8";
  if (l.includes("fp16")) return "fp16";
  return "unknown";
}

export default function App() {
  const [prompt, setPrompt] = useState<string>("Explain why an INT4 KV cache barely loses quality.");

  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [active, setActive] = useState<ChatTurn | null>(null);
  const [streaming, setStreaming] = useState<boolean>(false);
  const [chatFromMock, setChatFromMock] = useState<boolean>(false);

  const [health, setHealth] = useState<HealthResponse>(MOCK_HEALTH);
  const [metrics, setMetrics] = useState<CockpitMetrics>(EMPTY_METRICS);
  const [healthFromMock, setHealthFromMock] = useState<boolean>(true);
  const [metricsFromMock, setMetricsFromMock] = useState<boolean>(true);
  const [benchFromMock, setBenchFromMock] = useState<boolean>(true);

  const [liveTps, setLiveTps] = useState<number | undefined>();

  const cancelRef = useRef<(() => void) | null>(null);

  // Poll /health and /v1/metrics every 5 seconds.
  useEffect(() => {
    let cancelled = false;
    const refresh = async () => {
      const [h, m] = await Promise.all([fetchHealth(), fetchMetrics()]);
      if (cancelled) return;
      setHealth(h.data);
      setHealthFromMock(h.fromMock);
      setMetrics(summarizeMetricsText(m.raw));
      setMetricsFromMock(m.fromMock);
    };
    void refresh();
    const id = setInterval(refresh, 5000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  const send = () => {
    cancelRef.current?.();
    if (prompt.trim().length === 0) return;
    const userTurn: ChatTurn = {
      id: `u-${Date.now()}`,
      role: "user",
      content: prompt.trim(),
    };
    const assistantTurn: ChatTurn = {
      id: `a-${Date.now()}`,
      role: "assistant",
      content: "",
      tokens: [],
    };
    setTurns((t) => [...t, userTurn]);
    setActive(assistantTurn);
    setStreaming(true);
    setLiveTps(undefined);
    const startedAt = performance.now();

    const messages = [
      ...turns.map((t) => ({ role: t.role, content: t.content })),
      { role: userTurn.role, content: userTurn.content },
    ];
    const handle = chatStream(
      { model: health.model ?? "qwen3:8b-q4", messages, temperature: 0.7, max_tokens: 256 },
      (tok, opts) => {
        setChatFromMock(opts.fromMock);
        setActive((cur) => {
          if (!cur) return cur;
          const tokens = [...(cur.tokens ?? []), tok];
          const totalS = (performance.now() - startedAt) / 1000;
          const ttftS = cur.ttftS ?? (tokens.length === 1 ? tok.intervalMs / 1000 : undefined);
          setLiveTps(tok.tps);
          return {
            ...cur,
            content: cur.content + tok.text,
            tokens,
            ttftS,
            totalS,
          };
        });
      },
    );
    cancelRef.current = handle.cancel;
    handle.done.then((res) => {
      const totalS = (performance.now() - startedAt) / 1000;
      setActive((cur) => {
        if (!cur) return null;
        const finalised: ChatTurn = {
          ...cur,
          content: res.text || cur.content,
          finishReason: res.finishReason,
          totalS,
          fromMock: res.fromMock,
        };
        setTurns((arr) => [...arr, finalised]);
        return null;
      });
      setChatFromMock(res.fromMock);
      setStreaming(false);
      setLiveTps(undefined);
    }).catch(() => {
      setStreaming(false);
      setLiveTps(undefined);
    });

    setPrompt("");
  };

  const clearConvo = () => {
    cancelRef.current?.();
    setTurns([]);
    setActive(null);
    setStreaming(false);
    setLiveTps(undefined);
  };

  const mode = inferKVMode(health.loader);
  const lastAssistantTurn: ChatTurn | null =
    active ?? [...turns].reverse().find((t) => t.role === "assistant") ?? null;

  return (
    <KonjoApp
      product="squish"
      tagline="Inference Cockpit · local LLMs, made visible"
      status={
        streaming
          ? { label: "streaming", severity: "info" }
          : healthFromMock
          ? { label: "offline · mocks", severity: "warn" }
          : { label: "ready", severity: "ok" }
      }
    >
      <Hero />

      <div className="space-y-6 mt-10">
        <section className="grid lg:grid-cols-[1fr_360px] gap-4 items-start">
          <div className="space-y-3">
            <ChatPanel turns={turns} active={active} />
            <PromptBar
              prompt={prompt}
              onPromptChange={setPrompt}
              onSubmit={send}
              onClear={clearConvo}
              disabled={streaming}
              submitLabel={streaming ? "streaming…" : "send"}
            />
          </div>
          <ThroughputCard health={health} liveTps={liveTps} />
        </section>

        <KVCacheView metrics={metrics} mode={mode} />

        <QuantComparator serverMode={mode === "fp16" || mode === "unknown" ? "int4" : mode} onFromMockChange={setBenchFromMock} />

        <section className="grid lg:grid-cols-2 gap-4">
          <LatencyWaterfall turn={lastAssistantTurn} />
          <ThermalDial health={health} />
        </section>

        <ModelInfo health={health} mode={mode} />

        <MetaInspector
          health={health}
          healthFromMock={healthFromMock}
          metricsFromMock={metricsFromMock}
          benchFromMock={benchFromMock}
          chatFromMock={chatFromMock}
        />

        <Footer />
      </div>
    </KonjoApp>
  );
}

function Hero() {
  return (
    <section className="text-center pt-6 pb-2">
      <p className="text-konjo-mono uppercase tracking-[0.32em] text-konjo-violet" style={{ fontSize: 11 }}>
        squish · 局所 · local · ぎゅっ · squeeze
      </p>
      <h1
        className="text-konjo-display text-konjo-fg mt-4 mx-auto"
        style={{ fontSize: 52, fontWeight: 600, letterSpacing: "-0.025em", maxWidth: 920, lineHeight: 1.05 }}
      >
        Your model, on your laptop, <span style={{ color: "var(--color-konjo-accent)" }}>visible</span>.
      </h1>
      <p
        className="text-konjo-fg-muted mt-5 mx-auto"
        style={{ fontSize: 16, maxWidth: 640, lineHeight: 1.55 }}
      >
        Real-time chat. KV cache live from <span className="text-konjo-mono">/v1/metrics</span>. Quantization comparator. Latency waterfall. Apple Silicon power telemetry. Everything the SquishBar shows you in 13px — now in cinema.
      </p>
    </section>
  );
}

function Footer() {
  return (
    <footer
      className="mt-16 pt-8 border-t border-konjo-line/60 text-konjo-fg-muted text-konjo-mono"
      style={{ fontSize: 12 }}
    >
      <div className="flex flex-wrap gap-4 justify-between items-baseline">
        <span>
          built on{" "}
          <span className="text-konjo-fg">@konjoai/ui</span>
          {" · "}
          <span className="text-konjo-fg">/v1/chat/completions</span>
          {" · "}
          <span className="text-konjo-fg">/health</span>
          {" · "}
          <span className="text-konjo-fg">/v1/metrics</span>
          {" · "}
          <span className="text-konjo-fg">/api/benchmark</span>
        </span>
        <span className="text-konjo-fg-faint">
          part of the KonjoAI portfolio · vectro · kyro · miru · kohaku · kairu · toki · squash
        </span>
      </div>
    </footer>
  );
}
