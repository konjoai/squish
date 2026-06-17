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
import { AgentPlayground } from "./views/AgentPlayground";
import { TokenizerLab } from "./views/TokenizerLab";
import { QualityMonitor } from "./views/QualityMonitor";
import { EmbeddingsExplorer } from "./views/EmbeddingsExplorer";
import { SystemPanel } from "./views/SystemPanel";
import { ObservabilityPanel } from "./views/ObservabilityPanel";
import { StartupProfile } from "./views/StartupProfile";
import { SectionNav } from "./components/SectionNav";
import { CommandPalette, type Command } from "./components/CommandPalette";
import {
  chatStream, agentRun, fetchHealth, fetchMetrics, fetchQuality, fetchSysStats, fetchModelStatus,
  fetchObsReport, summarizeMetricsText,
} from "./lib/api";
import { applyAgentEvent } from "./lib/agent";
import {
  MOCK_HEALTH, MOCK_PROM_TEXT, MOCK_QUALITY, MOCK_SYS_STATS, MOCK_MODEL_STATUS, MOCK_OBS_REPORT,
} from "./lib/mock";
import { loadTurns, saveTurns, clearTurns } from "./lib/persist";
import { conversationToMarkdown, conversationToJSON } from "./lib/export";
import type {
  ChatTurn, HealthResponse, CockpitMetrics, KVMode, QualityReport, SysStats, ModelStatus, ObsReport,
} from "./lib/types";

const NAV_ITEMS = [
  { id: "chat", label: "chat" },
  { id: "agent", label: "agent" },
  { id: "kv", label: "kv cache" },
  { id: "quant", label: "quant" },
  { id: "tokenizer", label: "tokenizer" },
  { id: "embeddings", label: "embeddings" },
  { id: "latency", label: "latency" },
  { id: "quality", label: "quality" },
  { id: "observability", label: "observability" },
  { id: "thermal", label: "power" },
  { id: "system", label: "system" },
  { id: "startup", label: "startup" },
  { id: "model", label: "model" },
];

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

  // Restore any prior conversation from localStorage on first render.
  const [turns, setTurns] = useState<ChatTurn[]>(() => loadTurns());
  const [active, setActive] = useState<ChatTurn | null>(null);
  const [streaming, setStreaming] = useState<boolean>(false);
  const [chatFromMock, setChatFromMock] = useState<boolean>(false);
  const [agentMode, setAgentMode] = useState<boolean>(false);

  const [health, setHealth] = useState<HealthResponse>(MOCK_HEALTH);
  const [metrics, setMetrics] = useState<CockpitMetrics>(EMPTY_METRICS);
  const [quality, setQuality] = useState<QualityReport>(MOCK_QUALITY);
  const [sysStats, setSysStats] = useState<SysStats>(MOCK_SYS_STATS);
  const [modelStatus, setModelStatus] = useState<ModelStatus>(MOCK_MODEL_STATUS);
  const [obsReport, setObsReport] = useState<ObsReport>(MOCK_OBS_REPORT);
  const [healthFromMock, setHealthFromMock] = useState<boolean>(true);
  const [metricsFromMock, setMetricsFromMock] = useState<boolean>(true);
  const [benchFromMock, setBenchFromMock] = useState<boolean>(true);
  const [agentFromMock, setAgentFromMock] = useState<boolean>(true);
  const [tokFromMock, setTokFromMock] = useState<boolean>(true);
  const [qualityFromMock, setQualityFromMock] = useState<boolean>(true);
  const [embedFromMock, setEmbedFromMock] = useState<boolean>(true);
  const [sysFromMock, setSysFromMock] = useState<boolean>(true);
  const [obsFromMock, setObsFromMock] = useState<boolean>(true);
  const [startupFromMock, setStartupFromMock] = useState<boolean>(true);

  const [liveTps, setLiveTps] = useState<number | undefined>();

  const cancelRef = useRef<(() => void) | null>(null);

  // Poll /health, /v1/metrics, /v1/quality, /sys-stats, /model/status, /v1/obs-report every 5s.
  useEffect(() => {
    let cancelled = false;
    const refresh = async () => {
      const [h, m, q, s, ms, obs] = await Promise.all([
        fetchHealth(), fetchMetrics(), fetchQuality(), fetchSysStats(), fetchModelStatus(), fetchObsReport(),
      ]);
      if (cancelled) return;
      setHealth(h.data);
      setHealthFromMock(h.fromMock);
      setMetrics(summarizeMetricsText(m.raw));
      setMetricsFromMock(m.fromMock);
      setQuality(q.data);
      setQualityFromMock(q.fromMock);
      setSysStats(s.data);
      setModelStatus(ms.data);
      setSysFromMock(s.fromMock || ms.fromMock);
      setObsReport(obs.data);
      setObsFromMock(obs.fromMock);
    };
    void refresh();
    const id = setInterval(refresh, 5000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  // Persist completed turns so the conversation survives a reload.
  useEffect(() => { saveTurns(turns); }, [turns]);

  const send = () => {
    if (agentMode) { sendAgent(); return; }
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

  // Agentic chat: route the prompt through POST /v1/agent/run so the model
  // can call tools, and stream the live tool-execution timeline into the turn.
  const sendAgent = () => {
    cancelRef.current?.();
    if (prompt.trim().length === 0) return;
    const userTurn: ChatTurn = { id: `u-${Date.now()}`, role: "user", content: prompt.trim() };
    const assistantTurn: ChatTurn = { id: `a-${Date.now()}`, role: "assistant", content: "", steps: [] };
    setTurns((t) => [...t, userTurn]);
    setActive(assistantTurn);
    setStreaming(true);
    setLiveTps(undefined);
    const startedAt = performance.now();

    const messages = [
      ...turns.map((t) => ({ role: t.role, content: t.content })),
      { role: userTurn.role, content: userTurn.content },
    ];
    const handle = agentRun(
      { messages, max_steps: 8, max_tokens: 768, temperature: 0.4 },
      (ev, opts) => {
        setChatFromMock(opts.fromMock);
        setAgentFromMock(opts.fromMock);
        if (ev.type === "done" || ev.type === "error") return;
        setActive((cur) => {
          if (!cur) return cur;
          const steps = applyAgentEvent(cur.steps ?? [], ev);
          const totalS = (performance.now() - startedAt) / 1000;
          return { ...cur, steps, totalS };
        });
      },
    );
    cancelRef.current = handle.cancel;
    handle.done.then((res) => {
      const totalS = (performance.now() - startedAt) / 1000;
      setActive((cur) => {
        if (!cur) return null;
        // The model's final answer is the text of the last (tool-free) step.
        const lastText = (cur.steps ?? []).filter((s) => s.calls.length === 0).map((s) => s.text).join("").trim();
        const finalised: ChatTurn = { ...cur, content: lastText, totalS, fromMock: res.fromMock };
        setTurns((arr) => [...arr, finalised]);
        return null;
      });
      setStreaming(false);
      setLiveTps(undefined);
    }).catch(() => { setStreaming(false); setLiveTps(undefined); });

    setPrompt("");
  };

  const clearConvo = () => {
    cancelRef.current?.();
    setTurns([]);
    setActive(null);
    setStreaming(false);
    setLiveTps(undefined);
    clearTurns();
  };

  const copyMarkdown = () => {
    if (turns.length === 0) return;
    void navigator.clipboard?.writeText(conversationToMarkdown(turns));
  };

  const downloadJSON = () => {
    if (turns.length === 0) return;
    const blob = new Blob([conversationToJSON(turns)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `squish-conversation-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const mode = inferKVMode(health.loader);
  const lastAssistantTurn: ChatTurn | null =
    active ?? [...turns].reverse().find((t) => t.role === "assistant") ?? null;

  // Command palette (⌘K): every section + the key chat actions.
  const commands: Command[] = [
    ...NAV_ITEMS.map((it) => ({
      id: `goto-${it.id}`,
      title: `Go to ${it.label}`,
      group: "section",
      keywords: it.id,
      run: () => document.getElementById(it.id)?.scrollIntoView({ behavior: "smooth", block: "start" }),
    })),
    { id: "act-send", title: "Send message", group: "action", keywords: "chat submit prompt", run: send },
    { id: "act-clear", title: "Clear conversation", group: "action", keywords: "reset chat", run: clearConvo },
    { id: "act-copy-md", title: "Copy conversation as Markdown", group: "export", keywords: "share clipboard", run: copyMarkdown },
    { id: "act-dl-json", title: "Download conversation as JSON", group: "export", keywords: "share save export", run: downloadJSON },
    { id: "act-top", title: "Scroll to top", group: "action", keywords: "home hero", run: () => window.scrollTo({ top: 0, behavior: "smooth" }) },
  ];

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

      <CommandPalette commands={commands} />
      <SectionNav items={NAV_ITEMS} />

      <div className="space-y-12 mt-10">
        <section id="chat" className="grid lg:grid-cols-[1fr_360px] gap-4 items-start scroll-mt-24">
          <div className="space-y-3">
            <ChatPanel turns={turns} active={active} />
            <PromptBar
              prompt={prompt}
              onPromptChange={setPrompt}
              onSubmit={send}
              onClear={clearConvo}
              disabled={streaming}
              submitLabel={streaming ? (agentMode ? "running…" : "streaming…") : agentMode ? "run agent" : "send"}
              agentMode={agentMode}
              onAgentModeChange={setAgentMode}
            />
          </div>
          <ThroughputCard health={health} liveTps={liveTps} />
        </section>

        <div className="scroll-mt-24">
          <AgentPlayground onFromMockChange={setAgentFromMock} />
        </div>

        <div id="kv" className="scroll-mt-24">
          <KVCacheView metrics={metrics} mode={mode} />
        </div>

        <div id="quant" className="scroll-mt-24">
          <QuantComparator serverMode={mode === "fp16" || mode === "unknown" ? "int4" : mode} onFromMockChange={setBenchFromMock} />
        </div>

        <div className="scroll-mt-24">
          <TokenizerLab onFromMockChange={setTokFromMock} />
        </div>

        <div className="scroll-mt-24">
          <EmbeddingsExplorer onFromMockChange={setEmbedFromMock} />
        </div>

        <section className="grid lg:grid-cols-2 gap-4">
          <div id="latency" className="scroll-mt-24">
            <LatencyWaterfall turn={lastAssistantTurn} />
          </div>
          <div id="thermal" className="scroll-mt-24">
            <ThermalDial health={health} />
          </div>
        </section>

        <div className="scroll-mt-24">
          <SystemPanel stats={sysStats} status={modelStatus} fromMock={sysFromMock} />
        </div>

        <div className="scroll-mt-24">
          <QualityMonitor report={quality} fromMock={qualityFromMock} />
        </div>

        <div className="scroll-mt-24">
          <ObservabilityPanel report={obsReport} fromMock={obsFromMock} />
        </div>

        <div className="scroll-mt-24">
          <StartupProfile onFromMockChange={setStartupFromMock} />
        </div>

        <div id="model" className="scroll-mt-24">
          <ModelInfo health={health} mode={mode} />
        </div>

        <MetaInspector
          health={health}
          healthFromMock={healthFromMock}
          metricsFromMock={metricsFromMock}
          benchFromMock={benchFromMock}
          chatFromMock={chatFromMock}
          agentFromMock={agentFromMock}
          tokFromMock={tokFromMock}
          qualityFromMock={qualityFromMock}
          embedFromMock={embedFromMock}
          sysFromMock={sysFromMock}
          obsFromMock={obsFromMock}
          startupFromMock={startupFromMock}
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
        Real-time chat. A live agent that calls tools while you watch. Tokenizer lab. Embedding similarity. KV cache from <span className="text-konjo-mono">/v1/metrics</span>. Quantization comparator. Latency waterfall. P50/P95/P99 quality. APM traces &amp; bottlenecks. Power, disk &amp; load telemetry. Everything squish does — now visible, live, in cinema.
      </p>
      <button
        type="button"
        onClick={() => window.dispatchEvent(new Event("konjo:cmdk"))}
        className="mt-6 inline-flex items-center gap-2 px-3 py-1.5 rounded-konjo border border-konjo-line/60 bg-konjo-surface/50 text-konjo-fg-muted hover:text-konjo-fg hover:border-konjo-violet/60 transition-colors"
      >
        <span className="text-konjo-mono text-[11px]">jump to anything</span>
        <kbd className="text-konjo-mono text-[10px] text-konjo-violet border border-konjo-line/60 rounded px-1.5 py-0.5">⌘K</kbd>
      </button>
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
          <span className="text-konjo-fg">/v1/agent/run</span>
          {" · "}
          <span className="text-konjo-fg">/v1/tokenize</span>
          {" · "}
          <span className="text-konjo-fg">/v1/embeddings</span>
          {" · "}
          <span className="text-konjo-fg">/v1/quality</span>
          {" · "}
          <span className="text-konjo-fg">/v1/obs-report</span>
          {" · "}
          <span className="text-konjo-fg">/v1/startup-profile</span>
          {" · "}
          <span className="text-konjo-fg">/sys-stats</span>
          {" · "}
          <span className="text-konjo-fg">/health</span>
          {" · "}
          <span className="text-konjo-fg">/v1/metrics</span>
          {" · "}
          <span className="text-konjo-fg">/api/benchmark</span>
        </span>
        <span className="text-konjo-fg-faint">
          part of the KonjoAI portfolio
        </span>
      </div>
    </footer>
  );
}
