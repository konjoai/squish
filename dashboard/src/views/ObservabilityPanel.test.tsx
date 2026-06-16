import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ObservabilityPanel } from "./ObservabilityPanel";
import { MOCK_OBS_REPORT } from "../lib/mock";

describe("ObservabilityPanel", () => {
  it("renders the status badge and per-op latency rows", () => {
    render(<ObservabilityPanel report={MOCK_OBS_REPORT} fromMock />);
    expect(screen.getByText("degraded")).toBeInTheDocument();
    // model.prefill appears in both the latency rows and the bottleneck card;
    // sampler.sample is only a latency row, so it's unambiguous.
    expect(screen.getByText("sampler.sample")).toBeInTheDocument();
    expect(screen.getAllByText("model.prefill").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/per-operation latency/i)).toBeInTheDocument();
  });

  it("lists bottlenecks with hints", () => {
    render(<ObservabilityPanel report={MOCK_OBS_REPORT} fromMock />);
    expect(screen.getByText(/Prefill p99 high/i)).toBeInTheDocument();
  });

  it("shows a tracing-disabled note when there are no spans", () => {
    render(<ObservabilityPanel report={{ ...MOCK_OBS_REPORT, recent_spans: [] }} fromMock />);
    expect(screen.getByText(/span tracing disabled/i)).toBeInTheDocument();
  });

  it("celebrates a clean bill of health when there are no bottlenecks", () => {
    render(<ObservabilityPanel report={{ ...MOCK_OBS_REPORT, status: "ok", bottlenecks: [] }} fromMock />);
    expect(screen.getByText(/no operations over threshold/i)).toBeInTheDocument();
  });
});
