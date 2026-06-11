import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { QualityMonitor } from "./QualityMonitor";
import { MOCK_QUALITY } from "../lib/mock";

describe("QualityMonitor", () => {
  it("renders percentile groups and the request count", () => {
    render(<QualityMonitor report={MOCK_QUALITY} fromMock />);
    expect(screen.getByText(/latency/i)).toBeInTheDocument();
    expect(screen.getByText(/time to first token/i)).toBeInTheDocument();
    expect(screen.getByText(/rolling .* window/i)).toBeInTheDocument();
    expect(screen.getByText(/error rate/i)).toBeInTheDocument();
  });

  it("shows an empty state when there are no models", () => {
    render(<QualityMonitor report={{ ...MOCK_QUALITY, models: [] }} fromMock />);
    expect(screen.getByText(/no requests recorded/i)).toBeInTheDocument();
  });

  it("labels the source as mock or live", () => {
    const { rerender } = render(<QualityMonitor report={MOCK_QUALITY} fromMock />);
    expect(screen.getByText(/mock/)).toBeInTheDocument();
    rerender(<QualityMonitor report={MOCK_QUALITY} fromMock={false} />);
    expect(screen.getByText(/live/)).toBeInTheDocument();
  });
});
