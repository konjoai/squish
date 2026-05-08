import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ModelInfo } from "./ModelInfo";
import { MOCK_HEALTH } from "../lib/mock";

describe("ModelInfo", () => {
  it("renders the model name and ready state", () => {
    render(<ModelInfo health={MOCK_HEALTH} mode="int4" />);
    expect(screen.getByText(MOCK_HEALTH.model!)).toBeInTheDocument();
    expect(screen.getByText(/ready/i)).toBeInTheDocument();
  });

  it("falls back to 'no model loaded' when not loaded", () => {
    render(<ModelInfo health={{ ...MOCK_HEALTH, loaded: false, model: null, status: "no_model" }} mode="unknown" />);
    expect(screen.getByText(/no model loaded/i)).toBeInTheDocument();
  });

  it("shows the kv mode pill", () => {
    render(<ModelInfo health={MOCK_HEALTH} mode="int4" />);
    expect(screen.getByText("int4")).toBeInTheDocument();
  });
});
