import { describe, it, expect } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { AgentPlayground } from "./AgentPlayground";

describe("AgentPlayground", () => {
  it("renders the heading and an empty timeline state", () => {
    render(<AgentPlayground />);
    expect(screen.getByText(/agent playground/i)).toBeInTheDocument();
    expect(screen.getByText(/reasons, calls tools/i)).toBeInTheDocument();
  });

  it("discovers tools (falls back to mock palette when offline)", async () => {
    render(<AgentPlayground />);
    await waitFor(() => {
      expect(screen.getByText("squish_list_dir")).toBeInTheDocument();
    });
  });

  it("offers task suggestions", () => {
    render(<AgentPlayground />);
    expect(screen.getByText(/find every todo/i)).toBeInTheDocument();
  });
});
