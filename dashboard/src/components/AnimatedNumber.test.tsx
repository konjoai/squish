import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { AnimatedNumber } from "./AnimatedNumber";

describe("AnimatedNumber", () => {
  it("renders the value immediately on mount", () => {
    render(<AnimatedNumber value={42} />);
    expect(screen.getByText("42")).toBeInTheDocument();
  });

  it("applies the custom formatter", () => {
    render(<AnimatedNumber value={3.14159} format={(v) => v.toFixed(2)} />);
    expect(screen.getByText("3.14")).toBeInTheDocument();
  });
});
