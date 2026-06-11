import { describe, it, expect } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { StartupProfile } from "./StartupProfile";

describe("StartupProfile", () => {
  it("renders the heading", () => {
    render(<StartupProfile />);
    expect(screen.getByRole("heading", { name: /startup profile/i })).toBeInTheDocument();
  });

  it("renders the phase waterfall after fetch (mock fallback)", async () => {
    render(<StartupProfile />);
    await waitFor(() => {
      expect(screen.getByText("model_load")).toBeInTheDocument();
      expect(screen.getByText(/slowest/i)).toBeInTheDocument();
    });
  });
});
