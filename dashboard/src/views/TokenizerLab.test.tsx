import { describe, it, expect } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { TokenizerLab } from "./TokenizerLab";

describe("TokenizerLab", () => {
  it("renders the heading and a token count metric", () => {
    render(<TokenizerLab />);
    expect(screen.getByText(/tokenizer lab/i)).toBeInTheDocument();
    expect(screen.getByText(/chars \/ tok/i)).toBeInTheDocument();
  });

  it("tokenizes the seed text after debounce (mock fallback)", async () => {
    render(<TokenizerLab />);
    await waitFor(
      () => {
        // mock model id appears once tokenize resolves
        expect(screen.getByText(/mock bpe/i)).toBeInTheDocument();
      },
      { timeout: 1500 },
    );
  });
});
