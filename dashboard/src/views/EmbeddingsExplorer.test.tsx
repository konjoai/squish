import { describe, it, expect } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { EmbeddingsExplorer } from "./EmbeddingsExplorer";

describe("EmbeddingsExplorer", () => {
  it("renders the heading and the default text inputs", () => {
    render(<EmbeddingsExplorer />);
    expect(screen.getByText(/embeddings explorer/i)).toBeInTheDocument();
    expect(screen.getByText(/press embed to compare/i)).toBeInTheDocument();
  });

  it("computes a similarity heatmap on embed (mock fallback)", async () => {
    render(<EmbeddingsExplorer />);
    fireEvent.click(screen.getByRole("button", { name: /^embed$/i }));
    await waitFor(() => {
      expect(screen.getByLabelText(/cosine similarity matrix/i)).toBeInTheDocument();
    });
  });
});
