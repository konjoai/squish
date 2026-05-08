import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { PromptBar } from "./PromptBar";

const noop = () => {};

describe("PromptBar", () => {
  it("renders prompt textarea and buttons", () => {
    render(<PromptBar prompt="" onPromptChange={noop} onSubmit={noop} onClear={noop} disabled={false} />);
    expect(screen.getByPlaceholderText(/ask the model/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /clear/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /send/i })).toBeInTheDocument();
  });

  it("disables send when prompt is empty", () => {
    render(<PromptBar prompt="" onPromptChange={noop} onSubmit={noop} onClear={noop} disabled={false} />);
    expect(screen.getByRole("button", { name: /send/i })).toBeDisabled();
  });

  it("enables send when prompt is non-empty", () => {
    render(<PromptBar prompt="hi" onPromptChange={noop} onSubmit={noop} onClear={noop} disabled={false} />);
    expect(screen.getByRole("button", { name: /send/i })).toBeEnabled();
  });

  it("calls onClear when clear button is clicked", async () => {
    const onClear = vi.fn();
    render(<PromptBar prompt="hi" onPromptChange={noop} onSubmit={noop} onClear={onClear} disabled={false} />);
    await userEvent.click(screen.getByRole("button", { name: /clear/i }));
    expect(onClear).toHaveBeenCalledOnce();
  });

  it("calls onSubmit when the send button is clicked", async () => {
    const onSubmit = vi.fn();
    render(<PromptBar prompt="hi" onPromptChange={noop} onSubmit={onSubmit} onClear={noop} disabled={false} />);
    await userEvent.click(screen.getByRole("button", { name: /send/i }));
    expect(onSubmit).toHaveBeenCalledOnce();
  });
});
