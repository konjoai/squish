import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ChatPanel } from "./ChatPanel";
import { buildMockChatStream } from "../lib/mock";
import type { ChatTurn } from "../lib/types";

describe("ChatPanel", () => {
  it("renders empty hint when there are no turns", () => {
    render(<ChatPanel turns={[]} />);
    expect(screen.getByText(/start a conversation/i)).toBeInTheDocument();
  });

  it("renders user turn role label", () => {
    const turns: ChatTurn[] = [{ id: "u1", role: "user", content: "hi there" }];
    render(<ChatPanel turns={turns} />);
    expect(screen.getByText("hi there")).toBeInTheDocument();
    // Role pill renders the lowercase role text inside the header.
    expect(screen.getByText(/user/i)).toBeInTheDocument();
  });

  it("renders an assistant turn with token-level body", () => {
    const { turn } = buildMockChatStream();
    render(<ChatPanel turns={[turn]} />);
    expect(screen.getByText(/assistant/i)).toBeInTheDocument();
    // First token of the canned response is "Speculative".
    expect(screen.getByText(/Speculative/)).toBeInTheDocument();
  });

  it("highlights the active streaming turn", () => {
    const { turn } = buildMockChatStream();
    render(<ChatPanel turns={[]} active={turn} />);
    expect(screen.getByText(/streaming/i)).toBeInTheDocument();
  });
});
