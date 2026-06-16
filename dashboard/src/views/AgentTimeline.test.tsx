import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { AgentTimeline } from "./AgentTimeline";
import { ChatPanel } from "./ChatPanel";
import type { AgentStep, ChatTurn } from "../lib/types";

const STEPS: AgentStep[] = [
  {
    step: 1,
    text: "I'll list the directory.",
    complete: true,
    calls: [
      {
        callId: "c1",
        toolName: "squish_list_dir",
        arguments: { path: "." },
        result: "README.md\npyproject.toml",
        error: null,
        elapsedMs: 3.2,
        done: true,
      },
    ],
  },
];

describe("AgentTimeline", () => {
  it("renders step text, the tool name, and its result", () => {
    render(<AgentTimeline steps={STEPS} running={false} />);
    expect(screen.getByText(/list the directory/i)).toBeInTheDocument();
    expect(screen.getByText("squish_list_dir")).toBeInTheDocument();
    expect(screen.getByText(/pyproject\.toml/)).toBeInTheDocument();
  });
});

describe("ChatPanel agent mode", () => {
  it("renders the agent timeline for an assistant turn with steps", () => {
    const turns: ChatTurn[] = [
      { id: "u1", role: "user", content: "what files are here?" },
      { id: "a1", role: "assistant", content: "There is a README and a pyproject.", steps: STEPS },
    ];
    render(<ChatPanel turns={turns} />);
    // Tool call surfaced inline in the conversation.
    expect(screen.getByText("squish_list_dir")).toBeInTheDocument();
    // Final answer still rendered.
    expect(screen.getByText(/README and a pyproject/i)).toBeInTheDocument();
  });
});
