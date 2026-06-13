import { describe, it, expect, vi } from "vitest";
import { act } from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { CommandPalette, type Command } from "./CommandPalette";

function makeCommands(run = vi.fn()): Command[] {
  return [
    { id: "goto-agent", title: "Go to agent", group: "section", run },
    { id: "goto-quality", title: "Go to quality", group: "section", run },
    { id: "act-clear", title: "Clear conversation", group: "action", run },
  ];
}

function openPalette() {
  act(() => { window.dispatchEvent(new Event("konjo:cmdk")); });
}

describe("CommandPalette", () => {
  it("is closed until opened", () => {
    render(<CommandPalette commands={makeCommands()} />);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("opens on the konjo:cmdk event and lists commands", () => {
    render(<CommandPalette commands={makeCommands()} />);
    openPalette();
    expect(screen.getByRole("dialog", { name: /command palette/i })).toBeInTheDocument();
    expect(screen.getByText("Go to agent")).toBeInTheDocument();
    expect(screen.getByText("Clear conversation")).toBeInTheDocument();
  });

  it("fuzzy-filters as you type", () => {
    render(<CommandPalette commands={makeCommands()} />);
    openPalette();
    fireEvent.change(screen.getByLabelText(/command query/i), { target: { value: "quality" } });
    expect(screen.getByText("Go to quality")).toBeInTheDocument();
    expect(screen.queryByText("Clear conversation")).not.toBeInTheDocument();
  });

  it("runs the selected command on Enter and closes", async () => {
    const run = vi.fn();
    render(<CommandPalette commands={makeCommands(run)} />);
    openPalette();
    const input = screen.getByLabelText(/command query/i);
    fireEvent.change(input, { target: { value: "clear" } });
    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Enter" });
    expect(run).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });

  it("shows 'no matches' for an unmatched query", () => {
    render(<CommandPalette commands={makeCommands()} />);
    openPalette();
    fireEvent.change(screen.getByLabelText(/command query/i), { target: { value: "zzzzz" } });
    expect(screen.getByText(/no matches/i)).toBeInTheDocument();
  });
});
