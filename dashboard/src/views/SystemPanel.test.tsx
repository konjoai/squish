import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { SystemPanel } from "./SystemPanel";
import { MOCK_SYS_STATS, MOCK_MODEL_STATUS } from "../lib/mock";

describe("SystemPanel", () => {
  it("renders host metric tiles and the load mode", () => {
    render(<SystemPanel stats={MOCK_SYS_STATS} status={MOCK_MODEL_STATUS} fromMock />);
    expect(screen.getByText(/cpu load avg/i)).toBeInTheDocument();
    expect(screen.getByText(/process rss/i)).toBeInTheDocument();
    expect(screen.getByText(/disk/i)).toBeInTheDocument();
    expect(screen.getByText(/preload · async/i)).toBeInTheDocument();
  });

  it("shows the loaded model and load time", () => {
    render(<SystemPanel stats={MOCK_SYS_STATS} status={MOCK_MODEL_STATUS} fromMock={false} />);
    expect(screen.getByText(/loaded ·/i)).toBeInTheDocument();
    expect(screen.getByText(MOCK_MODEL_STATUS.model!)).toBeInTheDocument();
  });

  it("surfaces a load error state", () => {
    render(
      <SystemPanel
        stats={MOCK_SYS_STATS}
        status={{ ...MOCK_MODEL_STATUS, model_loaded: false, load_error: "OOM" }}
        fromMock
      />,
    );
    expect(screen.getByText(/load error/i)).toBeInTheDocument();
  });
});
