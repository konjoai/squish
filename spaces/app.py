"""squish — KV-cache quantization demo (Hugging Face Space, W107).

Zero-install browser demo: pick a synthetic activation distribution, run
INT8/INT4/INT2 KV-cache codecs through it, and see SNR + memory side-by-
side. The "Memory Budgeter" tab plans real-model KV cache footprints
across context lengths and tells you which tier fits a given RAM budget.

Heavy lifting lives in :mod:`spaces._logic`; this file is the Gradio shell.
"""

from __future__ import annotations

import gradio as gr

from spaces._logic import (
    EXAMPLES,
    KV_INT2_AUTO_THRESHOLD,
    KV_INT4_DEFAULT_THRESHOLD,
    MODEL_PRESETS,
    apply_hadamard,
    label_budget_fit,
    make_synthetic_activations,
    memory_table_rows,
    recommend_for_budget_mb,
    recommend_mode_for_context,
    run_all_tiers,
)

INTRO_MD = f"""
# squish · KV-cache quantization demo

Squish ships **3 quantization tiers for KV-cache storage** (INT8 / INT4 /
INT2) plus an optional Hadamard rotation that makes the low-bit modes
viable on real activations. This space runs them on synthetic tensors so
you can see SNR and memory at every tier — no model download, no GPU.

**Tier defaults (auto-picked by context length):**
- ≤ {KV_INT2_AUTO_THRESHOLD:,} tokens → **int8** (high quality, ~1.94× compression)
- {KV_INT2_AUTO_THRESHOLD:,} – {KV_INT4_DEFAULT_THRESHOLD:,} tokens → **int4** (balanced, ~3.76× compression)
- > {KV_INT4_DEFAULT_THRESHOLD:,} tokens → **int2** (long-context, ~7.11× compression)

→ [GitHub](https://github.com/konjoai/squish) · [PyPI](https://pypi.org/project/squish-ai/) · [Docs](https://squish.ai)
"""


def _tensor_inspector(
    n_tokens: int,
    head_dim: int,
    distribution: str,
    rotate: bool,
):
    """Build the (table, recommendation, summary) triple for the inspector tab."""
    arr = make_synthetic_activations(int(n_tokens), int(head_dim), distribution)
    if rotate:
        arr = apply_hadamard(arr)
    rows = run_all_tiers(arr)
    table = [
        [
            r.mode,
            f"{r.snr_db:.2f}" if r.snr_db != float("inf") else "∞",
            r.bytes_per_token,
            f"{r.compression_vs_fp16:.2f}×",
        ]
        for r in rows
    ]
    # Recommendation: smallest tier that still clears 6 dB SNR (1 bit ≈ 6 dB
    # Shannon — below that, reconstruction error dominates the signal).
    viable = [r for r in rows if r.snr_db >= 6.0]
    if viable:
        best = min(viable, key=lambda r: r.bytes_per_token)
        rec = f"### Recommended tier: **{best.mode}** ({best.snr_db:.1f} dB SNR, {best.compression_vs_fp16:.2f}× smaller than fp16)"
    else:
        rec = "### Recommended tier: **fp16** — every quantised tier fell below the 6 dB SNR floor on this input. Try enabling Hadamard rotation."
    summary = f"Input shape: ({int(n_tokens)}, {int(head_dim)}) · distribution: `{distribution}` · rotation: `{'on' if rotate else 'off'}`"
    return table, rec, summary


def _memory_budgeter(
    preset_label: str,
    context_tokens: int,
    budget_mb: float,
):
    """Build the memory table + recommendation for the budgeter tab."""
    n_layers, n_kv_heads, head_dim = MODEL_PRESETS[preset_label]
    rows = memory_table_rows(n_layers, n_kv_heads, head_dim, int(context_tokens))
    labelled = label_budget_fit(rows, float(budget_mb))
    table = [
        [
            r["mode"],
            f"{r['total_mb']:.1f}",
            f"{r['recent_window_mb']:.1f}",
            f"{r['compression_ratio']:.2f}×",
            r["fits"],
        ]
        for r in labelled
    ]
    by_ctx = recommend_mode_for_context(int(context_tokens))
    if budget_mb > 0:
        by_budget = recommend_for_budget_mb(
            n_layers, n_kv_heads, head_dim, int(context_tokens), float(budget_mb),
        )
        rec = (
            f"### Recommended tier\n"
            f"- by **context length** ({int(context_tokens):,} tok): **{by_ctx}**\n"
            f"- by **RAM budget** ({budget_mb:.0f} MB): **{by_budget}**"
        )
    else:
        rec = (
            f"### Recommended tier\n"
            f"- by **context length** ({int(context_tokens):,} tok): **{by_ctx}**\n"
            f"- by **RAM budget**: set a positive value to enable"
        )
    return table, rec


def build_demo() -> gr.Blocks:
    """Assemble the full Gradio app (kept as a function so tests can spin it up)."""
    with gr.Blocks(
        title="squish · KV-cache quantization",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(INTRO_MD)

        with gr.Tabs():
            # ----- Tab 1: tensor inspector --------------------------------
            with gr.Tab("Tensor Inspector"):
                gr.Markdown(
                    "Pick a distribution, choose whether to apply the "
                    "Hadamard rotation that makes low-bit tiers viable on "
                    "heavy-tailed activations, and watch SNR vs. compression."
                )
                with gr.Row():
                    n_tokens = gr.Slider(32, 1024, 256, step=32, label="n_tokens")
                    head_dim = gr.Slider(
                        16, 256, 128, step=16, label="head_dim (must be divisible by 4)",
                    )
                with gr.Row():
                    distribution = gr.Radio(
                        choices=["gaussian", "heavy_tailed", "outlier"],
                        value="heavy_tailed",
                        label="distribution",
                        info="gaussian = easy · heavy_tailed = realistic · outlier = bin-collapse demo",
                    )
                    rotate = gr.Checkbox(
                        value=False,
                        label="Apply Hadamard rotation",
                        info="QuaRot-style randomised rotation; whitens outliers before quantisation",
                    )
                run_btn = gr.Button("Run quantization", variant="primary")
                table = gr.Dataframe(
                    headers=["mode", "SNR (dB)", "B / token", "compression vs fp16"],
                    label="Per-tier reconstruction quality",
                    interactive=False,
                )
                rec = gr.Markdown()
                summary = gr.Markdown()

                run_btn.click(
                    _tensor_inspector,
                    inputs=[n_tokens, head_dim, distribution, rotate],
                    outputs=[table, rec, summary],
                )

                gr.Examples(
                    examples=[list(e) for e in EXAMPLES],
                    inputs=[n_tokens, head_dim, distribution, rotate],
                    label="Try these examples",
                )

            # ----- Tab 2: memory budgeter ---------------------------------
            with gr.Tab("Memory Budgeter"):
                gr.Markdown(
                    "Closed-form KV-cache memory across context lengths "
                    "and tiers, for real squishai model presets. "
                    "Same numbers `squish` uses internally to pick a tier "
                    "via `make_kv_cache(..., planned_context=...)`."
                )
                with gr.Row():
                    preset = gr.Dropdown(
                        choices=list(MODEL_PRESETS.keys()),
                        value=list(MODEL_PRESETS.keys())[3],   # Qwen2.5-7B
                        label="Model preset",
                    )
                with gr.Row():
                    context = gr.Slider(
                        512, 65_536, 16_384, step=512, label="context_tokens",
                    )
                    budget = gr.Number(
                        value=4096.0,
                        label="RAM budget (MB) — 0 to disable",
                        precision=0,
                    )
                budget_btn = gr.Button("Estimate memory", variant="primary")
                mem_table = gr.Dataframe(
                    headers=[
                        "mode", "total (MB)", "recent window (MB)",
                        "compression vs fp16", "fits budget?",
                    ],
                    label="KV-cache memory by tier",
                    interactive=False,
                )
                mem_rec = gr.Markdown()

                budget_btn.click(
                    _memory_budgeter,
                    inputs=[preset, context, budget],
                    outputs=[mem_table, mem_rec],
                )

                gr.Markdown(
                    "**The math:** "
                    "`total_bytes = n_layers · n_kv_heads · 2 (K+V) · context · "
                    "(head_dim_codes + 4-byte fp32 scale)`. "
                    "head_dim_codes is `head_dim` (int8), `head_dim/2` (int4), "
                    "or `head_dim/4` (int2). The recent-window column is the "
                    "fp16 sliding window that quality-critical recent tokens "
                    "live in (default 128 tokens)."
                )

        gr.Markdown(
            "---\n"
            "Built with `squish.kv.kv_cache` — the same codecs that ship in "
            "`pip install squish`. Source: "
            "[github.com/konjoai/squish/tree/main/spaces](https://github.com/konjoai/squish/tree/main/spaces)."
        )

    return demo


if __name__ == "__main__":
    build_demo().launch()
