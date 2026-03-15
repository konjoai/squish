#!/usr/bin/env python3
"""
dev/demos/record_agent_demo.py

Orchestrates a scripted 20-turn tool-call session for demo recording.

Connects to a running squish server via the OpenAI Python client and sends
20 predefined weather tool-call turns. Logs TTFT and token counts per turn
to stdout.

Prerequisites:
    squish serve <model> --agent --port 11435

Usage:
    python3 dev/demos/record_agent_demo.py [--port 11435] [--model squish]

Install openai client if needed:
    pip install openai
"""
from __future__ import annotations

import argparse
import sys
import time

# ── 20 predefined tool-call turns ────────────────────────────────────────────
_TURNS = [
    "What is the weather in London?",
    "How about Paris?",
    "And New York City?",
    "Compare London and Paris temperatures.",
    "Is it raining in Tokyo right now?",
    "What is the forecast for Sydney tomorrow?",
    "Is Chicago colder than Boston today?",
    "What is the humidity in Dubai?",
    "Show me the weather in Berlin.",
    "What about Madrid?",
    "Is it snowing in Oslo?",
    "What is the wind speed in Amsterdam?",
    "How warm is it in Singapore?",
    "What is the UV index in Los Angeles today?",
    "Is it foggy in San Francisco?",
    "What is the dew point in Phoenix?",
    "How cold is it in Moscow?",
    "What is the visibility in Mumbai?",
    "Is there a storm warning in Miami?",
    "Summarize the weather across all cities we discussed.",
]

# ── Weather tool schema ───────────────────────────────────────────────────────
_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather conditions for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g. 'London' or 'New York'",
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature units",
                },
            },
            "required": ["location"],
        },
    },
}


def run_demo(port: int = 11435, model: str = "squish") -> None:
    """Run the 20-turn scripted tool-call demo session."""
    try:
        from openai import OpenAI  # noqa: PLC0415
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)

    base_url = f"http://localhost:{port}/v1"
    client = OpenAI(base_url=base_url, api_key="squish")

    # Verify server is reachable
    try:
        client.models.list()
    except Exception as exc:
        print(
            f"\nERROR: Cannot connect to squish server at {base_url}\n"
            f"  {exc}\n\n"
            "Start the server first:\n"
            f"  squish serve <model> --agent --port {port}\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nSquish Agent Demo — {len(_TURNS)} turns")
    print(f"Server: {base_url}  Model: {model}")
    print("=" * 60)

    conversation: list[dict] = []
    total_tokens = 0

    for turn_idx, user_message in enumerate(_TURNS, 1):
        conversation.append({"role": "user", "content": user_message})

        t_start = time.perf_counter()
        ttft_ms = None
        turn_tokens = 0
        first_chunk = True
        response_text = ""

        try:
            stream = client.chat.completions.create(
                model=model,
                messages=conversation,
                tools=[_WEATHER_TOOL],
                tool_choice="auto",
                stream=True,
                max_tokens=256,
                temperature=0.0,
            )

            for chunk in stream:
                if first_chunk:
                    ttft_ms = (time.perf_counter() - t_start) * 1000
                    first_chunk = False

                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    response_text += delta.content
                    turn_tokens += 1

            elapsed_ms = (time.perf_counter() - t_start) * 1000

        except Exception as exc:
            print(f"  Turn {turn_idx:2d}  ERROR: {exc}", file=sys.stderr)
            conversation.pop()
            continue

        # Add assistant response to conversation history
        conversation.append({"role": "assistant", "content": response_text or "(tool call)"})
        total_tokens += turn_tokens

        ttft_str = f"{ttft_ms:.0f}ms" if ttft_ms is not None else "N/A"
        print(
            f"  Turn {turn_idx:2d}/{len(_TURNS)}"
            f"  TTFT={ttft_str:>7}"
            f"  tokens={turn_tokens:3d}"
            f"  elapsed={elapsed_ms:.0f}ms"
            f"  Q: {user_message[:50]}"
        )

    print("=" * 60)
    print(f"  Total turns : {len(_TURNS)}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Done.\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Scripted 20-turn agent demo for squish server",
    )
    ap.add_argument("--port", type=int, default=11435,
                    help="Squish server port (default: 11435)")
    ap.add_argument("--model", default="squish",
                    help="Model name to use (default: squish)")
    args = ap.parse_args()
    run_demo(port=args.port, model=args.model)


if __name__ == "__main__":
    main()
