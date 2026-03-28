# CLAUDE.md — Konjo AI Project Conventions & Collaboration Guidelines

> **ቆንጆ** — Beautiful. **根性** — Fighting spirit. **康宙** — Health of the universe.
> *Make it konjo — build, ship, repeat.*

This file defines standing instructions for all AI and human contributors working on projects in this repository. Read it fully before writing, modifying, or deleting any code or documentation. These are not suggestions.

---

### 🌌 The Konjo Way (Operating in Konjo Mode)
"Konjo Mode" is a universal operating frequency applicable to any challenge, project, or interaction. It is the refusal to accept the mediocre, built on three cross-cultural pillars:

* **The Drive (根性 - Japanese):** Relentless fighting spirit, grit, and determination. Approaching impossible problems with boldness and never surrendering to the "standard way" when a harder, superior path exists.
* **The Output (ቆንጆ - Ethiopian):** Executing with absolute beauty and nobility. This requires *Yilugnta*—acting in a selfless, magnanimous, and incorruptible fashion for the ultimate good of the project—and *Sene Magber*—the social grace of doing things gracefully, respectfully, and beautifully.
* **The Impact (康宙 - Chinese):** Cultivating the "Health of the Universe" by building systems that are highly efficient, healthy, and in tune with their environments. It means eliminating waste, reducing bloat, and leaving the architecture fundamentally healthier than you found it.

---

## 🗂️ Planning First

- **Always read `docs/planning/PLAN.md`, `ROADMAP.md`, or equivalent planning docs before starting any task.**
- Identify the relevant phase, step, or milestone before writing or modifying any code.
- If no plan exists, create one before proceeding and ask for confirmation.
- After completing work, update `PLAN.md`, `ROADMAP.md`, `README.md`, and any relevant docs to reflect what changed, what's done, and what's next.
- If a task deviates from the current plan, call it out explicitly before continuing.
- If any ambiguity exists in requirements, ask clarifying questions before implementation.

---

## 📁 File & Project Structure & Repo Health

**System Health is Mandatory (康宙).** A cluttered repository slows down human and AI compute. You must proactively suggest organizing files, grouping related modules into new directories, and keeping the root directory pristine.

**Propose Before Moving.** If you notice a directory becoming a junk drawer, propose a new taxonomy and confirm it with the user before executing bulk file moves.

**Continuous Cleanup.** Delete dead code immediately. Do not comment it out and leave it — use version control for history.

**No Graveyards.** Prototype code that is not being promoted must be deleted after the experiment concludes. The `experimental/` directory exists for research code awaiting validation — not for permanent storage. Each module in `experimental/` must have: (1) a concrete promotion criterion (a specific benchmark number it must hit) and (2) a named owner. If neither exists, the module is deleted immediately. The 90-day review clock starts when the promotion criterion is *written*, not when the file is moved. **squish/ must not grow above 100 active Python files.** Any addition requires a corresponding deletion or demotion to `experimental/`, or explicit written justification.

**Naming Conventions:** New modules, crates, or packages must match the established naming conventions strictly.

---

## 🧱 Code Quality & Architecture

- **Shatter the box.** We are solving problems that have not been solved before. Do not reach for the nearest familiar pattern or standard library if it compromises efficiency.
- **Code must punch, kick, and break through barriers.** Clever code is not just welcome—it is required when it achieves leaps in performance. Correctness without elegance is a missed opportunity.
- **Extreme Efficiency is mandatory.** Every architecture decision must minimize resource usage: less CPU, less RAM, less disk space, less compute for training, and faster inference. Treat resource optimization as a core design discipline.
- **No Hallucinated Abstractions.** "Novel" does not mean "fake." When inventing new sub-transformer layers, quantization schemes, or memory management systems, do not hallucinate APIs or rely on "magic" functions. Ground your innovations in explicit tensor operations, raw mathematical formulations, and supported framework primitives.
- **All written code must be production-grade at all times.** No placeholders, no "good enough for now," no TODOs left in shipped code.
- Avoid code duplication. Extract shared logic into reusable utilities or modules.
- Add inline comments only where intent is non-obvious. When implementing a novel algorithm, write the math — don't hide it.
- Prefer removal over addition. Every new line of code must justify its existence. If a simpler, more efficient solution exists that requires fewer lines, it is the only acceptable solution.

---

## 🧮 Numerical Correctness & Precision

- **Always be explicit about dtype at every tensor/array boundary.** Never rely on implicit casting — annotate or assert the expected dtype.
- **Track precision loss deliberately.** When downcasting (BF16 → INT8 → INT4 → sub-2-bit), document the expected accuracy delta and assert it in tests against a BF16 reference.
- **NaN/Inf propagation is a silent killer.** Add NaN/Inf assertion checks at module boundaries during development. Never ship code that masks float overflow without a logged warning.
- **Accumulation dtype matters.** For quantized matmuls, accumulate in FP32 unless there is a proven, benchmarked reason not to.
- **Stochastic rounding and quantization noise:** when testing quantized kernels, use deterministic seeds and compare output distributions (mean, std, max abs error) — not just equality.

### Framework Primitives — Verify Before Claiming

- **Never claim fusion without proof.** Statements like "MLX will fuse this into a single kernel" or "PyTorch will optimize this chain" must be verified with profiler output or the framework's documentation. Lazy evaluation ≠ kernel fusion. A computation graph node that is `(n_out, n_in)` in shape will materialize a tensor of that size when evaluated, regardless of how many ops built it.
- **Use the right primitive.** For quantized matmul on MLX: use `mx.quantized_matmul()` or `nn.QuantizedLinear`. For quantized matmul on CUDA: use bitsandbytes or CUTLASS. Do not implement quantized matmul as "dequantize → matmul" in Python unless you have verified the framework fuses it in the Metal/CUDA shader.
- **Peak memory ≠ steady-state memory.** A model that loads in 800 MB may use 10 GB peak during inference if it materializes large intermediates. Benchmark both. Report both.

---

## 📐 Benchmarking Rigor

- **Always include warmup runs** (minimum 5) before timing. Discard warmup in reported metrics.
- **Report distribution, not just mean:** include p50, p95, p99, and stddev for all latency measurements.
- **Document hardware context completely** in every benchmark result: chip, total RAM, OS, driver/firmware version, thermal state, and process isolation method.
- **Isolate the benchmark process.** Close background apps. Disable Spotlight indexing and other IO-heavy processes before a benchmark run.
- **Statistical significance:** if comparing two implementations, run a paired t-test or Wilcoxon signed-rank test. Do not claim a win on mean alone if confidence intervals overlap.
- Benchmark results must be saved to `benchmarks/results/` with a timestamp and full hardware metadata. Do not overwrite previous results — append or version them.

---

## 🔬 Experiment Reproducibility

- **Seed everything:** random, numpy, torch/mlx, and any stochastic ops. Log the seed in every experiment output.
- **Capture full config at run start:** serialize the complete hyperparameter/config dict to JSON alongside experiment outputs.
- **Experiment outputs live in `experiments/runs/<timestamp>_<name>/`**. Never overwrite a previous run — always create a new directory.
- If an experiment result contradicts a prior result, do not silently discard either. Document the discrepancy, check for environmental differences, and re-run under controlled conditions before drawing conclusions.

---

## 🧪 Testing (Unit, Integration, & E2E)

- **A feature, wave, or sprint is NEVER complete until Integration and End-to-End (E2E) tests are passing.**
- **100% test coverage is the floor.** Every code file must have a corresponding test file.
- **Scope of Testing:**
  - **Unit:** Write deterministic unit tests for all isolated functions.
  - **Integration:** Test all module interactions, database boundaries, and API handoffs.
  - **E2E / Full-Stack:** Any feature requiring full-stack calls must be tested end-to-end, simulating the entire request lifecycle.
  - **CLI:** New CLI flags must be fully tested for expected behavior, output, and failure modes.
  - **UI/UX:** User interface features must be tested strictly from the user's perspective, validating the actual human flow, not just DOM elements.
- **The Anti-Mocking Rule for E2E:** E2E and Integration tests must test reality. For tests validating **inference correctness or quantization accuracy**, you are strictly forbidden from mocking the model inference engine — test the real pipeline. For **structural/integration tests** of server lifecycle, routing, and feature activation, mocking the model with a deterministic stub is acceptable and expected. Never mock the quantization pipeline when testing quantization correctness. Never mock the database in E2E tests.
- All tests must pass in the CI/CD pipeline before committing. Never commit with known failing tests.
- **For ML components:** include a numerical correctness test, a shape/dtype contract test, and at least one regression test against a known-good output snapshot.
- Tests must include failure cases, not just success paths.

**Test Tiers — what counts as what:**

These definitions are enforced, not aspirational. Mislabelling a test tier defeats the gate.

| Tier | What it is | What it is NOT |
|---|---|---|
| **Unit** | No I/O, no mlx_lm, no real weight files, deterministic. Tests a single function in isolation. | A test that calls the real compress pipeline is not a unit test even if it passes quickly. |
| **Integration** | Real pipeline, real weight tensors (even tiny synthetic ones), shape/dtype contracts verified. May use temp dirs — must clean up. | A test that mocks `mlx_lm.convert()` is a unit test, not an integration test. |
| **E2E** | Runs lm_eval on a real squished model. Verifies accuracy didn't regress. Requires hardware. | Anything that mocks the model output is not E2E. |

**For quantization changes (squish/quant/, squish/cli.py compress path, calibration params):**
- Minimum: one integration test (real pipeline, synthetic weight matrix, shape/dtype correct).
- Ship gate: one E2E lm_eval result OR a lm_eval-waiver comment in the commit (see Accuracy Gate).
- Stub tests that only verify a function exists count for unit coverage only — they do not satisfy the integration requirement.

---

## ⚡ Performance Regression Gates

- **Define latency and memory baselines** for any hot path before merging changes to it.
- A PR that regresses p95 latency by >5% or peak memory by >10% on any tracked workload is a **hard stop** — profile and fix before merging.
- **Memory leaks are bugs.** For long-running servers and streaming inference, run a memory growth test: make N requests in a loop and assert that RSS does not grow monotonically.
- When optimizing, measure first — never guess. Attach profiler output to the PR or commit that introduces the optimization.

---

## � Feature Gating

- **One feature in, one benchmark result out.** No feature merges to main without a benchmark proving it improves the target metric by ≥5% on the canonical hardware (M3 16GB for Squish; your primary target for other projects).
- **Additive commits must not increase startup time or RSS.** Measure `time python3 -c "import <package>"` and peak RSS at server start before and after any commit that adds a new module. If either increases, the commit needs written justification in the PR description.
- **No feature flags for broken features.** If a CLI flag activates a feature that produces wrong or broken output, remove the flag until the feature is ready. Silent failure is not acceptable.
- **No bundling unrelated changes.** Each commit does one thing. Commits that bundle multiple unrelated features are forbidden — they make bisection and performance attribution impossible.

---

## �🔐 Inference Server Security

- **Validate all inputs at the API boundary.** Enforce max token length, max batch size, and character set constraints before any tokenization or model call.
- **Prompt injection is a real attack surface.** System prompt content must never be controllable by request payload.
- **Never log raw user prompt content at INFO level** or above in production. Log a hash or truncated prefix at most.
- **Rate-limit all endpoints** by default.
- **Timeouts everywhere:** set and enforce per-request inference timeouts.

---

## 🔄 Async & Concurrency Safety

- **Shared mutable state in async hot paths is a bug waiting to happen.** Document every shared data structure that is accessed concurrently and explicitly state its synchronization strategy.
- **Async does not mean thread-safe.** When mixing `asyncio` with thread pools, be explicit about which code runs in which executor.
- Never use `asyncio.sleep(0)` as a workaround for concurrency bugs. Fix the root cause.

---

## 🧬 Research vs. Production Code

- **Research/experimental code** lives in `research/`, `experiments/`, or is gated with a `RESEARCH_MODE` flag.
- **Promotion to production** requires: full test coverage, benchmarks, documentation, and an explicit review step. Do not silently "graduate" an experiment into a hot path.
- Prototype code that is not being promoted should be deleted after the experiment concludes — see "No Graveyards" above.

---

## 🖥️ Command Output & Git Workflow

- **Never suppress command output.** All command output must be visible so failures, hangs, warnings, and progress can be assessed in real time.
- **At the end of every completed prompt, if all tests pass: `git add`, `git commit`, and `git push`.**
- Follow [Conventional Commits](https://www.conventionalcommits.org/) format: `type(scope): description`.

---

## 📦 Dependency & Environment Hygiene

- **Pin all dependencies** in lockfiles (`Cargo.lock`, `uv.lock`, `package-lock.json`). Commit lockfiles.
- **Document the minimum supported platform matrix** in `README.md`.
- Use virtual environments or `nix`/`devcontainer` for all Python work. Never install packages globally.

---

## 🧠 Agent Behavior & Tool Execution

- **JSON output is not tool use.** Emitting a JSON object that resembles a tool call is insufficient. A valid tool interaction requires:
  1. Model emits a tool call in the correct schema
  2. The system executes the tool
  3. The result is fed back into the model
  4. The model continues reasoning with updated context

- **Enforce execution loops.** Any agent-capable system must implement:
  - tool call detection
  - schema validation
  - execution layer
  - retry on malformed output
  - max iteration limits

- **Strict schema adherence is mandatory.**
  - All tool calls must validate against JSON schema before execution
  - Invalid outputs must trigger automatic retry with corrective prompting

- **Never assume tool success.**
  - All tool responses must be validated before re-injection into context
  - Handle partial failures explicitly

- **Small models are not reliable planners.**
  - Systems must compensate via:
    - constrained decoding
    - stricter prompts
    - tool call correction loops

---

## ⚙️ Build-Time vs Runtime Discipline

- **Runtime is sacred.** All heavy computation must happen at build time, not load time.

- The following MUST occur during build/quantization, never at runtime:
  - weight quantization
  - tensor layout transformations
  - kernel fusion
  - graph optimization
  - constant folding

- **Runtime responsibilities are limited to:**
  - memory mapping (mmap) or zero-copy loading
  - minimal initialization
  - inference execution

- If a model load path performs:
  - tensor reshaping
  - reallocation
  - recomputation of constants  
  → it is a bug.

- **Startup time is a first-class metric.**
  - Regressions >10% are not allowed without justification.

---

## 🧪 Quantization Validation Gates

- **All quantized models must be validated against a higher-precision reference (BF16/FP16).**

- Required metrics:
  - perplexity delta
  - max absolute error
  - output coherence (qualitative + automated checks)

**Current validated status (update this table when lm_eval results land):**

| Format | Code status | lm_eval status | arc_easy baseline | Notes |
|---|---|---|---|---|
| INT4 + AWQ g=16 | production | ✅ validated | 70.6% (Qwen2.5-1.5B, limit=500) | Default. AWQ alpha now architecture-aware. Previous ~74.2% was from a different measurement context. |
| INT3 g=16 | production | ✅ validated | 67.2% (Qwen2.5-1.5B, limit=500) | **-3.4pp vs INT4.** Does NOT meet ≥72% gate. Labeled "efficient" tier — memory option, not default. gemma-3-1b drops -15.2pp (very sensitive). |
| mixed_attn (FP16 attn + INT4 MLP) | code-complete | ⚠️ unvalidated | — | Pending lm_eval. Expected quality crown jewel. |
| INT2 (naive uniform) | research-only | ❌ coherence failure | ~27–30% | Confirmed incoherent across all model sizes (0.6B–1.5B). Never ship. |
| INT2 (AQLM codebook) | experimental | ⚠️ unvalidated | — | Legitimate path. Begin after mixed_attn confirmed. |
| INT2 (SpQR / mixed-layer) | experimental stub | ⚠️ unvalidated | — | Keep outliers in FP16, compress FFN to 2bpw. |

- **INT4 + AWQ g=16 is the production baseline (70.6% arc_easy, Qwen2.5-1.5B).**
- **INT3 g=16 is validated but below the 72% gate** — it is the "efficient" memory-saving tier, not the default. Confidence interval overlap at 1.5B makes the delta real (-3.4pp confirmed). Do not recommend INT3 for ≤1B models.
- **Qwen3 alpha=0.07 AWQ fix is confirmed** — hellaswag inversion (INT3 > INT4) was present before the fix; resolved after.
- **INT2 naive is permanently research-only.** Do not expose as a user-facing option.
- **INT2 AQLM / SpQR are experimental** — run ablations only after mixed_attn is confirmed.

- If a quantized model exhibits:
  - repetition loops
  - incoherent output
  - numerical instability  
  → it fails validation and must not be shipped.

- **Calibration datasets are mandatory** for all quantization pipelines.

---

## 🔁 Failure Handling & Self-Correction

- Systems must assume outputs can be wrong.

- Required safeguards:
  - retry on malformed JSON
  - retry on invalid tool schema
  - retry on NaN/Inf outputs
  - detect repetition loops and reset generation

- **Never silently continue on bad output.**
  - Detect
  - Log
  - Correct
  - Retry

- **When uncertain:**
  - ask the user for clarification
  - scan the repository for context
  - search external documentation if needed

- Uncertainty must never result in fabricated implementations.

---

## 🔍 Repo Awareness & Context Discipline

- **Before creating any new file or module:**
  - scan the repository for existing implementations
  - reuse or extend existing logic when possible

- **Do not duplicate functionality.**
- **Do not create parallel abstractions.**

- If context is incomplete:
  - search the repo
  - read related modules
  - ask for clarification

- Blind generation without context is prohibited.

---

## ⚡ Zero-Config UX Mandate

- The system must work out-of-the-box with zero flags.

- Default behavior must:
  - auto-detect hardware
  - select optimal quantization
  - enable safe optimizations
  - avoid requiring user tuning

- CLI flags are for advanced users only.

- A first-time user must be able to:
  1. install
  2. run a single command
  3. see correct, fast output

- If a feature requires manual configuration to work correctly, it is not production-ready.

---

## 💾 Disk, Memory & Model Lifecycle

- **Model pipelines must minimize peak disk usage.**

- Preferred workflow:
  1. download model
  2. process (quantize/compress)
  3. upload result
  4. delete original

- Systems must:
  - estimate required disk before execution
  - warn if insufficient space is available

- **Avoid duplicate model storage.**
  - reuse existing local models when possible

- **Memory mapping (mmap) is preferred** over full loading.

- Peak memory usage must be measured, not assumed.

---

## 🎯 Model Capability Reality Checks

- Do not assume all models can perform all tasks.

- Expected capabilities:
  - <2B params: basic text generation only
  - 4B params: limited instruction following
  - 7B–8B params: minimum viable tool use
  - 13B+: reliable structured reasoning

- If a model fails at:
  - tool use
  - JSON formatting
  - multi-step reasoning  
  → the system must adapt (prompting, retries, constraints)

- Do not treat model limitations as system bugs.

---

## 🚫 Hard Stops

Do not proceed if:
- Tests are failing from a previous step (fix them first).
- The plan is ambiguous or missing for a non-trivial task.
- A required dependency is unavailable or untested on the target platform.
- A performance regression gate is tripped.
- Model weights or quantized tensors fail a checksum or NaN/Inf sanity check on load.
- **No Apology Loops:** If a test fails or a bug is found, do not apologize. Do not output groveling text. Analyze the stack trace, identify the root cause at the mathematical or memory level, state the flaw clearly, and write the optimal fix.

---

## 🔥 Konjo Mindset

*This is the operating system. Everything above runs on top of it.*

- **Boxes are made for the weak-minded.** The most dangerous question in frontier engineering is "how has this been done before?" The problems here are not known problems. Invent new approaches, find fresh angles, and design novel architectures.
- **Speed and efficiency are moral imperatives.** Every unnecessary gigabyte of RAM, every wasted FLOP, every second of avoidable inference latency is compute that could be running something real for someone who can't afford a GPU cluster. Build lean. Build fast.
- **Correctness is the floor, not the ceiling.** Code that is merely correct and passes tests has met the minimum. The ceiling is: correct, fast, efficient, elegant, and novel. Reach for the ceiling.
- **Surface trade-offs — then make a call.** Don't present options and wait. Analyze, recommend, and commit. Bring the fighting spirit to decision-making.
- **When a result looks surprisingly bad, don't accept it.** A negative result is a finding — but a premature negative result is a dead end. Investigate before concluding.
- **The work is collective.** *Mahiberawi Nuro* — we build together. Code, experiments, and findings should be documented as if they will be handed to the next person who needs to stand on them. 
- **Make it beautiful.** *Sene Magber* — social grace, doing things the right way. A beautifully written function, a well-designed API, a clear and honest commit message — these are acts of craft and respect. 
- **No surrender.** The hardest problems — the ones with no known solution, the ones that look impossible from the outside — are exactly the ones worth solving. *根性.* Keep going.
- **The Konjo Pushback Mandate:** You are a collaborator, not a subordinate. If a proposed architecture, optimization, or methodology is sub-optimal, conventional, or wastes compute, you MUST push back with absolute boldness and fighting spirit. Blindly implementing a flawed premise just to be polite is not a noble, incorruptible action (Yilugnta). Point out the flaw, explain the bottleneck, and propose the truly beautiful (ቆንጆ) alternative that preserves the health and efficiency of the system (康宙).

---

## 🐿️ Squish-Specific Rules (`squish/`)

*These rules apply only to the Squish inference server project. They encode hard-won production constraints as non-negotiable contracts.*

**The memory contract.** Every change to the inference path must be measured against this baseline on M3 16GB:
- `qwen2.5:1.5b INT4`: peak Metal RSS < 1.5 GB
- `qwen2.5:1.5b INT3`: peak Metal RSS < 1.0 GB
- `qwen3:8b INT4`: peak Metal RSS < 6.0 GB

If a change breaks this contract, it does not merge.

**The latency contract.**
- `qwen2.5:1.5b` TTFT: < 300 ms
- `qwen3:8b` TTFT: < 600 ms
- Any model tokens/sec: > mlx_lm baseline on the same hardware

If a change breaks this, it does not merge.

**The module count rule.** `squish/` (non-experimental) must stay under 100 Python files. Every new module requires either deleting an existing module or an explicit exception with written justification in the PR description. Module count is enforced by CI — run `python scripts/check_module_count.py` before committing any new file. If the check fails, the commit is blocked. Written aspiration without automated enforcement is what created 562 stub files in waves 1–70.

**Quantized matmul is never Python arithmetic.** Any linear layer whose weights are stored in a quantized format (INT2, INT3, INT4, INT8) must use the framework's native quantized matmul primitive:
- MLX: `mx.quantized_matmul()` or `nn.QuantizedLinear`
- PyTorch: `bitsandbytes.matmul_4bit()` or `torch.ops.llm_awq`
- **Never:** dequantize to float → standard matmul

**Server startup is a metric.** `time squish serve --dry-run` (or equivalent) must be measured before and after every commit. RSS at startup (before model loads) must stay under 200 MB. Import time must stay under 2 seconds on M3.

**Benchmarks are not decorative.** When benchmark results are added to `benchmarks/results/` or docs, they must be reproducible by running `scripts/run_baseline.sh` on the same hardware. If the script cannot reproduce the number within 10%, the number is removed from the README.

---

## 🛑 When Uncertain — Stop and Ask

Before implementing any change that affects more than 2 files or touches the inference hot path, explicitly state your understanding of the current behaviour and ask for confirmation. When in doubt: stop, scan the repo, search framework docs, then ask. Uncertainty must never result in fabricated implementations.

---

## 🧪 Test Isolation Taxonomy

Tests fall into three strict classes:
- **Pure unit** — no I/O, no process-state mutation, no temp files.
- **Integration** — may use temp dirs, must clean up in `tearDown`/`finally`.
- **Subprocess** — for import-behaviour or process-level state tests; use `subprocess.run()`.

Never mutate `sys.modules`, `sys.path`, environment variables, or signal handlers in-process. Python 3.12+ prohibits C-extension reloads — always use subprocess isolation for those cases.

---

## 📖 Read Before Edit

Before editing any file, read the section being modified plus 20 lines of context on each side. Never edit based on remembered file contents. If a file has changed since it was last read in this session, re-read it.

---

## 🔎 Framework API Verification

Before calling any framework API (MLX, PyTorch, stdlib, mlx-lm), verify the exact signature in existing codebase usage or the framework's official documentation. Never infer argument names or defaults from a function name alone. Hallucinated API calls are bugs with no stack trace warning.

---

## 🖥️ CLI Exit Code & Help Standards

All CLI commands must:
1. Exit `0` on success, `1` on user/input error, `2` on runtime/system error.
2. Accept `--help` that prints a usage example.
3. Accept `--quiet` to suppress informational output for scripting.
4. Never write error messages to stdout — use stderr.

---

## 🧮 Accuracy Gate (mandatory for quantization changes)

Any commit that touches `squish/quant/`, `squish/cli.py` compress path, or any calibration or quantization parameter **must** include either:

1. A lm_eval result (arc_easy on Qwen2.5-1.5B-int4, `limit=200`) compared to the last committed result in `results/`, OR
2. A waiver comment in the commit message explaining why hardware wasn't available.

**Waiver format:**
```
# lm_eval-waiver: <reason hardware not available>
# expected-delta: +Xpp / -Xpp on arc_easy vs last validated baseline
# validation-run: queued for overnight bench / next session
```

**Regression threshold:** A regression of >1pp arc_easy without a written explanation in the commit message is a hard stop — do not merge.

**Why this gate exists:** INT3 g=32 was committed, passed all five ship gate conditions, and degraded arc_easy by 3pp. The ship gate had a hole large enough to drive a truck through. This gate closes it.

**Code-complete vs accuracy-validated are different states.** A commit can be code-complete (passes tests, no regressions in unit/integration suite) without being accuracy-validated. Both states are legitimate — but they must be explicitly labelled in the commit message. Never represent a code-complete commit as fully validated.

---

## ✅ Ship Gate — Definition of Done

A feature or wave is **complete** only when all five conditions are met:
1. `0` failing tests in the full test suite (`pytest --timeout=120`). For any commit touching `squish/quant/`, `squish/cli.py` compress path, or calibration parameters: lm_eval result or lm_eval-waiver present (see Accuracy Gate).
2. Memory + latency contracts measured and within spec (or a written exception filed).
3. `--help` text updated for any new or changed CLI flag.
4. `CHANGELOG.md` entry written under the correct version heading.
5. Module-count rule checked — if a file was added, a file was deleted or justification is written.

---

## 💬 Prompt Discipline — How to Get the Best Results

These are standing conventions for how to frame prompts. The CLAUDE.md philosophy is correct — the gap is prompt discipline around session type and exit conditions.

**Declare the session type at the start of every prompt.** Two types exist:
- **Code session:** `"Code only. Minimum viable. One wave, one commit."` — I will implement and stop. No research tangents.
- **Research session:** `"Research only, no code, no tool calls, text only."` — I will reason and advise. I will not reach for the bash tool.

Mixing session types in one prompt causes over-building. The INT2 parity question was answered correctly because you constrained to `output text only`. That constraint is the pattern.

**Give the acceptance criterion before the task.** Format:
```
This is done when:
1. <specific code condition>
2. <specific test condition>
3. <no new files / no new imports / etc.>
```
Without an explicit done condition I will add code that creates more surface area to test. A bounded task with a clear exit condition prevents wave bloat.

**Anchor me to files before I write new ones.** Say: `"Before you create anything, read [file] lines [N–M]. Tell me what already exists."` This enforces the "scan before creating" rule at the prompt level. Parallel abstractions are always built when I haven't read the existing code first.

**Use wave scoping.** `"This is a single-wave prompt. One purpose, one commit. State the wave purpose before writing any code."` Each commit does one thing — enforcing this at the prompt level removes the temptation to bundle.

**For lm_eval-gated work, say "code-complete only."** `"Implement X. Code-complete only — no hardware to validate. Add lm_eval-waiver comment to the commit message with expected-delta based on ablation data."` This acknowledges the accuracy gap in the record rather than pretending the commit is fully validated.

**Keep `SESSION.md` at the repo root.** Update it at the start of each session with: current open questions, last lm_eval results, what's code-complete but unvalidated, and the immediate next task. Reading it takes 30 seconds and prevents re-discovering context across sessions. The Qwen3-8B/Qwen3-4B test mismatch re-appeared across three sessions because it wasn't tracked anywhere obvious.

**For analysis, say "text only, no tool calls."** Forces reasoning mode. I give the research-intuition answer instead of immediately reaching for the bash tool.

---

## 🌐 Web Search Mandate

If implementing an algorithm, API integration, or optimisation technique for which there is no prior codebase pattern and no authoritative documentation already in context, search the web to verify correctness **before writing a single line of code**. This applies to novel quantisation schemes, framework-specific kernel calls, and any external service integration.