# AGENTS.md — Konjo AI Project Conventions & Collaboration Guidelines

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

- **INT4 is the production baseline.**
- **INT3 is experimental and must be explicitly labeled as unstable.**
- **INT2 is research-only and must not be exposed as production-ready unless proven stable.**

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

**The module count rule.** `squish/` (non-experimental) must stay under 100 Python files. Every new module requires either deleting an existing module or an explicit exception with written justification in the PR description.

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

## ✅ Ship Gate — Definition of Done

A feature or wave is **complete** only when all five conditions are met:
1. `0` failing tests in the full test suite (`pytest --timeout=120`).
2. Memory + latency contracts measured and within spec (or a written exception filed).
3. `--help` text updated for any new or changed CLI flag.
4. `CHANGELOG.md` entry written under the correct version heading.
5. Module-count rule checked — if a file was added, a file was deleted or justification is written.

---

## 🌐 Web Search Mandate

If implementing an algorithm, API integration, or optimisation technique for which there is no prior codebase pattern and no authoritative documentation already in context, search the web to verify correctness **before writing a single line of code**. This applies to novel quantisation schemes, framework-specific kernel calls, and any external service integration.