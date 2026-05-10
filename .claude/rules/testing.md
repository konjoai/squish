---
paths: ["**/test_*.py", "**/tests/**"]
---
# Testing Rules
Every code file needs a corresponding test file. `python -m pytest` must be green.
Platform-specific tests (MLX) must be gated behind platform checks.
Optional-dependency tests must be gated behind env vars or feature flags.
Never mock the core inference logic in E2E tests.
