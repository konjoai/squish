# Git Workflow
- Conventional Commits: `type(scope): description`
- `python -m pytest` must be green before committing
- `ruff check` and `ruff format --check` must be clean
- Never commit with known failing tests
- Quantization accuracy gates must pass before any quant-related commit
