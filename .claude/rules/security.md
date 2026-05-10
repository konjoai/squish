---
paths: ["**/api*", "**/server*", "**/catalog*", "**/serving*"]
---
# Security Rules
- Pre-scan HF models before loading weights — at `squish pull hf:` time
- Prompt injection: system prompt content must never be controllable by request payload
- Never log raw user prompt content at INFO level — log a hash or truncated prefix
- Validate all inputs at API boundaries
- MLX imports must be gated behind platform check — never imported on Linux paths
- Rate-limit all endpoints by default
