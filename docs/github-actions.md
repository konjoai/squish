# GitHub Actions Integration

Wave 84 adds reusable composite GitHub Actions under `.github/actions/` so teams can run model
security checks and governance gates with standard `uses:` steps.

## Reusable Actions

| Action | Purpose | Required Inputs | Key Outputs |
|---|---|---|---|
| `./.github/actions/squash-scan` | Run model security scan (`squash scan`) | `model-path` | `scan-result`, `report-path` |
| `./.github/actions/squash-compress` | Compress model in CI (`squish compress`) | `model-path` | `compression-ratio`, `bom-path` |
| `./.github/actions/squash-attest` | Run policy attestation (`squash attest`) | `model-path` | `passed`, `attestation-path` |

## End-to-End Example

```yaml
name: squash-compliance

on:
  pull_request:
  workflow_dispatch:

jobs:
  compliance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Scan model artifact
        id: scan
        uses: ./.github/actions/squash-scan
        with:
          model-path: models/qwen3-8b
          strict: true

      - name: Compress model artifact
        id: compress
        if: ${{ steps.scan.outputs.scan-result != 'error' }}
        uses: ./.github/actions/squash-compress
        with:
          model-path: models/qwen3-8b
          method: awq
          nbits: 4

      - name: Run attestation
        id: attest
        uses: ./.github/actions/squash-attest
        with:
          model-path: models/qwen3-8b
          policies: enterprise-strict,eu-ai-act,nist-ai-rmf

      - name: Enforce pass gate
        if: ${{ steps.attest.outputs.passed != 'true' }}
        run: |
          echo "Attestation failed: ${{ steps.attest.outputs.attestation-path }}"
          exit 1
```

## Reuse from Another Repository

You can call these actions from external repositories by referencing this repo path and ref:

```yaml
uses: konjoai/squish/.github/actions/squash-attest@main
```

Pin to a commit SHA or release tag in production workflows.
