"""squish.squash.integrations — Platform SDK adapters for Squash attestation.

Each sub-module is a thin adapter that maps a platform's native artifact
or experiment concept to a Squash :class:`~squish.squash.attest.AttestConfig`.

Available adapters
------------------
- :mod:`~squish.squash.integrations.mlflow`      — MLflow run / model artifact
- :mod:`~squish.squash.integrations.wandb`        — W&B artifact attestation
- :mod:`~squish.squash.integrations.huggingface`  — HF Hub model-card SBOM push
- :mod:`~squish.squash.integrations.langchain`    — LangChain callback handler

Each adapter is import-guarded — only the adapters whose platform packages
are installed will load.
"""
