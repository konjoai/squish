"""squish.squash.integrations — Platform SDK adapters for Squash attestation.

Each sub-module is a thin adapter that maps a platform's native artifact
or experiment concept to a Squash :class:`~squish.squash.attest.AttestConfig`.

Available adapters
------------------
- :mod:`~squish.squash.integrations.mlflow`      — MLflow run / model artifact
- :mod:`~squish.squash.integrations.wandb`        — W&B artifact attestation
- :mod:`~squish.squash.integrations.huggingface`  — HF Hub model-card SBOM push
- :mod:`~squish.squash.integrations.langchain`    — LangChain callback handler
- :mod:`~squish.squash.integrations.sagemaker`    — AWS SageMaker model package
- :mod:`~squish.squash.integrations.vertex_ai`    — GCP Vertex AI Model Registry
- :mod:`~squish.squash.integrations.kubernetes`   — Kubernetes admission webhook
- :mod:`~squish.squash.integrations.ray`          — Ray Serve deployment decorator
- :mod:`~squish.squash.integrations.azure_devops` — Azure DevOps pipeline task

Each adapter is import-guarded — only the adapters whose platform packages
are installed will load.  The Kubernetes and Ray adapters have no mandatory
extra runtime dependencies; Ray itself is optional (only required at bind-time).
"""
