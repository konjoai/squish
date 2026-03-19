# squish-serve Helm Chart

Deploys the [Squish](https://github.com/squishai/squish) private local-inference server on Kubernetes, with optional NVIDIA GPU support, a persistent volume for model weights, and KEDA or native HPA autoscaling.

## Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| Kubernetes | ≥ 1.25 | |
| Helm | ≥ 3.12 | |
| NVIDIA device plugin | ≥ 0.14 | GPU nodes only |
| KEDA | ≥ 2.11 | Optional; for queue-depth autoscaling |

## Quick start

```bash
# 1. Add the OCI registry (or clone the repo)
cd helm/squish-serve

# 2. Install with defaults (GPU, ClusterIP service, 50 Gi PVC)
helm install squish-serve . \
  --namespace squish \
  --create-namespace \
  --set model.id=/models/Qwen2.5-7B-Instruct-bf16

# 3. Port-forward to test locally
kubectl -n squish port-forward svc/squish-serve 8080:8080

# 4. Call the API
curl http://localhost:8080/v1/models
```

## Configuration

See [values.yaml](values.yaml) for the full list of configurable options.

### Common overrides

```bash
# CPU-only (no GPU required)
helm install squish-serve . \
  --set image.flavour=cpu \
  --set resources.limits={} \
  --set resources.requests.cpu=4 \
  --set resources.requests.memory=16Gi \
  --set model.id=/models/Qwen2.5-1.5B-Instruct-bf16

# Expose via LoadBalancer
helm install squish-serve . --set service.type=LoadBalancer

# Enable KEDA autoscaling (requires KEDA operator)
helm install squish-serve . \
  --set autoscaling.enabled=false \
  --set keda.enabled=true \
  --set keda.prometheus.serverAddress=http://prometheus:9090
```

### GPU node scheduling

To pin GPU pods to labelled nodes and tolerate the standard NVIDIA GPU taint:

```yaml
# myvalues.yaml
nodeSelector:
  accelerator: nvidia-tesla-a100

tolerations:
  - key: "nvidia.com/gpu"
    operator: "Exists"
    effect: "NoSchedule"
```

```bash
helm install squish-serve . -f myvalues.yaml
```

### Using an existing model PVC

If you have already provisioned and populated a PVC with model weights:

```bash
helm install squish-serve . \
  --set persistence.enabled=false \
  --set persistence.existingClaim=my-model-pvc
```

## Upgrade

```bash
helm upgrade squish-serve . --namespace squish
```

## Uninstall

```bash
helm uninstall squish-serve --namespace squish
# The model PVC is NOT deleted automatically — delete manually when safe:
kubectl delete pvc squish-serve-models -n squish
```

## Templates

| Template | Kind | Condition |
|----------|------|-----------|
| `deployment.yaml` | Deployment | Always |
| `service.yaml` | Service | Always |
| `configmap.yaml` | ConfigMap | Always |
| `serviceaccount.yaml` | ServiceAccount | `serviceAccount.create=true` |
| `pvc.yaml` | PersistentVolumeClaim | `persistence.enabled=true` and no `existingClaim` |
| `hpa.yaml` | HorizontalPodAutoscaler | `autoscaling.enabled=true` |
| `keda-scaledobject.yaml` | ScaledObject (KEDA) | `keda.enabled=true` |
