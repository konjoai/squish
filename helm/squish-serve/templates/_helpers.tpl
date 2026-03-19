{{/*
Expand the name of the chart.
*/}}
{{- define "squish-serve.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this
(by the DNS naming spec).
*/}}
{{- define "squish-serve.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "squish-serve.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels applied to every resource.
*/}}
{{- define "squish-serve.labels" -}}
helm.sh/chart: {{ include "squish-serve.chart" . }}
{{ include "squish-serve.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels used by Deployment/Service matchLabels.
*/}}
{{- define "squish-serve.selectorLabels" -}}
app.kubernetes.io/name: {{ include "squish-serve.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Service account name to use.
*/}}
{{- define "squish-serve.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "squish-serve.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Resolve the full container image reference, appending the flavour suffix when set.
  repository=ghcr.io/squishai/squish  tag=latest  flavour=cuda  → ghcr.io/squishai/squish:latest-cuda
  repository=ghcr.io/squishai/squish  tag=latest  flavour=""    → ghcr.io/squishai/squish:latest
*/}}
{{- define "squish-serve.image" -}}
{{- $repo := .Values.image.repository -}}
{{- $tag  := .Values.image.tag | default .Chart.AppVersion -}}
{{- $flav := .Values.image.flavour -}}
{{- if $flav -}}
{{- printf "%s:%s-%s" $repo $tag $flav -}}
{{- else -}}
{{- printf "%s:%s" $repo $tag -}}
{{- end -}}
{{- end }}

{{/*
Resolve the PVC name — either an existing claim or the chart-managed one.
*/}}
{{- define "squish-serve.pvcName" -}}
{{- if .Values.persistence.existingClaim -}}
{{- .Values.persistence.existingClaim -}}
{{- else -}}
{{- printf "%s-models" (include "squish-serve.fullname" .) -}}
{{- end -}}
{{- end }}
