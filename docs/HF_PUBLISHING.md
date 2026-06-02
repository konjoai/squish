# Publishing squished models to Hugging Face

This guide covers publishing pre-compressed (squished) models to the
[`konjoai`](https://huggingface.co/konjoai) Hugging Face organization.

Squishing and uploading are **separate steps**. You compress a model locally
first, then publish the resulting directory with
[`scripts/publish_to_hf.py`](../scripts/publish_to_hf.py). The publish script
never quantizes — it validates, builds a model card, and uploads.

---

## 1. Squish a model locally

Start from a BF16 model directory and compress it to INT4 or INT3. INT3 ships
only for families that hold accuracy (Qwen3); the accuracy gate hard-blocks
families where INT3 collapses (e.g. Gemma-3).

```bash
# INT4 (AWQ calibration on by default)
python -m squish.cli compress ~/models/Qwen2.5-7B-Instruct-bf16 \
  --output ~/models/Qwen2.5-7B-Instruct-int4 \
  --format int4

# INT3 (Qwen3 family only)
python -m squish.cli compress ~/models/Qwen3-8B-bf16 \
  --output ~/models/Qwen3-8B-int3 \
  --format int3
```

The output directory is `mlx_lm`-compatible (`model.safetensors`,
`config.json`, `tokenizer.json`, chat template).

---

## 2. Validate the squished output

The publish script runs a one-token inference automatically before uploading
(via `mlx_lm`; see step 5), so a separate validation pass is usually not
needed. To sanity-check manually, load the directory with `mlx_lm` and
generate a few tokens:

```bash
python -c "import mlx_lm; m,t = mlx_lm.load('$HOME/models/Qwen2.5-7B-Instruct-int4'); print(mlx_lm.generate(m, t, prompt='Hello', max_tokens=20))"
```

If the model loads and generates coherent text, it's ready to publish. A load
error here means the upload would ship a broken model — fix the compression
before continuing.

> Note: `squish run <model>` starts the inference **server** and blocks; it is
> not a one-shot generator unless a daemon is already running. Use the `mlx_lm`
> snippet above (or just let the publish script validate) for a quick check.

---

## 3. Dry-run the upload (default)

The publish script defaults to dry-run: it validates, prints the model card,
lists the files and target URL, and writes **nothing** to Hugging Face.

```bash
python scripts/publish_to_hf.py \
  --local-path ~/models/Qwen2.5-7B-Instruct-int4 \
  --hf-name Qwen2.5-7B-Instruct-squished \
  --source-id Qwen/Qwen2.5-7B-Instruct \
  --quant INT4 \
  --context 128000 \
  --base-license apache-2.0
```

Review the output:
- All expected files are listed with sizes.
- The model card frontmatter and body render correctly.
- Target URL is `https://huggingface.co/konjoai/<hf-name>`.
- License attribution matches the base model.
- The validation inference succeeds (skipped with a warning off Apple Silicon
  or when `mlx_lm` is not installed — it never silently passes a broken model).

---

## 4. Authenticate with Hugging Face

A **write** token scoped to the `konjoai` org is required for live uploads
(dry-run needs no token).

1. Go to <https://huggingface.co/settings/tokens>.
2. Create a token with the **write** role (fine-grained tokens must include
   write access to the `konjoai` org).
3. Make it available to the script one of two ways:

   ```bash
   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
   # or
   huggingface-cli login
   ```

The script reads `HF_TOKEN` first, then falls back to the cached
`huggingface-cli login` token. If neither is present in live mode, it aborts
with instructions.

---

## 5. Publish for real

Add `--no-dry-run` to upload. The script creates `konjoai/<hf-name>` (if it
doesn't exist), writes the generated model card as `README.md`, and uploads the
directory.

```bash
python scripts/publish_to_hf.py \
  --local-path ~/models/Qwen2.5-7B-Instruct-int4 \
  --hf-name Qwen2.5-7B-Instruct-squished \
  --source-id Qwen/Qwen2.5-7B-Instruct \
  --quant INT4 \
  --context 128000 \
  --base-license apache-2.0 \
  --no-dry-run
```

On success it prints the model URL and a reminder to update the org README's
model table.

---

## 6. Recommended first batch

| HF name | Local source | Quant | Notes |
|---|---|---|---|
| `Qwen3-8B-squished` | `~/models/Qwen3-8B-int4` | INT4 | Flagship 8B |
| `Qwen3-8B-squished-int3` | `~/models/Qwen3-8B-int3` | INT3 | The differentiator — Ollama has no INT3 |
| `Qwen2.5-7B-Instruct-squished` | `~/models/Qwen2.5-7B-Instruct-int4` | INT4 | Benchmark reference model |
| `Qwen2.5-1.5B-Instruct-squished` | `~/models/Qwen2.5-1.5B-Instruct-int4` | INT4 | Small; useful as a spec-decode draft model |

Base model ids and licenses for the batch:
- Qwen3-8B → `Qwen/Qwen3-8B`, `apache-2.0`
- Qwen2.5-7B-Instruct → `Qwen/Qwen2.5-7B-Instruct`, `apache-2.0`
- Qwen2.5-1.5B-Instruct → `Qwen/Qwen2.5-1.5B-Instruct`, `apache-2.0`

All four use 128000-token context (`--context 128000`).

---

## 7. Estimated upload time

Disk sizes and rough upload times on consumer broadband. HF uses resumable
multipart uploads, so interrupted transfers can be re-run.

| Model | Disk size | ~50 Mbps up | ~100 Mbps up |
|---|---:|---:|---:|
| Qwen2.5-1.5B-Instruct-squished (INT4) | ~0.82 GB | ~2.5 min | ~1.5 min |
| Qwen3-8B-squished-int3 (INT3) | ~3.8 GB | ~11 min | ~6 min |
| Qwen2.5-7B-Instruct-squished (INT4) | ~4.0 GB | ~12 min | ~6 min |
| Qwen3-8B-squished (INT4) | ~4.3 GB | ~13 min | ~7 min |

Times are upload-only and ignore protocol overhead; budget extra for the
initial hashing pass on large `.safetensors` files.

---

## 8. Update the org README model table (manual for now)

After each publish, add a row to the **Pre-Compressed Models** table in the org
README at <https://huggingface.co/spaces/konjoai/README>:

```markdown
| Qwen3-8B | konjoai/Qwen3-8B-squished | INT4 | 4.3 GB | 128K |
```

The README space lives at `huggingface.co/spaces/konjoai/README`. Edit it via
the HF web UI or clone, edit, and push:

```bash
git clone https://huggingface.co/spaces/konjoai/README
# edit README.md, then
git add README.md && git commit -m "docs: add <model> to model table" && git push
```

Automating this table update from `publish_to_hf.py` is a tracked follow-up.
