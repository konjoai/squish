# SQUIZD Binary Format Specification — v1.0

**Status:** Finalised  
**Introduced:** Squish v44 (Wave 70)  
**Semver version:** `1.0.0`

---

## 1. Overview

SQUIZD (`.squizd`) is a self-describing binary format for compressed
transformer model weights produced by the Squish quantisation pipeline.  A
single file may activate one or more of the following feature layers:

| Flag bit | Symbol | Description |
|---|---|---|
| 0 | `ASTC` | ASTC hardware-texture compressed weight blocks |
| 1 | `TCA_TBE` | TCA-TBE lossless bitmap encoding |
| 2 | `INT4` | INT4 weight quantisation |
| 3 | `SPARSE` | Structured FFN sparsity masks |
| 4 | `EAGLE` | Trained EAGLE-3 draft head appendix |
| 5 | `INT2` | INT2 sub-4-bit hybrid weight blocks |
| 6 | `ANE_COREML` | ANE CoreML operator appendix (Wave 69) |

All active features are detected automatically from the header flags bitfield
at load time — no user-level flags are required at serve time.

---

## 2. File Layout

```
┌─────────────────────────────────────────────────────────┐
│  Section                          │  Offset/Size        │
├─────────────────────────────────────────────────────────┤
│  Fixed header                     │  0 – 255 (256 B)    │
│  Layer index table                │  256 – (256 + L×32) │
│  Weight blocks  [L layers]        │  variable           │
│    ┣ per-layer weight block       │  variable           │
│    ┗ sparsity mask block (opt)    │  variable           │
│  Scale / zero-point tables        │  variable           │
│  Draft head appendix  (optional)  │  variable           │
│  CoreML appendix      (optional)  │  variable           │
└─────────────────────────────────────────────────────────┘
```

`L` = number of transformer layers (`layer_count` field in the header).

---

## 3. Fixed Header (256 bytes)

All multi-byte integers are **little-endian**.

```
Offset  Size  Type     Field
──────  ────  ───────  ─────────────────────────────────────────────────────
  0      4    bytes    magic           "SQZD" (0x53 0x51 0x5A 0x44)
  4      2    uint16   version         Format version (currently 1)
  6      4    uint32   flags           Feature flags bitfield (see §1)
 10      2    uint16   layer_count     Number of transformer layers
 12      2    uint16   arch_id         Architecture identifier (0 = generic)
 14      4    uint32   sparsity_crc32  CRC32 of the sparsity metadata block at
                                       header offset 128–159; 0 if SPARSE unset
 18      8    uint64   eagle_hash      FNV-1a-64 hash of the draft head block at
                                       offset 160–191; 0 if EAGLE unset
 26    102    bytes    reserved        Must be zero for v1
128     32    bytes    sparsity_meta   Sparsity metadata (layer mask bitmask etc.)
160     32    bytes    eagle_meta      Draft head metadata (entry point, shape…)
192     64    bytes    reserved2       Reserved for future extensions
```

The header is always exactly `256` bytes.  Future minor versions may use the
`reserved` / `reserved2` bytes; `strict` mode validation will reject
non-zero reserved bytes.

### 3.1 Magic bytes

```
b"SQZD"  →  0x53 0x51 0x5A 0x44
```

A file whose first four bytes do not match is not a SQUIZD file.

### 3.2 Version

Version `1` is the only supported version in this specification.
Validators must reject files with `version < 1` or `version > 1` (until a
future specification revision raises the maximum).

### 3.3 Flags bitfield

Each bit independently enables a feature for the entire file.  Multiple bits
may be set simultaneously.  Bits 7–31 are reserved and must be zero.

Dispatch priority ordering (highest first):

1. `ANE_COREML` (bit 6) — delegates entire inference to CoreML
2. `ASTC` (bit 0) — hardware decode on Apple Silicon GPU
3. `TCA_TBE` (bit 1) — bitmap lossless decode
4. `INT2` (bit 5) — sub-4-bit hybrid blocks
5. `INT4` (bit 2) — standard INT4 Metal GEMV
6. NumPy fallback — always available (CI / non-Apple platforms)

The `SPARSE` (bit 3) and `EAGLE` (bit 4) flags augment the chosen kernel
stack; they do not independently select a path.

---

## 4. Layer Index Table

Immediately follows the 256-byte header.  Contains exactly `layer_count`
entries, each 32 bytes:

```
Offset  Size  Type    Field
──────  ────  ──────  ───────────────────────────────────────
  0      4    uint32  layer_idx       Zero-based layer index
  4      8    uint64  weight_offset   Byte offset to the weight block
 12      8    uint64  weight_length   Byte length of the weight block
 20      8    uint64  scale_offset    Byte offset to the scale/zero table
 28      4    uint32  reserved        Must be zero
```

All offsets are relative to the start of the file.  Layers are stored in
ascending `layer_idx` order.

---

## 5. Weight Block Layouts

The weight block format depends on the active flag bits.  Only one **primary**
weight encoding is used per file (flags priority from §3.3).

### 5.1 ASTC Weight Block (`ASTC` flag)

Each weight tensor is stored as an ASTC-compressed 2-D texture image:

```
[4B magic "ASTW"][4B width][4B height][4B block_footprint]
[NB ASTC bitstream]
```

- `block_footprint`: ASTC block size enum (e.g. `0x05` = 5×5, `0x08` = 8×5)
- The bitstream is standard ASTC as defined in ISO 19495

### 5.2 TCA-TBE Weight Block (`TCA_TBE` flag)

```
[4B magic "TCAW"][4B n_rows][4B word_width_bits][bitmap …]
```

Rows are 64-bit aligned.  The bitmap encodes non-zero positions; values
follow the bitmap in compressed form (see Wave 65 kernel documentation).

### 5.3 INT4 Weight Block (`INT4` flag)

```
[4B magic "INT4"][4B n_rows][4B n_cols]
[ceil(n_rows*n_cols/2) bytes]  ← packed nibbles, row-major
```

Each byte packs two INT4 values: `bits[3:0]` = element at even index,
`bits[7:4]` = element at odd index.  Signed two's complement.

### 5.4 INT2 Hybrid Block (`INT2` flag)

```
[4B magic "INT2"][4B n_rows][4B n_cols][4B sub_block_len]
[ceil(n_rows*n_cols/4) bytes]  ← packed crumbs
```

Two bits per weight, row-major.  Sub-blocks share a shared scale factor
stored in the scale/zero-point table.

### 5.5 NumPy fallback block (no compression flags)

```
[4B magic "NPCL"][4B dtype_code][4B ndim]
[ndim×4B shape][weight_data bytes]
```

`dtype_code`: 0 = float32, 1 = float16, 2 = bfloat16.
Weight data is C-contiguous row-major.

---

## 6. Sparsity Metadata Block (header offset 128–159)

Present and non-zero when `SPARSE` (bit 3) is set.

```
Offset  Size  Type    Field
──────  ────  ──────  ─────────────────────────────────────────
  0      4    uint32  sparsity_version    Currently 1
  4      4    uint32  block_size          FFN block granularity (e.g. 64)
  8      4    uint32  n_sparse_layers     Number of layers with masks
 12      4    float32 mean_sparsity       Mean fraction of zeroed blocks
 16     16    bytes   reserved            Must be zero
```

Per-layer sparsity masks are stored as part of each layer's weight block
(packed bitmaps following the weight data), pointed to by the layer index
table `scale_offset`.

---

## 7. Draft Head Appendix (header offset 160–191, then payload)

Present and non-zero when `EAGLE` (bit 4) is set.

### 7.1 Draft head metadata (header bytes 160–191)

```
Offset  Size  Type    Field
──────  ────  ──────  ───────────────────────────────────────
  0      8    uint64  draft_offset    File offset to draft head weights
  8      8    uint64  draft_length    Byte length of draft head section
 16      4    uint32  n_draft_layers  Number of draft speculative layers
 20      4    uint32  draft_vocab     Draft head vocabulary size
 24      8    bytes   reserved        Must be zero
```

### 7.2 Draft head payload format

```
[4B magic "EGDH"]
[4B n_draft_layers]
[per-layer draft weight blocks — same INT4/NumPy encoding as main model]
```

---

## 8. Scale / Zero-Point Tables

Immediately follows all weight blocks.  One entry per quantised weight matrix:

```
[4B magic "SCZT"]
[4B n_tensors]
[n_tensors × 16B entries]:
  [8B tensor_id: uint64 — hash of (layer_idx, tensor_name)]
  [4B scale: float32]
  [4B zero:  float32]
```

For INT4 and INT2, a second round of per-sub-block scales follows in the
same layout with `tensor_id` high bit set to distinguish them.

---

## 9. CoreML Appendix (`ANE_COREML` flag)

When `ANE_COREML` (bit 6) is set the file ends with a CoreML appendix
produced by the Wave 69 `convert_coreml` pipeline:

```
[4B magic "ANLX"]
[8B section_length]
[CoreML mlpackage archive bytes — zip-encoded]
```

The runtime extracts, caches, and loads the mlpackage via `coremltools`.
The main weight blocks are still present and used as fallback when ANE is
unavailable.

---

## 10. Appendix A — 2-Layer Toy Example

A minimal valid SQUIZD file (2 layers, INT4, no sparsity, no EAGLE):

```
Offset  Bytes (hex)
------  -----------
0       53 51 5A 44          "SQZD"
4       01 00                version = 1
6       04 00 00 00          flags = INT4 (bit 2)
10      02 00                layer_count = 2
12      00 00                arch_id = 0
14      5A 2B 00 00          sparsity_crc32 (computed over zero block = CRC32("\x00"*32))
18      00 00 00 00 00 00 00 00   eagle_hash = 0
26      [102 zero bytes]     reserved
128     [32 zero bytes]      sparsity_meta (all zero; SPARSE unset)
160     [32 zero bytes]      eagle_meta (all zero; EAGLE unset)
192     [64 zero bytes]      reserved2

256     [layer index table — 2 × 32 bytes]

320     [INT4 weight block for layer 0]
???     [INT4 weight block for layer 1]
???     [scale/zero tables]
```

The concrete block sizes depend on the model's hidden dimension.

---

## 11. Changelog

| Version | Change |
|---|---|
| 1.0.0 | Initial published specification (Wave 70 / Squish v44) |
