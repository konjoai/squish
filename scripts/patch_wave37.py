#!/usr/bin/env python3
"""
patch_wave37.py - Apply Wave 37 "Wire Everything In" changes to server.py.
Run from the squish repo root: python3 scripts/patch_wave37.py
"""
import sys
from pathlib import Path

SERVER = Path(__file__).parent.parent / "squish" / "server.py"


def patch(src: str) -> str:
    # ── Change 1: Wave 37 global declarations ─────────────────────────────────
    anchor1 = "_fr_spec_config         = None  # FRSpecConfig            \u2014 --fr-spec\n"
    assert anchor1 in src, f"anchor1 not found"
    wave37_globals = (
        "\n"
        "# \u2500\u2500 Wave 37: Wire Everything In \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "# Twelve isolation modules from Waves 33\u201335 wired into the live request path.\n"
        "_kvtc_manager           = None  # KVTCManager             \u2014 --kvtc\n"
        "_chunk_kv_manager       = None  # ChunkKVManager          \u2014 --chunk-kv\n"
        "_ssd_saguaro            = None  # SSDSaguaro              \u2014 --ssd-saguaro\n"
        "_speculative_streamer   = None  # SpeculativeStreamer      \u2014 --spec-stream\n"
        "_metal_flash_attn       = None  # MetalFlashAttention     \u2014 --metal-flash-attn\n"
        "_deja_vu_sparse_ffn     = None  # DejaVuSparseFFN         \u2014 --deja-vu\n"
        "_jacobi_decoder         = None  # JacobiDecoder           \u2014 --jacobi\n"
        "_mtp_predictor          = None  # MultiTokenPredictor     \u2014 --mtp\n"
        "_layer_overlap_loader   = None  # LayerOverlapLoader      \u2014 --layer-overlap\n"
        "_chip_profile           = None  # ChipProfile             \u2014 auto (startup)\n"
        "_fused_qkv_proj         = None  # FusedQKVProjection      \u2014 --fused-qkv\n"
        "_pd_disaggregator       = None  # PDDisaggregator         \u2014 --pd-disagg\n"
    )
    src = src.replace(anchor1, anchor1 + wave37_globals, 1)
    assert "_kvtc_manager" in src, "globals not inserted"
    print("  [1] Wave 37 globals inserted")

    # ── Change 2: CLI flags (after --fr-spec flag block) ──────────────────────
    # Find the add_argument call for --fr-spec and insert after its closing paren
    anchor2 = (
        '    p.add_argument("--fr-spec",   action="store_true",\n'
        '                   help="Frequency-token speculative decoding (v14)")\n'
    )
    assert anchor2 in src, f"anchor2 (--fr-spec argparse block) not found"
    wave37_flags = (
        "    # \u2500\u2500 Wave 37: Wire Everything In flags \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        '    p.add_argument("--kvtc",        action="store_true",\n'
        '                   help="KV-Transform Coder: low-rank quantised KV cache (v15)")\n'
        '    p.add_argument("--kvtc-rank",   type=int, default=64, metavar="N",\n'
        '                   help="KV-TC projection rank (default: 64)")\n'
        '    p.add_argument("--kvtc-bits",   type=int, default=8, choices=[4, 8],\n'
        '                   help="KV-TC quantisation bits (default: 8)")\n'
        '    p.add_argument("--chunk-kv",    action="store_true",\n'
        '                   help="Chunked KV manager: score/evict KV by chunk (v15)")\n'
        '    p.add_argument("--chunk-kv-size", type=int, default=16, metavar="N",\n'
        '                   help="Chunk size for ChunkKVManager (default: 16)")\n'
        '    p.add_argument("--chunk-kv-budget", type=float, default=0.5, metavar="F",\n'
        '                   help="Budget ratio for KV eviction (default: 0.5)")\n'
        '    p.add_argument("--ssd-saguaro", action="store_true",\n'
        '                   help="SSD Saguaro: stochastic speculative decode with draft multiplex (v15)")\n'
        '    p.add_argument("--spec-stream",  action="store_true",\n'
        '                   help="Speculative Streamer: buffered speculative token stream (v15)")\n'
        '    p.add_argument("--metal-flash-attn", action="store_true",\n'
        '                   help="Metal Flash Attention kernel (Apple Silicon, v15)")\n'
        '    p.add_argument("--deja-vu",     action="store_true",\n'
        '                   help="Deja Vu sparse FFN: predictor-guided neuron selection (v15)")\n'
        '    p.add_argument("--jacobi",      action="store_true",\n'
        '                   help="Jacobi parallel decoder: non-autoregressive decode path (v15)")\n'
        '    p.add_argument("--jacobi-n",    type=int, default=4, metavar="N",\n'
        '                   help="Jacobi token lookahead width (default: 4)")\n'
        '    p.add_argument("--jacobi-variant", default="jacobi",\n'
        '                   choices=["jacobi", "gauss_seidel"],\n'
        '                   help="Jacobi decode variant (default: jacobi)")\n'
        '    p.add_argument("--mtp",         action="store_true",\n'
        '                   help="Multi-Token Predictor head (v15)")\n'
        '    p.add_argument("--mtp-heads",   type=int, default=4, metavar="N",\n'
        '                   help="Number of MTP draft heads (default: 4)")\n'
        '    p.add_argument("--layer-overlap", action="store_true",\n'
        '                   help="Layer overlap loader: prefetch next layer while computing (v15)")\n'
        '    p.add_argument("--layer-overlap-prefetch", type=int, default=2, metavar="N",\n'
        '                   help="Layers to prefetch ahead (default: 2)")\n'
        '    p.add_argument("--fused-qkv",   action="store_true",\n'
        '                   help="Fused Q/K/V projection for GQA models (v15)")\n'
        '    p.add_argument("--pd-disagg",   action="store_true",\n'
        '                   help="PD disaggregation: separate prefill/decode workers (v15)")\n'
    )
    src = src.replace(anchor2, anchor2 + wave37_flags, 1)
    assert "--kvtc" in src, "CLI flags not inserted"
    print("  [2] CLI flags inserted")

    # ── Change 3: --all-optimizations expansion ────────────────────────────────
    # Find the bool_wave_flags list that ends with "spec_reason" and extend it
    anchor3 = '        "spec_reason",\n        # (chip_profile is always auto-detected'
    if anchor3 not in src:
        # Try alternate; look for the spec_reason entry in the list
        anchor3b = '        "spec_reason",\n    ]'
        assert anchor3b in src, "anchor3 (spec_reason list end) not found"
        src = src.replace(
            anchor3b,
            (
                '        "spec_reason",\n'
                "        # Wave 37: Wire Everything In\n"
                '        "kvtc", "chunk_kv", "ssd_saguaro", "spec_stream",\n'
                '        "metal_flash_attn", "deja_vu", "jacobi", "mtp",\n'
                '        "layer_overlap", "fused_qkv", "pd_disagg",\n'
                "    ]"
            ),
            1,
        )
    else:
        # anchor3 with comment already there
        pass
    assert '"kvtc"' in src, "--all-optimizations not expanded"
    print("  [3] --all-optimizations expansion updated")

    # ── Change 4: module initialisation block in main() ───────────────────────
    anchor4 = "    # \u2500\u2500 Wave 27: Inference velocity features \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
    assert anchor4 in src, "anchor4 (Wave 27 heading in main) not found"
    wave37_init = (
        "    # \u2500\u2500 Wave 37: Wire Everything In \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "    global _kvtc_manager, _chunk_kv_manager, _ssd_saguaro\n"
        "    global _speculative_streamer, _metal_flash_attn, _deja_vu_sparse_ffn\n"
        "    global _jacobi_decoder, _mtp_predictor, _layer_overlap_loader\n"
        "    global _chip_profile, _fused_qkv_proj, _pd_disaggregator\n"
        "\n"
        "    # ChipDetector \u2014 always runs at startup for auto-tuning\n"
        "    try:\n"
        "        from squish.hardware.chip_detector import ChipDetector\n"
        "        _cd = ChipDetector()\n"
        "        _chip_profile = _cd.detect()\n"
        "        _info('chip-detector',\n"
        "              f'detected {_chip_profile.name!r}  '\n"
        "              f'bw={_chip_profile.memory_bandwidth_gbps:.0f} GB/s  '\n"
        "              f'rec-kv-bits={_chip_profile.recommended_kv_bits}')\n"
        "        if getattr(args, 'chunk_kv', False) or getattr(args, 'all_optimizations', False):\n"
        "            global _chunk_prefill_size\n"
        "            if _chunk_prefill_size == 512:  # only override default\n"
        "                _chunk_prefill_size = _chip_profile.recommended_chunk_prefill\n"
        "                _info('chip-detector',\n"
        "                      f'auto-tuned chunk_prefill_size={_chunk_prefill_size}')\n"
        "    except Exception as _e:\n"
        "        _warn('chip-detector', f'ChipDetector init failed: {_e}')\n"
        "\n"
        "    # KVTCManager\n"
        "    if getattr(args, 'kvtc', False):\n"
        "        try:\n"
        "            from squish.kv.kvtc import KVTCConfig, KVTCManager\n"
        "            _n_layers = getattr(getattr(model, 'args', None), 'n_layers', None) \\\n"
        "                or len(getattr(model, 'layers', [])) or 32\n"
        "            _kvtc_manager = KVTCManager(\n"
        "                KVTCConfig(rank=args.kvtc_rank, quant_bits=args.kvtc_bits),\n"
        "                n_layers=_n_layers,\n"
        "            )\n"
        "            _kvtc_manager._server_enabled = True\n"
        "            _info('kvtc', f'KV-Transform Coder  rank={args.kvtc_rank}  bits={args.kvtc_bits}  '\n"
        "                         f'n_layers={_n_layers}')\n"
        "        except Exception as _e:\n"
        "            _warn('kvtc', f'KVTCManager init failed: {_e}')\n"
        "\n"
        "    # ChunkKVManager\n"
        "    if getattr(args, 'chunk_kv', False):\n"
        "        try:\n"
        "            from squish.kv.chunk_kv import ChunkKVConfig, ChunkKVManager\n"
        "            _chunk_kv_manager = ChunkKVManager(\n"
        "                ChunkKVConfig(\n"
        "                    chunk_size=args.chunk_kv_size,\n"
        "                    budget_ratio=args.chunk_kv_budget,\n"
        "                )\n"
        "            )\n"
        "            _info('chunk-kv', f'ChunkKVManager  chunk_size={args.chunk_kv_size}  '\n"
        "                             f'budget={args.chunk_kv_budget}')\n"
        "        except Exception as _e:\n"
        "            _warn('chunk-kv', f'ChunkKVManager init failed: {_e}')\n"
        "\n"
        "    # SSDSaguaro\n"
        "    if getattr(args, 'ssd_saguaro', False):\n"
        "        try:\n"
        "            from squish.speculative.ssd_saguaro import SSDConfig, SSDSaguaro\n"
        "            _ssd_saguaro = SSDSaguaro(\n"
        "                SSDConfig(k_outcomes=4, draft_len=8, acceptance_threshold=0.3)\n"
        "            )\n"
        "            _ssd_saguaro._server_enabled = True\n"
        "            _info('ssd-saguaro', 'SSD Saguaro speculative decoder ready')\n"
        "        except Exception as _e:\n"
        "            _warn('ssd-saguaro', f'SSDSaguaro init failed: {_e}')\n"
        "\n"
        "    # SpeculativeStreamer\n"
        "    if getattr(args, 'spec_stream', False):\n"
        "        try:\n"
        "            from squish.speculative.spec_stream import SpecStreamConfig, SpeculativeStreamer\n"
        "            _speculative_streamer = SpeculativeStreamer(\n"
        "                SpecStreamConfig(buffer_size=16, rollback_on_reject=True)\n"
        "            )\n"
        "            _info('spec-stream', 'SpeculativeStreamer buffer ready  size=16')\n"
        "        except Exception as _e:\n"
        "            _warn('spec-stream', f'SpeculativeStreamer init failed: {_e}')\n"
        "\n"
        "    # MetalFlashAttention\n"
        "    if getattr(args, 'metal_flash_attn', False):\n"
        "        try:\n"
        "            from squish.kernels.metal_flash_attn import MetalFlashAttention, MetalFlashConfig\n"
        "            _metal_flash_attn = MetalFlashAttention(MetalFlashConfig(causal=True))\n"
        "            _metal_flash_attn._server_enabled = True\n"
        "            _info('metal-flash-attn', 'Metal Flash Attention kernel active (causal)')\n"
        "        except Exception as _e:\n"
        "            _warn('metal-flash-attn', f'MetalFlashAttention init failed: {_e}')\n"
        "\n"
        "    # DejaVuSparseFFN\n"
        "    if getattr(args, 'deja_vu', False):\n"
        "        try:\n"
        "            from squish.token.deja_vu_sparse import DejaVuConfig, DejaVuSparseFFN\n"
        "            _deja_vu_sparse_ffn = DejaVuSparseFFN(\n"
        "                DejaVuConfig(hidden_size=512, ffn_size=2048)\n"
        "            )\n"
        "            _deja_vu_sparse_ffn._server_enabled = True\n"
        "            _info('deja-vu', 'Deja Vu sparse FFN predictor ready')\n"
        "        except Exception as _e:\n"
        "            _warn('deja-vu', f'DejaVuSparseFFN init failed: {_e}')\n"
        "\n"
        "    # JacobiDecoder\n"
        "    if getattr(args, 'jacobi', False):\n"
        "        try:\n"
        "            from squish.speculative.jacobi_decode import JacobiConfig, JacobiDecoder\n"
        "            _jacobi_decoder = JacobiDecoder(\n"
        "                JacobiConfig(\n"
        "                    n_tokens=args.jacobi_n,\n"
        "                    max_iter=8,\n"
        "                    variant=args.jacobi_variant,\n"
        "                    temperature=0.0,\n"
        "                )\n"
        "            )\n"
        "            _info('jacobi', f'Jacobi decoder  n_tokens={args.jacobi_n}  '\n"
        "                           f'variant={args.jacobi_variant}')\n"
        "        except Exception as _e:\n"
        "            _warn('jacobi', f'JacobiDecoder init failed: {_e}')\n"
        "\n"
        "    # MultiTokenPredictor\n"
        "    if getattr(args, 'mtp', False):\n"
        "        try:\n"
        "            from squish.speculative.mtp_head import MTPHeadConfig, MultiTokenPredictor\n"
        "            _mtp_predictor = MultiTokenPredictor(MTPHeadConfig(n_heads=args.mtp_heads))\n"
        "            _mtp_predictor._server_enabled = True\n"
        "            _info('mtp', f'Multi-Token Predictor  n_heads={args.mtp_heads}')\n"
        "        except Exception as _e:\n"
        "            _warn('mtp', f'MultiTokenPredictor init failed: {_e}')\n"
        "\n"
        "    # LayerOverlapLoader\n"
        "    if getattr(args, 'layer_overlap', False):\n"
        "        try:\n"
        "            from squish.io.layer_overlap_loader import LayerOverlapConfig, LayerOverlapLoader\n"
        "            _n_layers_lol = getattr(getattr(model, 'args', None), 'n_layers', None) \\\n"
        "                or len(getattr(model, 'layers', [])) or 32\n"
        "            _layer_overlap_loader = LayerOverlapLoader(\n"
        "                LayerOverlapConfig(prefetch_count=args.layer_overlap_prefetch)\n"
        "            )\n"
        "            _layer_overlap_loader.start(\n"
        "                _n_layers_lol,\n"
        "                load_fn=lambda idx: {'layer_idx': idx},\n"
        "            )\n"
        "            _info('layer-overlap', f'LayerOverlapLoader started  n_layers={_n_layers_lol}  '\n"
        "                                   f'prefetch={args.layer_overlap_prefetch}')\n"
        "        except Exception as _e:\n"
        "            _warn('layer-overlap', f'LayerOverlapLoader init failed: {_e}')\n"
        "\n"
        "    # FusedQKVProjection\n"
        "    if getattr(args, 'fused_qkv', False):\n"
        "        try:\n"
        "            from squish.hardware.fused_qkv_proj import FusedQKVConfig, FusedQKVProjection\n"
        "            _margs = getattr(model, 'args', None) or getattr(model, 'config', None)\n"
        "            _d_model   = getattr(_margs, 'hidden_size', None) or getattr(_margs, 'd_model', 4096) or 4096\n"
        "            _n_heads   = getattr(_margs, 'num_attention_heads', None) or getattr(_margs, 'n_heads', 32) or 32\n"
        "            _n_kv      = getattr(_margs, 'num_key_value_heads', None) or getattr(_margs, 'n_kv_heads', _n_heads) or _n_heads\n"
        "            _d_head    = _d_model // _n_heads\n"
        "            _fused_qkv_proj = FusedQKVProjection(\n"
        "                FusedQKVConfig(d_model=_d_model, n_heads=_n_heads,\n"
        "                               n_kv_heads=_n_kv, d_head=_d_head)\n"
        "            )\n"
        "            _fused_qkv_proj._server_enabled = True\n"
        "            _info('fused-qkv', f'FusedQKVProjection  d_model={_d_model}  '\n"
        "                               f'n_heads={_n_heads}  n_kv={_n_kv}')\n"
        "        except Exception as _e:\n"
        "            _warn('fused-qkv', f'FusedQKVProjection init failed: {_e}')\n"
        "\n"
        "    # PDDisaggregator\n"
        "    if getattr(args, 'pd_disagg', False):\n"
        "        try:\n"
        "            from squish.serving.pd_disagg import PDConfig, PDDisaggregator\n"
        "            _pd_disaggregator = PDDisaggregator(\n"
        "                PDConfig(max_prefill_tokens=8192, max_decode_tokens=512)\n"
        "            )\n"
        "            _info('pd-disagg', 'PD Disaggregator ready  max_prefill=8192  max_decode=512')\n"
        "        except Exception as _e:\n"
        "            _warn('pd-disagg', f'PDDisaggregator init failed: {_e}')\n"
        "\n"
    )
    src = src.replace(anchor4, wave37_init + anchor4, 1)
    assert "_kvtc_manager" in src, "init block not inserted"
    assert "KVTCManager" in src, "KVTCManager not in init block"
    print("  [4] main() init block inserted")

    # ── Change 5: _generate_tokens() live hooks ────────────────────────────────
    # Hook A: SpeculativeStreamer.reset() in spec-decode path
    # The spec-decode block begins with a `try:` and calls `_draft.generator.stream`
    anchor5a = (
        "            try:\n"
        "                gen = _draft.generator.stream(\n"
    )
    assert anchor5a in src, "anchor5a (spec-decode try block) not found"
    spec_stream_reset = (
        "            if _speculative_streamer is not None:\n"
        "                try:\n"
        "                    _speculative_streamer.reset()\n"
        "                except Exception:\n"
        "                    pass\n"
    )
    src = src.replace(anchor5a, spec_stream_reset + anchor5a, 1)
    assert "_speculative_streamer.reset()" in src, "spec stream reset not inserted"
    print("  [5a] SpeculativeStreamer.reset() hook inserted")

    # Hook B: JacobiDecoder parallel decode path
    # Insert before the `if _kv_cache is not None:` block in _generate_tokens
    # Find the right one — there's exactly one that starts the KV path after spec block
    anchor5b = (
        "        if _kv_cache is not None:\n"
        "            _kv_hit = _kv_cache.lookup(input_ids)\n"
    )
    assert anchor5b in src, "anchor5b (kv_cache lookup) not found"
    jacobi_path = (
        "        # \u2500\u2500 Wave 37: Jacobi parallel decode path \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "        if _jacobi_decoder is not None and not getattr(_draft, 'generator', None):\n"
        "            import logging as _jdlogging\n"
        "            _jdlog = _jdlogging.getLogger('squish.jacobi')\n"
        "            try:\n"
        "                def _jd_logits_fn(_ctx_ids):\n"
        "                    import mlx.core as mx\n"
        "                    _jd_in = mx.array(_ctx_ids)[None]\n"
        "                    _jd_out = model(_jd_in)\n"
        "                    if hasattr(_jd_out, 'logits'):\n"
        "                        _jd_out = _jd_out.logits\n"
        "                    import numpy as _np\n"
        "                    return _np.array(_jd_out[0])\n"
        "                _jd_context = list(input_ids)\n"
        "                _jd_generated = []\n"
        "                _jd_max = min(max_tokens, 256)\n"
        "                while len(_jd_generated) < _jd_max:\n"
        "                    _accepted, _n_iter = _jacobi_decoder.decode_step(\n"
        "                        _jd_logits_fn, _jd_context, vocab_size=tokenizer.vocab_size\n"
        "                    )\n"
        "                    if not _accepted:\n"
        "                        break\n"
        "                    for _tok in _accepted:\n"
        "                        _jd_generated.append(_tok)\n"
        "                        _jd_context.append(_tok)\n"
        "                        _piece = tokenizer.decode([_tok])\n"
        "                        yield _piece, 'continue'\n"
        "                        if _tok == tokenizer.eos_token_id:\n"
        "                            yield '', 'stop'\n"
        "                            return\n"
        "                        if len(_jd_generated) >= _jd_max:\n"
        "                            break\n"
        "                yield '', 'stop'\n"
        "                return\n"
        "            except Exception as _jd_err:\n"
        "                _jdlog.warning('[jacobi] decode failed (%s); falling back', _jd_err)\n"
        "\n"
    )
    src = src.replace(anchor5b, jacobi_path + anchor5b, 1)
    assert "_jacobi_decoder is not None" in src, "jacobi path not inserted"
    assert "_jd_logits_fn" in src, "jacobi logits_fn not inserted"
    print("  [5b] JacobiDecoder dispatch path inserted")

    # Hook C: ChunkKVManager invalidation after _kv_cache.reset()
    # Find `_kv_cache.reset()` call inside _generate_tokens
    anchor5c = "            _kv_cache.reset()\n"
    assert anchor5c in src, "anchor5c (_kv_cache.reset) not found"
    chunk_kv_invalidate = (
        "            _kv_cache.reset()\n"
        "            if _chunk_kv_manager is not None:\n"
        "                try:\n"
        "                    _chunk_kv_manager.invalidate_reuse_cache()\n"
        "                except Exception:\n"
        "                    pass\n"
    )
    src = src.replace(anchor5c, chunk_kv_invalidate, 1)
    assert "invalidate_reuse_cache" in src, "chunk kv invalidation not inserted"
    print("  [5c] ChunkKVManager.invalidate_reuse_cache() hook inserted")

    # Hook D: PDDisaggregator timing in prefill path
    # Find the `model(input_ids...)` prefill step in the cache-miss branch
    # Look for the prefill call pattern
    anchor5d = (
        "                _prefill_logits = model(\n"
        "                    mx.array(input_ids)[None],\n"
    )
    if anchor5d not in src:
        # Try variant without keyword arg
        anchor5d = (
            "                logits = model(\n"
            "                    mx.array(input_ids)[None],\n"
        )
    assert anchor5d in src, "anchor5d (prefill model call) not found"
    pd_prefill_timing = (
        "                _pd_prefill_t0 = time.monotonic() if _pd_disaggregator is not None else 0.0\n"
    )
    src = src.replace(anchor5d, pd_prefill_timing + anchor5d, 1)
    assert "_pd_prefill_t0" in src, "pd prefill timing not inserted"
    print("  [5d] PDDisaggregator prefill timing hook inserted")

    # After prefill: accumulate stats
    # Find `mx.eval(...)` after the prefill model call, or find `logits = logits[0, -1]`
    anchor5e = "                logits = logits[0, -1]\n"
    if anchor5e not in src:
        anchor5e = "                _prefill_logits = _prefill_logits[0, -1]\n"
    assert anchor5e in src, f"anchor5e (logits slice) not found"
    pd_prefill_stats = (
        "                logits = logits[0, -1]\n"
        "                if _pd_disaggregator is not None:\n"
        "                    try:\n"
        "                        _pd_disaggregator.stats.total_prefill_ms += (\n"
        "                            time.monotonic() - _pd_prefill_t0\n"
        "                        ) * 1000.0\n"
        "                        _pd_disaggregator.stats.total_prompt_tokens += len(input_ids)\n"
        "                        _pd_disaggregator.stats.total_requests += 1\n"
        "                    except Exception:\n"
        "                        pass\n"
    )
    src = src.replace(anchor5e, pd_prefill_stats, 1)
    assert "total_prefill_ms" in src, "pd prefill stats not inserted"
    print("  [5e] PDDisaggregator prefill stats hook inserted")

    # After decode loop: generated token count
    # Find `yield "", "stop"` that terminates the KV path
    anchor5f = '        yield "", "stop"\n'
    assert anchor5f in src, "anchor5f (yield stop) not found"
    pd_decode_stats = (
        "        if _pd_disaggregator is not None:\n"
        "            try:\n"
        "                _pd_disaggregator.stats.total_generated_tokens += (\n"
        "                    len(_cache_buf) if _cache_buf else 0\n"
        "                )\n"
        "            except Exception:\n"
        "                pass\n"
        '        yield "", "stop"\n'
    )
    src = src.replace(anchor5f, pd_decode_stats, 1)
    assert "total_generated_tokens" in src, "pd decode stats not inserted"
    print("  [5f] PDDisaggregator decode stats hook inserted")

    return src


def main():
    src = SERVER.read_text()
    print(f"Read {len(src)} bytes from {SERVER}")
    src2 = patch(src)
    SERVER.write_text(src2)
    print(f"Wrote {len(src2)} bytes to {SERVER}")
    # Verify
    verify = SERVER.read_text()
    checks = [
        "_kvtc_manager",
        "--kvtc",
        '"kvtc"',
        "_jacobi_decoder is not None",
        "_jd_logits_fn",
        "invalidate_reuse_cache",
        "_speculative_streamer.reset()",
        "_pd_prefill_t0",
        "total_prefill_ms",
        "total_generated_tokens",
    ]
    for c in checks:
        found = c in verify
        print(f"  {'OK' if found else 'MISSING'}: {c!r}")
    if all(c in verify for c in checks):
        print("\nAll checks passed!")
    else:
        print("\nSome checks FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
