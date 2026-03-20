#!/usr/bin/env python3
"""Wave 37 patch — writes Wave 37 additions to server.py on disk.

Safe to re-run: each change is guarded by a skip-if-present check.

Usage (from squish repo root):
    python3 scripts/patch_wave37.py
"""
import sys
from pathlib import Path

SERVER = Path(__file__).resolve().parent.parent / "squish" / "server.py"


def _check(msg, cond):
    status = "OK  " if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        sys.exit(1)


def main():
    src = SERVER.read_text()
    print(f"Read {len(src):,} bytes from {SERVER}")
    changed = False

    # ── 1. Global declarations ─────────────────────────────────────────────────
    if "_kvtc_manager" not in src:
        anchor = "_fr_spec_config         = None  # FRSpecConfig"
        assert anchor in src, f"anchor for globals not found in {SERVER}"
        # Find line end of that anchor
        idx = src.index(anchor)
        eol = src.index("\n", idx)
        insert_at = eol + 1  # after the newline

        globals_block = (
            "\n"
            "# Wave 37: Wire Everything In\n"
            "_kvtc_manager           = None  # KVTCManager\n"
            "_chunk_kv_manager       = None  # ChunkKVManager\n"
            "_ssd_saguaro            = None  # SSDSaguaro\n"
            "_speculative_streamer   = None  # SpeculativeStreamer\n"
            "_metal_flash_attn       = None  # MetalFlashAttention\n"
            "_deja_vu_sparse_ffn     = None  # DejaVuSparseFFN\n"
            "_jacobi_decoder         = None  # JacobiDecoder\n"
            "_mtp_predictor          = None  # MultiTokenPredictor\n"
            "_layer_overlap_loader   = None  # LayerOverlapLoader\n"
            "_chip_profile           = None  # ChipProfile (auto)\n"
            "_fused_qkv_proj         = None  # FusedQKVProjection\n"
            "_pd_disaggregator       = None  # PDDisaggregator\n"
        )
        src = src[:insert_at] + globals_block + src[insert_at:]
        _check("globals inserted", "_kvtc_manager" in src)
        changed = True
    else:
        print("  [SKIP] globals already present")

    # ── 2. CLI flags ───────────────────────────────────────────────────────────
    if '"--kvtc"' not in src and "'--kvtc'" not in src:
        # Find the --fr-spec add_argument line and insert after it
        anchor = 'ap.add_argument("--fr-spec"'
        assert anchor in src, f"anchor for CLI flags not found: {anchor!r}"
        idx = src.index(anchor)
        # Find end of this argument block (closing paren followed by newline)
        end = src.index(")\n", idx)
        insert_at = end + 2  # after ")\n"

        cli_block = (
            '    # Wave 37: Wire Everything In\n'
            '    ap.add_argument("--kvtc", action="store_true", default=False,\n'
            '                    help="KV-Transform Coder: low-rank quantised KV cache (v15)")\n'
            '    ap.add_argument("--kvtc-rank", type=int, default=64, metavar="N",\n'
            '                    help="KV-TC projection rank (default: 64)")\n'
            '    ap.add_argument("--kvtc-bits", type=int, default=8, choices=[4, 8],\n'
            '                    help="KV-TC quantisation bits (default: 8)")\n'
            '    ap.add_argument("--chunk-kv", action="store_true", default=False,\n'
            '                    help="ChunkKV: chunk-level KV eviction (v15)")\n'
            '    ap.add_argument("--chunk-kv-size", type=int, default=16, metavar="N",\n'
            '                    help="Tokens per KV chunk (default 16)")\n'
            '    ap.add_argument("--chunk-kv-budget", type=float, default=0.5, metavar="F",\n'
            '                    help="KV budget fraction to retain (default 0.5)")\n'
            '    ap.add_argument("--ssd-saguaro", action="store_true", default=False,\n'
            '                    help="SSD-Saguaro speculative decode with outcome prefetch (v15)")\n'
            '    ap.add_argument("--spec-stream", action="store_true", default=False,\n'
            '                    help="SpeculativeStreamer: buffered speculative token stream (v15)")\n'
            '    ap.add_argument("--metal-flash-attn", action="store_true", default=False,\n'
            '                    help="Metal Flash Attention kernel (Apple Silicon, v15)")\n'
            '    ap.add_argument("--deja-vu", action="store_true", default=False,\n'
            '                    help="Deja Vu sparse FFN predictor (v15)")\n'
            '    ap.add_argument("--jacobi", action="store_true", default=False,\n'
            '                    help="Jacobi parallel decoder (v15)")\n'
            '    ap.add_argument("--jacobi-n", type=int, default=4, metavar="N",\n'
            '                    help="Jacobi token lookahead width (default: 4)")\n'
            '    ap.add_argument("--jacobi-variant", default="jacobi",\n'
            '                    choices=["jacobi", "gauss_seidel"],\n'
            '                    help="Jacobi decode variant (default: jacobi)")\n'
            '    ap.add_argument("--mtp", action="store_true", default=False,\n'
            '                    help="Multi-Token Predictor head (v15)")\n'
            '    ap.add_argument("--mtp-heads", type=int, default=4, metavar="N",\n'
            '                    help="Number of MTP draft heads (default: 4)")\n'
            '    ap.add_argument("--layer-overlap", action="store_true", default=False,\n'
            '                    help="Layer overlap loader: prefetch next layer (v15)")\n'
            '    ap.add_argument("--layer-overlap-prefetch", type=int, default=2, metavar="N",\n'
            '                    help="Layers to prefetch ahead (default: 2)")\n'
            '    ap.add_argument("--fused-qkv", action="store_true", default=False,\n'
            '                    help="Fused Q/K/V projection for GQA models (v15)")\n'
            '    ap.add_argument("--pd-disagg", action="store_true", default=False,\n'
            '                    help="PD disaggregation: separate prefill/decode (v15)")\n'
        )
        src = src[:insert_at] + cli_block + src[insert_at:]
        _check("CLI flags inserted", '"--kvtc"' in src)
        changed = True
    else:
        print("  [SKIP] CLI flags already present")

    # ── 3. --all-optimizations expansion ──────────────────────────────────────
    if '"kvtc"' not in src:
        anchor = '"spec_reason",'
        assert anchor in src, f"anchor for all-opts expansion not found: {anchor!r}"
        # Find LAST occurrence (inside the list)
        idx = src.rindex(anchor)
        eol = src.index("\n", idx)
        insert_at = eol + 1

        allopt_block = (
            '        # Wave 37: Wire Everything In\n'
            '        "kvtc", "chunk_kv", "ssd_saguaro", "spec_stream",\n'
            '        "metal_flash_attn", "deja_vu", "jacobi", "mtp",\n'
            '        "layer_overlap", "fused_qkv", "pd_disagg",\n'
        )
        src = src[:insert_at] + allopt_block + src[insert_at:]
        _check("--all-opts expansion", '"kvtc"' in src)
        changed = True
    else:
        print("  [SKIP] --all-opts expansion already present")

    # ── 4. main() init block ───────────────────────────────────────────────────
    if "KVTCManager" not in src:
        anchor = "# -- Wave 27: Inference velocity features"
        if anchor not in src:
            anchor = "# \u2500\u2500 Wave 27: Inference velocity features"
        assert anchor in src, "anchor for Wave 27 heading in main() not found"
        idx = src.index(anchor)

        init_block = (
            "    # Wave 37: Wire Everything In\n"
            "    global _kvtc_manager, _chunk_kv_manager, _ssd_saguaro\n"
            "    global _speculative_streamer, _metal_flash_attn, _deja_vu_sparse_ffn\n"
            "    global _jacobi_decoder, _mtp_predictor, _layer_overlap_loader\n"
            "    global _chip_profile, _fused_qkv_proj, _pd_disaggregator\n"
            "\n"
            "    try:\n"
            "        from squish.hardware.chip_detector import ChipDetector\n"
            "        _chip_profile = ChipDetector().detect()\n"
            "        _info('chip-detector',\n"
            "              f'chip={_chip_profile.name!r} bw={_chip_profile.memory_bandwidth_gbps:.0f} GB/s')\n"
            "    except Exception as _e:\n"
            "        _warn('chip-detector', f'ChipDetector init failed: {_e}')\n"
            "\n"
            "    if getattr(args, 'kvtc', False):\n"
            "        try:\n"
            "            from squish.kv.kvtc import KVTCConfig, KVTCManager\n"
            "            _n = getattr(getattr(model, 'args', None), 'n_layers', None) \\\n"
            "                or len(getattr(model, 'layers', [])) or 32\n"
            "            _kvtc_manager = KVTCManager(\n"
            "                KVTCConfig(rank=args.kvtc_rank, quant_bits=args.kvtc_bits),\n"
            "                n_layers=_n)\n"
            "            _kvtc_manager._server_enabled = True\n"
            "            _info('kvtc', f'KVTCManager rank={args.kvtc_rank} bits={args.kvtc_bits} layers={_n}')\n"
            "        except Exception as _e:\n"
            "            _warn('kvtc', f'KVTCManager init failed: {_e}')\n"
            "\n"
            "    if getattr(args, 'chunk_kv', False):\n"
            "        try:\n"
            "            from squish.kv.chunk_kv import ChunkKVConfig, ChunkKVManager\n"
            "            _chunk_kv_manager = ChunkKVManager(\n"
            "                ChunkKVConfig(chunk_size=args.chunk_kv_size,\n"
            "                              budget_ratio=args.chunk_kv_budget))\n"
            "            _info('chunk-kv', f'ChunkKVManager chunk_size={args.chunk_kv_size} budget={args.chunk_kv_budget}')\n"
            "        except Exception as _e:\n"
            "            _warn('chunk-kv', f'ChunkKVManager init failed: {_e}')\n"
            "\n"
            "    if getattr(args, 'ssd_saguaro', False):\n"
            "        try:\n"
            "            from squish.speculative.ssd_saguaro import SSDConfig, SSDSaguaro\n"
            "            _ssd_saguaro = SSDSaguaro(SSDConfig(k_outcomes=4, draft_len=8, acceptance_threshold=0.3))\n"
            "            _ssd_saguaro._server_enabled = True\n"
            "            _info('ssd-saguaro', 'SSD Saguaro ready')\n"
            "        except Exception as _e:\n"
            "            _warn('ssd-saguaro', f'SSDSaguaro init failed: {_e}')\n"
            "\n"
            "    if getattr(args, 'spec_stream', False):\n"
            "        try:\n"
            "            from squish.speculative.spec_stream import SpecStreamConfig, SpeculativeStreamer\n"
            "            _speculative_streamer = SpeculativeStreamer(\n"
            "                SpecStreamConfig(buffer_size=16, rollback_on_reject=True))\n"
            "            _info('spec-stream', 'SpeculativeStreamer ready')\n"
            "        except Exception as _e:\n"
            "            _warn('spec-stream', f'SpeculativeStreamer init failed: {_e}')\n"
            "\n"
            "    if getattr(args, 'metal_flash_attn', False):\n"
            "        try:\n"
            "            from squish.kernels.metal_flash_attn import MetalFlashAttention, MetalFlashConfig\n"
            "            _metal_flash_attn = MetalFlashAttention(MetalFlashConfig(causal=True))\n"
            "            _metal_flash_attn._server_enabled = True\n"
            "            _info('metal-flash-attn', 'MetalFlashAttention active (causal)')\n"
            "        except Exception as _e:\n"
            "            _warn('metal-flash-attn', f'MetalFlashAttention init failed: {_e}')\n"
            "\n"
            "    if getattr(args, 'deja_vu', False):\n"
            "        try:\n"
            "            from squish.token.deja_vu_sparse import DejaVuConfig, DejaVuSparseFFN\n"
            "            _deja_vu_sparse_ffn = DejaVuSparseFFN(DejaVuConfig(hidden_size=512, ffn_size=2048))\n"
            "            _deja_vu_sparse_ffn._server_enabled = True\n"
            "            _info('deja-vu', 'DejaVu sparse FFN ready')\n"
            "        except Exception as _e:\n"
            "            _warn('deja-vu', f'DejaVuSparseFFN init failed: {_e}')\n"
            "\n"
            "    if getattr(args, 'jacobi', False):\n"
            "        try:\n"
            "            from squish.speculative.jacobi_decode import JacobiConfig, JacobiDecoder\n"
            "            _jacobi_decoder = JacobiDecoder(\n"
            "                JacobiConfig(n_tokens=args.jacobi_n, max_iter=8,\n"
            "                             variant=args.jacobi_variant, temperature=0.0))\n"
            "            _info('jacobi', f'JacobiDecoder n_tokens={args.jacobi_n} variant={args.jacobi_variant}')\n"
            "        except Exception as _e:\n"
            "            _warn('jacobi', f'JacobiDecoder init failed: {_e}')\n"
            "\n"
            "    if getattr(args, 'mtp', False):\n"
            "        try:\n"
            "            from squish.speculative.mtp_head import MTPHeadConfig, MultiTokenPredictor\n"
            "            _mtp_predictor = MultiTokenPredictor(MTPHeadConfig(n_heads=args.mtp_heads))\n"
            "            _mtp_predictor._server_enabled = True\n"
            "            _info('mtp', f'MultiTokenPredictor n_heads={args.mtp_heads}')\n"
            "        except Exception as _e:\n"
            "            _warn('mtp', f'MultiTokenPredictor init failed: {_e}')\n"
            "\n"
            "    if getattr(args, 'layer_overlap', False):\n"
            "        try:\n"
            "            from squish.io.layer_overlap_loader import LayerOverlapConfig, LayerOverlapLoader\n"
            "            _n2 = getattr(getattr(model, 'args', None), 'n_layers', None) \\\n"
            "                or len(getattr(model, 'layers', [])) or 32\n"
            "            _layer_overlap_loader = LayerOverlapLoader(\n"
            "                LayerOverlapConfig(prefetch_count=args.layer_overlap_prefetch))\n"
            "            _layer_overlap_loader.start(_n2, load_fn=lambda idx: {'layer_idx': idx})\n"
            "            _info('layer-overlap', f'LayerOverlapLoader started n={_n2}')\n"
            "        except Exception as _e:\n"
            "            _warn('layer-overlap', f'LayerOverlapLoader init failed: {_e}')\n"
            "\n"
            "    if getattr(args, 'fused_qkv', False):\n"
            "        try:\n"
            "            from squish.hardware.fused_qkv_proj import FusedQKVConfig, FusedQKVProjection\n"
            "            _ma = getattr(model, 'args', None) or getattr(model, 'config', None)\n"
            "            _dm = getattr(_ma, 'hidden_size', None) or getattr(_ma, 'd_model', 4096) or 4096\n"
            "            _nh = getattr(_ma, 'num_attention_heads', None) or getattr(_ma, 'n_heads', 32) or 32\n"
            "            _nk = getattr(_ma, 'num_key_value_heads', None) or getattr(_ma, 'n_kv_heads', _nh) or _nh\n"
            "            _fused_qkv_proj = FusedQKVProjection(\n"
            "                FusedQKVConfig(d_model=_dm, n_heads=_nh, n_kv_heads=_nk, d_head=_dm//_nh))\n"
            "            _fused_qkv_proj._server_enabled = True\n"
            "            _info('fused-qkv', f'FusedQKV d_model={_dm} n_heads={_nh} n_kv={_nk}')\n"
            "        except Exception as _e:\n"
            "            _warn('fused-qkv', f'FusedQKVProjection init failed: {_e}')\n"
            "\n"
            "    if getattr(args, 'pd_disagg', False):\n"
            "        try:\n"
            "            from squish.serving.pd_disagg import PDConfig, PDDisaggregator\n"
            "            _pd_disaggregator = PDDisaggregator(\n"
            "                PDConfig(max_prefill_tokens=8192, max_decode_tokens=512))\n"
            "            _info('pd-disagg', 'PDDisaggregator ready')\n"
            "        except Exception as _e:\n"
            "            _warn('pd-disagg', f'PDDisaggregator init failed: {_e}')\n"
            "\n"
        )
        src = src[:idx] + init_block + src[idx:]
        _check("main() init block", "KVTCManager" in src)
        changed = True
    else:
        print("  [SKIP] main() init block already present")

    # ── 5a. SpeculativeStreamer.reset() in spec-decode path ────────────────────
    if "_speculative_streamer.reset()" not in src:
        anchor = "            try:\n                gen = _draft.generator.stream(\n"
        assert anchor in src, "anchor5a (spec-decode try block) not found"
        reset_hook = (
            "            if _speculative_streamer is not None:\n"
            "                try:\n"
            "                    _speculative_streamer.reset()\n"
            "                except Exception:\n"
            "                    pass\n"
        )
        src = src.replace(anchor, reset_hook + anchor, 1)
        _check("spec-stream reset hook", "_speculative_streamer.reset()" in src)
        changed = True
    else:
        print("  [SKIP] spec-stream reset hook already present")

    # ── 5b. JacobiDecoder parallel decode path ─────────────────────────────────
    if "_jacobi_decoder is not None" not in src:
        anchor = "        if _kv_cache is not None:\n            _kv_hit = _kv_cache.lookup(input_ids)\n"
        assert anchor in src, "anchor5b (kv_cache lookup) not found"
        jacobi_block = (
            "        # Wave 37: Jacobi parallel decode path\n"
            "        if _jacobi_decoder is not None and not getattr(_draft, 'generator', None):\n"
            "            import logging as _jdlog_mod\n"
            "            _jdlog = _jdlog_mod.getLogger('squish.jacobi')\n"
            "            try:\n"
            "                def _jd_logits_fn(_ctx):\n"
            "                    import mlx.core as mx\n"
            "                    import numpy as _np\n"
            "                    _o = model(mx.array(_ctx)[None])\n"
            "                    if hasattr(_o, 'logits'):\n"
            "                        _o = _o.logits\n"
            "                    return _np.array(_o[0])\n"
            "                _jd_ctx = list(input_ids)\n"
            "                _jd_gen = []\n"
            "                _jd_max = min(max_tokens, 256)\n"
            "                while len(_jd_gen) < _jd_max:\n"
            "                    _acc, _ni = _jacobi_decoder.decode_step(\n"
            "                        _jd_logits_fn, _jd_ctx, vocab_size=tokenizer.vocab_size)\n"
            "                    if not _acc:\n"
            "                        break\n"
            "                    for _t in _acc:\n"
            "                        _jd_gen.append(_t)\n"
            "                        _jd_ctx.append(_t)\n"
            "                        yield tokenizer.decode([_t]), 'continue'\n"
            "                        if _t == tokenizer.eos_token_id:\n"
            "                            yield '', 'stop'\n"
            "                            return\n"
            "                        if len(_jd_gen) >= _jd_max:\n"
            "                            break\n"
            "                yield '', 'stop'\n"
            "                return\n"
            "            except Exception as _jd_ex:\n"
            "                _jdlog.warning('[jacobi] failed (%s); fallback', _jd_ex)\n"
            "\n"
        )
        src = src.replace(anchor, jacobi_block + anchor, 1)
        _check("jacobi decode path", "_jacobi_decoder is not None" in src)
        _check("jacobi logits_fn", "_jd_logits_fn" in src)
        changed = True
    else:
        print("  [SKIP] jacobi decode path already present")

    # ── 5c. ChunkKVManager invalidation after _kv_cache.reset() ───────────────
    if "invalidate_reuse_cache" not in src:
        anchor = "            _kv_cache.reset()\n"
        assert anchor in src, "anchor5c (_kv_cache.reset) not found"
        invalidate_block = (
            "            _kv_cache.reset()\n"
            "            if _chunk_kv_manager is not None:\n"
            "                try:\n"
            "                    _chunk_kv_manager.invalidate_reuse_cache()\n"
            "                except Exception:\n"
            "                    pass\n"
        )
        src = src.replace(anchor, invalidate_block, 1)
        _check("chunk-kv invalidation", "invalidate_reuse_cache" in src)
        changed = True
    else:
        print("  [SKIP] chunk-kv invalidation already present")

    # ── 5d/e. PDDisaggregator prefill timing ──────────────────────────────────
    if "total_prefill_ms" not in src:
        anchor = "                logits = logits[0, -1]\n"
        if anchor not in src:
            anchor = "                _prefill_logits = _prefill_logits[0, -1]\n"
        assert anchor in src, "anchor5e (logits slice) not found"
        pd_pre = (
            "                _pd_t0 = __import__('time').monotonic() if _pd_disaggregator is not None else 0.0\n"
        )
        pd_post = (
            "                logits = logits[0, -1]\n"
            "                if _pd_disaggregator is not None:\n"
            "                    try:\n"
            "                        _pd_disaggregator.stats.total_prefill_ms += (\n"
            "                            __import__('time').monotonic() - _pd_t0) * 1000.0\n"
            "                        _pd_disaggregator.stats.total_prompt_tokens += len(input_ids)\n"
            "                        _pd_disaggregator.stats.total_requests += 1\n"
            "                    except Exception:\n"
            "                        pass\n"
        )
        # Insert _pd_t0 before the prefill model call
        # Find model call before the logits slice
        model_call_idx = src.rfind("model(", 0, src.index(anchor))
        # Go to start of that line
        sol = src.rfind("\n", 0, model_call_idx) + 1
        src = src[:sol] + pd_pre + src[sol:]
        # Now find and replace the logits slice line
        src = src.replace(anchor, pd_post, 1)
        _check("pd-disagg prefill timing", "total_prefill_ms" in src)
        changed = True
    else:
        print("  [SKIP] pd-disagg prefill timing already present")

    # ── 5f. PDDisaggregator decode token count ─────────────────────────────────
    if "total_generated_tokens" not in src:
        anchor = '        yield "", "stop"\n'
        assert anchor in src, "anchor5f (yield stop) not found"
        pd_decode = (
            "        if _pd_disaggregator is not None:\n"
            "            try:\n"
            "                _pd_disaggregator.stats.total_generated_tokens += (\n"
            "                    len(_cache_buf) if _cache_buf else 0)\n"
            "            except Exception:\n"
            "                pass\n"
            '        yield "", "stop"\n'
        )
        src = src.replace(anchor, pd_decode, 1)
        _check("pd-disagg decode stats", "total_generated_tokens" in src)
        changed = True
    else:
        print("  [SKIP] pd-disagg decode stats already present")

    if changed:
        SERVER.write_text(src)
        print(f"\nWrote {len(src):,} bytes to {SERVER}")
    else:
        print("\nNo changes needed (all already present)")

    # Final verification
    final = SERVER.read_text()
    checks = [
        ("_kvtc_manager global        ", "_kvtc_manager"),
        ("--kvtc CLI flag             ", '"--kvtc"'),
        ('"kvtc" in all-opts          ', '"kvtc"'),
        ("KVTCManager init in main()  ", "KVTCManager"),
        ("_jacobi_decoder dispatch    ", "_jacobi_decoder is not None"),
        ("_jd_logits_fn defined       ", "_jd_logits_fn"),
        ("invalidate_reuse_cache      ", "invalidate_reuse_cache"),
        ("_speculative_streamer.reset ", "_speculative_streamer.reset()"),
        ("total_prefill_ms            ", "total_prefill_ms"),
        ("total_generated_tokens      ", "total_generated_tokens"),
    ]
    all_ok = True
    print("\n--- Final verification ---")
    for label, needle in checks:
        ok = needle in final
        print(f"  {'OK  ' if ok else 'FAIL'} {label}")
        all_ok = all_ok and ok
    if all_ok:
        print("\nAll checks passed!")
    else:
        print("\nSome checks FAILED — inspect server.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
