#!/usr/bin/env python3
"""
patch_wave37_v4.py  —  Fix IndentationError and add missing init block
"""
import sys
from pathlib import Path

SERVER = Path(__file__).resolve().parent.parent / "squish" / "server.py"


def main():
    src = SERVER.read_text()
    print(f"Read {len(src):,} bytes from {SERVER}")
    changed = False

    # ── Fix 1: Remove misplaced _pd_t0 line (IndentationError source) ─────────
    bad_line = "                _pd_t0 = __import__('time').monotonic() if _pd_disaggregator is not None else 0.0\n"
    if bad_line in src:
        src = src.replace(bad_line, "", 1)
        print("  [FIX1] Removed misplaced _pd_t0 line")
        changed = True
    else:
        print("  [SKIP FIX1] _pd_t0 line not found")

    # ── Fix 2: Remove misplaced total_prefill_ms block ─────────────────────────
    bad_block = (
        "                if _pd_disaggregator is not None:\n"
        "                    try:\n"
        "                        _pd_disaggregator.stats.total_prefill_ms += (\n"
        "                            __import__('time').monotonic() - _pd_t0) * 1000.0\n"
        "                        _pd_disaggregator.stats.total_prompt_tokens += len(input_ids)\n"
        "                        _pd_disaggregator.stats.total_requests += 1\n"
        "                    except Exception:\n"
        "                        pass\n"
    )
    if bad_block in src:
        src = src.replace(bad_block, "", 1)
        print("  [FIX2] Removed misplaced total_prefill_ms block")
        changed = True
    else:
        print("  [SKIP FIX2] total_prefill_ms block not found (may already be removed)")

    # ── Add missing main() init block ─────────────────────────────────────────
    if "from squish.kv.kvtc import KVTCConfig" not in src:
        needle = "# \u2500\u2500 Wave 27: Inference velocity features \u2500"
        if needle not in src:
            needle = "# -- Wave 27: Inference velocity features"
        if needle not in src:
            print("ABORT: Wave 27 heading not found"); sys.exit(1)
        idx = src.index(needle)
        sol = src.rfind("\n", 0, idx) + 1

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
            "            _nl = getattr(getattr(model, 'args', None), 'n_layers', None) \\\n"
            "                or len(getattr(model, 'layers', [])) or 32\n"
            "            _kvtc_manager = KVTCManager(\n"
            "                KVTCConfig(rank=args.kvtc_rank, quant_bits=args.kvtc_bits),\n"
            "                n_layers=_nl)\n"
            "            _kvtc_manager._server_enabled = True\n"
            "            _info('kvtc', f'KVTCManager rank={args.kvtc_rank} bits={args.kvtc_bits}')\n"
            "        except Exception as _e:\n"
            "            _warn('kvtc', f'KVTCManager init failed: {_e}')\n"
            "\n"
            "    if getattr(args, 'chunk_kv', False):\n"
            "        try:\n"
            "            from squish.kv.chunk_kv import ChunkKVConfig, ChunkKVManager\n"
            "            _chunk_kv_manager = ChunkKVManager(\n"
            "                ChunkKVConfig(chunk_size=args.chunk_kv_size,\n"
            "                              budget_ratio=args.chunk_kv_budget))\n"
            "            _info('chunk-kv', 'ChunkKVManager ready')\n"
            "        except Exception as _e:\n"
            "            _warn('chunk-kv', f'ChunkKVManager init failed: {_e}')\n"
            "\n"
            "    if getattr(args, 'ssd_saguaro', False):\n"
            "        try:\n"
            "            from squish.speculative.ssd_saguaro import SSDConfig, SSDSaguaro\n"
            "            _ssd_saguaro = SSDSaguaro(SSDConfig(k_outcomes=4, draft_len=8))\n"
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
            "            _info('metal-flash-attn', 'MetalFlashAttention active')\n"
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
            "            _info('jacobi', f'JacobiDecoder ready n_tokens={args.jacobi_n}')\n"
            "        except Exception as _e:\n"
            "            _warn('jacobi', f'JacobiDecoder init failed: {_e}')\n"
            "\n"
            "    if getattr(args, 'mtp', False):\n"
            "        try:\n"
            "            from squish.speculative.mtp_head import MTPHeadConfig, MultiTokenPredictor\n"
            "            _mtp_predictor = MultiTokenPredictor(MTPHeadConfig(n_heads=args.mtp_heads))\n"
            "            _mtp_predictor._server_enabled = True\n"
            "            _info('mtp', f'MultiTokenPredictor ready n_heads={args.mtp_heads}')\n"
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
            "            _layer_overlap_loader.start(_n2, load_fn=lambda i: {'layer_idx': i})\n"
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
            "            _nk = getattr(_ma, 'num_key_value_heads', _nh) or _nh\n"
            "            _fused_qkv_proj = FusedQKVProjection(\n"
            "                FusedQKVConfig(d_model=_dm, n_heads=_nh, n_kv_heads=_nk, d_head=_dm//_nh))\n"
            "            _fused_qkv_proj._server_enabled = True\n"
            "            _info('fused-qkv', f'FusedQKV ready d_model={_dm} n_heads={_nh}')\n"
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
        src = src[:sol] + init_block + src[sol:]
        print("  [ADD] main() init block inserted")
        changed = True
    else:
        print("  [SKIP] main() init block already present (KVTCConfig import found)")

    if changed:
        SERVER.write_text(src)
        print(f"\nWrote {len(src):,} bytes")

    # Verify
    final = SERVER.read_text()
    checks = [
        ("no _pd_t0 misplacement   ", "_pd_t0" not in final),
        ("no total_prefill_ms bad  ", "_pd_t0" not in final),
        ("syntax OK                ", _syntax_ok(final)),
        ("from squish.kv.kvtc imprt", "from squish.kv.kvtc import KVTCConfig" in final),
        ("KVTCManager in main()    ", "from squish.kv.kvtc import KVTCConfig" in final),
        ("total_generated_tokens   ", "total_generated_tokens" in final),
        ("_jacobi_decoder dispatch ", "_jacobi_decoder is not None" in final),
        ("invalidate_reuse_cache   ", "invalidate_reuse_cache" in final),
        ("spec-stream reset        ", "_speculative_streamer.reset()" in final),
    ]
    print("\n--- Verification ---")
    all_ok = True
    for label, ok in checks:
        print(f"  {'OK  ' if ok else 'FAIL'} {label}")
        all_ok = all_ok and ok

    if all_ok:
        print("\nAll checks passed!")
    else:
        print("\nSome checks FAILED")
        sys.exit(1)


def _syntax_ok(src: str) -> bool:
    import ast
    try:
        ast.parse(src)
        return True
    except SyntaxError as e:
        print(f"  SyntaxError: {e}")
        return False


if __name__ == "__main__":
    main()
