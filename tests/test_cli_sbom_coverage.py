"""Coverage for squish.cli cmd_sbom — CycloneDX ML-BOM sidecar inspect/verify/
bind/sign. squash-ai is optional and not installed, so fake squash modules are
injected for the action paths. Host-agnostic.
"""

from __future__ import annotations

import argparse
import json
import sys
import types

import pytest

from squish import cli


def _ns(**kw):
    kw.setdefault("result", None)
    kw.setdefault("baseline", None)
    return argparse.Namespace(**kw)


def _write_bom(path, *, hashes=True, metrics=None):
    comp = {
        "name": "qwen3-8b",
        "modelCard": {
            "modelParameters": {"quantizationLevel": "int4"},
            "quantitativeAnalysis": {"performanceMetrics": metrics or []},
        },
    }
    if hashes:
        comp["hashes"] = [{"content": "a" * 64}]
    path.write_text(json.dumps({"serialNumber": "urn:uuid:1234", "components": [comp]}))


@pytest.fixture
def fake_squash(monkeypatch):
    sb = types.ModuleType("squash.sbom_builder")
    osign = types.ModuleType("squash.oms_signer")

    class CycloneDXBuilder:
        @staticmethod
        def _hash_weight_files(d):
            return {"w.safetensors": "h"}

        @staticmethod
        def _composite_hash(files):
            return "a" * 64  # matches the stored hash by default

    class EvalBinder:
        @staticmethod
        def bind(bom_path, result_path, baseline_path):
            data = json.loads(bom_path.read_text())
            data["components"][0]["modelCard"]["quantitativeAnalysis"]["performanceMetrics"] = [
                {"type": "arc_easy", "value": 71}
            ]
            bom_path.write_text(json.dumps(data))

    class OmsSigner:
        result = "sig.json"

        @staticmethod
        def sign(bom_path):
            return OmsSigner.result

    sb.CycloneDXBuilder = CycloneDXBuilder
    sb.EvalBinder = EvalBinder
    osign.OmsSigner = OmsSigner
    monkeypatch.setitem(sys.modules, "squash", types.ModuleType("squash"))
    monkeypatch.setitem(sys.modules, "squash.sbom_builder", sb)
    monkeypatch.setitem(sys.modules, "squash.oms_signer", osign)
    return types.SimpleNamespace(builder=CycloneDXBuilder, signer=OmsSigner)


# ── squash not installed ─────────────────────────────────────────────────────


def test_sbom_squash_not_installed(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "squash.sbom_builder", None)  # → ImportError
    with pytest.raises(SystemExit) as exc:
        cli.cmd_sbom(_ns(model_dir=str(tmp_path), sbom_action="show"))
    assert exc.value.code == 1


# ── show ─────────────────────────────────────────────────────────────────────


def test_sbom_show_with_metrics(fake_squash, tmp_path, capsys):
    _write_bom(
        tmp_path / "cyclonedx-mlbom.json",
        metrics=[{"type": "arc_easy", "value": 71, "deltaFromBaseline": "-0.4"}],
    )
    cli.cmd_sbom(_ns(model_dir=str(tmp_path), sbom_action="show"))
    out = capsys.readouterr().out
    assert "qwen3-8b" in out and "arc_easy" in out


def test_sbom_show_no_metrics_no_hashes(fake_squash, tmp_path, capsys):
    _write_bom(tmp_path / "cyclonedx-mlbom.json", hashes=False, metrics=[])
    cli.cmd_sbom(_ns(model_dir=str(tmp_path), sbom_action="show"))
    assert "no performance metrics bound" in capsys.readouterr().out


def test_sbom_show_missing_sidecar(fake_squash, tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.cmd_sbom(_ns(model_dir=str(tmp_path), sbom_action="show"))
    assert exc.value.code == 1


# ── verify ───────────────────────────────────────────────────────────────────


def test_sbom_verify_match(fake_squash, tmp_path, capsys):
    _write_bom(tmp_path / "cyclonedx-mlbom.json")
    cli.cmd_sbom(_ns(model_dir=str(tmp_path), sbom_action="verify"))
    assert "integrity verified" in capsys.readouterr().out


def test_sbom_verify_mismatch(fake_squash, tmp_path):
    _write_bom(tmp_path / "cyclonedx-mlbom.json")
    fake_squash.builder._composite_hash = staticmethod(lambda f: "b" * 64)  # differ
    with pytest.raises(SystemExit) as exc:
        cli.cmd_sbom(_ns(model_dir=str(tmp_path), sbom_action="verify"))
    assert exc.value.code == 1


def test_sbom_verify_no_hashes(fake_squash, tmp_path):
    _write_bom(tmp_path / "cyclonedx-mlbom.json", hashes=False)
    with pytest.raises(SystemExit) as exc:
        cli.cmd_sbom(_ns(model_dir=str(tmp_path), sbom_action="verify"))
    assert exc.value.code == 1


def test_sbom_verify_missing_sidecar(fake_squash, tmp_path):
    with pytest.raises(SystemExit):
        cli.cmd_sbom(_ns(model_dir=str(tmp_path), sbom_action="verify"))


# ── bind ─────────────────────────────────────────────────────────────────────


def test_sbom_bind_success(fake_squash, tmp_path, capsys):
    _write_bom(tmp_path / "cyclonedx-mlbom.json")
    result = tmp_path / "eval.json"
    result.write_text("{}")
    cli.cmd_sbom(_ns(model_dir=str(tmp_path), sbom_action="bind", result=str(result)))
    assert "bound 1 metric" in capsys.readouterr().out


def test_sbom_bind_with_baseline(fake_squash, tmp_path, capsys):
    _write_bom(tmp_path / "cyclonedx-mlbom.json")
    result = tmp_path / "eval.json"
    result.write_text("{}")
    base = tmp_path / "base.json"
    base.write_text("{}")
    cli.cmd_sbom(
        _ns(model_dir=str(tmp_path), sbom_action="bind", result=str(result), baseline=str(base))
    )
    assert "bound" in capsys.readouterr().out


def test_sbom_bind_missing_result_arg(fake_squash, tmp_path):
    _write_bom(tmp_path / "cyclonedx-mlbom.json")
    with pytest.raises(SystemExit) as exc:
        cli.cmd_sbom(_ns(model_dir=str(tmp_path), sbom_action="bind", result=None))
    assert exc.value.code == 1


def test_sbom_bind_result_not_found(fake_squash, tmp_path):
    _write_bom(tmp_path / "cyclonedx-mlbom.json")
    with pytest.raises(SystemExit):
        cli.cmd_sbom(
            _ns(model_dir=str(tmp_path), sbom_action="bind", result=str(tmp_path / "absent.json"))
        )


def test_sbom_bind_missing_sidecar(fake_squash, tmp_path):
    with pytest.raises(SystemExit):
        cli.cmd_sbom(
            _ns(model_dir=str(tmp_path), sbom_action="bind", result=str(tmp_path / "r.json"))
        )


# ── sign ─────────────────────────────────────────────────────────────────────


def test_sbom_sign_success(fake_squash, tmp_path, capsys):
    _write_bom(tmp_path / "cyclonedx-mlbom.json")
    cli.cmd_sbom(_ns(model_dir=str(tmp_path), sbom_action="sign"))
    assert "signed" in capsys.readouterr().out


def test_sbom_sign_no_sigstore(fake_squash, tmp_path, capsys):
    _write_bom(tmp_path / "cyclonedx-mlbom.json")
    fake_squash.signer.result = None  # sign returns None → sigstore not installed
    cli.cmd_sbom(_ns(model_dir=str(tmp_path), sbom_action="sign"))
    assert "sigstore not installed" in capsys.readouterr().out


def test_sbom_sign_missing_sidecar(fake_squash, tmp_path):
    with pytest.raises(SystemExit):
        cli.cmd_sbom(_ns(model_dir=str(tmp_path), sbom_action="sign"))


def test_sbom_unknown_action_is_noop(fake_squash, tmp_path):
    # action matches no branch → falls through to function exit (no error)
    _write_bom(tmp_path / "cyclonedx-mlbom.json")
    cli.cmd_sbom(_ns(model_dir=str(tmp_path), sbom_action="other"))
