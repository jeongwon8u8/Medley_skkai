"""Environment-only preflight must not load models, download, or expose tokens."""
from pathlib import Path
import json
import sys

import pytest
import torch

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from multimodal.config import CONFIG_DIR, load_multimodal_config  # noqa: E402
from multimodal.llm_backbone import BackboneError  # noqa: E402
from multimodal.llm_preflight import collect_environment, estimate_parameter_storage, main, run_preflight  # noqa: E402


def test_default_preflight_never_accesses_model_or_tokens(monkeypatch, tmp_path):
    import multimodal.llm_preflight as module
    def prohibited(*args, **kwargs): raise AssertionError("Default preflight must not access model files/network")
    monkeypatch.setattr(module, "load_llm_backbone", prohibited)
    monkeypatch.setattr(module, "inspect_pretrained_config", prohibited)
    monkeypatch.setenv("HF_TOKEN", "synthetic-secret-do-not-report")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = load_multimodal_config(CONFIG_DIR / "fusion_b.json", [f"llm.cache_dir={tmp_path / 'absent-cache'}"])
    report = run_preflight(cfg)
    assert report["environment"]["gpu"] is None
    assert report["pretrained_load"] == report["quantized_gpu_forward"] == "not_run"
    assert report["peak_allocated_bytes"] is None
    assert "synthetic-secret" not in json.dumps(report)
    assert not (tmp_path / "absent-cache").exists()


def test_weight_estimates_are_not_fit_claims():
    report = estimate_parameter_storage(4_000_000_000)
    assert report["float16_bfloat16_bytes"] == 8_000_000_000
    assert report["nf4_packed_lower_bound_bytes"] == 2_000_000_000
    assert not report["fit_in_24gb_verified"] and "estimate" in report["kind"]
    with pytest.raises(BackboneError): estimate_parameter_storage(-1)


def test_cli_default_environment_only(capsys):
    assert main(["--config", str(CONFIG_DIR / "fusion_b.json")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["model_metadata"] == "not_inspected" and report["requested"]["model_id"] is None
    with pytest.raises(SystemExit):
        main(["--config", str(CONFIG_DIR / "fusion_b.json"), "--allow-download"])


def test_review_credential_patterns_in_cache_path_are_redacted(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    fake_token = "hf_" + "x" * 32
    report = collect_environment(str(tmp_path / fake_token / "cache"))
    assert fake_token not in json.dumps(report)
    assert report["disk_free_bytes"] > 0


def test_review_credential_patterns_in_model_path_are_redacted(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    fake_token = "hf_" + "x" * 32
    cfg = load_multimodal_config(CONFIG_DIR / "fusion_b.json", [f"llm.model_id={tmp_path / fake_token / 'model'}"])
    report = run_preflight(cfg)
    assert fake_token not in json.dumps(report)
    assert report["pretrained_load"] == "not_run"


def test_review_cli_error_paths_are_redacted(capsys, tmp_path):
    fake_token = "hf_" + "x" * 32
    with pytest.raises(SystemExit) as error:
        main(["--config", str(tmp_path / fake_token / "missing.json")])
    assert error.value.code == 2
    assert fake_token not in capsys.readouterr().err
