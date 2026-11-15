"""Configuration-only tests: no patient data, GPU, model loading or fitting."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from medley.config import load_config as load_medley  # noqa: E402
from multimodal.config import (  # noqa: E402
    CONFIG_DIR, ConfigError, apply_overrides, load_multimodal_config, main,
    recursive_merge, require_model_id, save_resolved_config, validate_config,
    validate_split_selection,
)


def load(name="fusion_b", overrides=()):
    return load_multimodal_config(CONFIG_DIR / f"{name}.json", overrides)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.mark.parametrize("name,fusion", [("base", "soft_token"), ("clinical_only", "none"),
                                        ("fusion_a", "representation"), ("fusion_b", "soft_token")])
def test_shipped_configs(name, fusion):
    config = load(name)
    assert config.settings["fusion"]["type"] == fusion
    assert config.settings["llm"]["tuning"] == "qlora"
    assert config.settings["llm"]["model_id"] is None
    assert config.effective_batch_size == 8
    assert config.clinical_variables == ["age", "sex", "updrs_part_iii", "disease_duration", "ledd"]
    assert config.split_validation == "validated"


def test_recursive_merge_and_copy():
    base = {"nested": {"a": 1, "b": [2]}, "list": [3, 4]}
    merged = recursive_merge(base, {"nested": {"a": 5}, "list": [9]})
    assert merged == {"nested": {"a": 5, "b": [2]}, "list": [9]}
    merged["nested"]["b"].append(8)
    assert base["nested"]["b"] == [2]


def test_relative_inheritance_and_override_precedence(tmp_path):
    base = load().settings
    write_json(tmp_path / "parent.json", base)
    child = write_json(tmp_path / "child.json", {"extends": "parent.json", "training": {"batch_size": 3}})
    config = load_multimodal_config(child, ["training.batch_size=2", "training.batch_size=4"])
    assert config.effective_batch_size == 32
    assert config.settings["training"]["max_epochs"] == 30


def test_type_conversion_and_scerebro():
    config = load(overrides=["eeg.encoder=scerebro", "eeg.feature_dim=3420", "llm.tuning=frozen",
                             "eeg.pooling.num_queries=8", "prediction.clip_range=null",
                             "training.gradient_checkpointing=false", "training.lr_lora=2.0e-5",
                             "evaluation.seeds=[0,2]", "evaluation.folds=[1,3]"])
    assert config.settings["prediction"]["clip_range"] is None
    assert config.settings["training"]["gradient_checkpointing"] is False
    assert config.settings["training"]["lr_lora"] == 0.00002
    assert config.medley["encoder"]["name"] == "scerebro"
    assert config.medley["encoder"]["reference"] == "CZ"
    assert config.settings["evaluation"]["folds"] == [1, 3]


@pytest.mark.parametrize("overrides,match", [
    (["architecture=clinical_only"], "incompatible"),
    (["fusion.type=unknown"], "fusion.type"),
    (["architecture=unknown"], "architecture"),
    (["clinical_set=C3"], "clinical_set"),
    (["task=ctt"], "task"),
    (["dataset=missing"], "dataset"),
    (["eeg.feature_dim=3420"], "feature_dim"),
    (["eeg.feature_dim=true"], "feature_dim"),
    (["eeg.pooling.hidden_dim=255"], "divisible"),
    (["eeg.pooling.num_heads=0"], "num_heads"),
    (["eeg.pooling.num_queries=0"], "num_queries"),
    (["eeg.pooling.type=mean"], "pooling.type"),
    (["eeg.projector.type=mlp"], "projector.type"),
    (["llm.lora_r=0"], "lora_r"),
    (["llm.lora_alpha=-1"], "lora_alpha"),
    (["llm.lora_dropout=1"], "lora_dropout"),
    (["llm.max_text_tokens=-1"], "max_text_tokens"),
    (["llm.compute_dtype=int8"], "compute_dtype"),
    (["llm.quantization=none"], "qlora requires nf4"),
    (["llm.tuning=lora"], "lora requires none"),
    (["llm.model_id=''"], "model_id"),
    (["llm.revision=12"], "revision"),
    (["training.batch_size=0"], "batch_size"),
    (["training.batch_size=1.5"], "batch_size"),
    (["training.gradient_accumulation=false"], "gradient_accumulation"),
    (["training.max_epochs=0"], "max_epochs"),
    (["training.early_stopping_patience=-1"], "patience"),
    (["training.lr_lora=0"], "lr_lora"),
    (["training.lr_new_modules=-1"], "lr_new_modules"),
    (["training.weight_decay=-1"], "weight_decay"),
    (["training.validation_fraction=1"], "validation_fraction"),
    (["training.validation_fraction=0"], "validation_fraction"),
    (["training.gradient_clip_norm=0"], "gradient_clip_norm"),
    (["training.optimizer=sgd"], "optimizer"),
    (["loss.primary=mse"], "loss.primary"),
    (["loss.huber_delta=0"], "huber_delta"),
    (["loss.auxiliary=true"], "auxiliary"),
    (["fusion.numeric_skip=true"], "numeric_skip"),
    (["fusion.uncertainty=true"], "uncertainty"),
    (["fusion.token_position=suffix"], "token_position"),
    (["prediction.clip_range=[30,0]"], "clip_range"),
    (["prediction.clip_range=[0,31]"], "clip_range"),
    (["prediction.clip_range=[false,30]"], "clip_range"),
    (["evaluation.use_existing_splits=false"], "use_existing_splits"),
    (["evaluation.seeds=[]"], "seeds"),
    (["evaluation.seeds=[0,0]"], "seeds"),
    (["evaluation.seeds=[-1]"], "seeds"),
    (["evaluation.seeds=[10]"], "existing Medley seeds"),
    (["evaluation.n_folds=3", "evaluation.folds=[1,2]"], "existing Medley protocol"),
    (["evaluation.folds=[0]"], "folds"),
    (["evaluation.folds=[6]"], "folds"),
    (["experiment_name=../escape"], "experiment_name"),
    (["experiment_name=/absolute"], "experiment_name"),
    (["training.unknown=1"], "Unknown override"),
    (["unknown.nested=1"], "Unknown override"),
    (["not-an-override"], "key=value"),
    (["llm.model_id=https://user:password@example.invalid/model"], "Credential"),
    (["llm.model_id=hf_fake_secret"], "Credential"),
])
def test_invalid_settings(overrides, match):
    with pytest.raises(ConfigError, match=match):
        load(overrides=overrides)


@pytest.mark.parametrize("mode,quant", [("frozen", "none"), ("frozen", "nf4"), ("lora", "none"), ("qlora", "nf4")])
def test_supported_tuning_pairs(mode, quant):
    assert load(overrides=[f"llm.tuning={mode}", f"llm.quantization={quant}"])


@pytest.mark.parametrize("field", ["training.lr_lora", "loss.huber_delta", "fusion.dropout"])
def test_nonfinite_values(field):
    for value in (".nan", ".inf", "-.inf"):
        with pytest.raises(ConfigError, match="finite"):
            load(overrides=[f"{field}={value}"])


def test_required_unknown_and_wrong_containers(tmp_path):
    for field, value, match in [("llm", [], "llm must be an object"), ("extra", 1, "Unknown configuration key")]:
        settings = load().settings
        settings[field] = value
        with pytest.raises(ConfigError, match=match):
            load_multimodal_config(write_json(tmp_path / "bad.json", settings))
    settings = load().settings
    del settings["training"]["batch_size"]
    with pytest.raises(ConfigError, match="Missing required setting: training.batch_size"):
        validate_config(settings)


@pytest.mark.parametrize("text", ['[]', '{', '{"a":1,"a":2}', 'null'])
def test_malformed_json(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(ConfigError):
        load_multimodal_config(path)


def test_inheritance_errors(tmp_path):
    for extends in (None, [], "", "missing.json", "cycle.json"):
        path = write_json(tmp_path / "cycle.json", {"extends": extends})
        with pytest.raises(ConfigError):
            load_multimodal_config(path)


def test_model_id_deferred():
    with pytest.raises(ConfigError, match="concrete llm.model_id"):
        require_model_id(load())
    # An identifier is syntax/configuration only, not a verified HF model.
    assert require_model_id(load(overrides=["llm.model_id=organization/unverified-model"])) == "organization/unverified-model"


def test_json_scientific_override_and_invalid_yaml():
    assert load(overrides=["training.lr_lora=2e-5"]).settings["training"]["lr_lora"] == 0.00002
    with pytest.raises(ConfigError, match="Invalid override value"):
        load(overrides=["evaluation.seeds=[unclosed"])


def test_medley_integration_rejects_leakage_and_dimension_drift(monkeypatch):
    import multimodal.config as module
    original = load().medley
    for mutate, message in [
        (lambda c: c["task"]["clinical_sets"].update(C1=["moca"]), "prohibited"),
        (lambda c: c["encoder"]["channels"].pop(), "feature dimension"),
    ]:
        changed = copy.deepcopy(original)
        mutate(changed)
        monkeypatch.setattr(module, "load_medley_config", lambda *args: changed)
        with pytest.raises(ConfigError, match=message):
            load()


def test_iowa_uses_available_clinical_sets():
    assert load(overrides=["dataset=iowa_rest", "clinical_set=C0"]).clinical_variables == ["age", "sex"]
    with pytest.raises(ConfigError, match="C1 is unavailable"):
        load(overrides=["dataset=iowa_rest"])


def test_existing_medley_loader_unchanged():
    before = load_medley("pd_eeg", "cbramod", "moca")
    config = load(overrides=["training.batch_size=2", "evaluation.seeds=[2]"])
    assert config.medley == before == load_medley("pd_eeg", "cbramod", "moca")
    assert before["evaluation"]["seeds"] == list(range(10))
    legacy_override = load_medley("pd_eeg", "cbramod", "moca", ["evaluation.max_segments=24"])
    assert legacy_override["evaluation"]["max_segments"] == 24
    config.settings["training"]["batch_size"] = 99
    assert load().settings["training"]["batch_size"] == 1


def test_snapshots_are_deterministic_and_roundtrip(tmp_path):
    config = load(overrides=["training.batch_size=2"])
    first = save_resolved_config(config, tmp_path / "one.json")
    second = save_resolved_config(config, tmp_path / "two.json")
    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text()) == config.to_dict()
    restored = load_multimodal_config(first)
    assert restored == config
    assert restored.effective_batch_size == 16
    assert load_multimodal_config(first, ["training.batch_size=3"]).effective_batch_size == 24
    with pytest.raises(ConfigError, match="overwrite requires explicit permission"):
        save_resolved_config(config, first)
    save_resolved_config(config, first, overwrite=True)
    assert first.read_bytes() == second.read_bytes()


def test_snapshot_uses_captured_medley_settings(tmp_path, monkeypatch):
    import multimodal.config as module
    config = load()
    path = save_resolved_config(config, tmp_path / "snapshot.json")
    def unexpected(*args, **kwargs):
        raise AssertionError("Snapshots must not reload changed local settings")
    monkeypatch.setattr(module, "load_medley_config", unexpected)
    assert load_multimodal_config(path) == config


def test_snapshot_security_and_cross_validation(tmp_path):
    config = load()
    for path in (tmp_path / "bad.csv", CONFIG_DIR / "never-write.json", APP / "splits" / "never-write.json"):
        with pytest.raises(ConfigError, match="output"):
            save_resolved_config(config, path, overwrite=True)
        assert not path.exists()
    target = tmp_path / "target.json"
    target.write_text("keep")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ConfigError, match="symbolic link"):
        save_resolved_config(config, link, overwrite=True)
    assert target.read_text() == "keep"
    payload = config.to_dict()
    payload["medley"]["encoder"]["api_key"] = "private"
    with pytest.raises(ConfigError, match="Credentials"):
        load_multimodal_config(write_json(tmp_path / "secret.json", payload))
    payload = config.to_dict()
    payload["settings"]["eeg"]["encoder"] = "scerebro"
    payload["settings"]["eeg"]["feature_dim"] = 3420
    with pytest.raises(ConfigError, match="does not match"):
        load_multimodal_config(write_json(tmp_path / "mismatch.json", payload))


def test_split_validation_is_read_only_or_deferred(tmp_path):
    config = load(overrides=["evaluation.seeds=[0]", "evaluation.folds=[1]"])
    medley = copy.deepcopy(config.medley)
    medley["paths"]["split_root"] = str(tmp_path)
    assert validate_split_selection(config.settings, medley) == "deferred"
    assert list(tmp_path.iterdir()) == []
    path = tmp_path / "pd_eeg__moca.csv"
    path.write_text("subject,seed,fold\nsynthetic-a,0,1\nsynthetic-b,0,2\n")
    original = path.read_bytes()
    assert validate_split_selection(config.settings, medley) == "validated"
    assert path.read_bytes() == original
    settings = apply_overrides(config.settings, ["evaluation.folds=[3]"])
    with pytest.raises(ConfigError, match="unavailable"):
        validate_split_selection(settings, medley)
    path.write_text("subject,seed,fold\nsynthetic-a,0,1\nsynthetic-a,0,2\n")
    with pytest.raises(ConfigError, match="duplicate"):
        validate_split_selection(config.settings, medley)


def test_cli_and_error_exit(tmp_path, capsys):
    output = tmp_path / "cli.json"
    assert main(["--config", str(CONFIG_DIR / "fusion_b.json"), "--set", "training.batch_size=2", "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["settings"]["training"]["batch_size"] == 2
    with pytest.raises(SystemExit) as error:
        main(["--config", str(CONFIG_DIR / "fusion_b.json"), "--set", "training.batch_size=0"])
    assert error.value.code == 2
    assert "training.batch_size" in capsys.readouterr().err


def test_module_cli_has_no_model_runtime_imports():
    script = """
import sys
from multimodal.config import CONFIG_DIR, load_multimodal_config
load_multimodal_config(CONFIG_DIR / 'fusion_b.json')
assert not any(name in sys.modules for name in ['torch','transformers','peft','bitsandbytes','pandas','sklearn','medley.splits'])
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=APP, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_existing_configs_and_splits_match_git():
    paths = subprocess.check_output(["git", "ls-files", "medley_pd/configs", "medley_pd/splits"], cwd=APP.parent, text=True).splitlines()
    for path in paths:
        original = subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=APP.parent)
        assert hashlib.sha256((APP.parent / path).read_bytes()).digest() == hashlib.sha256(original).digest()
