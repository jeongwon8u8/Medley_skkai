"""JSON 설정 점검 (데이터 없이). 실행: python -m pytest -q tests/test_configs.py"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from configs.config_loader import build_cfg, load_json  # noqa: E402

C = os.path.join(ROOT, "configs")


@pytest.mark.parametrize("ds", ["pd_eeg", "iowa_rest"])
def test_default_sets_match_task_yaml(ds):
    """JSON 기본 조합이 기존 yaml 의 clinical set 과 같아야 원본 결과가 재현됨"""
    from medley.config import load_config
    from medley.participants import clinical_sets
    cfg = build_cfg(load_json(os.path.join(C, f"train_config_{ds}.json")))
    assert clinical_sets(cfg) == clinical_sets(load_config(ds, "cbramod", "moca_impaired"))
    assert cfg["models"]["fusion"] == ["stack", "concat"]


def test_variable_off_and_guards():
    p = os.path.join(C, "train_config_pd_eeg.json")
    cfg = build_cfg(load_json(p, overrides=["clinical_sets.C1.ledd=false", "models.fusion.concat=false",
                                            "encoder=scerebro", "max_segments=24"]))
    sets = cfg["dataset"]["clinical_sets"]["moca_impaired"]
    assert "ledd" not in sets["C1"] and "ledd" in sets["C2"]
    assert cfg["models"]["fusion"] == ["stack"] and cfg["encoder"]["name"] == "scerebro"
    assert cfg["evaluation"]["max_segments"] == 24
    for bad in (["clinical_sets.C0.moca=true"], ["clinical_sets.C0.updrs_total=true"], ["clinical_sets.C0.nope=true"],
                ["clincal_sets.C0.age=false"]):
        with pytest.raises(SystemExit):
            build_cfg(load_json(p, overrides=bad))
    cfg = build_cfg(load_json(p, overrides=["clinical_sets.C0.updrs_total=true", "allow_excluded=true"]))
    assert "updrs_total" in cfg["dataset"]["clinical_sets"]["moca_impaired"]["C0"]


def test_iowa_reference_is_pz():
    cfg = build_cfg(load_json(os.path.join(C, "train_config_iowa_rest.json"), overrides=["encoder=scerebro"]))
    assert cfg["dataset"]["reference"] == "PZ" and cfg["encoder"]["reference"] == "PZ"
