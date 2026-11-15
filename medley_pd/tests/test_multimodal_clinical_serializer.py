"""Synthetic CPU-only serializer contracts; no clinical files or model assets."""
from __future__ import annotations

import copy
from dataclasses import replace
from decimal import Inexact, localcontext, ROUND_UP
import json
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from multimodal.config import (CONFIG_DIR, ConfigError, load_multimodal_config,
                               save_resolved_config, validate_config)
from multimodal.clinical_serializer import (ClinicalSerializationError, ClinicalTextSerializer,
                                           FEATURE_SETS, TEMPLATES)


VALUES = {"age": 68, "sex": "female", "updrs_part_iii": 24,
          "disease_duration": 6.5, "ledd": 450, "pigd_score": 2.5, "td_score": 1.2}


def config(feature_set="C1", *overrides):
    return load_multimodal_config(CONFIG_DIR / "base.json", [f"clinical_set={feature_set}", *overrides])


def record(feature_set="C1"):
    return {k: VALUES[k] for k in FEATURE_SETS[feature_set]}


def expected(feature_set="C1"):
    values = {**VALUES, "sex": "female"}
    return " ".join(TEMPLATES[k].format(values[k]) for k in FEATURE_SETS[feature_set])


@pytest.mark.parametrize("feature_set", ["C0", "C1", "C2"])
def test_feature_sets_exact_templates_order_determinism(feature_set):
    serializer = ClinicalTextSerializer(config(feature_set))
    clinical = record(feature_set)
    before = copy.deepcopy(clinical)
    # Literal oracle for all templates, independent of module template constants.
    sentences = ["The recorded age value is 68.", "The patient's recorded sex is female.",
                 "The recorded UPDRS Part III score is 24.",
                 "The recorded disease duration value is 6.5.", "The recorded LEDD value is 450.",
                 "The recorded PIGD score is 2.5.", "The recorded TD score is 1.2."]
    want = " ".join(sentences[:len(FEATURE_SETS[feature_set])])
    assert serializer.serialize(clinical) == want
    assert serializer.serialize(dict(reversed(list(clinical.items())))) == want
    assert serializer.serialize(clinical) == want
    assert clinical == before
    assert "  " not in want and want.endswith(".")
    assert "years" not in want and "mg/day" not in want


@pytest.mark.parametrize("value,want", [(68, "68"), (68.0, "68"), (6.5, "6.5"),
    (6.500000000000001, "6.5"), (np.int64(68), "68"), (np.uint32(450), "450"),
    (np.float32(2.3), "2.3"), (np.float64(6.5), "6.5"), (-0.0, "0"),
    (0.00000012, "0.00000012"), (1234567890123456789, "1234567890123456789"),
    (1.23456789123456, "1.23456789123456")])
def test_numeric_formatting(value, want):
    clinical = record()
    clinical["disease_duration"] = value
    assert f"duration value is {want}." in ClinicalTextSerializer(config()).serialize(clinical)


@pytest.mark.parametrize("value", [True, np.bool_(False), "68", "NaN", "", float("nan"),
    float("inf"), float("-inf"), np.float32("nan"), [], {}, complex(1, 0), np.array(68)])
def test_invalid_numeric_measurements(value):
    clinical = record()
    clinical["age"] = value
    with pytest.raises(ClinicalSerializationError):
        ClinicalTextSerializer(config()).serialize(clinical)


def test_explicit_precision_and_decimal_context_independence():
    clinical = record()
    clinical["age"] = 1.23456789
    serializer = ClinicalTextSerializer(config("C1", "clinical_text.numeric_significant_digits=6"))
    with localcontext() as ctx:
        ctx.rounding = ROUND_UP
        ctx.prec = 2
        ctx.traps[Inexact] = True
        ctx.Emax = 0
        ctx.Emin = 0
        assert "age value is 1.23457." in serializer.serialize(clinical)


@pytest.mark.parametrize("label,want", [(" F ", "female"), ("female", "female"),
                                       ("M", "male"), (" MALE ", "male")])
def test_canonical_sex_strings(label, want):
    clinical = record()
    clinical["sex"] = label
    assert f"sex is {want}." in ClinicalTextSerializer(config()).serialize(clinical)


@pytest.mark.parametrize("value", [0, 1, 2, np.float32(0), 0.5, "0", "unknown", "other", True, float("nan")])
def test_ambiguous_or_unsupported_sex(value):
    clinical = record()
    clinical["sex"] = value
    with pytest.raises(ClinicalSerializationError):
        ClinicalTextSerializer(config()).serialize(clinical)


@pytest.mark.parametrize("value,want", [(0, "female"), (np.float32(1), "male"), (2.0, "female")])
def test_explicit_verified_numeric_sex_mapping(value, want):
    cfg = config("C1", 'clinical_text.sex_code_mapping={"0":"female","1":"male","2":"female"}')
    serializer = ClinicalTextSerializer(cfg)
    cfg.settings["clinical_text"]["sex_code_mapping"]["0"] = "male"
    clinical = record()
    clinical["sex"] = value
    assert f"sex is {want}." in serializer.serialize(clinical)


@pytest.mark.parametrize("keys", [("ledd",), ("age", "sex", "ledd"), tuple(FEATURE_SETS["C1"])])
def test_skip_missing_and_all_missing(keys):
    clinical = record()
    for key in keys:
        clinical[key] = None
    text = ClinicalTextSerializer(config()).serialize(clinical)
    assert text == " ".join(TEMPLATES[k].format(VALUES[k]) for k in FEATURE_SETS["C1"] if k not in keys)
    assert not any(word in text for word in ("None", "NaN", "null", "missing"))
    if len(keys) == 5:
        assert text == ""
        assert ClinicalTextSerializer(config()).serialize_batch([clinical]) == [""]


@pytest.mark.parametrize("key", ["moca", "MoCA", "MOCA_score", "montreal_cognitive_assessment",
    "moca_label", "predicted_moca", "target", "ctt", "CTT_1", "Color Trails Test", "updrs_total",
    "UPDRS-Total", "total_updrs", "education", "subject_id", "visit_id", "split", "eeg_features", 42])
def test_prohibited_aliases_unknown_metadata_rejected(key):
    clinical = record()
    clinical[key] = "DO_NOT_DISCLOSE_TARGET_VALUE"
    for policy in ("reject", "ignore"):
        serializer = ClinicalTextSerializer(config("C1", f"clinical_text.extra_approved_policy={policy}"))
        with pytest.raises(ClinicalSerializationError) as error:
            serializer.serialize(clinical)
        assert "DO_NOT_DISCLOSE" not in str(error.value)


def test_extra_approved_policy_and_missing_feature_set_mismatch():
    serializer = ClinicalTextSerializer(config())
    with pytest.raises(ClinicalSerializationError, match="Extra approved"):
        serializer.serialize(record("C2"))
    ignore = ClinicalTextSerializer(config("C1", "clinical_text.extra_approved_policy=ignore"))
    assert ignore.serialize(record("C2")) == expected()
    with pytest.raises(ClinicalSerializationError, match="keys are missing"):
        ignore.serialize(record("C0"))
    contaminated_extra = record("C2")
    contaminated_extra["td_score"] = "invalid"
    with pytest.raises(ClinicalSerializationError):
        ignore.serialize(contaminated_extra)


def test_batch_order_duplicates_empty_and_contamination():
    serializer = ClinicalTextSerializer(config())
    first, second = record(), record()
    second["age"] = 72
    clinicals = [second, first, second]
    before = copy.deepcopy(clinicals)
    assert serializer.serialize_batch(clinicals) == [serializer.serialize(c) for c in clinicals]
    assert len(serializer.serialize_batch([first])) == 1
    assert serializer.serialize_batch([]) == []
    assert clinicals == before
    bad = {**first, "moca": "SECRET_LABEL"}
    with pytest.raises(ClinicalSerializationError):
        serializer.serialize_batch([first, bad, second])
    for malformed in (first, "text", None, [None]):
        with pytest.raises(ClinicalSerializationError):
            serializer.serialize_batch(malformed)
    for malformed in (None, [first], SimpleNamespace(clinical=first), {"clinical": first}):
        with pytest.raises(ClinicalSerializationError):
            serializer.serialize(malformed)


@pytest.mark.parametrize("name", ["base", "clinical_only", "fusion_a", "fusion_b"])
def test_legacy_json_and_snapshot_roundtrip(name, tmp_path):
    cfg = load_multimodal_config(CONFIG_DIR / f"{name}.json")
    assert cfg.settings["clinical_set"] == "C1"
    assert ClinicalTextSerializer(cfg).serialize(record()) == expected()
    snapshot = tmp_path / "resolved.json"
    save_resolved_config(cfg, snapshot)
    restored = load_multimodal_config(snapshot)
    assert restored.to_dict() == cfg.to_dict()
    assert ClinicalTextSerializer(restored).serialize(record()) == expected()
    legacy = cfg.to_dict()
    del legacy["settings"]["clinical_text"]
    snapshot.write_text(json.dumps(legacy))
    assert ClinicalTextSerializer(load_multimodal_config(snapshot)).serialize(record()) == expected()


def test_inheritance_cli_override_and_old_in_memory_config(tmp_path):
    child = tmp_path / "child.json"
    child.write_text(json.dumps({"extends": str(CONFIG_DIR / "base.json"),
                                "clinical_text": {"numeric_significant_digits": 12}}))
    cfg = load_multimodal_config(child, ["clinical_set=C0", "clinical_text.numeric_significant_digits=10"])
    assert cfg.settings["clinical_text"]["numeric_significant_digits"] == 10
    assert ClinicalTextSerializer(cfg).serialize(record("C0")) == expected("C0")
    del cfg.settings["clinical_text"]
    assert ClinicalTextSerializer(cfg).serialize(record("C0")) == expected("C0")
    cfg.medley["task"]["clinical_sets"]["C0"] = ["age", "moca"]
    with pytest.raises(ClinicalSerializationError):
        ClinicalTextSerializer(cfg)


@pytest.mark.parametrize("options", [None, {"template_version": "unknown"},
    {"numeric_significant_digits": 0}, {"numeric_significant_digits": True},
    {"numeric_significant_digits": 35}, {"extra_approved_policy": "silent"},
    {"sex_code_mapping": {"0": "woman"}}, {"sex_code_mapping": {}},
    {"sex_code_mapping": {"1.0": "male"}}, {"sex_code_mapping": {"01": "male"}},
    {"unexpected": True}])
def test_serializer_config_validation(options):
    cfg = config()
    cfg.settings["clinical_text"] = options
    with pytest.raises(ConfigError):
        validate_config(cfg.settings)


def test_three_subject_record_dataset_collator_smoke_and_label_invariance():
    from multimodal.data_adapter import SubjectRecord
    from multimodal.dataset import MedleySubjectDataset, MedleySubjectCollator
    import torch

    cfg = config("C1", 'clinical_text.sex_code_mapping={"0":"female","1":"male"}')
    clinicals = [{**record(), "sex": 0.0},
                {**record(), "age": 72.0, "sex": None, "ledd": None, "disease_duration": None},
                {"age": 61.0, "sex": 1.0, "updrs_part_iii": 32.0, "disease_duration": 3.25, "ledd": 300.0}]
    ids = ("synthetic-c", "synthetic-a", "synthetic-b")
    records = {sid: SubjectRecord(sid, np.full((i + 1, 4), 9876., dtype=np.float32),
                                 np.zeros(i + 1, dtype=bool), MappingProxyType(clinicals[i]),
                                 19.125 + i, "SYNTHETIC_METADATA") for i, sid in enumerate(ids)}
    adapter = SimpleNamespace(subject_ids=ids, clinical_variables=cfg.clinical_variables,
                              get_subject=records.__getitem__)
    dataset = MedleySubjectDataset(adapter, ids)
    samples = [dataset[i] for i in range(len(dataset))]
    assert tuple(s["subject_id"] for s in samples) == ids
    batch = MedleySubjectCollator()(samples)
    assert batch["subject_ids"] == list(ids)
    assert batch["clinical"] == clinicals
    before = copy.deepcopy(batch)
    serializer = ClinicalTextSerializer(cfg)
    texts = serializer.serialize_batch(batch["clinical"])
    assert len(texts) == 3 and len(set(texts)) == 3
    assert texts[0] == expected()
    assert texts[1] == "The recorded age value is 72. The recorded UPDRS Part III score is 24."
    assert "duration value is 3.25." in texts[2]
    assert texts == serializer.serialize_batch(batch["clinical"])
    for sid, text in zip(ids, texts):
        assert sid not in text
        assert all(x not in text for x in ("9876", "SYNTHETIC_METADATA", "19.125", "20.125", "21.125", "MoCA"))
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            assert torch.equal(batch[key], before[key])
        else:
            assert batch[key] == before[key]
    for sid in ids:
        records[sid] = replace(records[sid], moca=29.875)
    changed = MedleySubjectCollator()([dataset[i] for i in range(len(dataset))])
    assert serializer.serialize_batch(changed["clinical"]) == texts
    assert not torch.equal(changed["moca"], batch["moca"])


def test_import_and_use_without_torch_or_hugging_face():
    code = '''
import sys
sys.path.insert(0, sys.argv[1])
from multimodal.config import CONFIG_DIR, load_multimodal_config
from multimodal.clinical_serializer import ClinicalTextSerializer
s = ClinicalTextSerializer(load_multimodal_config(CONFIG_DIR / 'base.json'))
assert s.serialize(dict(age=68, sex=None, updrs_part_iii=24, disease_duration=6.5, ledd=None))
assert not any(k.split('.')[0] in {'torch','transformers','peft','tokenizers'} for k in sys.modules)
'''
    subprocess.run([sys.executable, "-c", code, str(APP)], check=True, capture_output=True, text=True)
