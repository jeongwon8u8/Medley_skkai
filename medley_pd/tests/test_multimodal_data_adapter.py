"""Synthetic disk-backed Medley adapter tests; no clinical data or GPU needed."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from medley.config import preprocess_spec  # noqa: E402
from multimodal.config import (  # noqa: E402
    CONFIG_DIR, ConfigError, ResolvedConfig, load_multimodal_config, save_resolved_config,
)
from multimodal.data_adapter import (  # noqa: E402
    DataAdapterError, MedleyDataAdapter, SubjectRecord, SubjectSplit, validate_subject_partition,
)


@pytest.fixture
def files(tmp_path):
    """Create ten synthetic PD subjects, real mapped table, caches and 2 seeds."""
    config = load_multimodal_config(CONFIG_DIR / "fusion_b.json", ["evaluation.seeds=[0,1]"])
    medley = copy.deepcopy(config.medley)
    for name, folder in (("data_root", "data"), ("feature_root", "features"), ("split_root", "splits")):
        medley["paths"][name] = str(tmp_path / folder)
    cfg = ResolvedConfig(copy.deepcopy(config.settings), medley, "deferred")
    data = tmp_path / "data" / "ds007526"
    data.mkdir(parents=True)
    ids = [f"synthetic-{i:02d}" for i in range(10)]
    rows = []
    for i, sid in enumerate(ids):
        rows.append({"participant_id": sid, "group": "PD", "subject_id": f"PD{i:04d}",
                     "age": 60 + i, "sex": "M" if i % 2 else "F", "moca": 20 + i,
                     "ctt": 90, "updrs_total": 50, "updrs_part_iii": 20,
                     "disease_duration": 5, "ledd": 300, "pigd_score": 1.5, "td_score": 0.5})
    table = data / "participants.tsv"
    pd.DataFrame(rows).to_csv(table, sep="\t", index=False)
    cache = tmp_path / "features" / "pd_eeg" / "cbramod"
    cache.mkdir(parents=True)
    for i, sid in enumerate(ids):
        length = i + 2
        feat = np.repeat(np.arange(length, dtype=np.float32)[:, None], 3800, axis=1)
        np.savez(cache / f"{sid}.npz", feat=feat, flag=np.arange(length) % 2 == 1)
    splits = tmp_path / "splits" / "pd_eeg__moca.csv"
    splits.parent.mkdir()
    pd.DataFrame([{"subject": sid, "seed": seed, "fold": i % 5 + 1}
                  for seed in (0, 1) for i, sid in enumerate(ids)]).to_csv(splits, index=False)
    return cfg, table, cache, splits, ids


def mutate_table(path, field, value, index=0):
    table = pd.read_csv(path, sep="\t")
    # Object storage allows malformed-value fixtures under strict pandas dtypes.
    table[field] = table[field].astype(object)
    table.loc[index, field] = value
    table.to_csv(path, sep="\t", index=False, na_rep="n/a")


def assert_code(code, operation):
    with pytest.raises(DataAdapterError) as error:
        operation()
    assert error.value.code == code


def test_complete_subject_preserves_windows_flags_and_order(files):
    cfg, _, cache, _, ids = files
    cfg.medley["evaluation"]["max_segments"] = 1  # ignored: return full sequence
    adapter = MedleyDataAdapter(cfg)
    record = adapter.get_subject(ids[3])
    assert isinstance(record, SubjectRecord)
    with np.load(cache / f"{ids[3]}.npz") as source:
        np.testing.assert_array_equal(record.eeg_features, source["feat"])
        np.testing.assert_array_equal(record.artifact_flags, source["flag"])
    assert record.eeg_features.shape == (5, 3800)
    assert record.eeg_features.dtype == np.float32
    assert record.artifact_flags.dtype == np.bool_
    assert record.moca == 23.0 and record.subgroup == "PD"
    assert len(adapter.get_subject(ids[0]).eeg_features) == 2
    assert len(adapter.get_subject(ids[-1]).eeg_features) == 11
    assert ids[3] not in repr(record) and "23.0" not in repr(record)
    assert adapter.subject_ids == tuple(ids)
    record.eeg_features[:] = 0
    assert adapter.get_subject(ids[3]).eeg_features[-1, 0] == 4


@pytest.mark.parametrize("clinical_set,fields", [
    ("C0", ["age", "sex"]),
    ("C1", ["age", "sex", "updrs_part_iii", "disease_duration", "ledd"]),
    ("C2", ["age", "sex", "updrs_part_iii", "disease_duration", "ledd", "pigd_score", "td_score"]),
])
def test_clinical_sets_and_no_target_leakage(files, clinical_set, fields):
    cfg, _, _, _, ids = files
    cfg.settings["clinical_set"] = clinical_set
    record = MedleyDataAdapter(cfg).get_subject(ids[0])
    assert list(record.clinical) == fields
    assert not {"moca", "ctt", "updrs_total", "group", "subject"} & record.clinical.keys()
    assert record.clinical["sex"] == 0.0
    with pytest.raises(TypeError):
        record.clinical["moca"] = 24


def test_missing_optional_values_are_none(files):
    cfg, table, _, _, ids = files
    for name in ("age", "sex", "updrs_part_iii", "disease_duration", "ledd"):
        mutate_table(table, name, None)
    record = MedleyDataAdapter(cfg).get_subject(ids[0])
    assert all(value is None for value in record.clinical.values())
    assert record.moca == 20.0


@pytest.mark.parametrize("field", ["age", "sex", "updrs_part_iii", "disease_duration", "ledd"])
def test_missing_required_clinical_columns_fail(files, field):
    cfg, table, _, _, _ = files
    frame = pd.read_csv(table, sep="\t").drop(columns=field)
    frame.to_csv(table, sep="\t", index=False)
    assert_code("participant_schema", lambda: MedleyDataAdapter(cfg))


@pytest.mark.parametrize("encoder,dim", [("cbramod", 3800), ("scerebro", 3420)])
def test_encoder_dimensions_and_spec(files, encoder, dim):
    cfg, _, _, _, ids = files
    other = load_multimodal_config(CONFIG_DIR / "fusion_b.json", [f"eeg.encoder={encoder}", f"eeg.feature_dim={dim}"])
    cfg.settings["eeg"] = copy.deepcopy(other.settings["eeg"])
    cfg.medley["encoder"] = copy.deepcopy(other.medley["encoder"])
    cache = Path(cfg.medley["paths"]["feature_root"]) / "pd_eeg" / encoder
    cache.mkdir(exist_ok=True)
    np.savez(cache / f"{ids[0]}.npz", feat=np.ones((4, dim), np.float32), flag=[False, True, True, False])
    (cache / "spec.json").write_text(json.dumps(preprocess_spec(cfg.medley)))
    adapter = MedleyDataAdapter(cfg)
    assert adapter.cache_spec_status == "validated"
    assert adapter.get_subject(ids[0]).eeg_features.shape == (4, dim)


@pytest.mark.parametrize("shape,code", [((3, 3799), "feature_dimension"), ((0, 3800), "feature_shape"),
                                       ((3800,), "feature_shape"), ((2, 3, 3800), "feature_shape")])
def test_malformed_sequence_shapes(files, shape, code):
    cfg, _, cache, _, ids = files
    np.savez(cache / f"{ids[0]}.npz", feat=np.ones(shape, np.float32), flag=np.zeros(3, bool))
    assert_code(code, lambda: MedleyDataAdapter(cfg).get_subject(ids[0]))


@pytest.mark.parametrize("flags,code", [([True], "artifact_length"), ([[True], [False]], "artifact_length"),
                                       ([0, 2], "artifact_flags"), ([0, np.nan], "artifact_flags"),
                                       (["yes", "no"], "artifact_flags")])
def test_bad_flags(files, flags, code):
    cfg, _, cache, _, ids = files
    np.savez(cache / f"{ids[0]}.npz", feat=np.ones((2, 3800), np.float32), flag=flags)
    assert_code(code, lambda: MedleyDataAdapter(cfg).get_subject(ids[0]))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_eeg_rejected(files, value):
    cfg, _, cache, _, ids = files
    feat = np.ones((2, 3800), np.float32)
    feat[1, 4] = value
    np.savez(cache / f"{ids[0]}.npz", feat=feat, flag=[False, True])
    assert_code("feature_values", lambda: MedleyDataAdapter(cfg).get_subject(ids[0]))


def test_all_artifact_windows_are_preserved(files):
    cfg, _, cache, _, ids = files
    np.savez(cache / f"{ids[0]}.npz", feat=np.ones((3, 3800), np.float32), flag=[True]*3)
    record = MedleyDataAdapter(cfg).get_subject(ids[0])
    assert record.eeg_features.shape[0] == 3 and record.artifact_flags.all()


@pytest.mark.parametrize("label,reason", [(None, "missing_moca"), (-1, "invalid_moca"),
                                        (31, "invalid_moca"), (np.inf, "invalid_moca")])
def test_invalid_labels_explicit_eligibility(files, label, reason):
    cfg, table, _, _, ids = files
    mutate_table(table, "moca", label)
    adapter = MedleyDataAdapter(cfg)
    assert ids[0] not in adapter.subject_ids
    assert adapter.exclusions[0].reason == reason
    assert_code(reason, lambda: adapter.get_subject(ids[0]))
    assert_code("split_ineligible", lambda: adapter.get_split(0, 1))


@pytest.mark.parametrize("score", [0, 30, 23.5])
def test_valid_moca_endpoints_and_continuous_values(files, score):
    cfg, table, _, _, ids = files
    mutate_table(table, "moca", score)
    assert MedleyDataAdapter(cfg).get_subject(ids[0]).moca == score


def test_nonpatient_missing_table_unknown_and_duplicate_subjects(files):
    cfg, table, _, _, ids = files
    mutate_table(table, "group", "Control")
    adapter = MedleyDataAdapter(cfg)
    assert_code("not_patient", lambda: adapter.get_subject(ids[0]))
    assert_code("unknown_subject", lambda: adapter.get_subject("synthetic-unknown"))
    mutate_table(table, "participant_id", ids[1])
    assert_code("duplicate_subject", lambda: MedleyDataAdapter(cfg))
    table.unlink()
    assert_code("missing_participants", lambda: MedleyDataAdapter(cfg))


@pytest.mark.parametrize("subject_id", [None, "", "../outside", "path/subject", "nan", "bad\\id"])
def test_invalid_subject_identifiers(files, subject_id):
    cfg, _, _, _, _ = files
    assert_code("invalid_subject_id", lambda: MedleyDataAdapter(cfg).get_subject(subject_id))


def test_cache_failures_are_explicit(files):
    cfg, _, cache, _, ids = files
    path = cache / f"{ids[0]}.npz"
    path.unlink()
    adapter = MedleyDataAdapter(cfg)
    assert_code("missing_cache", lambda: adapter.get_subject(ids[0]))
    np.savez(path, feat=np.zeros((2, 3800)))  # missing flag
    assert_code("malformed_cache", lambda: adapter.get_subject(ids[0]))
    path.write_bytes(b"not an NPZ")
    assert_code("malformed_cache", lambda: adapter.get_subject(ids[0]))
    with path.open("wb") as stream:
        np.save(stream, np.zeros((2, 3800)))  # mislabeled NPY, not NPZ
    assert_code("malformed_cache", lambda: adapter.get_subject(ids[0]))


@pytest.mark.parametrize("source", ["moca", "MOCA_score", "MoCA_Z", "ctt", "UPDRS Total", "ctt_log"])
def test_alias_leakage_rejected_before_loading(files, source):
    cfg, _, _, _, _ = files
    cfg.medley["dataset"]["columns"]["age"] = source
    assert_code("clinical_leakage", lambda: MedleyDataAdapter(cfg))


def test_unapproved_predictor_and_unknown_set_rejected(files):
    cfg, _, _, _, _ = files
    cfg.medley["task"]["clinical_sets"]["C1"].append("unreviewed_proxy")
    assert_code("clinical_leakage", lambda: MedleyDataAdapter(cfg))
    cfg.settings["clinical_set"] = "unknown"
    with pytest.raises(ConfigError, match="clinical_set"):
        MedleyDataAdapter(cfg)


def test_fixed_splits_exact_order_disjoint_and_unchanged(files):
    cfg, _, _, splits, ids = files
    before = hashlib.sha256(splits.read_bytes()).digest()
    adapter = MedleyDataAdapter(cfg)
    for seed in (0, 1):
        for fold in range(1, 6):
            result = adapter.get_split(seed, fold)
            assert isinstance(result, SubjectSplit)
            assert result.test_subjects == tuple(sid for i, sid in enumerate(ids) if i % 5 + 1 == fold)
            assert result.train_subjects == tuple(sid for i, sid in enumerate(ids) if i % 5 + 1 != fold)
            assert not set(result.train_subjects) & set(result.test_subjects)
    assert hashlib.sha256(splits.read_bytes()).digest() == before
    assert_code("split_selection", lambda: adapter.get_split(9, 1))
    assert_code("split_selection", lambda: adapter.get_split(0, 0))


def test_partition_overlap_and_duplicates():
    assert_code("split_overlap", lambda: validate_subject_partition(["a"], ["a"]))
    assert_code("duplicate_split_subject", lambda: validate_subject_partition(["a", "a"], ["b"]))
    assert_code("empty_partition", lambda: validate_subject_partition([], ["b"]))


@pytest.mark.parametrize("text", ["", "wrong,columns\na,b\n", "subject,seed,fold\na,notint,1\n",
                                   "subject,seed,fold\na,0,1\na,0,2\n"])
def test_bad_fixed_split_schema(files, text):
    cfg, _, _, splits, _ = files
    splits.write_text(text)
    assert_code("split_schema", lambda: MedleyDataAdapter(cfg).get_split(0, 1))


def test_missing_splits_never_created_and_unknown_manifest_subjects(files):
    cfg, _, _, splits, _ = files
    adapter = MedleyDataAdapter(cfg)
    splits.unlink()
    assert_code("missing_splits", lambda: adapter.get_split(0, 1))
    assert not splits.exists()
    rows = [{"subject": f"absent-{fold}", "seed": seed, "fold": fold} for seed in (0, 1) for fold in range(1, 6)]
    pd.DataFrame(rows).to_csv(splits, index=False)
    assert_code("split_subject", lambda: adapter.get_split(0, 1))


def test_adapter_reuses_medley_apis(files, monkeypatch):
    from medley import features, participants
    cfg, _, _, _, ids = files
    calls = []
    for module, name in [(features, "load_segments"), (participants, "load_participants"),
                         (participants, "clinical_sets"), (participants, "task_cohort")]:
        original = getattr(module, name)
        def wrapper(*args, _original=original, _name=name, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)
        monkeypatch.setattr(module, name, wrapper)
    def prohibited(*args, **kwargs):
        raise AssertionError("Averaging/extraction must never run")
    monkeypatch.setattr(features, "load_features", prohibited)
    monkeypatch.setattr(features, "run_extract", prohibited)
    adapter = MedleyDataAdapter(cfg)
    adapter.get_subject(ids[0])
    assert set(calls) == {"load_segments", "load_participants", "clinical_sets", "task_cohort"}


def test_cache_provenance_and_config_independence(files):
    cfg, _, cache, _, ids = files
    adapter = MedleyDataAdapter(cfg)
    assert adapter.cache_spec_status == "absent"
    cfg.settings["clinical_set"] = "C0"
    assert len(adapter.get_subject(ids[0]).clinical) == 5
    spec = preprocess_spec(cfg.medley)
    spec["l_freq"] = 5
    (cache / "spec.json").write_text(json.dumps(spec))
    assert_code("cache_spec", lambda: MedleyDataAdapter(cfg))


def test_normalization_notices_do_not_print_clinical_values(files, capsys):
    cfg, table, _, _, _ = files
    frame = pd.read_csv(table, sep="\t")
    frame["sex"] = [0, 1] * 5  # legacy loader prints a numeric coding warning
    frame.to_csv(table, sep="\t", index=False)
    adapter = MedleyDataAdapter(cfg)
    assert adapter.metadata_notices
    assert capsys.readouterr().out == ""


def test_relative_paths_are_bound_to_construction_cwd(files, monkeypatch):
    cfg, table, _, _, ids = files
    root = table.parents[2]
    monkeypatch.chdir(root)
    cfg.medley["paths"].update(data_root="data", feature_root="features", split_root="splits")
    adapter = MedleyDataAdapter(cfg)
    monkeypatch.chdir(APP)
    assert adapter.get_subject(ids[0]).eeg_features.shape == (2, 3800)
    assert len(adapter.get_split(0, 1).test_subjects) == 2


def test_missing_moca_column_and_unknown_encoder(files):
    cfg, table, _, _, _ = files
    cfg.settings["eeg"]["encoder"] = "unknown"
    with pytest.raises(ConfigError, match="encoder"):
        MedleyDataAdapter(cfg)
    cfg.settings["eeg"]["encoder"] = "cbramod"
    pd.read_csv(table, sep="\t").drop(columns="moca").to_csv(table, sep="\t", index=False)
    assert_code("participant_schema", lambda: MedleyDataAdapter(cfg))


def test_unparseable_labels_never_imputed(files):
    cfg, table, _, _, ids = files
    mutate_table(table, "moca", "not-a-score")
    adapter = MedleyDataAdapter(cfg)
    assert ids[0] not in adapter.subject_ids
    # Existing Medley numeric coercion interprets this row as missing.
    assert_code("missing_moca", lambda: adapter.get_subject(ids[0]))


def test_cache_float_dtype_is_preserved_and_integer_cache_rejected(files):
    cfg, _, cache, _, ids = files
    path = cache / f"{ids[0]}.npz"
    np.savez(path, feat=np.ones((2, 3800), np.float64), flag=[0, 1])
    adapter = MedleyDataAdapter(cfg)
    assert adapter.get_subject(ids[0]).eeg_features.dtype == np.float64
    np.savez(path, feat=np.ones((2, 3800), np.int32), flag=[0, 1])
    assert_code("feature_values", lambda: adapter.get_subject(ids[0]))


@pytest.mark.parametrize("source", ["score_MoCA", "z_CTT", "total_UPDRS"])
def test_review_leakage_alias_in_nonprefix_position(files, source):
    cfg, table, _, _, _ = files
    frame = pd.read_csv(table, sep="\t")
    frame[source] = frame["moca"]  # explicit synthetic target-derived alias
    frame.to_csv(table, sep="\t", index=False)
    cfg.medley["dataset"]["columns"]["age"] = source
    assert_code("clinical_leakage", lambda: MedleyDataAdapter(cfg))


def test_review_duplicate_split_headers_rejected(files):
    cfg, _, _, splits, _ = files
    frame = pd.read_csv(splits)
    # DictReader silently keeps only the last duplicate field unless checked.
    lines = ["subject,seed,fold,fold"]
    lines += [f"{r.subject},{r.seed},{r.fold},{r.fold}" for r in frame.itertuples()]
    splits.write_text("\n".join(lines) + "\n")
    assert_code("split_schema", lambda: MedleyDataAdapter(cfg).get_split(0, 1))


def test_review_split_extra_csv_values_rejected(files):
    cfg, _, _, splits, _ = files
    lines = splits.read_text().splitlines()
    lines[1] += ",unexpected"
    splits.write_text("\n".join(lines) + "\n")
    assert_code("split_schema", lambda: MedleyDataAdapter(cfg).get_split(0, 1))


def test_review_numpy_id_sequences_supported():
    validate_subject_partition(np.array(["synthetic-a", "synthetic-b"]), np.array(["synthetic-c"]))


def test_review_all_ten_seeds_and_five_folds(files):
    cfg, _, _, splits, ids = files
    cfg.settings["evaluation"]["seeds"] = list(range(10))
    pd.DataFrame([{"subject": sid, "seed": seed, "fold": (i + seed) % 5 + 1}
                  for seed in range(10) for i, sid in enumerate(ids)]).to_csv(splits, index=False)
    original = splits.read_bytes()
    adapter = MedleyDataAdapter(cfg)
    for seed in range(10):
        for fold in range(1, 6):
            result = adapter.get_split(seed, fold)
            expected = tuple(sid for i, sid in enumerate(ids) if (i + seed) % 5 + 1 == fold)
            assert result.test_subjects == expected
            assert set(result.train_subjects) | set(result.test_subjects) == set(ids)
            assert not set(result.train_subjects) & set(result.test_subjects)
    assert splits.read_bytes() == original


def test_review_session_suffix_not_inferred_as_patient_identity(files):
    cfg, table, _, _, ids = files
    mutate_table(table, "participant_id", ids[0] + "_ses-1")
    adapter = MedleyDataAdapter(cfg)
    assert_code("unknown_subject", lambda: adapter.get_subject(ids[0]))
    assert_code("missing_cache", lambda: adapter.get_subject(ids[0] + "_ses-1"))
    assert_code("split_subject", lambda: adapter.get_split(0, 1))


def test_review_resolved_snapshot_integrates_with_adapter(files, tmp_path):
    cfg, _, _, _, ids = files
    output = save_resolved_config(cfg, tmp_path / "resolved.json")
    restored = load_multimodal_config(output, ["clinical_set=C0"])
    adapter = MedleyDataAdapter(restored)
    assert adapter.clinical_variables == ("age", "sex")
    assert adapter.get_subject(ids[0]).moca == 20.0
    assert adapter.get_split(0, 1).test_subjects == (ids[0], ids[5])
