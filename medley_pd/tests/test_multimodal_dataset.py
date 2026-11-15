"""CPU-only subject Dataset/Collator contracts, using synthetic patients only."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import sys
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from multimodal.config import CONFIG_DIR, ResolvedConfig, load_multimodal_config  # noqa: E402
from multimodal.data_adapter import DataAdapterError, MedleyDataAdapter, SubjectRecord, SubjectSplit  # noqa: E402
from multimodal.dataset import MedleySubjectCollator, MedleySubjectDataset, SubjectDataError  # noqa: E402


class SyntheticAdapter:
    """Minimal implementation of Task 2 public record/property contracts."""

    def __init__(self, lengths=(3, 5, 2), width=4):
        self.subject_ids = tuple(f"synthetic-{i}" for i in range(len(lengths)))
        self.clinical_variables = ("age", "sex")
        self.calls = []
        self.records = {}
        for i, (sid, length) in enumerate(zip(self.subject_ids, lengths)):
            self.records[sid] = SubjectRecord(
                sid, np.arange(length * width, dtype=np.float64).reshape(length, width) + 100 * i,
                np.arange(length) % 2 == 1, MappingProxyType({"age": 60.0 + i, "sex": None}),
                20.0 + i, None if i == 2 else "PD",
            )

    def get_subject(self, subject_id):
        self.calls.append(subject_id)
        return self.records[subject_id]


@pytest.fixture
def adapter():
    return SyntheticAdapter()


def dataset(adapter):
    return MedleySubjectDataset(adapter, adapter.subject_ids)


def assert_same_sample(a, b):
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], torch.Tensor):
            assert torch.equal(a[key], b[key]), key
        else:
            assert a[key] == b[key], key


def test_dataset_one_patient_per_item_and_order(adapter):
    ids = [adapter.subject_ids[2], adapter.subject_ids[0]]
    data = MedleySubjectDataset(adapter, ids)
    assert adapter.calls == []  # no eager cache loading
    ids.reverse()  # constructor retains its own immutable order
    assert len(data) == 2
    assert data.subject_ids == ("synthetic-2", "synthetic-0")
    assert data[0]["subject_id"] == "synthetic-2"
    assert data[1]["subject_id"] == "synthetic-0"
    assert data[-1]["subject_id"] == "synthetic-0"
    assert adapter.calls == ["synthetic-2", "synthetic-0", "synthetic-0"]


def test_sample_dtypes_values_flags_and_metadata(adapter):
    data = dataset(adapter)
    for i in range(len(data)):
        sample = data[i]
        source = adapter.records[adapter.subject_ids[i]]
        assert sample["eeg_features"].dtype == torch.float32
        assert sample["moca"].dtype == torch.float32 and sample["moca"].shape == ()
        assert sample["artifact_flags"].dtype == torch.bool
        assert sample["eeg_features"].device.type == "cpu"
        torch.testing.assert_close(sample["eeg_features"], torch.tensor(source.eeg_features, dtype=torch.float32))
        assert torch.equal(sample["artifact_flags"], torch.tensor(source.artifact_flags))
        assert sample["moca"].item() == source.moca
        assert sample["clinical"] == dict(source.clinical)
        assert sample["clinical"]["sex"] is None
        assert sample["subgroup"] == source.subgroup


def test_repeated_indexing_deterministic_and_independent(adapter):
    data = dataset(adapter)
    first, second = data[0], data[0]
    assert_same_sample(first, second)
    first["eeg_features"].zero_()
    first["artifact_flags"].fill_(True)
    first["moca"].zero_()
    first["clinical"]["age"] = 1.0
    assert_same_sample(second, data[0])
    assert adapter.records["synthetic-0"].eeg_features[-1, -1] == 11


@pytest.mark.parametrize("ids,match", [([], "nonempty"), ("synthetic-0", "sequence"),
                                      (["synthetic-0", "synthetic-0"], "Duplicate"),
                                      ([None], "canonical"), ([""], "canonical"),
                                      ([" synthetic-0"], "canonical"), (["unknown"], "Unknown"),
                                      (["../outside"], "Unknown")])
def test_invalid_subject_lists(adapter, ids, match):
    with pytest.raises(SubjectDataError, match=match):
        MedleySubjectDataset(adapter, ids)


def test_invalid_indices(adapter):
    data = dataset(adapter)
    for index in (3, -4):
        with pytest.raises(IndexError):
            data[index]
    for index in (True, slice(None), 1.5):
        with pytest.raises(TypeError):
            data[index]
    assert data[np.int64(0)]["subject_id"] == "synthetic-0"


@pytest.mark.parametrize("field,value,match", [
    ("subject_id", "different-subject", "identity"),
    ("eeg_features", np.ones(4), "EEG"),
    ("eeg_features", np.ones((0, 4)), "EEG"),
    ("eeg_features", np.ones((3, 0)), "EEG"),
    ("eeg_features", np.ones((3, 4), dtype=np.int32), "EEG"),
    ("eeg_features", np.full((3, 4), np.nan), "EEG"),
    ("eeg_features", np.full((3, 4), np.inf), "EEG"),
    ("eeg_features", np.full((3, 4), 1e100), "float32"),
    ("artifact_flags", np.array([True]), "artifact_flags"),
    ("artifact_flags", np.ones(3, dtype=np.int32), "artifact_flags"),
    ("moca", float("nan"), "moca"), ("moca", float("inf"), "moca"),
    ("moca", -1, "moca"), ("moca", 31, "moca"), ("moca", True, "moca"),
    ("moca", "24", "moca"),
    ("clinical", {"age": 60.0, "moca": 24.0}, "allowlisted"),
    ("clinical", {"age": 60.0}, "selected feature set"),
    ("clinical", {"age": 60.0, "sex": "M"}, "numeric-or-None"),
    ("clinical", {"age": float("inf"), "sex": None}, "numeric-or-None"),
    ("subgroup", 123, "subgroup"),
])
def test_malformed_adapter_records(adapter, field, value, match):
    adapter.records["synthetic-0"] = replace(adapter.records["synthetic-0"], **{field: value})
    with pytest.raises(SubjectDataError, match=match):
        dataset(adapter)[0]


def test_missing_record_and_adapter_error_propagation(adapter, monkeypatch):
    adapter.records["synthetic-0"] = None
    with pytest.raises(SubjectDataError, match="SubjectRecord"):
        dataset(adapter)[0]
    error = DataAdapterError("missing_cache", "Synthetic missing cache")
    def fail(sid):
        raise error
    monkeypatch.setattr(adapter, "get_subject", fail)
    with pytest.raises(DataAdapterError) as caught:
        dataset(adapter)[0]
    assert caught.value is error


def test_negative_strides_and_readonly_source(adapter):
    record = adapter.records["synthetic-0"]
    eeg = record.eeg_features[::-1]
    eeg.setflags(write=False)
    flags = record.artifact_flags[::-1]
    flags.setflags(write=False)
    adapter.records["synthetic-0"] = replace(record, eeg_features=eeg, artifact_flags=flags)
    sample = dataset(adapter)[0]
    torch.testing.assert_close(sample["eeg_features"], torch.tensor(eeg.copy(), dtype=torch.float32))
    assert sample["artifact_flags"].tolist() == flags.tolist()


@pytest.mark.parametrize("lengths", [(3, 5, 2), (4, 4, 4), (1,), (1, 6)])
def test_collator_padding_alignment_and_nonmutation(lengths):
    adapter = SyntheticAdapter(lengths=lengths)
    data = dataset(adapter)
    samples = [data[i] for i in range(len(data))]
    before = copy.deepcopy(samples)
    batch = MedleySubjectCollator()(samples)
    assert batch["eeg_features"].shape == (len(lengths), max(lengths), 4)
    assert batch["eeg_mask"].dtype == torch.bool
    assert batch["artifact_flags"].dtype == torch.bool
    assert batch["eeg_features"].dtype == batch["moca"].dtype == torch.float32
    assert batch["moca"].shape == (len(lengths),)
    assert batch["subject_ids"] == list(adapter.subject_ids)
    assert batch["clinical"] == [sample["clinical"] for sample in samples]
    assert batch["subgroups"] == [sample["subgroup"] for sample in samples]
    for i, length in enumerate(lengths):
        assert batch["eeg_mask"][i].tolist() == [True]*length + [False]*(max(lengths)-length)
        assert torch.equal(batch["eeg_features"][i, :length], samples[i]["eeg_features"])
        assert torch.equal(batch["artifact_flags"][i, :length], samples[i]["artifact_flags"])
        assert torch.count_nonzero(batch["eeg_features"][i, length:]) == 0
        assert not batch["artifact_flags"][i, length:].any()
        assert batch["moca"][i] == samples[i]["moca"]
        assert_same_sample(samples[i], before[i])
    # Output tensors/dicts must also be independent of caller samples.
    batch["eeg_features"].zero_()
    batch["artifact_flags"].fill_(True)
    batch["moca"].zero_()
    batch["clinical"][0]["age"] = 10.0
    for sample, original in zip(samples, before):
        assert_same_sample(sample, original)


def test_flagged_or_zero_segments_are_still_real_positions(adapter):
    record = adapter.records["synthetic-0"]
    adapter.records["synthetic-0"] = replace(record, eeg_features=np.zeros((3, 4)), artifact_flags=np.ones(3, bool))
    batch = MedleySubjectCollator()([dataset(adapter)[0], dataset(adapter)[1]])
    assert batch["eeg_mask"][0].tolist() == [True, True, True, False, False]
    assert batch["artifact_flags"][0].tolist() == [True, True, True, False, False]


@pytest.mark.parametrize("field,value,match", [
    ("eeg_features", torch.ones(3, 4, dtype=torch.float64), "eeg_features"),
    ("eeg_features", torch.ones(3, 4, 1), "eeg_features"),
    ("eeg_features", torch.empty(0, 4), "nonempty"),
    ("eeg_features", torch.full((3, 4), float("nan")), "finite"),
    ("artifact_flags", torch.zeros(3, dtype=torch.int64), "artifact_flags"),
    ("artifact_flags", torch.ones(2, dtype=torch.bool), "length"),
    ("moca", torch.tensor([24.0]), "moca"),
    ("moca", torch.tensor(24, dtype=torch.int64), "moca"),
    ("moca", torch.tensor(float("inf")), "moca"),
    ("moca", torch.tensor(31.0), "moca"),
    ("clinical", {"moca": 24.0}, "allowlisted"),
    ("clinical", {"ctt": 90.0}, "allowlisted"),
    ("clinical", {"updrs_total": 50.0}, "allowlisted"),
    ("subgroup", [], "subgroup"),
    ("subject_id", None, "subject_id"),
])
def test_invalid_collator_samples(adapter, field, value, match):
    sample = dataset(adapter)[0]
    sample[field] = value
    with pytest.raises(SubjectDataError, match=match):
        MedleySubjectCollator()([sample])


def test_empty_missing_fields_duplicate_ids_and_incompatible_metadata(adapter):
    collate = MedleySubjectCollator()
    with pytest.raises(SubjectDataError, match="nonempty"):
        collate([])
    for key in dataset(adapter)[0]:
        sample = dataset(adapter)[0]
        del sample[key]
        with pytest.raises(SubjectDataError, match="exactly"):
            collate([sample])
    with pytest.raises(SubjectDataError, match="Duplicate"):
        collate([dataset(adapter)[0], dataset(adapter)[0]])
    a, b = dataset(adapter)[0], dataset(adapter)[1]
    b["eeg_features"] = torch.ones(5, 7)
    with pytest.raises(SubjectDataError, match="widths"):
        collate([a, b])
    b = dataset(adapter)[1]
    b["clinical"] = {"age": 61.0}
    with pytest.raises(SubjectDataError, match="clinical keys"):
        collate([a, b])


def test_cpu_dataloader_smoke(adapter):
    data = dataset(adapter)
    loader = DataLoader(data, batch_size=3, shuffle=False, num_workers=0, collate_fn=MedleySubjectCollator())
    batches = list(loader)
    assert len(batches) == 1
    batch = batches[0]
    assert batch["eeg_features"].shape == (3, 5, 4)
    assert batch["eeg_mask"].tolist() == [[True, True, True, False, False], [True]*5, [True, True, False, False, False]]
    assert batch["moca"].tolist() == [20.0, 21.0, 22.0]
    assert batch["subject_ids"] == list(adapter.subject_ids)
    assert batch["subgroups"] == ["PD", "PD", None]
    assert batch["artifact_flags"].tolist() == [
        [False, True, False, False, False],
        [False, True, False, True, False],
        [False, True, False, False, False],
    ]
    assert batch["eeg_mask"].sum().item() == 10
    assert batch["clinical"] == [{"age": 60.0+i, "sex": None} for i in range(3)]
    assert all(batch[key].device.type == "cpu" for key in ("eeg_features", "eeg_mask", "artifact_flags", "moca"))


@pytest.mark.parametrize("encoder,width", [("cbramod", 3800), ("scerebro", 3420)])
def test_real_adapter_dataset_collator_and_fixed_split(tmp_path, encoder, width):
    cfg = load_multimodal_config(CONFIG_DIR / "fusion_b.json", [f"eeg.encoder={encoder}", f"eeg.feature_dim={width}", "clinical_set=C0", "evaluation.seeds=[0]"])
    medley = copy.deepcopy(cfg.medley)
    for key, name in (("data_root", "data"), ("feature_root", "features"), ("split_root", "splits")):
        medley["paths"][key] = str(tmp_path / name)
    cfg = ResolvedConfig(copy.deepcopy(cfg.settings), medley, "deferred")
    root = tmp_path / "data" / "ds007526"
    root.mkdir(parents=True)
    ids = [f"synthetic-{i}" for i in range(5)]
    rows = [{"participant_id": sid, "group": "PD", "subject_id": f"PD{i:03d}",
             "age": 60+i, "sex": "F", "moca": 20+i, "ctt": 90, "updrs_total": 50,
             "updrs_part_iii": 20, "disease_duration": 5, "ledd": 300,
             "pigd_score": 1.5, "td_score": 0.5} for i, sid in enumerate(ids)]
    pd.DataFrame(rows).to_csv(root / "participants.tsv", sep="\t", index=False)
    cache = tmp_path / "features" / "pd_eeg" / encoder
    cache.mkdir(parents=True)
    for i, sid in enumerate(ids):
        np.savez(cache / f"{sid}.npz", feat=np.ones((i+2, width), np.float32), flag=np.zeros(i+2, bool))
    split_path = tmp_path / "splits" / "pd_eeg__moca.csv"
    split_path.parent.mkdir()
    pd.DataFrame([{"subject": sid, "seed": 0, "fold": i+1} for i, sid in enumerate(ids)]).to_csv(split_path, index=False)
    split_bytes = split_path.read_bytes()
    adapter = MedleyDataAdapter(cfg)
    split = adapter.get_split(0, 1)
    assert isinstance(split, SubjectSplit)
    train = MedleySubjectDataset(adapter, split.train_subjects)
    test = MedleySubjectDataset(adapter, split.test_subjects)
    assert not set(train.subject_ids) & set(test.subject_ids)
    assert len(train) == 4 and len(test) == 1
    batch = next(iter(DataLoader(train, batch_size=4, collate_fn=MedleySubjectCollator())))
    assert batch["subject_ids"] == list(split.train_subjects)
    assert batch["eeg_features"].shape == (4, 6, width)
    assert split_path.read_bytes() == split_bytes


@pytest.mark.parametrize("hidden_value", [np.nan, 123.0])
def test_review_masked_eeg_is_not_silently_unmasked(adapter, hidden_value):
    record = adapter.records["synthetic-0"]
    values = record.eeg_features.copy()
    values[0, 0] = hidden_value
    mask = np.zeros(values.shape, dtype=bool)
    mask[0, 0] = True
    adapter.records["synthetic-0"] = replace(record, eeg_features=np.ma.array(values, mask=mask))
    with pytest.raises(SubjectDataError, match="masked"):
        dataset(adapter)[0]


def test_review_masked_flags_are_not_silently_unmasked(adapter):
    record = adapter.records["synthetic-0"]
    flags = np.ma.array(record.artifact_flags, mask=[True, False, False])
    adapter.records["synthetic-0"] = replace(record, artifact_flags=flags)
    with pytest.raises(SubjectDataError, match="masked"):
        dataset(adapter)[0]


def test_review_numpy_extended_precision_target_conversion(adapter):
    adapter.records["synthetic-0"] = replace(adapter.records["synthetic-0"], moca=np.longdouble("23.5"))
    target = dataset(adapter)[0]["moca"]
    assert target.dtype == torch.float32 and target.shape == ()
    assert target.item() == 23.5
