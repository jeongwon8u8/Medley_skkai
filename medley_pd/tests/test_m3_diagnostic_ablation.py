from pathlib import Path
import sys

import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from m3_ablation import ARMS, planned, subgroup_metrics
from multimodal.data_adapter import SubjectSplit
from multimodal.eeg_pooling import build_eeg_pooling
from multimodal.experiment_protocol import select_inner_validation


def test_stratified_quotas_and_no_outer_test_labels():
    subjects = tuple(f"sub-{i:03d}" for i in range(20))
    outer = SubjectSplit(0, 1, subjects, ("heldout",))
    targets = dict(zip(subjects, [20] * 10 + [26] * 5 + [29] * 5))
    a = select_inner_validation(outer, policy="sha256_moca_bands_v1", targets=targets)
    b = select_inner_validation(outer, policy="sha256_moca_bands_v1", targets=dict(reversed(list(targets.items()))))
    assert a == b and a.test_subjects == ("heldout",)
    assert sorted(targets[s] for s in a.validation_subjects) == [20, 20, 26, 29]
    with pytest.raises(ValueError, match="outer-train"):
        select_inner_validation(outer, policy="sha256_moca_bands_v1", targets={**targets, "heldout": 30})


def test_fusion_arms_receive_identical_frozen_pooling_and_shared_training():
    late = planned("E_fixed_mean_late", 1).config
    soft = planned("D_fixed_mean_soft", 1).config
    assert late.settings["training"] == soft.settings["training"]
    assert late.settings["llm"] == soft.settings["llm"]
    late_pool, soft_pool = build_eeg_pooling(late), build_eeg_pooling(soft)
    assert not list(late_pool.parameters()) and not list(soft_pool.parameters())
    features = torch.randn(2, 3, 3800)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    torch.testing.assert_close(late_pool(features, mask), soft_pool(features, mask), atol=0, rtol=0)
    assert len(ARMS) == 5


def test_tail_metrics_include_counts_and_signed_bias():
    rows = pd.DataFrame({"y":[20,26,30], "yhat":[22,26,27]})
    all_rows, low, high = list(subgroup_metrics(rows, arm="probe", fold=5))
    assert all_rows["n"] == 3
    assert (low["n"], low["MAE"], low["bias"]) == (1,2,2)
    assert (high["n"], high["MAE"], high["bias"]) == (1,3,-3)


@pytest.mark.parametrize("arm", ["D_fixed_mean_soft", "E_fixed_mean_late"])
def test_fixed_projection_checkpoint_is_reconstructed_without_training(arm, adapter_files, tmp_path):
    from multimodal.experiment_protocol import ExperimentProtocol
    from multimodal.model_registry import create_runtime
    from multimodal.training_control import verify_restoration
    from test_multimodal_experiment_runner import execution_inputs
    from test_multimodal_fusion_b import backbone

    model = ARMS[arm][0]
    cfg, adapter, _ = execution_inputs(adapter_files, tmp_path, model)
    cfg.settings["ablation"]["enabled"] = True
    cfg.settings["eeg"]["pooling"].update(type="mean", num_queries=1, fixed_projection=True)
    prepared = ExperimentProtocol(cfg, adapter).prepare(seed=0, fold=1)
    def factory(config):
        from multimodal.llm_backbone import prepare_llm_backbone
        tiny = backbone()
        return prepare_llm_backbone(config, tiny.model, tiny.tokenizer)
    runtime = create_runtime(model, cfg, prepared, backbone_factory=factory)
    first = runtime.model.pooling.input_projection.weight.clone()
    assert not runtime.model.pooling.input_projection.weight.requires_grad
    checkpoint = tmp_path / "checkpoint"
    runtime.save(checkpoint)
    fresh = create_runtime(model, cfg, prepared, backbone_factory=factory)
    verify_restoration(fresh, checkpoint, best_epoch=0)
    torch.testing.assert_close(first, fresh.model.pooling.input_projection.weight, atol=0, rtol=0)


from test_multimodal_data_adapter import files as adapter_files


def test_de45_profile_launches_only_four_jobs(tmp_path, monkeypatch):
    import m3_ablation as a

    seen, reads, reports = [], [], []

    class Process:
        def __init__(self, command, **options):
            seen.append(tuple(command[-4:]))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def poll(self):
            return 0

        def wait(self):
            return 0

    monkeypatch.setattr(a, "OUT", tmp_path)
    monkeypatch.setattr(a, "job", lambda arm, fold: reads.append((arm, fold)) or True)
    monkeypatch.setattr(a, "gpu_capacity", lambda workers: workers)
    monkeypatch.setattr(a.shutil, "which", lambda program: None)
    monkeypatch.setattr(a.subprocess, "Popen", Process)
    monkeypatch.setattr(a, "report", reports.append)
    a.launch(1, "de45")
    assert reads == [(a.ANCHOR, 4), (a.ANCHOR, 5)]
    assert seen == [("--arm", arm, "--fold", str(fold)) for arm in a.FOCUSED_ARMS for fold in (4, 5)]
    assert reports == ["de45"]
    with pytest.raises(ValueError, match="max-parallel 1"):
        a.launch(2, "de45")


def test_de45_report_pairs_only_the_44_shared_patients(tmp_path, monkeypatch):
    import m3_ablation as a
    from types import SimpleNamespace

    def fake_series(arm, fold):
        offset = {a.ANCHOR: 0, a.FOCUSED_ARMS[0]: 1, a.FOCUSED_ARMS[1]: 2}[arm]
        return SimpleNamespace(
            runs={(0, fold): {"completion": {"execution_id": f"{arm}-{fold}"}}},
            frame=pd.DataFrame([{"seed": 0, "fold": fold, "subject": f"{fold}-{i}",
                                 "y": 20 + i % 10, "yhat": 20 + i % 10 + offset} for i in range(22)]),
        )

    monkeypatch.setattr(a, "OUT", tmp_path)
    monkeypatch.setattr(a, "job", fake_series)
    a.report("de45")
    report, = tmp_path.glob("report_*")
    rows = pd.read_csv(report / "paired_predictions.csv")
    assert len(rows) == 44 and set(rows.fold) == {4, 5}
    assert rows.delta_absolute_error_E_minus_D.eq(1).all()
    metrics = pd.read_csv(report / "metrics.csv")
    assert set(metrics.fold.astype(str)) == {"4", "5", "matched_4_5"}
