from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from moca_controlled_baselines import MODELS, RidgePath, fit_fold


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_ridge_matches_reference_and_stack_never_uses_test_targets(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(2)
    data = rng.normal(size=(22, 8))
    target = rng.normal(size=22) + 25
    train, val, test = np.arange(12), np.arange(12, 17), np.arange(17, 22)

    def tensor(values):
        return torch.as_tensor(values, device=device, dtype=torch.float64)

    alphas = tensor([.01, 1., 10.])
    x, y = tensor(data), tensor(target)
    actual = RidgePath(x[train], y[train], alphas).predict(x[test]).cpu().numpy()
    scaler = StandardScaler().fit(data[train])
    for i, alpha in enumerate((.01, 1., 10.)):
        reference = Ridge(alpha=alpha).fit(scaler.transform(data[train]), target[train])
        np.testing.assert_allclose(actual[:, i], reference.predict(scaler.transform(data[test])), atol=1e-10)

    clinical = {"C0": tensor(data[:, :2]), "C1": tensor(data[:, :4])}
    clinical["C1"][0, 0] = float("nan")
    predictions, selection = fit_fold(x, clinical, y, train, val, test, alphas, alphas, 3)
    changed = y.clone()
    changed[test] += 20
    again, selection_again = fit_fold(x, clinical, changed, train, val, test, alphas, alphas, 3)
    assert selection == selection_again
    assert set(predictions) == set(MODELS)
    for model in MODELS:
        np.testing.assert_allclose(predictions[model], again[model], atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_parallel_cuda_folds_match_serial(adapter_files, tmp_path):
    import pandas as pd
    from moca_controlled_baselines import controlled_baselines
    from test_multimodal_experiment_runner import execution_inputs

    _, _, protocol = execution_inputs(adapter_files, tmp_path)
    cohort = protocol.validate_cohort()
    splits = {(0, fold): protocol.get_outer_split(seed=0, fold=fold) for fold in (1, 2)}
    serial, _ = controlled_baselines(protocol, cohort, splits, tmp_path / "serial", device="cuda:0")
    parallel, _ = controlled_baselines(protocol, cohort, splits, tmp_path / "parallel", device="cuda:0", workers=5)
    pd.testing.assert_frame_equal(serial, parallel, atol=1e-10, rtol=0)


from test_multimodal_data_adapter import files as adapter_files
