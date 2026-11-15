"""Analytic paired/cluster/multiplicity tests; no LLMs, data downloads or training."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from multimodal.prediction_matching import MatchedPair
from multimodal.paired_bootstrap import paired_bootstrap,cluster_row_indices
from multimodal.evaluation_metrics import core_metrics,paired_metrics
from multimodal.experiment_evaluation import AnalysisError


def pair_fixture():
    frame=pd.DataFrame(dict(seed=[0,0,1,1],fold=[1,1,2,1],subject=["a","b","a","c"],
        target=[20.,22.,20.,21.],subgroup=[None]*4,candidate_prediction=[21.,22.,23.,22.],reference_prediction=[22.,23.,21.,24.]))
    return MatchedPair("candidate","reference",frame,{}, {})


def test_cluster_membership_and_duplicate_draw_multiplicity():
    pair=pair_fixture();rows=cluster_row_indices(pair.frame.subject.to_numpy(),np.array([0,0,2]))
    assert rows.tolist()==[0,2,0,2,3]
    frame=pair.frame.iloc[rows]
    a=core_metrics(frame.target,frame.candidate_prediction);b=core_metrics(frame.target,frame.reference_prediction)
    assert a["MAE"]==b["MAE"]==1.8
    assert core_metrics(pair.frame.iloc[[0,2,3]].target,pair.frame.iloc[[0,2,3]].candidate_prediction)["MAE"]!=1.8


def test_bootstrap_exact_draws_percentiles_and_signed_estimate():
    pair=pair_fixture();result=paired_bootstrap(pair,replicates=20,seed=7,min_subjects=2,confidence=.8)
    rng=np.random.default_rng(7);manual=[]
    for _ in range(20):
        rows=cluster_row_indices(pair.frame.subject.to_numpy(),rng.integers(0,3,size=3))
        frame=pair.frame.iloc[rows]
        a=core_metrics(frame.target,frame.candidate_prediction);b=core_metrics(frame.target,frame.reference_prediction)
        manual.append([a["MAE"]-b["MAE"],a["RMSE"]-b["RMSE"]])
    np.testing.assert_array_equal(result.draws,np.array(manual))
    assert result.metadata["delta_MAE"]==-.5
    assert result.metadata["delta_RMSE"]==pytest.approx(np.sqrt(11/4)-np.sqrt(15/4))
    endpoints=np.percentile(manual,[10,90],axis=0,method="linear")
    assert result.metadata["delta_MAE_lower"]==pytest.approx(endpoints[0,0])
    assert result.metadata["delta_RMSE_upper"]==pytest.approx(endpoints[1,1])
    assert result.metadata["scope"]=="pooled_repeated_cv" and result.metadata["n_unique_subjects"]==3


def test_reproducibility_seed_variation_swap_and_zero():
    pair=pair_fixture();a=paired_bootstrap(pair,replicates=40,seed=19,min_subjects=2)
    b=paired_bootstrap(pair,replicates=40,seed=19,min_subjects=2)
    c=paired_bootstrap(pair,replicates=40,seed=20,min_subjects=2)
    np.testing.assert_array_equal(a.draws,b.draws);assert not np.array_equal(a.draws,c.draws)
    frame=pair.frame.rename(columns={"candidate_prediction":"reference_prediction","reference_prediction":"candidate_prediction"})
    swapped=paired_bootstrap(MatchedPair("reference","candidate",frame,{},{}),replicates=40,seed=19,min_subjects=2)
    np.testing.assert_array_equal(swapped.draws,-a.draws)
    assert swapped.metadata["delta_MAE_lower"]==pytest.approx(-a.metadata["delta_MAE_upper"])
    frame["candidate_prediction"]=frame.reference_prediction
    zero=paired_bootstrap(MatchedPair("equalA","equalB",frame,{},{}),replicates=20,min_subjects=2)
    assert np.count_nonzero(zero.draws)==0 and zero.metadata["delta_RMSE_upper"]==0.


@pytest.mark.parametrize("options",[{"replicates":1},{"confidence":1.},{"seed":True},{"min_subjects":1},
    {"min_subjects":5},{"aggregation":"row_bootstrap"},{"aggregation":"mean_runs"}])
def test_invalid_or_unsupported_bootstrap_is_explicit(options):
    with pytest.raises(AnalysisError): paired_bootstrap(pair_fixture(),**options)


def test_fixed_seed_fold_scope_and_bad_pair_metadata():
    pair=pair_fixture();frame=pair.frame.iloc[:2].copy()
    fixed=paired_bootstrap(MatchedPair("a","b",frame,{},{}),replicates=10,min_subjects=2,aggregation="mean_runs")
    assert fixed.metadata["scope"]=="single_fixed_run"
    bad=pair.frame;bad.loc[2,"target"]=19.
    with pytest.raises(AnalysisError): paired_bootstrap(MatchedPair("a","b",bad,{},{}),replicates=10,min_subjects=2)
    bad=pair.frame;bad=pd.concat([bad,bad.iloc[:1]])
    with pytest.raises(AnalysisError): paired_bootstrap(MatchedPair("a","b",bad,{},{}),replicates=10,min_subjects=2)


@pytest.mark.parametrize("draw",[[0,0],[0,0,9],[0.,0.,2.],[True,True,False]])
def test_invalid_cluster_draws(draw):
    with pytest.raises(AnalysisError): cluster_row_indices(pair_fixture().frame.subject,draw)
