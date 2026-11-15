"""C2 isolated early-stopping boundaries; real model acceptance lives in runner tests."""
import pytest

from multimodal.training_control import EarlyStopping
from multimodal.model_registry import RunnerError


@pytest.mark.parametrize("scores,patience,delta,best,epochs",[
    ([4.,3.,3.,3.,2.],2,0.,2,4),([4.,3.9,3.8],2,.2,1,3),([2.],5,0.,1,1),([3.,2.,1.],1,0.,3,3)])
def test_early_stopping_boundaries(scores,patience,delta,best,epochs):
    run=EarlyStopping(patience=patience,min_delta=delta)
    for score in scores:
        improved,stop=run.observe(score)
        if stop: break
    assert run.best_epoch==best and run.epochs==epochs


@pytest.mark.parametrize("score",[float("nan"),float("inf"),-1.,True])
def test_nonfinite_or_invalid_validation_fails(score):
    with pytest.raises(RunnerError): EarlyStopping().observe(score)


@pytest.mark.parametrize("patience,delta",[(0,0.),(True,0.),(1,-1.),(1,float("nan"))])
def test_invalid_stopping_options(patience,delta):
    with pytest.raises(RunnerError): EarlyStopping(patience=patience,min_delta=delta)
