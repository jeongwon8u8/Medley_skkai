import numpy as np
from codl.fusion.pd_direct_concat import fit_concat,predict_concat


def test_concat_preprocessing_is_training_only_and_state_replays():
    rng = np.random.default_rng(0)
    eeg = rng.normal(size=(8,256))
    clinical = np.column_stack([np.arange(30,38),np.arange(8)%2]).astype(float)
    clinical[2,0] = np.nan
    settings = {'nn_token_dim':8,'nn_hidden':32,'nn_dropout':.2,'nn_lr':.001,
                'nn_wd':.01,'nn_epochs':3,'nn_patience':2,'min_delta':1e-5}
    payload,log = fit_concat(eeg,clinical,np.arange(20,28,dtype=float),
                             np.ones((2,256))*1000,np.array([[9999,0],[8888,1]]),
                             np.array([20.,25.]),['age','sex'],settings,0)
    np.testing.assert_allclose(payload['eeg_mean'],eeg.mean(axis=0))
    np.testing.assert_allclose(payload['prep']['mu'][0],np.nanmean(clinical[:,0]))
    assert payload['clinical_pretraining'] is False
    a = predict_concat(payload,eeg,clinical)
    b = predict_concat(payload,eeg,clinical)
    np.testing.assert_array_equal(a,b)
    assert np.isfinite(a).all() and len(log)<=3
