import numpy as np
import torch
from codl.fusion.ft_architecture_a import FTArchitectureA
from codl.fusion.pd_direct_concat import fit_concat, predict_concat


def test_real_projections_and_tokenizer_receive_gradients():
    torch.manual_seed(7)
    for cat in ([False, True], [False, True, False, False, False]):
        model = FTArchitectureA(cat)
        x = torch.randn(4,len(cat)); x[:,1] = torch.tensor([0.,1.,0.,1.])
        missing = torch.zeros_like(x,dtype=torch.bool); missing[0,-1]=True
        e = torch.randn(4,256)
        model(e,x,missing).square().mean().backward()
        probe = model.verify_gradients(e,x,missing)
        assert probe['fusion_shape'] == [4,512]
        assert model.clinical_projection.linear.in_features == len(cat)*8
        assert model.head.mlp[2].p == .1


def test_train_only_statistics_and_checkpoint_replay(tmp_path):
    torch.set_num_threads(2)
    rng=np.random.default_rng(1)
    eeg=rng.normal(size=(12,256))
    clinical=np.column_stack([np.arange(12)+40,np.arange(12)%2]).astype(float)
    settings={'nn_token_dim':8,'nn_hidden':32,'nn_dropout':.2,'nn_lr':.001,
              'nn_wd':.01,'nn_epochs':3,'nn_patience':2,'min_delta':1e-5}
    y=np.arange(12,dtype=float)
    payload,_=fit_concat(eeg[:8],clinical[:8],y[:8],eeg[8:]+100,
                         clinical[8:]+[100,0],y[8:]+100,['age','sex'],settings,0,'architecture_a')
    np.testing.assert_allclose(payload['eeg_mean'],eeg[:8].mean(axis=0))
    assert payload['target_mean']==y[:8].mean()
    assert payload['prep']['mu'][0]==clinical[:8,0].mean()
    before=predict_concat(payload,eeg,clinical)
    path=tmp_path/'synthetic.pt';torch.save(payload,path)
    loaded=torch.load(path,weights_only=True)
    np.testing.assert_array_equal(before,predict_concat(loaded,eeg,clinical))
