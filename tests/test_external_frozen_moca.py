import json
from pathlib import Path
import numpy as np
import torch
from codl.eeg import run_external_frozen_moca as run
from codl.clinical.run_pd_ft import inner_indices
from codl.clinical.ft_clinical import _modules


def test_frozen_head_train_only_statistics_and_reload(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    (tmp_path/'checkpoint.pt').write_bytes(b'synthetic encoder identity')
    monkeypatch.setattr(run,'PRE',tmp_path)
    settings=json.loads(Path('src/codl/clinical/pd_ft_seed0.json').read_text())
    settings.update(nn_epochs=3,nn_patience=2)
    e=np.random.default_rng(1).normal(size=(12,256));y=np.arange(12,dtype=float)
    ids=[f'fake{i}' for i in range(12)];test=np.ones((2,256))*50
    pred=run.fit_head(e,y,ids,test,['held1','held2'],settings,tmp_path/'head')
    p=torch.load(tmp_path/'head/checkpoint.pt',weights_only=True)
    tr,va=inner_indices(12,0,.2)
    np.testing.assert_allclose(p['eeg_mean'],e[tr].mean(0))
    assert p['target_mean']==y[tr].mean()
    assert not set(p['partitions']['test'])&set(p['partitions']['train']+p['partitions']['validation'])
    _,Net=_modules();m=Net(256,None,8,32,.2);m.load_state_dict(p['model']);m.eval()
    x=torch.tensor((test-np.array(p['eeg_mean']))/np.array(p['eeg_scale']),dtype=torch.float32)
    with torch.no_grad():replay=m(x,None,None).numpy()*p['target_std']+p['target_mean']
    np.testing.assert_allclose(pred,replay,atol=0,rtol=0)
