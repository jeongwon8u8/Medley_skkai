"""Reusable nested clinical OOF training, independent of EEG training runners."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from codl.clinical.run_pd_ft import inner_indices, write_json
from codl.clinical.ft_clinical import fit_member, infer_member
from medley.tasks import clin_model
ROOT = Path(__file__).resolve().parents[3]
FT = ROOT/'outputs/clinical_ft/clinical_ft_c0_c1_seed0_20261126'
ALPHAS = np.logspace(0,5,11)
def ridge(alpha):
    return make_pipeline(StandardScaler(),Ridge(alpha=float(alpha)))

def inner_clinical(cfg,d,fit_ids,hold_ids,columns,settings,path,fold,inner,name):
    path.mkdir(parents=True,exist_ok=True)
    if (path/'predictions.csv').exists():
        saved = pd.read_csv(path/'predictions.csv').set_index('subject')
        assert set(saved.index)==set(hold_ids)
        return saved.loc[hold_ids,'ridge'].to_numpy(),saved.loc[hold_ids,'ft'].to_numpy()
    x,y = d.loc[fit_ids,columns].to_numpy(float),d.loc[fit_ids,'target'].to_numpy()
    held_x = d.loc[hold_ids,columns].to_numpy(float)
    model = clin_model(cfg).fit(x,y)
    rp = model.predict(held_x); joblib.dump(model,path/'clinical_ridge.joblib')
    predictions = []
    member_meta = []
    for seed in range(5):
        tr,va = inner_indices(len(fit_ids),seed,.2)
        payload,history = fit_member(x[tr],y[tr],x[va],y[va],columns,settings,seed)
        groups = {'train':[fit_ids[i] for i in tr],'validation':[fit_ids[i] for i in va],'test':hold_ids}
        assert not set(hold_ids)&(set(groups['train'])|set(groups['validation']))
        payload['partitions'] = groups
        fp = path/f'ft_member{seed}.pt'; torch.save(payload,fp)
        pd.DataFrame(history).to_csv(path/f'ft_member{seed}_history.csv',index=False)
        predictions.append(infer_member(fp,held_x)[0])
        member_meta.append({'seed':seed,'selected_epoch':payload['selected_epoch'],'seconds':payload['seconds'],'partitions':groups})
    mean = np.mean(predictions,axis=0)
    pd.DataFrame({'subject':hold_ids,'ridge':rp,'ft':mean}).to_csv(path/'predictions.csv',index=False)
    write_json(path/'metadata.json',{'fit_subjects':fit_ids,'held_subjects':hold_ids,'members':member_meta,'outer_fold':fold,'inner_fold':inner,'clinical_set':name})
    return rp,mean
