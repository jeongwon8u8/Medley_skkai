"""Audit actual 3-cohort checkpoint and every downstream outer-test prediction."""
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'medley_pd')]
from codl.eeg.run_external_frozen_moca import OUT,PRE,OLD,load_reference,sha,write_json
from codl.clinical.ft_clinical import _modules,infer_member
from codl.fusion.pd_direct_concat import predict_concat
from medley.metrics import regression_metrics
from medley.participants import clinical_sets


def replay_head(folder,e,ids,d):
    p=torch.load(folder/'checkpoint.pt',weights_only=True)
    groups=p['partitions'];assert not set(groups['test'])&set(groups['train']+groups['validation'])
    assert not set(groups['train'])&set(groups['validation'])
    assert set(ids)==set(groups['test'])
    scaler=StandardScaler().fit(np.stack([e[s] for s in groups['train']]))
    np.testing.assert_allclose(scaler.mean_,p['eeg_mean'])
    np.testing.assert_allclose(scaler.scale_,p['eeg_scale'])
    y=d.loc[groups['train'],'target'].to_numpy()
    np.testing.assert_allclose([y.mean(),y.std()],[p['target_mean'],p['target_std']])
    assert p['encoder_checkpoint_sha256']==sha(PRE/'checkpoint.pt')
    history=pd.read_csv(folder/'history.csv')
    assert p['selected_epoch'] in history.epoch.values
    _,Net=_modules();s=p['settings'];m=Net(256,None,s['nn_token_dim'],s['nn_hidden'],s['nn_dropout'])
    m.load_state_dict(p['model']);m.eval()
    x=torch.tensor(scaler.transform(np.stack([e[i] for i in ids])),dtype=torch.float32)
    with torch.no_grad():pred=m(x,None,None).numpy()*p['target_std']+p['target_mean']
    return pred,p


def main():
    torch.set_num_threads(1)
    cfg,frame,splits,br,_=load_reference();d=frame.set_index('subject',drop=False)
    people=pd.read_csv(PRE/'participants_and_split.tsv',sep='\t')
    assert len(people)==121 and people.person_key.nunique()==121
    assert set(people.dataset)=={'turku','unm','ucsd'}
    assert people.label_pd.value_counts().to_dict()=={0:61,1:60}
    ck=torch.load(PRE/'checkpoint.pt',weights_only=True)
    assert set(ck['provenance']['train_people'])==set(people.loc[people.split.eq('train'),'person_key'])
    assert set(ck['provenance']['validation_people'])==set(people.loc[people.split.eq('validation'),'person_key'])
    assert not set(ck['provenance']['train_people'])&set(ck['provenance']['validation_people'])
    history=pd.read_csv(PRE/'history.csv')
    assert ck['provenance']['selected_epoch']==int(history.loc[history.validation_person_bce.idxmin(),'epoch'])
    with np.load(OUT/'patient_vectors.npz') as z:
        assert z['vectors'].shape==(113,256) and np.isfinite(z['vectors']).all()
        e=dict(zip(z['subject_id'],z['vectors']))
    assert set(e)==set(d.index)
    pred=pd.read_csv(OUT/'predictions.csv.gz')
    expected={'GNN_MLP','GNN_Ridge'}|{f'GNN_{m}_{c}' for m in ['ClinicalRidge_Stack','FT_Stack','FT_Concat'] for c in ['C0','C1']}
    assert set(pred.model)==expected and len(pred)==8*113
    assert not pred.duplicated(['model','subject_id']).any()
    for name,g in pred.groupby('model'):
        assert set(g.subject_id)==set(d.index)
        np.testing.assert_array_equal(g.fold,splits.set_index('subject').loc[g.subject_id,'fold'])
        np.testing.assert_array_equal(g.y,d.loc[g.subject_id,'target'])
    foldmetrics=[];selected=[];replays=0
    for fold in range(1,6):
        folder=OUT/f'outer{fold}'
        test=d.index[d.index.isin(splits.loc[splits.fold.eq(fold),'subject'])].tolist()
        train=d.index[~d.index.isin(test)].tolist();E=lambda ids:np.stack([e[s] for s in ids])
        get=lambda name:pred[pred.model.eq(name)&pred.fold.eq(fold)].set_index('subject_id').loc[test,'yhat'].to_numpy()
        model=joblib.load(folder/'ridge.joblib');np.testing.assert_allclose(model.predict(E(test)),get('GNN_Ridge'),atol=1e-6)
        np.testing.assert_allclose(model.best_estimator_[0].mean_,E(train).mean(0),rtol=1e-6,atol=1e-8)
        replay,p=replay_head(folder/'mlp',e,test,d);np.testing.assert_allclose(replay,get('GNN_MLP'),atol=1e-5)
        assert set(p['partitions']['train']+p['partitions']['validation'])==set(train)
        selected.append({'fold':fold,'model':'GNN_MLP','member':0,'epoch':p['selected_epoch'],'n_train':len(p['partitions']['train']),'n_validation':len(p['partitions']['validation']),'n_test':len(test),'seconds':p['seconds']})
        inner_oof=pd.read_csv(folder/'inner_oof.csv').set_index('subject_id')
        assert set(inner_oof.index)==set(train) and inner_oof.index.is_unique
        for inner,(fi,hi) in enumerate(KFold(3,shuffle=True,random_state=0).split(train),1):
            fitids=[train[i] for i in fi];held=[train[i] for i in hi]
            replay,p=replay_head(folder/f'inner{inner}_mlp',e,held,d)
            assert set(p['partitions']['train']+p['partitions']['validation'])==set(fitids)
            np.testing.assert_allclose(replay,inner_oof.loc[held,'gnn'],atol=1e-5)
            for c,cols in clinical_sets(cfg).items():
                source=OLD/f'outer{fold}'/f'inner{inner}_{c}'
                if not (source/'metadata.json').exists():source=OUT/f'outer{fold}'/f'inner{inner}_{c}'
                clinical=joblib.load(source/'clinical_ridge.joblib')
                np.testing.assert_allclose(clinical.predict(d.loc[held,cols].to_numpy(float)),inner_oof.loc[held,'ridge_'+c],atol=1e-6)
                ft=[]
                for member in range(5):
                    path=source/f'ft_member{member}.pt';p=torch.load(path,weights_only=True)
                    assert set(p['partitions']['train']+p['partitions']['validation'])==set(fitids)
                    assert not set(held+test)&set(fitids)
                    ft.append(infer_member(path,d.loc[held,cols].to_numpy(float))[0])
                np.testing.assert_allclose(np.mean(ft,axis=0),inner_oof.loc[held,'ft_'+c],atol=1e-5)
        from codl.fusion.run_pd_direct_fusion import FT
        bf=pd.read_csv(FT/'predictions.csv.gz')
        for c,cols in clinical_sets(cfg).items():
            for label,base,bname in [('ClinicalRidge',br,'clinical:'),('FT',bf,'clinical_ft:')]:
                name=f'GNN_{label}_Stack_{c}';meta=joblib.load(folder/(name+'.joblib'))
                cp=base[base.model.eq(bname+c)].set_index('subject').loc[test,'yhat']
                replay=meta.predict(np.column_stack([get('GNN_MLP'),cp]))
                np.testing.assert_allclose(replay,get(name),atol=1e-6)
            members=[]
            for seed in range(5):
                p=torch.load(folder/f'concat_{c}_member{seed}.pt',weights_only=True)
                parts=p['partitions'];assert set(parts['test'])==set(test)
                assert set(parts['train']+parts['validation'])==set(train)
                assert not set(parts['train'])&set(parts['validation'])
                scale=StandardScaler().fit(E(parts['train']))
                np.testing.assert_allclose(p['eeg_mean'],scale.mean_)
                np.testing.assert_allclose(p['eeg_scale'],scale.scale_)
                np.testing.assert_allclose(p['target_mean'],d.loc[parts['train'],'target'].mean())
                members.append(predict_concat(p,E(test),d.loc[test,cols].to_numpy(float)))
                selected.append({'fold':fold,'model':'GNN_FT_Concat_'+c,'member':seed,'epoch':p['selected_epoch'],'n_train':len(parts['train']),'n_validation':len(parts['validation']),'n_test':len(test),'seconds':p['seconds']})
            np.testing.assert_allclose(np.mean(members,axis=0),get('GNN_FT_Concat_'+c),atol=1e-5)
        for name,g in pred[pred.fold.eq(fold)].groupby('model'):
            foldmetrics.append({'model':name,'fold':fold,**regression_metrics(g.y,g.yhat),'true_mean':float(g.y.mean()),'true_std':float(g.y.std(ddof=0)),'pred_mean':float(g.yhat.mean()),'pred_std':float(g.yhat.std(ddof=0))})
        replays+=1
    pd.DataFrame(foldmetrics).to_csv(OUT/'fold_metrics.csv',index=False)
    pd.DataFrame(selected).to_csv(OUT/'selected_epochs.csv',index=False)
    audit={'status':'passed','new_models':8,'people_per_model':113,'outer_folds':5,
        'checkpoint_replay_folds':replays,'external_people':121,'external_train':96,'external_validation':25,
        'external_selected_epoch':ck['provenance']['selected_epoch'],'outer_test_used_for_selection':False,
        'internal_oof_excludes_training_and_early_stopping':True,'encoder_and_BN_frozen':True,
        'clinical_inner_predictions_replayed_from_saved_models':True,
        'checkpoint_sha256':sha(PRE/'checkpoint.pt'),'patient_vectors_sha256':sha(OUT/'patient_vectors.npz'),
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),Path(__file__).with_name('run_external_frozen_moca.py'),ROOT/'src/codl/fusion/pd_direct_concat.py',ROOT/'src/codl/clinical/ft_clinical.py']}}
    write_json(OUT/'audit.json',audit);print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
