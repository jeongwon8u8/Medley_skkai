"""Three-cohort external PD/HC encoder frozen for seed0 PD-EEG regression/fusion."""
import argparse
import json
from pathlib import Path
import sys
import time
import copy
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'medley_pd')]
from codl.clinical.run_pd_ft import load_reference, inner_indices, write_json, sha
from codl.clinical.ft_clinical import _modules
from codl.eeg.external_pd_hc import load_checkpoint
from codl.eeg.cache_pd_graphs import context
from codl.eeg.directed_pdc import operator
from codl.fusion.pd_direct_concat import fit_concat,predict_concat
from codl.fusion.clinical_oof import ridge,ALPHAS,inner_clinical,FT
from medley.participants import clinical_sets
from medley.metrics import regression_metrics

OUT=ROOT/'outputs/eeg_gnn/external_3cohort121_frozen_pd_eeg_moca_seed0'
PRE=ROOT/'outputs/external_pd_hc/external_3cohort121_pd_hc_directed_cheb_seed0'
OLD=ROOT/'outputs/fusion/pd_eeg_moca_direct_supervised_seed0_7settings'


def fit_head(e,y,ids,test_e,test_ids,settings,path,seed=0):
    """Team EEG-only Net; scaler/target stats fit only inner training people."""
    path.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    tr,va=inner_indices(len(ids),seed,.2)
    scaler=StandardScaler().fit(e[tr])
    x=torch.tensor(scaler.transform(e),dtype=torch.float32)
    mean=float(y[tr].mean());std=max(float(y[tr].std()),1e-8)
    target=torch.tensor((y-mean)/std,dtype=torch.float32)
    torch.manual_seed(seed);_,Net=_modules()
    model=Net(256,None,settings['nn_token_dim'],settings['nn_hidden'],settings['nn_dropout'])
    opt=torch.optim.AdamW(model.parameters(),lr=settings['nn_lr'],weight_decay=settings['nn_wd'])
    loss_fn=torch.nn.HuberLoss(delta=1.)
    best=float('inf');selected=0;history=[];state=None
    for epoch in range(1,settings['nn_epochs']+1):
        model.train();opt.zero_grad();loss=loss_fn(model(x[tr],None,None),target[tr])
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step();model.eval()
        with torch.no_grad():vl=float(loss_fn(model(x[va],None,None),target[va]))
        assert np.isfinite(vl)
        history.append({'epoch':epoch,'train_huber':float(loss.detach()),'validation_huber':vl})
        if vl<best-settings['min_delta']:
            best=vl;selected=epoch;state=copy.deepcopy(model.state_dict())
        if epoch-selected>=settings['nn_patience']:break
    model.load_state_dict(state);model.eval()
    xt=torch.tensor(scaler.transform(test_e),dtype=torch.float32)
    with torch.no_grad():pred=model(xt,None,None).numpy()*std+mean
    groups={'train':[ids[i] for i in tr],'validation':[ids[i] for i in va],'test':list(test_ids)}
    assert not set(test_ids)&set(ids)
    payload={'model':state,'settings':settings,'eeg_mean':scaler.mean_.tolist(),
             'eeg_scale':scaler.scale_.tolist(),'target_mean':mean,'target_std':std,
             'selected_epoch':selected,'partitions':groups,'seed':seed,
             'seconds':time.perf_counter()-started,'encoder_checkpoint_sha256':sha(PRE/'checkpoint.pt')}
    torch.save(payload,path/'checkpoint.pt');pd.DataFrame(history).to_csv(path/'history.csv',index=False)
    write_json(path/'metadata.json',{k:v for k,v in payload.items() if k!='model'})
    return pred


def extract(d):
    out=OUT/'features';out.mkdir(exist_ok=True)
    model,payload=load_checkpoint(PRE/'checkpoint.pt');model.eval()
    for p in model.parameters():p.requires_grad_(False)
    before={k:v.clone() for k,v in model.state_dict().items()}
    settings,cfg,frame,splits,cache,_=context()
    assert set(frame.subject)==set(d.index)
    for k in ['fs','seg_sec','channels','filter','orders','mvar_ridge','graph_band','lambda_proxy']:
        assert settings[k]==payload['settings'][k],k
    identity={'checkpoint_sha256':sha(PRE/'checkpoint.pt'),'cache_config_sha256':sha(cache/'config.json'),
              'mode':'eval; all parameters and BatchNorm statistics frozen; mean of all valid windows',
              'dimension':256,'external_people':121,'source_cache':str(cache)}
    if (out/'identity.json').exists():assert json.loads((out/'identity.json').read_text())==identity
    else:write_json(out/'identity.json',identity)
    vectors=[];timings=[];started=time.perf_counter()
    with torch.inference_mode():
        for n,sid in enumerate(d.index,1):
            target=out/(sid+'.npz');source=cache/(sid+'.npz');digest=sha(source)
            if target.exists():
                with np.load(target) as z:
                    assert z['source_sha256'].item()==digest
                    vector=z['vector'];nw=int(z['n_windows'])
            else:
                t=time.perf_counter()
                with np.load(source) as z:
                    x=torch.tensor(z['x']);lap=torch.tensor(np.stack([operator(a) for a in z['adjacency'][:,1]]),dtype=torch.float32)
                chunks=[]
                for i in range(0,len(x),64):chunks.append(model.encode(x[i:i+64],lap[i:i+64]).numpy())
                vector=np.concatenate(chunks).mean(axis=0);nw=len(x)
                assert vector.shape==(256,) and np.isfinite(vector).all()
                np.savez_compressed(target,vector=vector,n_windows=nw,source_sha256=digest)
                timings.append({'subject_id':sid,'seconds':time.perf_counter()-t,'n_windows':nw})
            vectors.append(vector)
            if n%10==0:print(f'Features {n}/113',flush=True)
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    np.savez_compressed(OUT/'patient_vectors.npz',subject_id=np.array(d.index.tolist(),dtype=str),vectors=np.stack(vectors))
    pd.DataFrame(timings).to_csv(out/'timings.csv',index=False)
    write_json(out/'completed.json',{'n_people':len(d),'seconds':time.perf_counter()-started,
                                   'encoder_state_unchanged':True,'BatchNorm_unchanged':True})


def save_prediction(name,fold,ids,pred,d):
    path=OUT/name;path.mkdir(exist_ok=True)
    assert np.isfinite(pred).all()
    pd.DataFrame({'subject_id':ids,'fold':fold,'seed':0,'model':name,
                  'y':d.loc[ids,'target'].to_numpy(),'yhat':pred}).to_csv(path/f'fold{fold}.csv',index=False)


def collect(d,splits,baselines):
    rows=[];complete=[]
    for name,base in baselines.items():
        rows.append({'model':name,'status':'completed_reused','n':113,**regression_metrics(base.y,base.yhat)})
    for folder in OUT.iterdir():
        if not folder.is_dir():continue
        paths=sorted(folder.glob('fold[1-5].csv'))
        if not paths:continue
        p=pd.concat([pd.read_csv(f) for f in paths],ignore_index=True)
        assert not p.subject_id.duplicated().any()
        assert set(p.subject_id)<=set(d.index)
        np.testing.assert_array_equal(p.y,d.loc[p.subject_id,'target'])
        np.testing.assert_array_equal(p.fold,splits.set_index('subject').loc[p.subject_id,'fold'])
        p.to_csv(folder/'predictions.csv.gz',index=False)
        if len(p)==113:
            complete.append(p);rows.append({'model':folder.name,'status':'completed','n':113,**regression_metrics(p.y,p.yhat)})
        else:rows.append({'model':folder.name,'status':'incomplete','n':len(p)})
    table=pd.DataFrame(rows)
    eeg=table.loc[table.model.eq('GNN_MLP'),'MAE']
    for i,r in table.iterrows():
        if r.model.startswith('GNN_') and r.model.endswith(('C0','C1')) and r.status=='completed':
            ref=('Ridge_' if 'ClinicalRidge' in r.model else 'FT_')+r.model[-2:]
            table.loc[i,'delta_MAE_clinical']=r.MAE-float(table.loc[table.model.eq(ref),'MAE'].iloc[0])
            if len(eeg):table.loc[i,'delta_MAE_EEG_MLP']=r.MAE-float(eeg.iloc[0])
    table.to_csv(OUT/'comparison.csv',index=False)
    if complete:pd.concat(complete).to_csv(OUT/'predictions.csv.gz',index=False)
    return table


def main(stage):
    OUT.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    cfg,frame,splits,br,_=load_reference();d=frame.set_index('subject',drop=False)
    settings=json.loads((ROOT/'src/codl/clinical/pd_ft_seed0.json').read_text())
    bf=pd.read_csv(FT/'predictions.csv.gz')
    baselines={}
    for c in ('C0','C1'):
        for prefix,source,modelname in [('Ridge_',br,'clinical:'),('FT_',bf,'clinical_ft:')]:
            p=source[source.model.eq(modelname+c)].copy()
            assert len(p)==113 and p.subject.nunique()==113
            np.testing.assert_array_equal(p.y,d.loc[p.subject,'target'])
            np.testing.assert_array_equal(p.fold,splits.set_index('subject').loc[p.subject,'fold'])
            baselines[prefix+c]=p
    if stage=='extract':extract(d);return
    if stage=='report':print(collect(d,splits,baselines).to_string(index=False));return
    with np.load(OUT/'patient_vectors.npz') as z:
        assert set(z['subject_id'])==set(d.index)
        mapping=dict(zip(z['subject_id'],z['vectors']))
    E=lambda ids:np.stack([mapping[s] for s in ids])
    Y=lambda ids:d.loc[ids,'target'].to_numpy()
    write_json(OUT/'protocol.json',{'scope':'external 3-cohort N121 frozen encoder; PD-EEG N113 MoCA',
        'seed':0,'split_sha256':sha(ROOT/'medley_pd/splits/pd_eeg__moca.csv',True),
        'checkpoint_sha256':sha(PRE/'checkpoint.pt'),'eeg_mlp_members':1,'fusion_concat_members':5,
        'stack_alpha':1.,'inner_folds':3,'clinical_settings':settings,
        'head':'team Net: LayerNorm(256), Linear(256,32), GELU, Dropout(.2), Linear(32,1)',
        'concat':'frozen EEG256 + flattened FT tokens; fresh team Net per member; predictions averaged',
        'encoder_pretraining_exclusions':['BrainLat','PD-EEG','Iowa']})
    for fold in range(1,6):
        test=d.index[d.index.isin(splits.loc[splits.fold.eq(fold),'subject'])].tolist()
        train=d.index[~d.index.isin(test)].tolist()
        folder=OUT/f'outer{fold}';folder.mkdir(exist_ok=True)
        if stage=='eeg':
            if not (OUT/'GNN_Ridge'/f'fold{fold}.csv').exists():
                model=GridSearchCV(ridge(1.),{'ridge__alpha':ALPHAS},cv=KFold(3,shuffle=True,random_state=0),scoring='neg_mean_squared_error').fit(E(train),Y(train))
                joblib.dump(model,folder/'ridge.joblib')
                write_json(folder/'ridge_partitions.json',{'outer_train':train,'outer_test':test,'selected_alpha':float(model.best_params_['ridge__alpha'])})
                save_prediction('GNN_Ridge',fold,test,model.predict(E(test)),d);collect(d,splits,baselines)
            if not (OUT/'GNN_MLP'/f'fold{fold}.csv').exists():
                pred=fit_head(E(train),Y(train),train,E(test),test,settings,folder/'mlp')
                save_prediction('GNN_MLP',fold,test,pred,d);collect(d,splits,baselines)
        if stage=='fusion':
            scores={'gnn':np.full(len(train),np.nan)}
            for c in ('C0','C1'):scores['ridge_'+c]=np.full(len(train),np.nan);scores['ft_'+c]=np.full(len(train),np.nan)
            lineage=[]
            for inner,(fi,hi) in enumerate(KFold(3,shuffle=True,random_state=0).split(train),1):
                fitids=[train[i] for i in fi];held=[train[i] for i in hi]
                ipath=folder/f'inner{inner}_eeg_predictions.csv'
                if ipath.exists():
                    p=pd.read_csv(ipath).set_index('subject');assert set(p.index)==set(held);ep=p.loc[held,'yhat'].to_numpy()
                else:
                    ep=fit_head(E(fitids),Y(fitids),fitids,E(held),held,settings,folder/f'inner{inner}_mlp')
                    pd.DataFrame({'subject':held,'yhat':ep}).to_csv(ipath,index=False)
                scores['gnn'][hi]=ep
                for c,cols in clinical_sets(cfg).items():
                    previous=OLD/f'outer{fold}'/f'inner{inner}_{c}'
                    if not (previous/'metadata.json').exists():
                        previous=folder/f'inner{inner}_{c}'
                        inner_clinical(cfg,d,fitids,held,cols,settings,previous,fold,inner,c)
                    meta=json.loads((previous/'metadata.json').read_text())
                    assert meta['fit_subjects']==fitids and meta['held_subjects']==held
                    for member in range(5):
                        ck=torch.load(previous/f'ft_member{member}.pt',map_location='cpu',weights_only=True)
                        assert ck['settings']==settings and ck['columns']==cols
                        assert set(ck['partitions']['train']+ck['partitions']['validation'])==set(fitids)
                        assert set(ck['partitions']['test'])==set(held)
                        assert not set(test)&set(fitids+held)
                    rp,fp=inner_clinical(cfg,d,fitids,held,cols,settings,previous,fold,inner,c)
                    scores['ridge_'+c][hi]=rp;scores['ft_'+c][hi]=fp
                    lineage.append({'inner':inner,'clinical':c,'source':str(previous),'metadata_sha256':sha(previous/'metadata.json'),'predictions_sha256':sha(previous/'predictions.csv')})
            assert all(np.isfinite(v).all() for v in scores.values())
            pd.DataFrame({'subject_id':train,**scores}).to_csv(folder/'inner_oof.csv',index=False)
            write_json(folder/'reused_clinical_lineage.json',lineage)
            eeg=pd.read_csv(OUT/'GNN_MLP'/f'fold{fold}.csv').set_index('subject_id').loc[test,'yhat'].to_numpy()
            for c,cols in clinical_sets(cfg).items():
                for label,key,base in [('ClinicalRidge','ridge_',baselines['Ridge_'+c]),('FT','ft_',baselines['FT_'+c])]:
                    name=f'GNN_{label}_Stack_{c}'
                    model=ridge(1.).fit(np.column_stack([scores['gnn'],scores[key+c]]),Y(train))
                    cp=base.set_index('subject').loc[test,'yhat'].to_numpy()
                    pred=model.predict(np.column_stack([eeg,cp]));joblib.dump(model,folder/(name+'.joblib'))
                    save_prediction(name,fold,test,pred,d);collect(d,splits,baselines)
                name='GNN_FT_Concat_'+c
                if not (OUT/name/f'fold{fold}.csv').exists():
                    predictions=[]
                    for seed in range(5):
                        tr,va=inner_indices(len(train),seed,.2);ti=[train[i] for i in tr];vi=[train[i] for i in va]
                        payload,history=fit_concat(E(ti),d.loc[ti,cols].to_numpy(float),Y(ti),E(vi),d.loc[vi,cols].to_numpy(float),Y(vi),cols,settings,seed)
                        payload['partitions']={'train':ti,'validation':vi,'test':test};payload['encoder_checkpoint_sha256']=sha(PRE/'checkpoint.pt')
                        torch.save(payload,folder/f'concat_{c}_member{seed}.pt')
                        pd.DataFrame(history).to_csv(folder/f'concat_{c}_member{seed}_history.csv',index=False)
                        predictions.append(predict_concat(payload,E(test),d.loc[test,cols].to_numpy(float)))
                    np.savez_compressed(folder/f'concat_{c}_member_predictions.npz',subject_id=np.array(test),predictions=np.stack(predictions))
                    save_prediction(name,fold,test,np.mean(predictions,axis=0),d);collect(d,splits,baselines)
        print(f'{stage}: outer fold {fold}/5 complete',flush=True)
    write_json(OUT/(stage+'_completed.json'),{'seconds':time.perf_counter()-started,'folds':5})
    print(collect(d,splits,baselines).to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['extract','eeg','fusion','report']);args=parser.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1):main(args.stage)
