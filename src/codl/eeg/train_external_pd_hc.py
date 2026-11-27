"""External-only PD/HC training with explicit audited 3- or 4-cohort scope.

This entry point never imports the PD-EEG/MoCA cohort loader or uses its scores.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'medley_pd')]
from codl.eeg.directed_pdc import operator
from codl.eeg.external_graphs import sha
from codl.eeg.external_pd_hc import ExternalPDHC, load_checkpoint, person_metrics, save_checkpoint, split_people
from codl.eeg.finite_monitor import FiniteMonitor


def infer(model, signals, subjects):
    model.eval(); rows=[]; vectors=[]
    with torch.no_grad():
        for person in subjects:
            x,lap=signals[person]; probabilities=[]; embeddings=[]
            for start in range(0,len(x),64):
                z=model.encode(x[start:start+64],lap[start:start+64])
                probabilities.extend(model.classifier(z).squeeze(-1).sigmoid().tolist())
                embeddings.append(z.numpy())
            rows.append({'person_key':person,'probability':float(np.mean(probabilities)),
                         'n_windows':len(x)})
            vectors.append(np.concatenate(embeddings).mean(axis=0))
    return pd.DataFrame(rows),np.stack(vectors)


def fit(signals, people, settings, output, provenance):
    """Training engine; caller must establish source eligibility before calling."""
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    seed=settings['seed'];torch.manual_seed(seed);np.random.seed(seed)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(seed)
    train=people.loc[people.split.eq('train'),'person_key'].tolist()
    val=people.loc[people.split.eq('validation'),'person_key'].tolist()
    if set(train)&set(val) or set(train+val)!=set(signals):raise ValueError('Signal/partition mismatch')
    if set(people.split)!={'train','validation'}:raise ValueError('Unexpected partition')
    labels=people.set_index('person_key').label_pd.astype(float).to_dict()
    if len({labels[p] for p in val})!=2:raise ValueError('Validation requires both classes')
    model=ExternalPDHC(settings)
    optimizer=torch.optim.Adam(model.parameters(),lr=settings['lr'])
    monitor=FiniteMonitor(model.encoder)
    people.to_csv(output/'participants_and_split.tsv',sep='\t',index=False)
    (output/'settings.json').write_text(json.dumps(settings,indent=2))
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    history=[];best=float('inf');best_state=None;selected=0;t=time.perf_counter()
    for epoch in range(1,settings['epochs']+1):
        model.train();samples=[]
        for p in train:
            samples.extend((p,int(i)) for i in rng.integers(len(signals[p][0]),size=settings['windows_per_patient_per_epoch']))
        rng.shuffle(samples);loss_total=0.
        for start in range(0,len(samples),settings['batch_size']):
            batch=samples[start:start+settings['batch_size']]
            x=torch.stack([signals[p][0][i] for p,i in batch]);lap=torch.stack([signals[p][1][i] for p,i in batch])
            target=torch.tensor([labels[p] for p,_ in batch],dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits=model(x,lap)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,target)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError('Nonfinite gradient')
            optimizer.step();loss_total+=float(loss.detach())*len(batch)
        predictions,_=infer(model,signals,val)
        y=np.array([labels[p] for p in predictions.person_key]);p=predictions.probability.to_numpy().clip(1e-7,1-1e-7)
        validation_bce=float(-(y*np.log(p)+(1-y)*np.log1p(-p)).mean())
        history.append({'epoch':epoch,'train_window_bce':loss_total/len(samples),
                        'validation_person_bce':validation_bce,'elapsed_seconds':time.perf_counter()-t})
        pd.DataFrame(history).to_csv(output/'history.csv',index=False)
        if validation_bce<best:
            best=validation_bce;selected=epoch;best_state=copy.deepcopy(model.state_dict())
            torch.save({'model':best_state,'optimizer':optimizer.state_dict(),'epoch':epoch,
                        'settings':settings,'provenance':provenance},output/'best_resume.pt')
        print(json.dumps(history[-1]),flush=True)
        if epoch-selected>=settings['patience']:break
        if time.perf_counter()-t>=settings.get('time_budget_seconds',float('inf')):break
    training_seconds=time.perf_counter()-t
    model.load_state_dict(best_state);model.eval()
    (output/'numerics.json').write_text(json.dumps(monitor.report(),indent=2));monitor.close()
    predictions,vectors=infer(model,signals,val)
    predictions['label_pd']=[int(labels[p]) for p in predictions.person_key]
    _,metrics=person_metrics(predictions)
    checkpoint=output/'checkpoint.pt'
    provenance={**provenance,'task':'external_PDHC_pretraining','target':'label_pd',
                'train_people':train,'validation_people':val,'selected_epoch':selected,
                'training_seconds':training_seconds,'pretrained':False,'initialization':'random'}
    save_checkpoint(checkpoint,model,settings,provenance)
    # Also make head-free downstream loading explicit as two independent files.
    torch.save(model.encoder.state_dict(),output/'encoder.pt')
    torch.save(model.classifier.state_dict(),output/'classifier.pt')
    reloaded,payload=load_checkpoint(checkpoint)
    replay,replay_vectors=infer(reloaded,signals,val)
    np.testing.assert_allclose(replay.probability,predictions.probability,rtol=0,atol=0)
    np.testing.assert_allclose(replay_vectors,vectors,rtol=0,atol=0)
    predictions.to_csv(output/'validation_predictions.csv',index=False)
    if settings.get('save_all_external_vectors',True):
        all_people=train+val;_,all_vectors=infer(model,signals,all_people)
        np.savez_compressed(output/'eeg_vectors.npz',person_uid=np.array(all_people),vectors=all_vectors)
    summary={'status':'completed','selected_epoch':selected,'training_seconds':training_seconds,
             'n_train':len(train),'n_validation':len(val),'metrics':metrics,
             'checkpoint_sha256':sha(checkpoint),'reload_and_features':'exact_match',
             'model':'paper-based modified DirectedGNN; random initialization; external PD/HC only'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--recordings',required=True,type=Path)
    parser.add_argument('--three-cohort',action='store_true')
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('external_pd_hc_seed0.json'))
    args=parser.parse_args()
    store=ROOT/'artifacts/external_pd_hc_inventory'
    audit=json.loads((store/'pool_audit.json').read_text())
    if not args.three_cohort and (not audit.get('brainlat_14_exclusions_verified') or not audit.get('brainlat_moca_absence_verified')):
        raise ValueError('Complete source MoCA/exclusion audit required before training')
    pool_path=ROOT/'data/external_pd_hc_private'/('pool_base_3cohorts_private.tsv' if args.three_cohort else 'pool_gnn_brainlat_cl_private.tsv')
    if sha(pool_path)!=audit['files_sha256'][pool_path.name]:
        raise ValueError('Source roster/clinical records changed since audit')
    people=pd.read_csv(pool_path,sep='\t',dtype={'subject_uid':str,'source_subject_id':str})
    people=people.rename(columns={'person_uid':'person_key'})
    expected_n=121 if args.three_cohort else 163
    if len(people)!=expected_n or people.person_key.nunique()!=expected_n:raise ValueError('Wrong pool size')
    if args.three_cohort:
        actual=people.groupby(['dataset','label_pd']).size().to_dict()
        assert actual=={('turku',0):20,('turku',1):20,('unm',0):25,('unm',1):25,('ucsd',0):16,('ucsd',1):15},actual
    settings=json.loads(args.config.read_text())
    people=split_people(people,settings['seed'],settings['validation_fraction'])
    corpus=json.loads(args.recordings.read_text())
    rows=corpus['recordings']
    if len({(r['person_uid'],r['recording']) for r in rows})!=len(rows):raise ValueError('Duplicate recording entry')
    if corpus.get('failures'):raise ValueError('Unresolved recordings')
    if {r['person_uid'] for r in rows}!=set(people.person_key):raise ValueError('Missing or additional people in caches')
    signals={}
    for person in people.person_key:
        xs=[];laps=[]
        for row in [r for r in rows if r['person_uid']==person]:
            path=ROOT/row['cache']
            if sha(path)!=row['cache_sha256']:raise ValueError('Cache content mismatch')
            meta=json.loads(path.with_name('metadata.json').read_text())
            import medley.eeg
            import codl.eeg.external_graphs
            import codl.eeg.directed_pdc
            current_code={Path(module.__file__).name:sha(module.__file__) for module in
                          (medley.eeg,codl.eeg.external_graphs,codl.eeg.directed_pdc)}
            if meta['identity']['code']!=current_code:raise ValueError('Graph implementation changed; rebuild caches')
            if not all(sha(source)==digest for source,digest in meta['identity']['files'].items()):
                raise ValueError('Source EEG changed since graph generation')
            cfg=meta['identity']['cfg'];enc=cfg['encoder'];pp=cfg['preprocess']
            assert [pp['l_freq'],pp['h_freq']]==settings['filter']
            assert all(enc[k]==settings[k] for k in ('fs','seg_sec','amp_div','channels'))
            assert pp['rereference'] is None and pp['flag_uv']==100
            assert meta['identity']['graph']=={k:settings[k] for k in ('orders','mvar_ridge','graph_band','lambda_proxy')}
            with np.load(path) as z:
                xs.append(z['x']);laps.append(np.stack([operator(a[1]) for a in z['adjacency']]))
        x=np.concatenate(xs);lap=np.concatenate(laps).astype(np.float32)
        if x.shape[1:]!=(19,200) or not np.isfinite(x).all() or not np.isfinite(lap).all():raise ValueError('Invalid model inputs')
        signals[person]=(torch.tensor(x),torch.tensor(lap))
    settings={**settings,'status':'running'}
    provenance={'pool_sha256':sha(pool_path),'recordings_manifest_sha256':sha(args.recordings),
                'audit_sha256':sha(store/'pool_audit.json'),'source_recordings':rows,
                'recruitment_review_sha256':sha(store/'recruitment_review.json'),
                'code':{p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('external_pd_hc.py'),Path(__file__).with_name('directed_gnn.py'))}}
    print(json.dumps(fit(signals,people,settings,ROOT/'outputs/external_pd_hc'/settings['experiment_name'],provenance)))


if __name__=='__main__':main()
