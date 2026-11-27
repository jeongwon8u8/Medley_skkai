"""Verify every completed public recording cache and aggregate person statistics."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'medley_pd')]
from codl.eeg.external_graphs import sha


def main():
    store=ROOT/'artifacts/external_pd_hc_inventory'
    manifest=json.loads((store/'public_graph_manifest.json').read_text())
    assert manifest['status']=='completed' and not manifest['failures']
    candidates=[ROOT/'data/external_pd_hc_private'/name for name in ('pool_base_3cohorts_private.tsv','pool_gnn_brainlat_cl_private.tsv')]
    pool_path=next(p for p in candidates if p.exists() and sha(p)==manifest['pool_sha256'])
    assert sha(pool_path)==manifest['pool_sha256']
    pool=pd.read_csv(pool_path,sep='\t',dtype=str)
    expected=set(pool.loc[pool.dataset.ne('brainlat'),'person_uid'])
    rows=manifest['recordings']
    assert len(rows)==200 and {r['person_uid'] for r in rows}==expected and len(expected)==121
    assert len({(r['person_uid'],r['recording']) for r in rows})==len(rows)
    statistics=[];configs=set()
    for row in rows:
        path=ROOT/row['cache'];meta=json.loads(path.with_name('metadata.json').read_text())
        assert sha(path)==row['cache_sha256']==meta['npz_sha256']
        assert all(sha(p)==h for p,h in meta['identity']['files'].items())
        cfg=meta['identity']['cfg'];enc=cfg['encoder'];pp=cfg['preprocess']
        assert enc['fs']==200 and enc['seg_sec']==1 and enc['amp_div']==1
        assert pp['l_freq']==.3 and pp['h_freq']==45 and pp['rereference'] is None
        configs.add(json.dumps({'encoder':enc,'preprocess':pp,'graph':meta['identity']['graph']},sort_keys=True))
        with np.load(path) as data:
            n=len(data['x']);assert data['x'].shape==(n,19,200)
            assert data['adjacency'].shape==(n,4,19,19)
            assert n==meta['n_retained']==row['n_valid']
            assert np.isfinite(data['x']).all() and np.isfinite(data['adjacency']).all()
            assert len(np.unique(data['window_index']))==n
            assert data['window_index'].min()>=0 and data['window_index'].max()<meta['n_total']
            retained_flags=int(data['amplitude_flag'].sum())
        assert len(meta['windows'])==n and all(w['spectral_radius']<1 for w in meta['windows'])
        assert n+len(meta['failures'])==meta['n_total'] and meta['n_amplitude_excluded']==0
        statistics.append({'person_uid':row['person_uid'],'dataset':row['dataset'],
            'n_recordings':1,'n_total':meta['n_total'],'n_valid':n,
            'n_flagged':meta['n_amplitude_flagged'],'n_retained_flagged':retained_flags,
            'n_graph_failures':len(meta['failures']),'graph_seconds':meta['seconds'],
            'max_abs_microvolts':meta['max_abs_microvolts']})
    assert len(configs)==1
    d=pd.DataFrame(statistics)
    aggregate={k:'sum' for k in ('n_recordings','n_total','n_valid','n_flagged','n_retained_flagged','n_graph_failures','graph_seconds')}
    aggregate['max_abs_microvolts']='max'
    persons=d.groupby(['dataset','person_uid']).agg(aggregate).reset_index()
    persons['amplitude_flag_fraction']=persons.n_flagged/persons.n_total
    assert persons.n_valid.gt(0).all()
    persons.to_csv(store/'public_patient_graph_summary.csv',index=False)
    cohort=persons.groupby('dataset').agg(n_people=('person_uid','size'),
        n_recordings=('n_recordings','sum'),n_total=('n_total','sum'),n_valid=('n_valid','sum'),
        n_flagged=('n_flagged','sum'),n_graph_failures=('n_graph_failures','sum')).reset_index()
    cohort.to_csv(store/'public_cohort_graph_summary.csv',index=False)
    report={'status':'passed','n_people':len(persons),'n_recordings':len(rows),
            'cohorts':cohort.to_dict('records'),'graph_elapsed_seconds':manifest['seconds'],
            'source_and_cache_hashes':'verified','window_shape':[19,200],
            'preprocessing_and_graph_config':'same across public recordings; dataset line frequency differs',
            'trained':False,'brainlat':'not included; requires authorized EEG download and probe'}
    (store/'public_graph_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
