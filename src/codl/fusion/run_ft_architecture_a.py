"""Train/replay Architecture A on verified external-121 frozen EEG vectors."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'medley_pd')]
from codl.clinical.run_pd_ft import load_reference, inner_indices, sha, write_json
from codl.fusion.pd_direct_concat import fit_concat, predict_concat
from codl.fusion.team_representation import SOURCE_COMMIT
from medley.participants import clinical_sets
from medley.metrics import regression_metrics

BASE = ROOT/'outputs/eeg_gnn/external_3cohort121_frozen_pd_eeg_moca_seed0'
PRE = ROOT/'outputs/external_pd_hc/external_3cohort121_pd_hc_directed_cheb_seed0'
OUT = ROOT/'outputs/fusion/external_3cohort121_frozen_gnn_ft_architecture_a_seed0'
FT = ROOT/'outputs/clinical_ft/clinical_ft_c0_c1_seed0_20261126'


def main(folds, audit_only=False):
    started = time.perf_counter()
    cfg, frame, splits, _, _ = load_reference()
    d = frame.set_index('subject', drop=False)
    settings = json.loads((ROOT/'src/codl/clinical/pd_ft_seed0.json').read_text())
    identity = json.loads((BASE/'features/identity.json').read_text())
    summary = json.loads((PRE/'summary.json').read_text())
    assert identity['external_people'] == 121 and identity['dimension'] == 256
    assert identity['checkpoint_sha256'] == summary['checkpoint_sha256'] == sha(PRE/'checkpoint.pt')
    assert identity['checkpoint_sha256'] == '7815522dd24d183ca6d8e1672768f3eb7cfe6277cc15675565284733aea5dae4'
    completed = json.loads((BASE/'features/completed.json').read_text())
    assert completed['encoder_state_unchanged'] and completed['BatchNorm_unchanged']
    with np.load(BASE/'patient_vectors.npz') as z:
        ids, vectors = z['subject_id'], z['vectors']
        assert len(set(ids)) == 113 and set(ids) == set(d.index)
        assert vectors.shape == (113,256) and np.isfinite(vectors).all()
        mapping = dict(zip(ids, vectors))
    for sid, vector in mapping.items():
        with np.load(BASE/'features'/f'{sid}.npz') as z:
            np.testing.assert_array_equal(vector, z['vector'])
            assert z['source_sha256'].item() == sha(Path(identity['source_cache'])/f'{sid}.npz')
    E = lambda subjects: np.stack([mapping[s] for s in subjects])
    Y = lambda subjects: d.loc[subjects,'target'].to_numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {'experiment_name': OUT.name, 'settings': settings, 'split_seed': 0,
        'member_seeds': list(range(5)), 'team_source_commit': SOURCE_COMMIT,
        'encoder_sha256': identity['checkpoint_sha256'], 'vector_sha256': sha(BASE/'patient_vectors.npz'),
        'split_sha256': sha(ROOT/'medley_pd/splits/pd_eeg__moca.csv', True),
        'architecture': 'EEG256 -> Linear256; FT8 flattened16/40 -> Linear256; concat512 -> LN -> Linear256 -> GELU -> Dropout0.1 -> Linear1',
        'encoder': 'external three-cohort121 PD/HC frozen eval; BatchNorm unchanged; window mean',
        'preprocessing': 'clinical, EEG scaler and target standardization fit inner train only',
        'clinical_pretraining': False}
    if (OUT/'protocol.json').exists():
        assert json.loads((OUT/'protocol.json').read_text()) == protocol
    else:
        write_json(OUT/'protocol.json', protocol)
    for fold in folds:
        test = d.index[d.index.isin(splits.loc[splits.fold.eq(fold),'subject'])].tolist()
        train = d.index[~d.index.isin(test)].tolist()
        for c, cols in clinical_sets(cfg).items():
            path = OUT/c/f'fold{fold}'; path.mkdir(parents=True, exist_ok=True)
            predictions = []
            for seed in range(5):
                tr, va = inner_indices(len(train),seed,settings['nn_val_frac'])
                ti, vi = [train[i] for i in tr], [train[i] for i in va]
                groups = {'train':ti, 'validation':vi, 'test':test}
                assert not (set(ti)&set(vi) or set(ti)&set(test) or set(vi)&set(test))
                target = path/f'member{seed}.pt'
                if not target.exists():
                    assert not audit_only, 'Missing checkpoint'
                    payload, history = fit_concat(E(ti),d.loc[ti,cols].to_numpy(float),Y(ti),
                        E(vi),d.loc[vi,cols].to_numpy(float),Y(vi),cols,settings,seed,'architecture_a')
                    payload.update(partitions=groups, encoder_checkpoint_sha256=identity['checkpoint_sha256'],
                                   vector_sha256=protocol['vector_sha256'])
                    before = predict_concat(payload,E(test),d.loc[test,cols].to_numpy(float))
                    torch.save(payload,target)
                    pd.DataFrame(history).to_csv(path/f'member{seed}_history.csv',index=False)
                else:
                    before = None
                payload = torch.load(target,map_location='cpu',weights_only=True)
                assert payload['partitions'] == groups and payload['settings'] == settings
                assert payload['columns'] == cols and payload['seed'] == seed
                assert payload['architecture'] == 'architecture_a'
                assert payload['encoder_checkpoint_sha256'] == identity['checkpoint_sha256']
                assert payload['vector_sha256'] == protocol['vector_sha256']
                assert payload['forward_gradient_probe']['fusion_shape'][1] == 512
                np.testing.assert_allclose(payload['target_mean'],Y(ti).mean())
                np.testing.assert_allclose(payload['target_std'],Y(ti).std())
                np.testing.assert_allclose(payload['eeg_mean'],E(ti).mean(axis=0),atol=1e-5)
                from codl.clinical.ft_clinical import ClinicalPrep, prep_state
                assert payload['prep'] == prep_state(ClinicalPrep().fit(d.loc[ti,cols].to_numpy(float)))
                history = pd.read_csv(path/f'member{seed}_history.csv')
                best, selected = float('inf'), 0
                for item in history.itertuples():
                    if item.val_huber < best-settings['min_delta']:
                        best, selected = item.val_huber, item.epoch
                assert payload['selected_epoch'] == selected
                pred = predict_concat(payload,E(test),d.loc[test,cols].to_numpy(float))
                if before is not None: np.testing.assert_array_equal(pred,before)
                predictions.append(pred)
            pred = np.mean(predictions,axis=0).astype(np.float64)
            result = pd.DataFrame({'subject_id':test,'fold':fold,'y':Y(test),'yhat':pred})
            dest = path/'predictions.csv'
            if dest.exists(): np.testing.assert_allclose(pd.read_csv(dest).yhat,pred,rtol=0,atol=1e-12)
            else: result.to_csv(dest,index=False)
            np.savez_compressed(path/'member_predictions.npz',predictions=np.stack(predictions))
            print(f'{c} fold {fold}: 5 checkpoints reloaded, predictions verified',flush=True)
    rows=[]; fold_rows=[]
    ft = pd.read_csv(FT/'predictions.csv.gz')
    for c in ('C0','C1'):
        paths = sorted((OUT/c).glob('fold*/predictions.csv'))
        p = pd.concat([pd.read_csv(f) for f in paths],ignore_index=True)
        assert not p.subject_id.duplicated().any()
        np.testing.assert_array_equal(p.y,Y(p.subject_id))
        np.testing.assert_array_equal(p.fold,splits.set_index('subject').loc[p.subject_id,'fold'])
        if len(p) != 113: continue
        p.to_csv(OUT/c/'oof.csv',index=False)
        refs = {'GNN_MLP':pd.read_csv(BASE/'GNN_MLP/predictions.csv.gz'),
                'FT_only':ft[ft.model.eq('clinical_ft:'+c)].rename(columns={'subject':'subject_id'}),
                'raw_Concat':pd.read_csv(BASE/f'GNN_FT_Concat_{c}/predictions.csv.gz')}
        metrics = regression_metrics(p.y,p.yhat)
        row = {'model':'Architecture_A_'+c,'n':len(p),**metrics}
        for name, ref in refs.items():
            assert len(ref)==113 and ref.subject_id.nunique()==113
            matched = ref.set_index('subject_id').loc[p.subject_id]
            np.testing.assert_array_equal(matched.y,p.y)
            np.testing.assert_array_equal(matched.fold,p.fold)
            row['delta_MAE_vs_'+name] = metrics['MAE']-regression_metrics(matched.y,matched.yhat)['MAE']
        rows.append(row)
        for f, group in p.groupby('fold'):
            epochs=[torch.load(OUT/c/f'fold{f}'/f'member{s}.pt',weights_only=True)['selected_epoch'] for s in range(5)]
            fold_rows.append({'model':c,'fold':int(f),'n':len(group),**regression_metrics(group.y,group.yhat),
                              'selected_epochs':epochs})
    pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(OUT/'fold_metrics.csv',index=False)
    if len(rows)==2:
        payloads=[torch.load(p,weights_only=True) for p in OUT.glob('C*/fold*/member*.pt')]
        write_json(OUT/'audit.json',{'status':'completed','people_per_model':113,'outer_folds':5,
            'reloaded_checkpoints':len(payloads),'oof_per_person':1,'lineage_verified':True,
            'training_seconds':sum(p['seconds'] for p in payloads),'audit_seconds':time.perf_counter()-started,
            'inner_train_only_statistics':True,'projection_shapes_and_gradients_verified':True})
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--folds',nargs='+',type=int,default=[1,2,3,4,5],choices=range(1,6))
    parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--output',type=Path,default=OUT,
                        help='Use a new directory for a fresh independent rerun')
    args=parser.parse_args()
    OUT=args.output.resolve()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1):main(args.folds,args.audit_only)
