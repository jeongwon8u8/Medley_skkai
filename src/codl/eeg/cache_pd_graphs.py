"""Private deterministic per-window graph cache and training-patient probe."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from codl.clinical.run_pd_ft import load_reference, write_json, sha
from codl.eeg.directed_pdc import design, fit_var, radius, make_graph, synthetic_direction
from medley.eeg import preprocess

SETTINGS = Path(__file__).with_name('gnn_config.json')


def context():
    settings = json.loads(SETTINGS.read_text())
    cfg, d, splits, _, meta = load_reference()
    cfg['encoder'] = {k: settings[k] for k in ('fs', 'seg_sec', 'amp_div', 'channels')}
    cfg['encoder']['name'] = 'directed_cheb_pilot'
    # Explicitly use shared preprocessing, without LEAD overrides.
    assert cfg['preprocess']['rereference'] is None
    assert [cfg['preprocess']['l_freq'], cfg['preprocess']['h_freq']] == settings['filter']
    identity = {'settings': settings, 'preprocess': cfg['preprocess'], 'dataset': cfg['dataset'],
                'code': {str(p.relative_to(ROOT)): sha(p) for p in
                         (Path(__file__), Path(__file__).with_name('directed_pdc.py'), ROOT/'medley_pd/medley/eeg.py')}}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    cache = ROOT/'artifacts/gnn_cache'/digest
    cache.mkdir(parents=True, exist_ok=True)
    write_json(cache/'config.json', identity)
    manifest = json.loads((ROOT/'artifacts/eeg_inventory/rest_manifest.json').read_text())
    return settings, cfg, d, splits, cache, manifest


def build(item, cfg, settings, cache):
    out = cache/(item['subject']+'.npz')
    if out.exists():
        return json.loads(out.with_suffix('.json').read_text())
    t = time.perf_counter()
    path = ROOT/'data/ds007526'/item['relative']
    if hashlib.new(item['algorithm'], path.read_bytes()).hexdigest() != item['digest']:
        raise ValueError('Raw EEG version mismatch')
    x, flags, info = preprocess(str(path), cfg)
    assert x.shape[1:] == (19, 200) and np.isfinite(x).all()
    indices, graphs, diagnostics, failed = [], [], [], []
    for idx in range(len(x)):
        try:
            a, diag = make_graph(x[idx], settings['orders'], settings['mvar_ridge'])
            indices.append(int(idx)); graphs.append(a); diagnostics.append(diag)
        except (ValueError, np.linalg.LinAlgError) as e:
            failed.append({'window': int(idx), 'error': str(e)})
    if not indices:
        raise ValueError('No valid graph windows for patient')
    np.savez_compressed(out, x=x[indices], adjacency=np.array(graphs), window_index=indices,
                        amplitude_flag=flags[indices])
    result = {'preprocessing': info, 'n_total': len(x), 'n_artifact': int(flags.sum()),
              'n_amplitude_excluded': 0, 'n_valid': len(indices), 'failures': failed, 'windows': diagnostics,
              'source_digest': item['digest'], 'seconds': time.perf_counter()-t}
    write_json(out.with_suffix('.json'), result)
    return result


def probe(cfg, settings, cache, manifest):
    item = next(i for i in manifest['items'] if i['subject'] == manifest['sample_subject'])
    x, flags, info = preprocess(str(ROOT/'data/ds007526'/item['relative']), cfg)
    valid = x[~flags]
    if not len(valid):
        raise ValueError('Probe has no artifact-free windows')
    w = valid[0].astype(float)
    centered = w-w.mean(axis=1, keepdims=True)
    car = centered-centered.mean(axis=0, keepdims=True)
    orders = []
    for p in range(1, 11):
        z, y = design(centered, p)
        coef, diag = fit_var(w, p, settings['mvar_ridge'])
        orders.append({**diag, 'design_rank': int(np.linalg.matrix_rank(z, tol=1e-5)),
                       'design_condition': float(np.linalg.cond(z)),
                       'nominal_residual_df_without_intercept': len(y)-19*p,
                       'spectral_radius': radius(coef), 'finite': bool(np.isfinite(coef).all())})
    result = {'shape': list(w.shape), 'original_rank': int(np.linalg.matrix_rank(centered, tol=1e-5)),
              'CAR_rank': int(np.linalg.matrix_rank(car, tol=1e-5)), 'preprocessing': info,
              'n_valid_amplitude': len(valid), 'orders': orders, 'synthetic': synthetic_direction(),
              'role': 'outer fold 1 inner training, seed 0'}
    write_json(cache/'probe.json', result)
    built = build(item, cfg, settings, cache)
    print(json.dumps({**{k:v for k,v in result.items() if k not in ('orders','preprocessing')},
                      'graphs': built['n_valid'], 'failed': len(built['failures']), 'seconds': built['seconds']}), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['probe','all']); args = ap.parse_args()
    settings, cfg, d, splits, cache, manifest = context()
    with threadpool_limits(limits=1):
        if args.stage == 'probe':
            probe(cfg, settings, cache, manifest)
        else:
            if not (cache/'probe.json').exists():
                raise ValueError('Run training-patient probe first')
            results = []
            for i, item in enumerate(manifest['items'], 1):
                results.append(build(item, cfg, settings, cache))
                if i%5 == 0 or i == len(manifest['items']):
                    print(f'Graph cache {i}/{len(manifest["items"])}', flush=True)
            write_json(cache/'summary.json', {'n_patients': len(results),
                       'n_windows': sum(r['n_valid'] for r in results),
                       'n_artifact': sum(r['n_artifact'] for r in results),
                       'n_failed': sum(len(r['failures']) for r in results),
                       'graph_preprocess_seconds': sum(r['seconds'] for r in results)})
    print('Cache:', cache, flush=True)


if __name__ == '__main__':
    main()
