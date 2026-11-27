"""Build reviewed public-source recording graphs; no BrainLat substitution."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'medley_pd')]
from codl.eeg.external_graphs import build_recording, sha


def one_recording(job):
    item, dependencies = job
    settings = json.loads(Path(__file__).with_name('external_pd_hc_seed0.json').read_text())
    common = runpy.run_path(str(ROOT/'medley_pd/configs/config.py'))['COMMON']
    ds = item['dataset']
    cfg = {'preprocess': common['preprocess'].copy(),
           'dataset': {'name': ds, 'line_freq': 50 if ds=='turku' else 60, 'ref_aliases': []},
           'encoder': {k: settings[k] for k in ('fs', 'seg_sec', 'amp_div', 'channels')}}
    cfg['encoder']['name'] = 'external_pd_hc_directed_cheb'
    assert [cfg['preprocess']['l_freq'],cfg['preprocess']['h_freq']] == settings['filter']
    assert cfg['preprocess']['rereference'] is None
    folder = ROOT/'data/external_pd_hc_public'/ds
    path = folder/item['relative']
    source = {'version':item['version'], 'native_reference':'CPz' if ds=='unm' else 'unestablished; preserved',
              'native_filter':'source raw; no PREP derivative', 'source_digest':item['digest'],
              'source_algorithm':item['algorithm']}
    import hashlib
    for entry in [item]+dependencies:
        p = folder/entry['relative']
        if p.stat().st_size!=entry['size'] or hashlib.new(entry['algorithm'],p.read_bytes()).hexdigest()!=entry['digest']:
            raise ValueError('Raw version/hash mismatch')
    out, meta = build_recording(path, [folder/d['relative'] for d in dependencies], cfg, settings,
                                ROOT/'artifacts/external_pd_hc_graphs', source)
    return {'person_uid':item['person_uid'], 'dataset':ds, 'recording':item['relative'],
            'cache':str(out.relative_to(ROOT)), 'cache_sha256':meta['npz_sha256'],
            'n_total':meta['n_total'], 'n_valid':meta['n_retained'],
            'n_flagged':meta['n_amplitude_flagged'], 'seconds':meta['seconds']}


def main():
    store = ROOT/'artifacts/external_pd_hc_inventory'
    source = store/'public_download_manifest.json'
    manifest = json.loads(source.read_text()); items = manifest['items']
    jobs=[]
    for item in items:
        if not item['relative'].endswith(('.set','.bdf')):
            continue
        deps=[d for d in items if d['dataset']==item['dataset'] and
              d['relative']==str(Path(item['relative']).with_suffix('.fdt')).replace('\\','/')]
        jobs.append((item,deps))
    t=time.perf_counter(); rows=[]; failed=[]
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures=[executor.submit(one_recording,job) for job in jobs]
        for job,future in zip(jobs,futures):
            try:
                rows.append(future.result())
            except Exception as error:
                failed.append({'person_uid':job[0]['person_uid'],'recording':job[0]['relative'],'error':str(error)})
            (store/'public_graph_progress.json').write_text(json.dumps({'rows':rows,'failures':failed},indent=2))
            if (len(rows)+len(failed))%10==0:print(f'Graph recordings {len(rows)}/{len(jobs)}; failed {len(failed)}',flush=True)
    output={'recordings':rows,'failures':failed,'seconds':time.perf_counter()-t,
            'download_manifest_sha256':sha(source),'pool_sha256':manifest['pool_sha256'],
            'status':'completed' if not failed else 'incomplete'}
    (store/'public_graph_manifest.json').write_text(json.dumps(output,indent=2))
    print(json.dumps({k:v for k,v in output.items() if k not in ('recordings','failures')}),flush=True)
    if failed:raise RuntimeError('Graph failures require review before training')


if __name__ == '__main__':
    main()
