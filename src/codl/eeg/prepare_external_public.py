"""Inventory/download only supplied-pool public EEG; pinned hashes, no labels fitted."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[3]
STORE = ROOT/'artifacts/external_pd_hc_inventory'


def get(url):
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(attempt+1)


def inventory():
    pool_path = ROOT/'data/external_pd_hc_private/pool_base_3cohorts_private.tsv'
    pool = pd.read_csv(pool_path, sep='\t', dtype=str)
    STORE.mkdir(parents=True,exist_ok=True)
    items = []
    for cohort, ds, version in [('unm', 'ds003490', '1.1.0'), ('ucsd', 'ds002778', '1.0.5')]:
        people = pool[pool.dataset.eq(cohort)].set_index('source_subject_id').person_uid.to_dict()
        tree_path = STORE/f'{ds}_{version}_tree.json'
        if not tree_path.exists():
            tree_path.write_text(json.dumps(get(f'https://api.github.com/repos/OpenNeuroDatasets/{ds}/git/trees/{version}?recursive=1').json()))
        tree = json.loads(tree_path.read_text())
        if tree.get('truncated'):
            raise ValueError('Truncated source tree')
        paths = [x['path'] for x in tree['tree'] if x['path'].split('/')[0] in people
                 and x['path'].endswith(('.set', '.fdt', '.bdf'))]
        def one(relative):
            pointer = get(f'https://raw.githubusercontent.com/OpenNeuroDatasets/{ds}/{version}/{relative}').text
            m = re.search(r'(SHA256E|MD5E)-s(\d+)--([0-9a-f]+)\.', pointer)
            if not m:
                raise ValueError('Unrecognized annex pointer')
            return {'dataset': cohort, 'source': ds, 'version': version,
                    'person_uid': people[relative.split('/')[0]], 'relative': relative,
                    'size': int(m[2]), 'algorithm': 'sha256' if m[1]=='SHA256E' else 'md5',
                    'digest': m[3], 'url': f'https://s3.amazonaws.com/openneuro.org/{ds}/{relative}'}
        with ThreadPoolExecutor(max_workers=5) as executor:
            items.extend(executor.map(one, paths))
    url = 'https://api.osf.io/v2/nodes/pehj9/files/osfstorage/5ead625a62d4ab004d6cec1c/'
    osf = []
    while url:
        page = get(url).json(); osf.extend(page['data']); url = page['links'].get('next')
    (STORE/'osf_all_data_files.json').write_text(json.dumps(osf, indent=2))
    people = pool[pool.dataset.eq('turku')].set_index('source_subject_id').person_uid.to_dict()
    # Compare original strings first; numeric normalization is recorded only for
    # OSF filename zero-padding, with a bijection check to avoid collisions.
    normalized = {str(int(k)): v for k,v in people.items()}
    if len(normalized) != len(people):
        raise ValueError('Ambiguous Turku numeric source IDs')
    for x in osf:
        a = x['attributes']; match = re.fullmatch(r'(\d+)_(eyesOpen|eyesClosed)\.set', a['name'])
        if not match or str(int(match[1])) not in normalized:
            continue  # PREP derivatives are not additional independent recordings.
        items.append({'dataset': 'turku', 'source': 'pehj9', 'version': 'OSF hash-pinned',
                      'person_uid': normalized[str(int(match[1]))], 'relative': a['name'],
                      'size': a['size'], 'algorithm': 'sha256', 'digest': a['extra']['hashes']['sha256'],
                      'url': x['links']['download']})
    expected = set(pool.loc[pool.dataset.ne('brainlat'), 'person_uid'])
    if {x['person_uid'] for x in items} != expected:
        raise ValueError('Public inventory does not cover exactly the 121 supplied people')
    summary = {ds: {'people': len({x['person_uid'] for x in items if x['dataset']==ds}),
                    'files': sum(x['dataset']==ds for x in items),
                    'bytes': sum(x['size'] for x in items if x['dataset']==ds)}
               for ds in ('turku','unm','ucsd')}
    result = {'pool_sha256': hashlib.sha256(pool_path.read_bytes()).hexdigest(),
              'items': items, 'summary': summary, 'policy': 'all raw listed rest sessions; no PREP duplicates'}
    (STORE/'public_download_manifest.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(summary), flush=True)


def download():
    manifest = json.loads((STORE/'public_download_manifest.json').read_text())
    def one(item):
        dest = ROOT/'data/external_pd_hc_public'/item['dataset']/item['relative']
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            # Reuse already verified technical probes when they are the exact source.
            ds = {'unm':'ds003490', 'ucsd':'ds002778', 'turku':'turku'}[item['dataset']]
            probe = ROOT/'data/external_pd_hc_probe'/ds/item['relative']
            data = probe.read_bytes() if probe.exists() else get(item['url']).content
            if len(data)!=item['size'] or hashlib.new(item['algorithm'], data).hexdigest()!=item['digest']:
                raise ValueError('Downloaded signal hash/size mismatch')
            partial = dest.with_name(dest.name+'.partial')
            partial.write_bytes(data); partial.replace(dest)
        if dest.stat().st_size!=item['size'] or hashlib.new(item['algorithm'],dest.read_bytes()).hexdigest()!=item['digest']:
            raise ValueError('Existing signal hash/size mismatch')
        return item['size']
    t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures=[executor.submit(one,item) for item in manifest['items']]
        for i,future in enumerate(as_completed(futures),1):
            future.result()
            (STORE/'public_download_progress.json').write_text(json.dumps({
                'verified_files':i,'total_files':len(futures),'elapsed_seconds':time.perf_counter()-t}))
            if i%10==0 or i==len(manifest['items']): print(f'Verified files {i}/{len(manifest["items"])}',flush=True)
    (STORE/'public_download_completed.json').write_text(json.dumps({
        'files': len(manifest['items']), 'people':121, 'seconds':time.perf_counter()-t,
        'manifest_sha256':hashlib.sha256((STORE/'public_download_manifest.json').read_bytes()).hexdigest()},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['inventory','download'])
    args=parser.parse_args()
    inventory() if args.stage=='inventory' else download()
