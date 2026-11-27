"""Fetch only fixed-cohort resting EEG; verify v1.0.2 git-annex content hashes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import xml.etree.ElementTree as ET

import requests

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from codl.clinical.run_pd_ft import load_reference, inner_indices

STORE = ROOT / 'artifacts/eeg_inventory'
BASE = 'https://s3.amazonaws.com/openneuro.org'


def get(url, **kwargs):
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=120, **kwargs)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def inventory():
    STORE.mkdir(parents=True, exist_ok=True)
    cfg, d, splits, _, _ = load_reference()
    found, token = {}, None
    ns = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    while True:
        params = {'list-type': '2', 'prefix': 'ds007526/', 'max-keys': 1000}
        if token:
            params['continuation-token'] = token
        root = ET.fromstring(get(BASE, params=params).content)
        for item in root.findall('s:Contents', ns):
            found[item.find('s:Key', ns).text] = int(item.find('s:Size', ns).text)
        token = root.findtext('s:NextContinuationToken', None, ns)
        if not token:
            break
    def entry(sid):
        relative = f'{sid}/eeg/{sid}_task-rest_eeg.set'
        pointer = get(f'https://raw.githubusercontent.com/OpenNeuroDatasets/ds007526/1.0.2/{relative}').text
        match = re.search(r'(SHA256E|MD5E)-s(\d+)--([0-9a-f]+)\.set', pointer)
        if not match:
            raise ValueError('Unsupported annex pointer')
        size = int(match[2])
        if found.get('ds007526/' + relative) != size:
            raise ValueError('S3 and version-pinned size differ')
        return {'subject': sid, 'relative': relative, 'size': size,
                'algorithm': 'sha256' if match[1] == 'SHA256E' else 'md5', 'digest': match[3]}
    with ThreadPoolExecutor(max_workers=6) as pool:
        items = list(pool.map(entry, d.subject.tolist()))
    outer = d.loc[~d.subject.isin(splits.loc[splits.fold == 1, 'subject'])].reset_index(drop=True)
    tr, _ = inner_indices(len(outer), 0, .2)
    sample = outer.loc[tr[0], 'subject']
    result = {'dataset_version': '1.0.2', 'n': len(items), 'bytes': sum(x['size'] for x in items),
              'sample_subject': sample, 'items': items}
    (STORE / 'rest_manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'n': len(items), 'bytes': result['bytes'], 'GiB': result['bytes'] / 2**30,
                      'sample_MiB': next(x['size'] for x in items if x['subject'] == sample) / 2**20}))


def download(sample):
    manifest = json.loads((STORE / 'rest_manifest.json').read_text())
    items = manifest['items']
    if sample:
        items = [x for x in items if x['subject'] == manifest['sample_subject']]
    def one(item):
        path = ROOT / 'data/ds007526' / item['relative']
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if hashlib.new(item['algorithm'], path.read_bytes()).hexdigest() != item['digest']:
                raise ValueError('Existing source hash mismatch')
            return
        data = get(BASE + '/ds007526/' + item['relative']).content
        if len(data) != item['size'] or hashlib.new(item['algorithm'], data).hexdigest() != item['digest']:
            raise ValueError('Downloaded file does not match v1.0.2')
        path.write_bytes(data)
    t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, _ in enumerate(pool.map(one, items), 1):
            if i % 10 == 0 or i == len(items):
                print(f'Verified EEG files: {i}/{len(items)}', flush=True)
    print(f'Download/check seconds: {time.perf_counter()-t:.2f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['inventory', 'sample', 'all'])
    args = ap.parse_args()
    inventory() if args.stage == 'inventory' else download(args.stage == 'sample')
