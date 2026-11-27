"""Per-recording external EEG cache. Never consumes labels or cohort statistics."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from medley.eeg import preprocess, read_raw
from codl.eeg import directed_pdc


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def build_recording(path, dependencies, cfg, settings, cache_root, source_metadata):
    """Dependencies must include sidecar signal files (.fdt/.eeg) when present.

    source_metadata records version, native filters/reference, and units; never
    labels or clinical metadata. Caller is responsible for source provenance.
    """
    path = Path(path).resolve()
    files = [path] + [Path(p).resolve() for p in dependencies]
    if len(set(files)) != len(files):
        raise ValueError('Duplicate source dependency')
    from medley import eeg
    identity = {
        'schema': 'external_recording_v1', 'cfg': cfg,
        'graph': {k: settings[k] for k in ('orders', 'mvar_ridge', 'graph_band', 'lambda_proxy')},
        'source': source_metadata,
        'files': {str(p): sha(p) for p in files},
        'code': {Path(p).name: sha(p) for p in (__file__, eeg.__file__, directed_pdc.__file__)},
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    folder = Path(cache_root) / key
    output = folder / 'graphs.npz'
    meta_path = folder / 'metadata.json'
    if output.exists() or meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta['identity'] != identity or meta['npz_sha256'] != sha(output):
            raise ValueError('Cache identity/content mismatch')
        return output, meta
    t = time.perf_counter()
    raw = read_raw(str(path))
    raw_info = {'channels': raw.ch_names, 'fs': float(raw.info['sfreq']),
                'max_abs_volts': float(np.max(np.abs(raw.get_data(picks='eeg')))),
                'loader_units': 'MNE volts; common preprocessing multiplies by 1e6'}
    del raw
    x, flags, info = preprocess(str(path), cfg)
    if x.shape[1:] != (19, 200) or not np.isfinite(x).all():
        raise ValueError('Expected finite [windows,19,200]')
    indices, graphs, diagnostics, failed = [], [], [], []
    with threadpool_limits(limits=1):
        for i, w in enumerate(x):
            try:
                a, diag = directed_pdc.make_graph(w, settings['orders'], settings['mvar_ridge'])
            except (ValueError, np.linalg.LinAlgError) as error:
                failed.append({'window': i, 'error': str(error)})
                continue
            indices.append(i); graphs.append(a); diagnostics.append(diag)
    if not indices:
        raise ValueError('No stable graph windows')
    folder.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output, x=x[indices], adjacency=np.asarray(graphs),
                        window_index=np.asarray(indices), amplitude_flag=flags[indices])
    first = x[indices[0]].astype(float)
    centered = first - first.mean(axis=1, keepdims=True)
    meta = {
        'identity': identity, 'npz_sha256': sha(output), 'raw': raw_info,
        'preprocessing': info, 'input_shape': list(x.shape), 'n_total': len(x),
        'n_amplitude_flagged': int(flags.sum()), 'amplitude_flag_fraction': float(flags.mean()),
        'n_retained': len(indices), 'n_amplitude_excluded': 0,
        'max_abs_microvolts': float(np.max(np.abs(x))),
        'probe_rank': int(np.linalg.matrix_rank(centered)),
        'probe_CAR_rank': int(np.linalg.matrix_rank(centered-centered.mean(axis=0, keepdims=True))),
        'windows': diagnostics, 'failures': failed, 'seconds': time.perf_counter()-t,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return output, meta
